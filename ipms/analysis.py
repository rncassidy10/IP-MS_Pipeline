"""
Core analysis functions for IP-MS pipeline
"""

import pandas as pd
import numpy as np
import yaml
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import sys 















def prep_ip(config_path):
    """
    Load and prepare IP-MS data for analysis.
    
    This function:
    1. Loads the YAML configuration file
    2. Reads the proteomics Excel file
    3. Identifies and organizes intensity columns by condition
    4. Filters by minimum peptide count
    5. Removes low-quality proteins (detected in too few samples)
    6. Maps gene symbols using mygene
    7. Removes contaminants using CRAPome database
    8. Creates output directory structure
    
    Parameters:
    -----------
    config_path : str
        Path to YAML configuration file
        
    Returns:
    --------
    dict containing:
        'df' : pd.DataFrame
            Cleaned protein data with intensity values
        'config' : dict
            Loaded configuration dictionary
        'intensity_cols' : dict
            Maps condition names to their intensity column names
            e.g., {'EV': ['LFQ intensity EV_1', 'LFQ intensity EV_2', ...], 
                   'WT': [...], ...}
        'metadata' : dict
            Summary statistics about the data
    
    Example:
    --------
    >>> data = prep_ip('config/experiment.yaml')
    >>> df = data['df']
    >>> print(f"Loaded {len(df)} proteins")
    >>> print(f"Conditions: {list(data['intensity_cols'].keys())}")
    """
    
    # =========================================================================
    # 1. LOAD CONFIGURATION
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 1: LOADING DATA AND CONFIGURATION")
    print("="*80)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"\n✓ Configuration loaded")
    print(f"  Experiment: {config['experiment']['name']}")
    print(f"  Control: {config['conditions']['control']}")
    print(f"  Treatments: {', '.join(config['conditions']['treatments'])}")
    
    # =========================================================================
    # 2. LOAD PROTEOMICS DATA
    # =========================================================================
    print(f"\n[1/8] Loading proteomics data...")
    
    input_file = config['data_paths']['input_file']
    df = pd.read_excel(input_file)
    
    initial_protein_count = len(df)
    print(f"  ✓ Loaded {df.shape[0]} proteins, {df.shape[1]} columns")
    
    # =========================================================================
    # 3. IDENTIFY INTENSITY COLUMNS
    # =========================================================================
    print(f"\n[2/8] Identifying intensity columns...")
    
    intensity_prefix = config['data_columns']['intensity_prefix']
    
    # Create dict to store columns for each condition
    intensity_cols = {}
    
    # Get control columns
    control = config['conditions']['control']
    cols = [c for c in df.columns if intensity_prefix in c and control in c]
    intensity_cols[control] = sorted(cols)
    print(f"  {control}: {len(cols)} replicates")
    
    # Get treatment columns
    for treatment in config['conditions']['treatments']:
        cols = [c for c in df.columns if intensity_prefix in c and treatment in c]
        intensity_cols[treatment] = sorted(cols)
        print(f"  {treatment}: {len(cols)} replicates")
    
    # Flatten all intensity columns for easier access
    all_intensity_cols = []
    for cols in intensity_cols.values():
        all_intensity_cols.extend(cols)
    
    print(f"  ✓ Total intensity columns: {len(all_intensity_cols)}")
    
    # =========================================================================
    # 4. FILTER BY MINIMUM PEPTIDES
    # =========================================================================
    print(f"\n[3/8] Filtering by minimum peptides...")

    peptide_col = config['data_columns']['peptides']
    min_peptides = config['qc_parameters']['min_peptides']

    if peptide_col in df.columns:
        before = len(df)
        df = df[df[peptide_col] >= min_peptides].copy()
        removed = before - len(df)
        print(f"  ✓ Removed {removed} proteins with < {min_peptides} peptides")
        print(f"    Remaining: {len(df)} proteins")
    else:
        print(f"  ⚠ Warning: Column '{peptide_col}' not found, skipping peptide filter")

    # =========================================================================
    # 5. FILTER LOW QUALITY PROTEINS (BY MISSINGNESS IN TREATMENTS)
    # =========================================================================
    print(f"\n[4/8] Filtering proteins by missingness in treatments...")

    before = len(df)

    # Get treatment conditions (exclude control)
    treatment_conditions = config['conditions']['treatments']
    control_condition = config['conditions']['control']

    # Calculate valid count for each treatment condition
    keep_protein = pd.Series(False, index=df.index)

    for treatment in treatment_conditions:
        treatment_cols = intensity_cols[treatment]
        n_replicates = len(treatment_cols)
        min_required = int(np.ceil(n_replicates * 0.5))  # 50% of replicates
        
        # Count valid values for this condition
        valid_count = df[treatment_cols].notna().sum(axis=1)
        
        # Mark proteins that pass threshold for this condition
        passes_threshold = valid_count >= min_required
        keep_protein = keep_protein | passes_threshold
        
        n_passing = passes_threshold.sum()
        print(f"  {treatment}: {n_passing} proteins have ≥{min_required}/{n_replicates} valid values")

    # Apply filter
    df = df[keep_protein].copy()
    removed = before - len(df)

    print(f"\n  ✓ Removed {removed} low-quality proteins")
    print(f"    Remaining: {len(df)} proteins")
    print(f"    (Kept proteins present in ≥50% of replicates in at least one treatment)")

    # =========================================================================
    # 6. ADD GENE SYMBOLS (if missing)
    # =========================================================================
    print(f"\n[5/8] checking for gene symbols ...")

    gene_col = config['data_columns']['gene_symbol']
    protein_col = config['data_columns']['protein_id']
    
    if gene_col not in df.columns or df[gene_col].isna().all():
        print(f"\n Gene symbols missing - mapping from protein IDs using mygene...")
        
        try:
            import mygene
            mg = mygene.MyGeneInfo()
            
            # Get unique protein IDs
            protein_ids = df[protein_col].dropna().unique().tolist()
            print(f"  Querying mygene for {len(protein_ids)} proteins...")
            
            # Clean protein IDs
            cleaned_ids = {}
            for pid in protein_ids:
                clean_pid = str(pid).split('-')[0]
                clean_pid = clean_pid.split('.')[0]
                cleaned_ids[pid] = clean_pid
            
            unique_clean_ids = list(set(cleaned_ids.values()))
            print(f"  Cleaned to {len(unique_clean_ids)} unique base IDs")
            
            scopes_to_try = [
                'uniprot',
                'accession,uniprot,refseq,ensembl.protein',
            ]
            
            protein_to_gene = {}
            unmapped_ids = set(unique_clean_ids)
            
            for scope in scopes_to_try:
                if not unmapped_ids:
                    break
                
                print(f"  Trying scope: {scope}...")
                
                results = mg.querymany(
                    list(unmapped_ids), 
                    scopes=scope, 
                    fields='symbol,name',
                    species='human',
                    returnall=True
                )
                
                newly_mapped = 0
                for result in results['out']:
                    query_id = result['query']
                    
                    if 'symbol' in result and query_id in unmapped_ids:
                        protein_to_gene[query_id] = result['symbol']
                        unmapped_ids.remove(query_id)
                        newly_mapped += 1
                    elif 'name' in result and query_id in unmapped_ids and query_id not in protein_to_gene:
                        gene_name = result['name'].split(',')[0].split('(')[0].strip()
                        if len(gene_name) < 50:
                            protein_to_gene[query_id] = gene_name
                            unmapped_ids.remove(query_id)
                            newly_mapped += 1
                
                if newly_mapped > 0:
                    print(f"    ✓ Mapped {newly_mapped} proteins")
            
            # Map back to original IDs
            original_to_gene = {}
            for orig_id, clean_id in cleaned_ids.items():
                if clean_id in protein_to_gene:
                    original_to_gene[orig_id] = protein_to_gene[clean_id]
            
            # Apply mapping to dataframe
            df[gene_col] = df[protein_col].map(original_to_gene)
            
            # For proteins still without gene symbols, use protein ID
            still_unmapped = df[gene_col].isna().sum()
            if still_unmapped > 0:
                df[gene_col] = df[gene_col].fillna(df[protein_col])
            
            successful = len(original_to_gene)
            print(f"\n  ✓ Successfully mapped {successful}/{len(protein_ids)} proteins to gene symbols")
            if still_unmapped > 0:
                print(f"  ⚠ {still_unmapped} proteins kept protein ID as identifier")
            
        except ImportError:
            print(f"  ⚠ Warning: mygene not installed")
            print(f"  Installing mygene... (this may take a minute)")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "mygene", "--quiet"])
            print(f"  ✓ mygene installed - please run prep_ip() again")
            return None
            
        except Exception as e:
            print(f"  ⚠ Warning: Could not map gene symbols: {e}")
            print(f"  Using protein IDs as gene names instead")
            df[gene_col] = df[protein_col]
    else:
        print(f"\n[5/8] Gene symbols found in data")
        valid_symbols = df[gene_col].notna().sum()
        print(f"  ✓ {valid_symbols} proteins have gene symbols")

    # =========================================================================
    # 6. REMOVE MANUAL CONTAMINANTS (before CRAPome)
    # =========================================================================
    print(f"\n[6/8] Removing manual contaminants...")
    
    # Get manual contaminant list from config (with defaults)
    manual_contaminants = config.get('manual_contaminants', [
        'KRT', 'Keratin', 'keratin',  # Keratins (skin contamination)
        'TRYP', 'Trypsin', 'trypsin',  # Trypsin (from digestion)
        'ALB', 'Albumin', 'albumin',   # Albumin (from serum)
        'IGG', 'Immunoglobulin', 'immunoglobulin'  # Antibody (from IP)
    ])
    
    if len(manual_contaminants) > 0:
        # Combine Accession and Description for searching
        desc_col = 'Description' if 'Description' in df.columns else None
        
        if desc_col:
            df['Search_String'] = df[protein_col].fillna('') + ' ' + df[desc_col].fillna('')
        else:
            df['Search_String'] = df[protein_col].fillna('')
        
        # Track removed proteins
        before = len(df)
        removed_proteins = []
        
        for contaminant in manual_contaminants:
            # Find proteins matching this contaminant
            mask = df['Search_String'].str.contains(contaminant, case=False, na=False)
            removed = df[mask]
            
            if len(removed) > 0:
                print(f"    Found {len(removed)} proteins matching '{contaminant}'")
                removed_proteins.extend(removed[protein_col].tolist())
                df = df[~mask].copy()
        
        # Remove the temporary search column
        df = df.drop(columns=['Search_String'])
        
        removed = before - len(df)
        print(f"\n  ✓ Removed {removed} manual contaminant proteins")
        print(f"    Remaining: {len(df)} proteins")
    else:
        print(f"  No manual contaminants specified, skipping")


    # =========================================================================
    # 7. REMOVE CONTAMINANTS USING CRAPOME DATABASE
    # =========================================================================
    print(f"\n[7/8] Filtering contaminants using CRAPome database...")
    
    crapome_file = config['data_paths'].get('crapome_list', 'data/raw/Crapome_list.xlsx')
    
    if os.path.exists(crapome_file):
        # Load CRAPome data (Excel file)
        crapome_df = pd.read_excel(crapome_file)
        
        # This file has columns: PROTID, GENEID, FREQ, ALL_NUMSPECSTOT, etc.
        # FREQ = frequency protein appears in negative controls (0-1)
        # ALL_NUMSPECSTOT = total spectral counts across all experiments
        
        # Get thresholds from config (with defaults)
        frequency_threshold = config.get('crapome_filtering', {}).get('freq_threshold', 0.30)
        spectral_count_threshold = config.get('crapome_filtering', {}).get('spectral_count_threshold', 50)
        
        contaminant_mask = (
            (crapome_df['FREQ'] > frequency_threshold) & 
            (crapome_df['ALL_NUMSPECSTOT'] > spectral_count_threshold)
        )
        
        # Get list of contaminant protein IDs and gene symbols
        contaminant_ids = set(crapome_df[contaminant_mask]['PROTID'].tolist())
        contaminant_genes = set(crapome_df[contaminant_mask]['GENEID'].tolist())
        
        print(f"  CRAPome database loaded: {len(crapome_df)} proteins")
        print(f"  Identified {len(contaminant_ids)} contaminant protein IDs")
        print(f"  Identified {len(contaminant_genes)} contaminant gene symbols")
        print(f"  Thresholds: FREQ > {frequency_threshold:.0%}, Spectral Count > {spectral_count_threshold}")
        
        # Filter dataframe
        before = len(df)
        
        # Remove by protein ID (check if cleaned IDs match)
        # CRAPome has "NP_005336.3" format, ours might be "NP_005336"
        crapome_clean_ids = set([pid.split('.')[0] for pid in contaminant_ids])
        df_clean_ids = df[protein_col].str.split('.').str[0]
        
        remove_by_id = df_clean_ids.isin(crapome_clean_ids)
        
        # Also remove by gene symbol
        if gene_col in df.columns:
            remove_by_gene = df[gene_col].isin(contaminant_genes)
            remove_mask = remove_by_id | remove_by_gene
        else:
            remove_mask = remove_by_id
        
        df = df[~remove_mask].copy()
        
        removed = before - len(df)
        print(f"\n  ✓ Removed {removed} CRAPome contaminants")
        print(f"    Remaining: {len(df)} proteins")
        
        # Show examples of what was removed
        if removed > 0:
            removed_examples = crapome_df[contaminant_mask].nlargest(10, 'FREQ')[
                ['GENEID', 'FREQ', 'ALL_NUMSPECSTOT']
            ]
            print(f"\n  Top contaminants removed (examples):")
            for _, row in removed_examples.iterrows():
                gene = row['GENEID']
                freq = row['FREQ']
                sc = row['ALL_NUMSPECSTOT']
                print(f"    {gene:12} - FREQ: {freq:.1%}, Total SC: {sc}")
    
    
    else:
        print(f"  ⚠ Warning: CRAPome file not found at {crapome_file}")
        print(f"  Skipping CRAPome filtering")

    # =========================================================================
    # 8. SAVE FILTERED DATA AS CSV (for reference)
    # =========================================================================
    print(f"\n[8/8] Saving filtered data as CSV for reference...")

    csv_output_dir = os.path.join(config['data_paths']['output_dir'], 'tables')
    os.makedirs(csv_output_dir, exist_ok=True)

    # Save full filtered dataset
    csv_path = os.path.join(csv_output_dir, 'filtered_proteins_after_prep.csv')
    df.to_csv(csv_path, index=False)

    print(f"  ✓ Saved: filtered_proteins_after_prep.csv")
    print(f"    Location: {csv_output_dir}")
    print(f"    {len(df)} proteins × {len(df.columns)} columns")

    # Save summary version
    summary_cols = [
        config['data_columns']['protein_id'],
        config['data_columns']['gene_symbol'],
        config['data_columns']['peptides']
    ]
    summary_cols.extend(all_intensity_cols)
    summary_cols_present = [c for c in summary_cols if c in df.columns]
    df_summary = df[summary_cols_present].copy()

    summary_path = os.path.join(csv_output_dir, 'filtered_proteins_summary.csv')
    df_summary.to_csv(summary_path, index=False)

    print(f"  ✓ Saved: filtered_proteins_summary.csv")
    print(f"    (Protein info + intensities only)")

    # =========================================================================
    # 9. CHECK DATA QUALITY
    # =========================================================================
    print(f"\nData quality summary...")
    
    # Missing values per condition
    print(f"\n  Missing values by condition:")
    for condition, cols in intensity_cols.items():
        total_values = len(df) * len(cols)
        missing = df[cols].isna().sum().sum()
        pct_missing = (missing / total_values) * 100
        print(f"    {condition}: {pct_missing:.1f}% missing")
    
    # Proteins detected per condition
    print(f"\n  Proteins detected per condition:")
    for condition, cols in intensity_cols.items():
        detected = (df[cols].notna().any(axis=1)).sum()
        print(f"    {condition}: {detected} proteins")
    
    # =========================================================================
    # 10. CREATE OUTPUT DIRECTORIES
    # =========================================================================
    output_dir = config['data_paths']['output_dir']
    output_dirs = _create_output_dirs(output_dir)
    
    print(f"\n✓ Output directories created at: {output_dir}")
    
    # =========================================================================
    # 11. CREATE METADATA SUMMARY
    # =========================================================================
    metadata = {
        'n_proteins': len(df),
        'n_samples': len(all_intensity_cols),
        'n_conditions': len(intensity_cols),
        'proteins_removed': initial_protein_count - len(df),
        'conditions': list(intensity_cols.keys()),
        'replicates_per_condition': {k: len(v) for k, v in intensity_cols.items()}
    }
    
    # =========================================================================
    # 12. FINAL SUMMARY
    # =========================================================================
    print("\n" + "="*80)
    print("DATA PREPARATION COMPLETE")
    print("="*80)
    print(f"\nInitial proteins:        {initial_protein_count}")
    print(f"Final proteins:          {len(df)}")
    print(f"Proteins removed:        {metadata['proteins_removed']}")
    print(f"\nConditions analyzed:     {', '.join(metadata['conditions'])}")
    print(f"Total samples:           {metadata['n_samples']}")
    print("\n" + "="*80 + "\n")
    
    # Create return dictionary
    return_data = {
        'df': df,
        'config': config,
        'intensity_cols': intensity_cols,
        'metadata': metadata,
        'output_dirs': output_dirs
    }
    
    # =========================================================================
    # 13. AUTO-SAVE FOR SEQUENTIAL WORKFLOW
    # =========================================================================
    save_path = os.path.join(output_dir, 'data_after_prep.pkl')
    save_data(return_data, save_path)
    
    return return_data


































# ============================================================================
# ============================================================================
# ============================================================================
########################
# == QC IP FUNCTION == #
########################


def qc_ip(data, output_suffix=''):
    """
    Generate quality control plots and metrics.
    
    Creates:
    - Missing value heatmap
    - Sample correlation heatmap (0 to 1, white to red)
    - PCA plot (larger points and labels)
    
    Parameters:
    -----------
    data : dict
        Output from prep_ip()
    output_suffix : str, optional
        Suffix to add to output filenames. Use this to distinguish QC runs.
        Example: output_suffix='_after_drop' creates '01_missing_values_after_drop.pdf'
        
    Returns:
    --------
    None (saves plots to results/figures/qc/)
    
    Examples:
    ---------
    >>> # Initial QC
    >>> qc_ip(data)
    >>> # Creates: 01_missing_values.pdf, 02_correlation_heatmap.pdf, etc.
    
    >>> # After dropping samples - different filenames!
    >>> data = drop_samples(data)
    >>> qc_ip(data, output_suffix='_after_drop')
    >>> # Creates: 01_missing_values_after_drop.pdf, 02_correlation_heatmap_after_drop.pdf, etc.
    """
    
    print("\n" + "="*80)
    print("QUALITY CONTROL ANALYSIS")
    if output_suffix:
        print(f"Output suffix: {output_suffix}")
    print("="*80)
    
    df = data['df']
    config = data['config']
    intensity_cols = data['intensity_cols']
    output_dirs = data['output_dirs']
    
    qc_dir = output_dirs['qc']
    
    # Get all intensity columns in a flat list
    all_intensity = []
    for cols in intensity_cols.values():
        all_intensity.extend(cols)
    
    print(f"\nGenerating QC plots...")
    print(f"  Output directory: {qc_dir}")
    
    # =========================================================================
    # 1. MISSING VALUES HEATMAP
    # =========================================================================
    print(f"\n[1/3] Creating missing values heatmap...")
    
    # Create binary missing matrix (1 = missing, 0 = present)
    presence_data = df[all_intensity].notna().astype(int)  # notna() instead of isnull()

    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot heatmap
    sns.heatmap(
        presence_data.T,  # Changed from missing_data
        cmap='RdYlGn',    # Keep as-is (now 1=present=green, 0=missing=red)
        cbar_kws={'label': 'Present (1) vs Missing (0)'},  # Update label
        yticklabels=all_intensity,
        xticklabels=False,
        ax=ax
    )
    
    ax.set_title('Missing Value Pattern Across Samples', fontsize=14, fontweight='bold')
    ax.set_xlabel('Proteins', fontsize=12)
    ax.set_ylabel('Samples', fontsize=12)
    
    # Fix label alignment
    ax.set_yticks(np.arange(len(all_intensity)) + 0.5)
    ax.set_yticklabels(all_intensity, fontsize=8, rotation=0)
    
    plt.tight_layout()
    plt.savefig(f"{qc_dir}/01_missing_values{output_suffix}.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: 01_missing_values{output_suffix}.pdf")
    
    # Print missing value statistics
    total_values = len(df) * len(all_intensity)
    total_present = presence_data.sum().sum()  # Changed
    total_missing = total_values - total_present  # Calculate from present
    pct_missing = (total_missing / total_values) * 100
    print(f"    Overall: {pct_missing:.1f}% missing values")
    
    # =========================================================================
    # 2. CORRELATION HEATMAP
    # =========================================================================
    print(f"\n[2/3] Creating sample correlation heatmap...")
    
    # Calculate correlation
    corr_data = df[all_intensity].corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot heatmap - Red-White-Blue with white at 0.5
    sns.heatmap(
        corr_data,
        annot=False,
        cmap='RdBu_r',  # Red-White-Blue
        vmin=0,
        vmax=1,
        center=0.5,     # White at 0.5
        square=True,
        cbar_kws={'label': 'Pearson Correlation'},
        ax=ax
    )
    
    ax.set_title('Sample-to-Sample Correlation', fontsize=14, fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f"{qc_dir}/02_correlation_heatmap{output_suffix}.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: 02_correlation_heatmap{output_suffix}.pdf")
    
    # Print correlation statistics
    print(f"\n  Average correlations within conditions:")
    for condition, cols in intensity_cols.items():
        if len(cols) > 1:
            condition_corr = corr_data.loc[cols, cols]
            mask = np.triu(np.ones_like(condition_corr), k=1).astype(bool)
            avg_corr = condition_corr.where(mask).stack().mean()
            print(f"    {condition}: {avg_corr:.3f}")
    
    # =========================================================================
    # 3. PCA PLOT
    # =========================================================================
    print(f"\n[3/3] Creating PCA plot...")
    
    pca_data = df[all_intensity].dropna()
    
    if len(pca_data) < 10:
        print(f"  ⚠ Warning: Only {len(pca_data)} complete proteins")
    
    pca_data_t = pca_data.T
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(pca_data_t)
    
    pca = PCA(n_components=min(2, scaled_data.shape[1]))
    pca_coords = pca.fit_transform(scaled_data)
    
    # Colors
    colors = []
    labels = []
    condition_colors = {
        list(intensity_cols.keys())[0]: 'blue',
        list(intensity_cols.keys())[1]: 'green' if len(intensity_cols) > 1 else 'blue',
        list(intensity_cols.keys())[2]: 'red' if len(intensity_cols) > 2 else 'blue'
    }
    
    for condition, cols in intensity_cols.items():
        for col in cols:
            colors.append(condition_colors.get(condition, 'gray'))
            labels.append(condition)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    for condition in intensity_cols.keys():
        mask = np.array(labels) == condition
        ax.scatter(
            pca_coords[mask, 0],
            pca_coords[mask, 1],
            c=condition_colors[condition],
            label=condition,
            s=200,
            alpha=0.7,
            edgecolors='black',
            linewidth=2
        )
    
    # Labels
    for i, label in enumerate(all_intensity):
        short_label = label.split(',')[-1].strip()[:15]
        
        x, y = pca_coords[i, 0], pca_coords[i, 1]
        offset_scale = 0.05
        x_range = pca_coords[:, 0].max() - pca_coords[:, 0].min()
        y_range = pca_coords[:, 1].max() - pca_coords[:, 1].min()
        
        x_offset = x_range * offset_scale * np.sign(x) if x != 0 else x_range * 0.1
        y_offset = y_range * offset_scale * np.sign(y) if y != 0 else y_range * 0.1
        
        ax.annotate(
            short_label,
            (x, y),
            xytext=(x + x_offset, y + y_offset),
            fontsize=10,
            color='black',
            fontweight='bold',
            ha='center'
        )
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
    ax.set_title('PCA - Sample Clustering', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12, loc='best')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{qc_dir}/03_pca_plot{output_suffix}.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: 03_pca_plot{output_suffix}.pdf")
    print(f"\n  PC1 explains {pca.explained_variance_ratio_[0]*100:.1f}% of variance")
    print(f"  PC2 explains {pca.explained_variance_ratio_[1]*100:.1f}% of variance")

    
    # =========================================================================
    # 4. PCA PLOT - TREATMENTS ONLY (without control)
    # =========================================================================
    print(f"\n[4/4] Creating PCA plot (treatments only, no control)...")
    
    # Get treatment columns only (exclude control)
    control_condition = config['conditions']['control']
    treatment_intensity = []
    treatment_labels = []
    treatment_condition_labels = []
    
    for condition, cols in intensity_cols.items():
        if condition != control_condition:
            treatment_intensity.extend(cols)
            for col in cols:
                treatment_labels.append(col)
                treatment_condition_labels.append(condition)
    
    if len(treatment_intensity) > 0:
        # PCA on treatment samples only
        pca_data = df[treatment_intensity].dropna()
        
        if len(pca_data) < 10:
            print(f"  ⚠ Warning: Only {len(pca_data)} complete proteins")
        
        pca_data_t = pca_data.T
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(pca_data_t)
        
        pca = PCA(n_components=min(2, scaled_data.shape[1]))
        pca_coords = pca.fit_transform(scaled_data)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Get unique treatment conditions
        unique_treatments = [c for c in intensity_cols.keys() if c != control_condition]
        treatment_colors_map = {
            unique_treatments[0]: 'green' if len(unique_treatments) > 0 else 'blue',
            unique_treatments[1]: 'red' if len(unique_treatments) > 1 else 'blue'
        }
        
        for condition in unique_treatments:
            mask = np.array(treatment_condition_labels) == condition
            ax.scatter(
                pca_coords[mask, 0],
                pca_coords[mask, 1],
                c=treatment_colors_map[condition],
                label=condition,
                s=200,
                alpha=0.7,
                edgecolors='black',
                linewidth=2
            )
        
        # Labels
        for i, label in enumerate(treatment_labels):
            short_label = label.split(',')[-1].strip()[:15]
            
            x, y = pca_coords[i, 0], pca_coords[i, 1]
            offset_scale = 0.05
            x_range = pca_coords[:, 0].max() - pca_coords[:, 0].min()
            y_range = pca_coords[:, 1].max() - pca_coords[:, 1].min()
            
            x_offset = x_range * offset_scale * np.sign(x) if x != 0 else x_range * 0.1
            y_offset = y_range * offset_scale * np.sign(y) if y != 0 else y_range * 0.1
            
            ax.annotate(
                short_label,
                (x, y),
                xytext=(x + x_offset, y + y_offset),
                fontsize=10,
                color='black',
                fontweight='bold',
                ha='center'
            )
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
        ax.set_title('PCA - Treatments Only (Control Excluded)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=12, loc='best')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{qc_dir}/04_pca_treatments_only{output_suffix}.pdf", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: 04_pca_treatments_only{output_suffix}.pdf")
        print(f"\n  PC1 explains {pca.explained_variance_ratio_[0]*100:.1f}% of variance")
        print(f"  PC2 explains {pca.explained_variance_ratio_[1]*100:.1f}% of variance")
    else:
        print(f"  ⚠ No treatment samples found (only control)")


    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*80)
    print("QC COMPLETE")
    print("="*80)
    print(f"\nPlots saved to: {qc_dir}")
    print(f"  • 01_missing_values{output_suffix}.pdf")
    print(f"  • 02_correlation_heatmap{output_suffix}.pdf")
    print(f"  • 03_pca_plot{output_suffix}.pdf (all samples)")
    print(f"  • 04_pca_treatments_only{output_suffix}.pdf (no control)")
    print("="*80 + "\n")






















# ============================================================================
# ============================================================================
# ============================================================================
##################################
# == Drop Samples IP FUNCTION == #
##################################

def drop_samples(data, samples_to_drop=None):
    """
    Remove problematic samples from the dataset after QC review.
    
    Use this after reviewing QC plots to exclude samples that:
    - Have poor correlation with replicates
    - Cluster away from replicates in PCA
    - Have unusual distributions
    - Failed during sample prep
    
    Parameters:
    -----------
    data : dict
        Output from prep_ip()
    samples_to_drop : list of str or dict, optional
        Samples to remove. Can be:
        - List of full column names: ['Abundances (Normalized): F6: Sample, EV, Gel1', ...]
        - Dict mapping conditions to samples: {'EV': [1, 2], 'WT': [3]}
        - None: Interactive mode (prints samples and asks which to drop)
        
    Returns:
    --------
    dict : Updated data dictionary with samples removed
    
    Examples:
    ---------
    >>> # After reviewing QC, drop specific columns
    >>> data = drop_samples(data, samples_to_drop=[
    ...     'Abundances (Normalized): F7: Sample, EV, Gel2',
    ...     'Abundances (Normalized): F14: Sample, WT, Gel4'
    ... ])
    
    >>> # Or drop by condition and replicate number
    >>> data = drop_samples(data, samples_to_drop={
    ...     'EV': [2],      # Drop EV replicate 2
    ...     'WT': [4]       # Drop WT replicate 4
    ... })
    
    >>> # Interactive mode - prints all samples, prompts you to select
    >>> data = drop_samples(data)
    """
    
    import copy
    
    df = data['df']
    config = data['config']
    intensity_cols = data['intensity_cols']
    
    print("\n" + "="*80)
    print("DROP SAMPLES (MANUAL QC)")
    print("="*80)
    
    # Show all current samples
    print("\nCurrent samples:")
    sample_list = []
    for condition, cols in intensity_cols.items():
        print(f"\n{condition}:")
        for i, col in enumerate(cols, 1):
            sample_list.append(col)
            # Extract a readable name from the column
            short_name = col.split(',')[-1].strip() if ',' in col else col
            print(f"  {len(sample_list)}. {short_name} [{col}]")
    
    # Determine which samples to drop
    if samples_to_drop is None:
        # Interactive mode
        print("\n" + "="*80)
        print("Enter sample numbers to drop (comma-separated), or press Enter to keep all:")
        print("Example: 2,7,14")
        user_input = input("> ").strip()
        
        if user_input:
            try:
                indices = [int(x.strip()) - 1 for x in user_input.split(',')]
                cols_to_drop = [sample_list[i] for i in indices if 0 <= i < len(sample_list)]
            except (ValueError, IndexError):
                print("Invalid input. No samples dropped.")
                return data
        else:
            print("No samples dropped.")
            return data
            
    elif isinstance(samples_to_drop, dict):
        # Dict format: {'EV': [1, 2], 'WT': [3]}
        cols_to_drop = []
        for condition, replicate_nums in samples_to_drop.items():
            if condition in intensity_cols:
                for rep_num in replicate_nums:
                    if 0 < rep_num <= len(intensity_cols[condition]):
                        cols_to_drop.append(intensity_cols[condition][rep_num - 1])
                    else:
                        print(f"  ⚠ Warning: {condition} replicate {rep_num} doesn't exist")
            else:
                print(f"  ⚠ Warning: Condition '{condition}' not found")
    
    elif isinstance(samples_to_drop, list):
        # List of full column names
        cols_to_drop = samples_to_drop
    
    else:
        print("Invalid samples_to_drop format. No samples dropped.")
        return data
    
    # Verify columns exist
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    
    if not cols_to_drop:
        print("\nNo valid samples to drop.")
        return data
    
    # Show what will be dropped
    print(f"\nDropping {len(cols_to_drop)} sample(s):")
    for col in cols_to_drop:
        print(f"  - {col}")
    
    # Confirm (if interactive)
    if samples_to_drop is None:
        confirm = input("\nConfirm? (yes/no): ").strip().lower()
        if confirm not in ['yes', 'y']:
            print("Cancelled. No samples dropped.")
            return data
    
    # Drop the samples
    df_updated = df.drop(columns=cols_to_drop)
    
    # Update intensity_cols
    intensity_cols_updated = {}
    for condition, cols in intensity_cols.items():
        updated_cols = [c for c in cols if c not in cols_to_drop]
        if updated_cols:  # Only keep conditions with remaining samples
            intensity_cols_updated[condition] = updated_cols
        else:
            print(f"  ⚠ Warning: All {condition} samples dropped! Condition removed.")
    
    # Update metadata
    all_intensity_cols = []
    for cols in intensity_cols_updated.values():
        all_intensity_cols.extend(cols)
    
    metadata_updated = data['metadata'].copy()
    metadata_updated['n_samples'] = len(all_intensity_cols)
    metadata_updated['n_conditions'] = len(intensity_cols_updated)
    metadata_updated['conditions'] = list(intensity_cols_updated.keys())
    metadata_updated['replicates_per_condition'] = {
        k: len(v) for k, v in intensity_cols_updated.items()
    }
    metadata_updated['samples_dropped'] = cols_to_drop
    
    # Create updated data dict
    data_updated = {
        'df': df_updated,
        'config': config,
        'intensity_cols': intensity_cols_updated,
        'metadata': metadata_updated,
        'output_dirs': data['output_dirs']
    }
    
    # Print summary
    print("\n" + "="*80)
    print("SAMPLES DROPPED")
    print("="*80)
    print(f"\nRemaining samples: {metadata_updated['n_samples']}")
    print(f"Remaining conditions: {metadata_updated['conditions']}")
    print(f"\nReplicates per condition:")
    for condition, n_reps in metadata_updated['replicates_per_condition'].items():
        print(f"  {condition}: {n_reps} replicates")
    
    print("\n" + "="*80)
    print("⚠ IMPORTANT: Save this updated data!")
    print("="*80)
    print("Run this to save:")
    print("  save_data(data, '../results/data_after_qc.pkl')")
    print("="*80 + "\n")
    
    return data_updated














# ============================================================================
# ============================================================================
# ============================================================================
##########################
# == NORM IP FUNCTION == #
##########################
def norm_ip(data, method='log2', imputation='mindet'):
    """
    Normalize intensity data and impute missing values.
    
    Normalization options:
    - 'log2': Log2 transformation (default, recommended for proteomics)
    - 'zscore': Z-score normalization (mean=0, std=1)
    - 'quantile': Quantile normalization
    - 'median': Median normalization
    
    Imputation options:
    - 'mindet': Minimum detection (left-censored, default for proteomics)
    - 'zero': Replace with zero
    - 'median': Median imputation per sample
    - 'knn': K-nearest neighbors
    
    Parameters:
    -----------
    data : dict
        Output from prep_ip() or drop_samples()
    method : str, optional
        Normalization method (default: 'log2')
    imputation : str, optional
        Imputation method for missing values (default: 'mindet')
        
    Returns:
    --------
    dict : Updated data dictionary with normalized/imputed values
    
    Example:
    --------
    >>> data = prep_ip('config/experiment.yaml')
    >>> data = norm_ip(data, method='log2', imputation='mindet')
    >>> # Data is now normalized and saved to data_after_norm.pkl
    """
    
    print("\n" + "="*80)
    print("NORMALIZATION AND IMPUTATION")
    print("="*80)
    
    df = data['df'].copy()  # Work on a copy
    config = data['config']
    intensity_cols = data['intensity_cols']
    
    # Get all intensity columns
    all_intensity = []
    for cols in intensity_cols.values():
        all_intensity.extend(cols)
    
    print(f"\nMethod: {method}")
    print(f"Imputation: {imputation}")
    print(f"Processing {len(df)} proteins across {len(all_intensity)} samples")
    
    # =========================================================================
    # 1. CHECK DATA BEFORE NORMALIZATION
    # =========================================================================
    print(f"\n[1/3] Data before normalization:")
    
    for condition, cols in intensity_cols.items():
        values = df[cols].values.flatten()
        values = values[~np.isnan(values)]
        print(f"  {condition}:")
        print(f"    Range: {values.min():.1f} to {values.max():.1f}")
        print(f"    Median: {np.median(values):.1f}")
        print(f"    Missing: {df[cols].isna().sum().sum()} values")
    
    # =========================================================================
    # 2. NORMALIZATION
    # =========================================================================
    print(f"\n[2/3] Applying {method} normalization...")
    
    if method == 'log2':
        # Log2 transform
        # Add small constant to avoid log(0)
        df[all_intensity] = np.log2(df[all_intensity] + 1)
        print(f"  ✓ Log2 transformation applied")
        
    elif method == 'zscore':
        # Z-score normalization (per sample)
        for col in all_intensity:
            values = df[col].dropna()
            mean = values.mean()
            std = values.std()
            df[col] = (df[col] - mean) / std
        print(f"  ✓ Z-score normalization applied")
        
    elif method == 'quantile':
        # Quantile normalization
        from sklearn.preprocessing import quantile_transform
        # Only on non-missing values
        mask = df[all_intensity].notna()
        df.loc[:, all_intensity] = df[all_intensity].where(
            ~mask,
            quantile_transform(df[all_intensity].fillna(0))
        )
        print(f"  ✓ Quantile normalization applied")
        
    elif method == 'median':
        # Median normalization (per sample)
        for col in all_intensity:
            median = df[col].median()
            global_median = df[all_intensity].median().median()
            df[col] = df[col] - median + global_median
        print(f"  ✓ Median normalization applied")
        
    else:
        print(f"  ⚠ Warning: Unknown method '{method}', skipping normalization")
    
    # =========================================================================
    # 3. IMPUTATION
    # =========================================================================
    print(f"\n[3/3] Applying {imputation} imputation...")
    
    missing_before = df[all_intensity].isna().sum().sum()
    
    if imputation == 'mindet':
        # Minimum detection - impute with low values (left-censored)
        # For each sample, use min value - 1.8*std (typical for proteomics)
        for col in all_intensity:
            if df[col].isna().any():
                valid_values = df[col].dropna()
                if len(valid_values) > 0:
                    min_val = valid_values.min()
                    std_val = valid_values.std()
                    impute_val = min_val - 1.8 * std_val
                    df[col] = df[col].fillna(impute_val)
        print(f"  ✓ MinDet imputation applied (min - 1.8*std per sample)")
        
    elif imputation == 'zero':
        # Replace with zero
        df[all_intensity].fillna(0, inplace=True)
        print(f"  ✓ Zero imputation applied")
        
    elif imputation == 'median':
        # Median imputation (per sample)
        for col in all_intensity:
            median = df[col].median()
            df[col].fillna(median, inplace=True)
        print(f"  ✓ Median imputation applied")
        
    elif imputation == 'knn':
        # K-nearest neighbors
        from sklearn.impute import KNNImputer
        imputer = KNNImputer(n_neighbors=5)
        df[all_intensity] = imputer.fit_transform(df[all_intensity])
        print(f"  ✓ KNN imputation applied (k=5)")
        
    else:
        print(f"  ⚠ Warning: Unknown imputation '{imputation}', skipping")
    
    missing_after = df[all_intensity].isna().sum().sum()
    print(f"    Missing values: {missing_before} → {missing_after}")
    
    # =========================================================================
    # 4. CHECK DATA AFTER NORMALIZATION
    # =========================================================================
    print(f"\n" + "="*80)
    print("NORMALIZATION SUMMARY")
    print("="*80)
    
    print(f"\nData after normalization:")
    for condition, cols in intensity_cols.items():
        values = df[cols].values.flatten()
        values = values[~np.isnan(values)]
        print(f"  {condition}:")
        print(f"    Range: {values.min():.2f} to {values.max():.2f}")
        print(f"    Median: {np.median(values):.2f}")
        print(f"    Std: {np.std(values):.2f}")
    
    # =========================================================================
    # 5. CREATE COMPARISON PLOTS
    # =========================================================================
    print(f"\nCreating before/after comparison plots...")
    
    output_dir = data['output_dirs']['qc']
    
    # Get original data for comparison
    df_original = data['df'][all_intensity].copy()
    
    # Create figure with 2 rows: before and after
    n_conditions = len(intensity_cols)
    fig, axes = plt.subplots(2, n_conditions, figsize=(5*n_conditions, 10))
    
    # Handle single condition case
    if n_conditions == 1:
        axes = axes.reshape(2, 1)
    
    for idx, (condition, cols) in enumerate(intensity_cols.items()):
        # BEFORE (top row)
        ax_before = axes[0, idx]
        for col in cols:
            values = df_original[col].dropna()
            ax_before.hist(values, bins=50, alpha=0.5, label=col.split(',')[-1].strip()[:10])
        ax_before.set_title(f'{condition} - Before Normalization', fontweight='bold')
        ax_before.set_xlabel('Intensity (Raw)', fontsize=10)
        ax_before.set_ylabel('Frequency', fontsize=10)
        ax_before.legend(fontsize=8, loc='upper right')
        ax_before.grid(alpha=0.3)
        
        # AFTER (bottom row)
        ax_after = axes[1, idx]
        for col in cols:
            values = df[col].dropna()
            ax_after.hist(values, bins=50, alpha=0.5, label=col.split(',')[-1].strip()[:10])
        ax_after.set_title(f'{condition} - After {method} + {imputation}', fontweight='bold')
        ax_after.set_xlabel(f'Intensity ({method})', fontsize=10)
        ax_after.set_ylabel('Frequency', fontsize=10)
        ax_after.legend(fontsize=8, loc='upper right')
        ax_after.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/normalization_comparison.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: normalization_comparison.pdf")
    
    # =========================================================================
    # 6. UPDATE DATA DICTIONARY
    # =========================================================================
    data_updated = data.copy()
    data_updated['df'] = df
    data_updated['normalization'] = {
        'method': method,
        'imputation': imputation
    }
    
    # =========================================================================
    # 7. AUTO-SAVE FOR SEQUENTIAL WORKFLOW
    # =========================================================================
    output_path = os.path.join(config['data_paths']['output_dir'], 'data_after_norm.pkl')
    save_data(data_updated, output_path)
    
    print("\n" + "="*80)
    print("NORMALIZATION COMPLETE")
    print("="*80)
    print(f"\nNext step: stat_ip() for statistical analysis")
    print("="*80 + "\n")
    
    return data_updated








# ============================================================================
# ============================================================================
# ============================================================================
##########################
# == STAT IP FUNCTION == #
##########################










def stat_ip(data, p_threshold=0.05, log2fc_threshold=1.0, correction='fdr_bh'):
    """
    Perform statistical analysis comparing treatments to control.
    
    For each comparison (treatment vs control):
    - Calculates log2 fold change
    - Performs t-test
    - Applies multiple testing correction
    - Identifies significant proteins
    
    Parameters:
    -----------
    data : dict
        Output from norm_ip()
    p_threshold : float, optional
        P-value threshold for significance (default: 0.05)
    log2fc_threshold : float, optional
        Log2 fold change threshold for significance (default: 1.0)
        Use absolute value: |log2FC| > threshold
    correction : str, optional
        Multiple testing correction method (default: 'fdr_bh')
        Options: 'fdr_bh' (Benjamini-Hochberg), 'bonferroni', 'none'
        
    Returns:
    --------
    dict : Updated data dictionary with statistical results
        - 'stats_results': DataFrame with all proteins and statistics
        - 'significant_proteins': Dict of significant proteins per comparison
        - 'stats_params': Parameters used for analysis
    
    Example:
    --------
    >>> data = norm_ip(data)
    >>> data = stat_ip(data, p_threshold=0.05, log2fc_threshold=1.0)
    >>> # Results saved to: results/tables/stats_results_pval05_l2fc1.csv
    """
    
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS")
    print("="*80)
    
    df = data['df'].copy()
    config = data['config']
    intensity_cols = data['intensity_cols']
    
    # Get control and treatments
    control = config['conditions']['control']
    treatments = config['conditions']['treatments']
    
    print(f"\nControl: {control}")
    print(f"Treatments: {', '.join(treatments)}")
    print(f"\nThresholds:")
    print(f"  P-value: {p_threshold}")
    print(f"  Log2 FC: {log2fc_threshold}")
    print(f"  Correction: {correction}")
    
    # =========================================================================
    # 1. PREPARE RESULTS DATAFRAME
    # =========================================================================
    print(f"\n[1/4] Preparing data...")
    
    # Start with protein info columns
    info_cols = [
        config['data_columns']['protein_id'],
        config['data_columns']['gene_symbol'],
        config['data_columns']['peptides']
    ]
    
    results_df = df[info_cols].copy()
    print(f"  ✓ Starting with {len(results_df)} proteins")
    
    # =========================================================================
    # 2. CALCULATE STATISTICS FOR EACH COMPARISON
    # =========================================================================
    print(f"\n[2/4] Calculating statistics...")
    
    from scipy.stats import ttest_ind
    from statsmodels.stats.multitest import multipletests
    
    significant_proteins = {}
    
    for treatment in treatments:
        comparison_name = f"{treatment}_vs_{control}"
        print(f"\n  Analyzing: {comparison_name}")
        
        # Get columns
        control_cols = intensity_cols[control]
        treatment_cols = intensity_cols[treatment]
        
        # Calculate means
        control_mean = df[control_cols].mean(axis=1)
        treatment_mean = df[treatment_cols].mean(axis=1)
        
        # Calculate log2 fold change
        # Already log2 transformed, so just subtract
        log2fc = treatment_mean - control_mean
        
        # Calculate t-test
        pvalues = []
        for idx in df.index:
            control_vals = df.loc[idx, control_cols].dropna()
            treatment_vals = df.loc[idx, treatment_cols].dropna()
            
            # Need at least 2 values in each group
            if len(control_vals) >= 2 and len(treatment_vals) >= 2:
                stat, pval = ttest_ind(treatment_vals, control_vals)
                pvalues.append(pval)
            else:
                pvalues.append(np.nan)
        
        pvalues = np.array(pvalues)
        
        # Apply multiple testing correction
        if correction == 'fdr_bh':
            # Remove NaN for correction
            mask = ~np.isnan(pvalues)
            adj_pvalues = np.full(len(pvalues), np.nan)
            if mask.any():
                _, adj_p, _, _ = multipletests(pvalues[mask], method='fdr_bh')
                adj_pvalues[mask] = adj_p
            correction_label = 'FDR'
        elif correction == 'bonferroni':
            mask = ~np.isnan(pvalues)
            adj_pvalues = np.full(len(pvalues), np.nan)
            if mask.any():
                _, adj_p, _, _ = multipletests(pvalues[mask], method='bonferroni')
                adj_pvalues[mask] = adj_p
            correction_label = 'Bonferroni'
        else:  # no correction
            adj_pvalues = pvalues
            correction_label = 'None'
        
        # Add to results dataframe
        results_df[f'{comparison_name}_log2FC'] = log2fc
        results_df[f'{comparison_name}_pvalue'] = pvalues
        results_df[f'{comparison_name}_adj_pvalue'] = adj_pvalues
        results_df[f'{comparison_name}_control_mean'] = control_mean
        results_df[f'{comparison_name}_treatment_mean'] = treatment_mean
        
        # Determine significance
        sig_mask = (
            (adj_pvalues < p_threshold) & 
            (np.abs(log2fc) > log2fc_threshold)
        )
        
        results_df[f'{comparison_name}_significant'] = sig_mask
        results_df[f'{comparison_name}_significant'] = results_df[f'{comparison_name}_significant'].map({
            True: 'significant',
            False: 'not_significant'
        })
        
        # Count significant proteins
        n_sig = sig_mask.sum()
        n_enriched = ((log2fc > log2fc_threshold) & (adj_pvalues < p_threshold)).sum()
        n_depleted = ((log2fc < -log2fc_threshold) & (adj_pvalues < p_threshold)).sum()
        
        print(f"    Total significant: {n_sig}")
        print(f"      Enriched in {treatment}: {n_enriched}")
        print(f"      Depleted in {treatment}: {n_depleted}")
        
        # Store significant protein list
        significant_proteins[comparison_name] = {
            'total': n_sig,
            'enriched': n_enriched,
            'depleted': n_depleted,
            'protein_ids': results_df[sig_mask][config['data_columns']['protein_id']].tolist()
        }
    
    # =========================================================================
    # 3. SAVE RESULTS
    # =========================================================================
    print(f"\n[3/4] Saving results...")
    
    # Create filename with thresholds
    threshold_label = f"pval{str(p_threshold).replace('.', '')}_l2fc{str(log2fc_threshold).replace('.', '')}"
    
    # Save complete results
    output_dir = os.path.join(config['data_paths']['output_dir'], 'tables')
    os.makedirs(output_dir, exist_ok=True)
    
    results_path = os.path.join(output_dir, f'stats_results_{threshold_label}.csv')
    results_df.to_csv(results_path, index=False)
    print(f"  ✓ Saved: stats_results_{threshold_label}.csv")
    print(f"    Location: {output_dir}")
    print(f"    {len(results_df)} proteins × {len(results_df.columns)} columns")
    
    # Save significant proteins only
    for comparison_name in significant_proteins.keys():
        sig_df = results_df[results_df[f'{comparison_name}_significant'] == 'significant'].copy()
        
        if len(sig_df) > 0:
            sig_path = os.path.join(output_dir, f'{comparison_name}_significant_{threshold_label}.csv')
            sig_df.to_csv(sig_path, index=False)
            print(f"  ✓ Saved: {comparison_name}_significant_{threshold_label}.csv ({len(sig_df)} proteins)")
    
    # =========================================================================
    # 4. CREATE SUMMARY TABLE
    # =========================================================================
    print(f"\n[4/4] Creating summary...")
    
    summary_data = []
    for comparison_name, stats in significant_proteins.items():
        summary_data.append({
            'Comparison': comparison_name,
            'Total_Significant': stats['total'],
            'Enriched': stats['enriched'],
            'Depleted': stats['depleted'],
            'P_threshold': p_threshold,
            'Log2FC_threshold': log2fc_threshold,
            'Correction': correction
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, f'summary_{threshold_label}.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"  ✓ Saved: summary_{threshold_label}.csv")
    
    # =========================================================================
    # 5. UPDATE DATA DICTIONARY
    # =========================================================================
    data_updated = data.copy()
    data_updated['stats_results'] = results_df
    data_updated['significant_proteins'] = significant_proteins
    data_updated['stats_params'] = {
        'p_threshold': p_threshold,
        'log2fc_threshold': log2fc_threshold,
        'correction': correction,
        'threshold_label': threshold_label
    }
    
    # =========================================================================
    # 6. AUTO-SAVE FOR SEQUENTIAL WORKFLOW
    # =========================================================================
    output_path = os.path.join(config['data_paths']['output_dir'], 'data_after_stat.pkl')
    save_data(data_updated, output_path)
    
    # =========================================================================
    # 7. PRINT SUMMARY
    # =========================================================================
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS COMPLETE")
    print("="*80)
    
    print(f"\nResults Summary:")
    for comparison_name, stats in significant_proteins.items():
        print(f"\n{comparison_name}:")
        print(f"  Total significant: {stats['total']}")
        print(f"  Enriched: {stats['enriched']}")
        print(f"  Depleted: {stats['depleted']}")
    
    print(f"\nThresholds used:")
    print(f"  P-value < {p_threshold}")
    print(f"  |Log2FC| > {log2fc_threshold}")
    print(f"  Correction: {correction}")
    
    print(f"\nFiles saved:")
    print(f"  • stats_results_{threshold_label}.csv (all proteins)")
    print(f"  • [comparison]_significant_{threshold_label}.csv (sig proteins only)")
    print(f"  • summary_{threshold_label}.csv (overview)")
    
    print("\n" + "="*80)
    print("Next step: viz_ip() for volcano plots and visualizations")
    print("="*80 + "\n")
    
    return data_updated





























# ============================================================================
# ============================================================================
# ============================================================================
##########################
# == VIZ IP FUNCTION == #
##########################

def viz_ip(data, create_volcano=True, create_heatmap=True, top_n=100):
    """
    Create visualization plots for IP-MS results.
    
    Creates:
    - Volcano plots (labeled and unlabeled versions)
    - Clustered heatmap of top significant proteins (intensities)
    - Clustered heatmap of log2 fold changes (sample-level)
    
    Parameters:
    -----------
    data : dict
        Output from stat_ip()
    create_volcano : bool, optional
        Create volcano plots (default: True)
    create_heatmap : bool, optional
        Create heatmap of top proteins (default: True)
    top_n : int, optional
        Number of top proteins to show in heatmap (default: 50)
        
    Returns:
    --------
    None (saves plots to results/figures/viz/)
    
    Example:
    --------
    >>> data = stat_ip(data)
    >>> viz_ip(data)
    >>> # Volcano plots and heatmaps saved to results/figures/viz/
    """
    
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    df = data['df']
    config = data['config']
    intensity_cols = data['intensity_cols']
    stats_results = data['stats_results']
    sig_proteins = data['significant_proteins']
    stats_params = data['stats_params']
    
    # Create viz output directory
    viz_dir = os.path.join(config['data_paths']['output_dir'], 'figures', 'viz')
    os.makedirs(viz_dir, exist_ok=True)
    
    print(f"\nOutput directory: {viz_dir}")
    
    # Get comparisons
    comparisons = list(sig_proteins.keys())
    
    # =========================================================================
    # 1. VOLCANO PLOTS (labeled and unlabeled versions)
    # =========================================================================
    if create_volcano:
        print(f"\n[1/3] Creating volcano plots...")
        
        for comparison in comparisons:
            print(f"  Creating volcano plot for {comparison}...")
            
            # Get data for this comparison
            log2fc = stats_results[f'{comparison}_log2FC']
            pval = stats_results[f'{comparison}_adj_pvalue']
            significant = stats_results[f'{comparison}_significant']
            gene_symbols = stats_results['Gene_Symbol']
            
            # -log10 transform p-values
            neg_log10_pval = -np.log10(pval.replace(0, 1e-300))  # Avoid log(0)
            
            # Create categories for coloring
            colors = []
            for i in range(len(stats_results)):
                if significant.iloc[i] == 'significant':
                    if log2fc.iloc[i] > 0:
                        colors.append('enriched')
                    else:
                        colors.append('depleted')
                else:
                    colors.append('not_significant')
            
            # =====================================================================
            # CREATE BOTH LABELED AND UNLABELED VERSIONS
            # =====================================================================
            for labeled in [True, False]:
                
                # Create plot
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Plot points by category
                for category, color, label in [
                    ('not_significant', '#CCCCCC', 'Not Significant'),
                    ('depleted', '#CCCCCC', 'Depleted'),
                    ('enriched', '#E74C3C', 'Enriched')
                ]:
                    mask = np.array(colors) == category
                    ax.scatter(
                        log2fc[mask],
                        neg_log10_pval[mask],
                        c=color,
                        label=label,
                        s=30,
                        alpha=0.6,
                        edgecolors='none'
                    )
                
                # Add threshold lines
                p_threshold = stats_params['p_threshold']
                fc_threshold = stats_params['log2fc_threshold']
                
                ax.axhline(-np.log10(p_threshold), color='black', linestyle='--', 
                          linewidth=1, alpha=0.5, label=f'p = {p_threshold}')
                ax.axvline(fc_threshold, color='black', linestyle='--', 
                          linewidth=1, alpha=0.5)
                ax.axvline(-fc_threshold, color='black', linestyle='--', 
                          linewidth=1, alpha=0.5)
                
                # Label top enriched proteins (only for labeled version)
                if labeled:
                    enriched_mask = (np.array(colors) == 'enriched')
                    if enriched_mask.any():
                        enriched_data = stats_results[enriched_mask].copy()
                        enriched_data['neg_log10_pval'] = neg_log10_pval[enriched_mask]
                        top_enriched = enriched_data.nlargest(25, f'{comparison}_log2FC')
                        
                        # Use adjustText for automatic label spacing (if available)
                        try:
                            from adjustText import adjust_text
                            texts = []
                            for idx, row in top_enriched.iterrows():
                                texts.append(ax.text(
                                    row[f'{comparison}_log2FC'], 
                                    row['neg_log10_pval'],
                                    row['Gene_Symbol'],
                                    # x_pos + 0.3,  # ← Start 0.3 units to the right
                                    # y_pos + 0.5,  # ← Start 0.5 units up
                                    fontsize=8,
                                    alpha=0.8
                                ))  
                            
                            # Print to confirm it's running
                            print(f"      Using adjustText for {len(texts)} labels")
                            
                            adjust_text(
                                texts, 
                                arrowprops=dict(arrowstyle='-', color='black', lw=0.5),
                                expand_points=(2.0, 2.0),
                                expand_text=(2.0, 4.0),
                                force_points=2.0,
                                force_text=2.0,
                                shrinkA=10,      # ← Add this: space between label and arrow start
                                shrinkB=10 
                            )  
                            
                        except ImportError as e:
                            print(f"      adjustText not available, using manual spacing")
                            # Fallback to manual spacing
                            for idx, row in top_enriched.iterrows():
                                ax.annotate(
                                    row['Gene_Symbol'],
                                    xy=(row[f'{comparison}_log2FC'], row['neg_log10_pval']),
                                    xytext=(20, 20),
                                    textcoords='offset points',
                                    fontsize=8,
                                    alpha=0.8,
                                    arrowprops=dict(arrowstyle='->', lw=0.5, color='black')
                                )
                        
                        except Exception as e:
                            print(f"      Error with adjustText: {e}")
                            # Fallback
                            for idx, row in top_enriched.iterrows():
                                ax.annotate(
                                    row['Gene_Symbol'],
                                    xy=(row[f'{comparison}_log2FC'], row['neg_log10_pval']),
                                    xytext=(20, 20),
                                    textcoords='offset points',
                                    fontsize=8,
                                    alpha=0.8,
                                    arrowprops=dict(arrowstyle='->', lw=0.5, color='black')
                                )
                
                # Labels and title
                ax.set_xlabel('Log2 Fold Change', fontsize=12, fontweight='bold')
                ax.set_ylabel('-Log10 Adjusted P-value', fontsize=12, fontweight='bold')
                ax.set_title(f'Volcano Plot: {comparison}', fontsize=14, fontweight='bold')
                ax.legend(loc='lower right', fontsize=10)
                ax.grid(alpha=0.3)
                
                # Save with appropriate filename
                suffix = '' if labeled else '_clean'
                volcano_path = os.path.join(viz_dir, f'volcano_{comparison}_{stats_params["threshold_label"]}{suffix}.pdf')
                plt.tight_layout()
                plt.savefig(volcano_path, dpi=300, bbox_inches='tight')
                plt.close()
            
            print(f"    ✓ Saved: volcano_{comparison}_{stats_params['threshold_label']}.pdf (labeled)")
            print(f"    ✓ Saved: volcano_{comparison}_{stats_params['threshold_label']}_clean.pdf (unlabeled)")
    
        # =========================================================================
        # 2. HEATMAP OF TOP PROTEINS (with clustering)
        # =========================================================================
        if create_heatmap:
            print(f"\n[2/3] Creating heatmap of top {top_n} proteins...")
            
            # Collect top proteins from all comparisons
            all_sig_proteins = set()
            for comparison in comparisons:
                sig_mask = (stats_results[f'{comparison}_significant'] == 'significant') & \
                        (stats_results[f'{comparison}_log2FC'] > 0)  # Only enriched
                sig_data = stats_results[sig_mask].copy()
                
                # Get top proteins by fold change
                top_proteins = sig_data.nlargest(top_n, f'{comparison}_log2FC')
                all_sig_proteins.update(top_proteins['Accession'].tolist())
            
            print(f"  Total unique significant proteins: {len(all_sig_proteins)}")
            
            if len(all_sig_proteins) > 0:
                # Filter to significant proteins
                heatmap_df = stats_results[stats_results['Accession'].isin(all_sig_proteins)].copy()
                
                # Get intensity data for these proteins
                all_intensity_cols = []
                for cols in intensity_cols.values():
                    all_intensity_cols.extend(cols)
                
                # Get normalized intensities from original df
                heatmap_data = df.loc[heatmap_df.index, all_intensity_cols]
                
                # Add gene symbols as row labels
                heatmap_data.index = heatmap_df['Gene_Symbol'].values
                
                # Create clustered heatmap WITHOUT colorbar
                g = sns.clustermap(
                    heatmap_data,
                    cmap='RdBu_r',
                    center=heatmap_data.max().max() / 2,
                    # center=0,
                    cbar_kws={'label': 'Normalized Intensity (log2)'},  
                    yticklabels=True,
                    xticklabels=True,
                    figsize=(20, 16),
                    row_cluster=True,
                    col_cluster=False,
                    method='average',
                    metric='euclidean'
                )
                # # Remove the colorbar axes completely
                # if g.ax_cbar is not None:
                #     g.ax_cbar.remove()

                # Set title and labels on the heatmap axes
                g.ax_heatmap.set_title(f'Top {len(heatmap_data)} Enriched Proteins', 
                            fontsize=14, fontweight='bold', pad=20)
                g.ax_heatmap.set_xlabel('Samples', fontsize=12)
                g.ax_heatmap.set_ylabel('Proteins', fontsize=12)
                
                # Rotate x labels
                g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=45, ha='right', fontsize=8)
                g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=8)
                
                # Save
                heatmap_path = os.path.join(viz_dir, f'heatmap_top_proteins_{stats_params["threshold_label"]}.pdf')
                g.savefig(heatmap_path, dpi=300)
                plt.close()
                
                print(f"  ✓ Saved: heatmap_top_proteins_{stats_params['threshold_label']}.pdf")
            else:
                print(f"  ⚠ No significant proteins to plot")
    
        # =========================================================================
        # 3. HEATMAP OF LOG2 FOLD CHANGES (with clustering)
        # =========================================================================
        if create_heatmap:
            print(f"\n[3/3] Creating log2 fold change heatmap...")
            
            # Collect top proteins from all comparisons (same as before)
            all_sig_proteins = set()
            for comparison in comparisons:
                sig_mask = (stats_results[f'{comparison}_significant'] == 'significant') & \
                        (stats_results[f'{comparison}_log2FC'] > 0)  # Only enriched
                sig_data = stats_results[sig_mask].copy()
                top_proteins = sig_data.nlargest(top_n, f'{comparison}_log2FC')
                all_sig_proteins.update(top_proteins['Accession'].tolist())
            
            if len(all_sig_proteins) > 0:
                # Filter to significant proteins
                heatmap_df = stats_results[stats_results['Accession'].isin(all_sig_proteins)].copy()
                
                # Calculate log2FC for each individual sample vs control average
                control_condition = config['conditions']['control']
                control_cols = intensity_cols[control_condition]
                
                # Get control average for each protein
                control_avg = df.loc[heatmap_df.index, control_cols].mean(axis=1)
                
                # Calculate log2FC for each treatment sample
                log2fc_data = pd.DataFrame(index=heatmap_df.index)
                
                for condition, cols in intensity_cols.items():
                    if condition != control_condition:  # Skip control
                        for col in cols:
                            # Sample intensity - control average = log2FC
                            sample_name = col.split(',')[-1].strip()[:20]  # Shorter label
                            col_name = f"{condition}_{sample_name}"
                            log2fc_data[col_name] = df.loc[heatmap_df.index, col] - control_avg
                
                # Add gene symbols as row labels
                log2fc_data.index = heatmap_df['Gene_Symbol'].values
                
                # Create clustered heatmap - COMPLETELY SIMPLE
                g = sns.clustermap(
                    log2fc_data,
                    cmap='RdBu_r',
                    center=0,
                    # center=log2fc_data.median().median(),
                    # vmin=-15,  # ← Minimum value (dark blue)
                    # vmax=15,  # ← Maximum value (dark red) 
                    cbar_kws={'label': 'Log2 Fold Change (vs Control Avg)'}, 
                    #     'shrink': 0.3,  # ← Makes it 50% of default size (try 0.3-0.8)
                    #     'aspect': 5,   # ← Width to height ratio (higher = wider, try 20-40)
                    #     'pad': 0.6     # ← Distance from heatmap (try 0.02-0.1)
                    # },
                    yticklabels=True,
                    xticklabels=True,
                    figsize=(20, 16),  # ← Fixed size, very large
                    row_cluster=True,
                    col_cluster=False,
                    method='average',
                    metric='euclidean'
                    # NO dendrogram_ratio, NO cbar_pos - all defaults!
                )

                # Set title and labels
                g.ax_heatmap.set_title(f'Log2 Fold Changes - Top {len(log2fc_data)} Enriched Proteins', 
                            fontsize=14, fontweight='bold', pad=20)
                g.ax_heatmap.set_xlabel('Samples (vs Control Average)', fontsize=12)
                g.ax_heatmap.set_ylabel('Proteins', fontsize=12)

                # Rotate x labels
                g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=45, ha='right', fontsize=5)
                g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=8)

                # Save with padding
                fc_heatmap_path = os.path.join(viz_dir, f'heatmap_log2fc_{stats_params["threshold_label"]}.pdf')
                plt.savefig(fc_heatmap_path, dpi=300, bbox_inches=None, pad_inches=2.0)  # ← Use plt.savefig with padding
                plt.close('all')
                
                print(f"  ✓ Saved: heatmap_log2fc_{stats_params['threshold_label']}.pdf")
            else:
                print(f"  ⚠ No significant proteins to plot")

    # =========================================================================
    # 4. SUMMARY
    # =========================================================================
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    
    print(f"\nPlots saved to: {viz_dir}")
    
    if create_volcano:
        print(f"\nVolcano plots:")
        for comparison in comparisons:
            print(f"  • volcano_{comparison}_{stats_params['threshold_label']}.pdf (labeled)")
            print(f"  • volcano_{comparison}_{stats_params['threshold_label']}_clean.pdf (unlabeled)")
    
    if create_heatmap:
        print(f"\nHeatmaps:")
        print(f"  • heatmap_top_proteins_{stats_params['threshold_label']}.pdf (clustered intensities)")
        print(f"  • heatmap_log2fc_{stats_params['threshold_label']}.pdf (clustered fold changes)")
    
    print("\n" + "="*80)
    print("Analysis complete! Review your plots and results.")
    print("="*80 + "\n")













































def venn_ip(data, show_names=False, top_n=10):
    """
    Create Venn diagrams showing overlap between significant proteins across comparisons.
    
    Creates:
    - 2-way or 3-way Venn diagrams depending on number of comparisons
    - List of proteins unique to each comparison
    - List of proteins shared between comparisons
    
    Parameters:
    -----------
    data : dict
        Output from stat_ip()
    show_names : bool, optional
        Print protein names in each category (default: False)
    top_n : int, optional
        Number of top proteins to show per category (default: 10)
        
    Returns:
    --------
    dict containing:
        'overlaps' : dict with protein sets for each region
        'unique_counts' : dict with counts per comparison
        'shared_counts' : dict with shared protein counts
    
    Example:
    --------
    >>> data = stat_ip(data)
    >>> venn_results = venn_ip(data, show_names=True)
    >>> # Venn diagram saved to results/figures/viz/venn_diagram.pdf
    """
    
    print("\n" + "="*80)
    print("CREATING VENN DIAGRAMS")
    print("="*80)
    
    try:
        from matplotlib_venn import venn2, venn3
    except ImportError:
        print("\n⚠ matplotlib-venn not installed!")
        print("Installing matplotlib-venn...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib-venn", "--quiet"])
        print("✓ Installed! Please run venn_ip() again.")
        return None
    
    config = data['config']
    stats_results = data['stats_results']
    sig_proteins = data['significant_proteins']
    stats_params = data['stats_params']
    

    
    # Create viz output directory
    viz_dir = os.path.join(config['data_paths']['output_dir'], 'figures', 'viz')
    os.makedirs(viz_dir, exist_ok=True)
    
    print(f"\nOutput directory: {viz_dir}")
    
    # Get comparisons and their significant proteins (enriched only)
    comparisons = list(sig_proteins.keys())
    protein_sets = {}
    
    for comparison in comparisons:
        # Get enriched proteins only
        sig_mask = (stats_results[f'{comparison}_significant'] == 'significant') & \
                   (stats_results[f'{comparison}_log2FC'] > 0)
        protein_sets[comparison] = set(stats_results[sig_mask]['Accession'].tolist())
        print(f"\n{comparison}: {len(protein_sets[comparison])} enriched proteins")
    
    # =========================================================================
    # CREATE VENN DIAGRAM
    # =========================================================================
    n_comparisons = len(comparisons)
    
    if n_comparisons == 2:
        print(f"\nCreating 2-way Venn diagram...")
        
        # 2-way Venn
        comp1, comp2 = comparisons[0], comparisons[1]
        set1, set2 = protein_sets[comp1], protein_sets[comp2]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        v = venn2([set1, set2], set_labels=(comp1, comp2), ax=ax)
        
        # Style
        if v.get_patch_by_id('10'):
            v.get_patch_by_id('10').set_color('#E74C3C')
            v.get_patch_by_id('10').set_alpha(0.6)
        if v.get_patch_by_id('01'):
            v.get_patch_by_id('01').set_color('#3498DB')
            v.get_patch_by_id('01').set_alpha(0.6)
        if v.get_patch_by_id('11'):
            v.get_patch_by_id('11').set_color('#9B59B6')
            v.get_patch_by_id('11').set_alpha(0.6)
        
        ax.set_title(f'Enriched Protein Overlap\n({stats_params["threshold_label"]})', 
                    fontsize=14, fontweight='bold')
        
        # Save
        venn_path = os.path.join(viz_dir, f'venn_diagram_{stats_params["threshold_label"]}.pdf')
        plt.tight_layout()
        plt.savefig(venn_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved: venn_diagram_{stats_params['threshold_label']}.pdf")
        
        # Calculate overlaps
        unique1 = set1 - set2
        unique2 = set2 - set1
        shared = set1 & set2
        
        overlaps = {
            f'{comp1}_only': unique1,
            f'{comp2}_only': unique2,
            'shared': shared
        }
        
    elif n_comparisons == 3:
        print(f"\nCreating 3-way Venn diagram...")
        
        # 3-way Venn
        comp1, comp2, comp3 = comparisons[0], comparisons[1], comparisons[2]
        set1, set2, set3 = protein_sets[comp1], protein_sets[comp2], protein_sets[comp3]
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        v = venn3([set1, set2, set3], set_labels=(comp1, comp2, comp3), ax=ax)
        
        # Style patches
        colors = {
            '100': '#E74C3C',  # Only set1
            '010': '#3498DB',  # Only set2
            '001': '#2ECC71',  # Only set3
            '110': '#F39C12',  # set1 & set2
            '101': '#9B59B6',  # set1 & set3
            '011': '#1ABC9C',  # set2 & set3
            '111': '#34495E'   # All three
        }
        
        for region, color in colors.items():
            if v.get_patch_by_id(region):
                v.get_patch_by_id(region).set_color(color)
                v.get_patch_by_id(region).set_alpha(0.6)
        
        ax.set_title(f'Enriched Protein Overlap\n({stats_params["threshold_label"]})', 
                    fontsize=14, fontweight='bold')
        
        # Save
        venn_path = os.path.join(viz_dir, f'venn_diagram_{stats_params["threshold_label"]}.pdf')
        plt.tight_layout()
        plt.savefig(venn_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved: venn_diagram_{stats_params['threshold_label']}.pdf")
        
        # Calculate all overlaps
        overlaps = {
            f'{comp1}_only': set1 - set2 - set3,
            f'{comp2}_only': set2 - set1 - set3,
            f'{comp3}_only': set3 - set1 - set2,
            f'{comp1}_{comp2}_only': (set1 & set2) - set3,
            f'{comp1}_{comp3}_only': (set1 & set3) - set2,
            f'{comp2}_{comp3}_only': (set2 & set3) - set1,
            'all_three': set1 & set2 & set3
        }
    else:
        print(f"\n⚠ Can only create Venn diagrams for 2 or 3 comparisons")
        print(f"  You have {n_comparisons} comparisons")
        return None
    
    # =========================================================================
    # PRINT SUMMARY
    # =========================================================================
    print("\n" + "="*80)
    print("VENN DIAGRAM SUMMARY")
    print("="*80)
    
    for category, proteins in overlaps.items():
        print(f"\n{category}: {len(proteins)} proteins")
        
        if show_names and len(proteins) > 0:
            # Get gene symbols for these proteins
            protein_info = stats_results[stats_results['Accession'].isin(proteins)][
                ['Accession', 'Gene_Symbol']
            ].head(top_n)
            
            print(f"  Top {min(len(proteins), top_n)}:")
            for _, row in protein_info.iterrows():
                print(f"    • {row['Gene_Symbol']} ")
            
            if len(proteins) > top_n:
                print(f"    ... and {len(proteins) - top_n} more")
    
    # =========================================================================
    # SAVE OVERLAP LISTS TO CSV
    # =========================================================================
    csv_output_dir = os.path.join(config['data_paths']['output_dir'], 'tables')
    
    for category, proteins in overlaps.items():
        if len(proteins) > 0:
            # Get full info for these proteins
            overlap_df = stats_results[stats_results['Accession'].isin(proteins)].copy()
            
            # Sort by average log2FC across comparisons
            fc_cols = [c for c in overlap_df.columns if '_log2FC' in c]
            if len(fc_cols) > 0:
                overlap_df['avg_log2FC'] = overlap_df[fc_cols].mean(axis=1)
                overlap_df = overlap_df.sort_values('avg_log2FC', ascending=False)
            
            # Save
            csv_path = os.path.join(csv_output_dir, f'venn_{category}_{stats_params["threshold_label"]}.csv')
            overlap_df.to_csv(csv_path, index=False)
            print(f"\n✓ Saved: venn_{category}_{stats_params['threshold_label']}.csv")
    
    print("\n" + "="*80)
    print("VENN ANALYSIS COMPLETE")
    print("="*80 + "\n")
    
    # Return results
    return {
        'overlaps': overlaps,
        'protein_sets': protein_sets,
        'n_comparisons': n_comparisons
    }






















def boxplot_ip(data, top_n=20, group_by='category'):
    """
    Create boxplots showing log2 intensities of enriched proteins across samples.
    
    Shows the distribution of protein intensities for each condition, making it easy
    to see which proteins are truly enriched vs background.
    
    Parameters:
    -----------
    data : dict
        Output from stat_ip()
    top_n : int, optional
        Number of top enriched proteins to plot per category (default: 20)
        Set to None to plot all significant proteins
    group_by : str, optional
        How to group boxplots:
        - 'condition': One boxplot per condition (WT, d2d3, EV)
        - 'protein': One panel per protein showing all conditions
        - 'category': Separate plots for WT-only, Shared, d2d3-only
        Default: 'condition'
        
    Returns:
    --------
    None (saves plots to results/figures/viz/)
    
    Example:
    --------
    >>> data = stat_ip(data)
    >>> boxplot_ip(data, top_n=20, group_by='condition')
    >>> # Creates boxplot showing top 20 proteins across conditions
    
    >>> boxplot_ip(data, top_n=10, group_by='category')
    >>> # Creates separate plots for WT-only, Shared, and d2d3-only proteins
    """
    
    print("\n" + "="*80)
    print("CREATING BOXPLOTS OF ENRICHED PROTEINS")
    print("="*80)
    
    df = data['df']
    config = data['config']
    intensity_cols = data['intensity_cols']
    stats_results = data['stats_results']
    sig_proteins = data['significant_proteins']
    stats_params = data['stats_params']
    
    # Create viz output directory
    viz_dir = os.path.join(config['data_paths']['output_dir'], 'figures', 'viz')
    os.makedirs(viz_dir, exist_ok=True)
    
    print(f"\nOutput directory: {viz_dir}")
    
    # Get comparisons
    comparisons = list(sig_proteins.keys())
    
    # =========================================================================
    # COLLECT ALL UNIQUE ENRICHED PROTEINS
    # =========================================================================
    print(f"\nCollecting enriched proteins...")
    
    all_enriched = set()
    for comparison in comparisons:
        sig_mask = (stats_results[f'{comparison}_significant'] == 'significant') & \
                   (stats_results[f'{comparison}_log2FC'] > 0)
        enriched = stats_results[sig_mask]['Accession'].tolist()
        all_enriched.update(enriched)
        print(f"  {comparison}: {len(enriched)} enriched")
    
    print(f"\nTotal unique enriched proteins: {len(all_enriched)}")
    
    if len(all_enriched) == 0:
        print("  ⚠ No enriched proteins found!")
        return
    
    # Filter to enriched proteins
    enriched_df = stats_results[stats_results['Accession'].isin(all_enriched)].copy()
    
    # Sort by average log2FC
    fc_cols = [c for c in enriched_df.columns if '_log2FC' in c]
    enriched_df['avg_log2FC'] = enriched_df[fc_cols].mean(axis=1)
    enriched_df = enriched_df.sort_values('avg_log2FC', ascending=False)
    
    # =========================================================================
    # CATEGORIZE PROTEINS (for category grouping)
    # =========================================================================
    if group_by == 'category':
        print(f"\nCategorizing proteins by enrichment pattern...")
        
        protein_categories = {}
        for idx in enriched_df.index:
            accession = enriched_df.loc[idx, 'Accession']
            
            # Check which comparisons this protein is significant in
            is_sig = {}
            for comp in comparisons:
                sig_mask = (stats_results[f'{comp}_significant'] == 'significant') & \
                           (stats_results[f'{comp}_log2FC'] > 0)
                is_sig[comp] = accession in stats_results[sig_mask]['Accession'].values
            
            # Categorize (assuming comparisons are 'WT_vs_EV' and 'd2d3_vs_EV')
            if all(is_sig.values()):
                category = 'Shared'
            elif is_sig.get('WT_vs_EV', False):
                category = 'WT only'
            elif is_sig.get('d2d3_vs_EV', False):
                category = 'd2d3 only'
            else:
                category = 'Other'
            
            protein_categories[accession] = category
        
        enriched_df['category'] = enriched_df['Accession'].map(protein_categories)
        
        # Count proteins per category
        category_counts = enriched_df['category'].value_counts()
        print(f"  WT only: {category_counts.get('WT only', 0)} proteins")
        print(f"  Shared: {category_counts.get('Shared', 0)} proteins")
        print(f"  d2d3 only: {category_counts.get('d2d3 only', 0)} proteins")
    
    # =========================================================================
    # PREPARE DATA FOR PLOTTING
    # =========================================================================
    
    # Get intensities for enriched proteins
    plot_data = []
    
    for idx in enriched_df.index:
        accession = enriched_df.loc[idx, 'Accession']
        gene = enriched_df.loc[idx, 'Gene_Symbol']
        category = enriched_df.loc[idx, 'category'] if 'category' in enriched_df.columns else None
        
        # Get intensities from all samples
        for condition, cols in intensity_cols.items():
            for col in cols:
                intensity = df.loc[idx, col]
                if pd.notna(intensity):  # Only include non-missing values
                    plot_data.append({
                        'Protein': gene,
                        'Condition': condition,
                        'Log2_Intensity': intensity,
                        'Accession': accession,
                        'Category': category
                    })
    
    plot_df = pd.DataFrame(plot_data)
    
    print(f"\n  Data points: {len(plot_df)}")
    print(f"  Conditions: {plot_df['Condition'].unique().tolist()}")
    
    # =========================================================================
    # CREATE BOXPLOTS
    # =========================================================================
    
    if group_by == 'condition':
        print(f"\n[1/1] Creating boxplot grouped by condition...")
        
        # Take top N proteins overall
        if top_n is not None:
            proteins_to_plot = enriched_df.head(top_n)['Gene_Symbol'].tolist()
            plot_df = plot_df[plot_df['Protein'].isin(proteins_to_plot)]
            print(f"  Plotting top {len(proteins_to_plot)} proteins")
        else:
            print(f"  Plotting all {len(enriched_df)} proteins")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, max(8, len(plot_df['Protein'].unique()) * 0.4)))
        
        # Create boxplot
        sns.boxplot(
            data=plot_df,
            y='Protein',
            x='Log2_Intensity',
            hue='Condition',
            ax=ax,
            palette={'EV': '#95A5A6', 'WT': '#E74C3C', 'd2d3': '#3498DB'},
            linewidth=1.5
        )
        
        # Overlay individual points
        sns.stripplot(
            data=plot_df,
            y='Protein',
            x='Log2_Intensity',
            hue='Condition',
            ax=ax,
            dodge=True,
            alpha=0.6,
            size=4,
            palette={'EV': '#95A5A6', 'WT': '#E74C3C', 'd2d3': '#3498DB'},
            legend=False
        )
        
        # Labels and title
        ax.set_xlabel('Log2 Intensity', fontsize=12, fontweight='bold')
        ax.set_ylabel('Protein', fontsize=12, fontweight='bold')
        ax.set_title(f'Top {len(plot_df["Protein"].unique())} Enriched Proteins - Log2 Intensities', 
                    fontsize=14, fontweight='bold')
        ax.legend(title='Condition', fontsize=10, title_fontsize=11)
        ax.grid(axis='x', alpha=0.3)
        
        # Save
        boxplot_path = os.path.join(viz_dir, f'boxplot_enriched_proteins_{stats_params["threshold_label"]}.pdf')
        plt.tight_layout()
        plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: boxplot_enriched_proteins_{stats_params['threshold_label']}.pdf")
    
    elif group_by == 'category':
        print(f"\n[1/1] Creating boxplots grouped by category...")
        
        # Get categories
        categories = ['WT only', 'Shared', 'd2d3 only']
        categories = [cat for cat in categories if cat in plot_df['Category'].unique()]
        
        n_categories = len(categories)
        fig, axes = plt.subplots(1, n_categories, figsize=(8*n_categories, 10), sharey=True)
        if n_categories == 1:
            axes = [axes]
        
        for i, category in enumerate(categories):
            ax = axes[i]
            
            # Filter to this category
            cat_df = plot_df[plot_df['Category'] == category].copy()
            
            # Take top N proteins for this category
            category_proteins = enriched_df[enriched_df['category'] == category].head(top_n)['Gene_Symbol'].tolist()
            cat_df = cat_df[cat_df['Protein'].isin(category_proteins)]
            
            n_proteins = len(cat_df['Protein'].unique())
            print(f"  {category}: plotting {n_proteins} proteins")
            
            if len(cat_df) == 0:
                ax.text(0.5, 0.5, f'No proteins in\n{category}', 
                       ha='center', va='center', fontsize=12)
                ax.set_title(f'{category} (0 proteins)', fontsize=12, fontweight='bold')
                continue
            
            # Create boxplot
            sns.boxplot(
                data=cat_df,
                y='Protein',
                x='Log2_Intensity',
                hue='Condition',
                ax=ax,
                palette={'EV': '#95A5A6', 'WT': '#E74C3C', 'd2d3': '#3498DB'},
                linewidth=1.5
            )
            
            # Overlay points
            sns.stripplot(
                data=cat_df,
                y='Protein',
                x='Log2_Intensity',
                hue='Condition',
                ax=ax,
                dodge=True,
                alpha=0.6,
                size=4,
                palette={'EV': '#95A5A6', 'WT': '#E74C3C', 'd2d3': '#3498DB'},
                legend=False
            )
            
            # Labels
            ax.set_xlabel('Log2 Intensity', fontsize=12, fontweight='bold')
            if i == 0:
                ax.set_ylabel('Protein', fontsize=12, fontweight='bold')
            else:
                ax.set_ylabel('')
            ax.set_title(f'{category}\n({n_proteins} proteins)', fontsize=12, fontweight='bold')
            ax.legend(title='Condition', fontsize=10, title_fontsize=11)
            ax.grid(axis='x', alpha=0.3)
        
        # Overall title
        fig.suptitle('Enriched Proteins by Category', fontsize=14, fontweight='bold', y=0.98)
        
        # Save
        boxplot_path = os.path.join(viz_dir, f'boxplot_by_category_{stats_params["threshold_label"]}.pdf')
        plt.tight_layout()
        plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: boxplot_by_category_{stats_params['threshold_label']}.pdf")
    
    elif group_by == 'protein':
        print(f"\n[1/1] Creating individual boxplots per protein...")
        
        # Take top N proteins
        if top_n is not None:
            proteins_to_plot = enriched_df.head(top_n)
            print(f"  Plotting top {len(proteins_to_plot)} proteins")
        else:
            proteins_to_plot = enriched_df
            print(f"  Plotting all {len(enriched_df)} proteins")
        
        plot_df = plot_df[plot_df['Protein'].isin(proteins_to_plot['Gene_Symbol'])]
        
        # Determine grid layout
        n_proteins = len(proteins_to_plot)
        n_cols = min(3, n_proteins)
        n_rows = int(np.ceil(n_proteins / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
        if n_proteins == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        # Plot each protein
        for i, (idx, row) in enumerate(proteins_to_plot.iterrows()):
            gene = row['Gene_Symbol']
            protein_data = plot_df[plot_df['Protein'] == gene]
            
            ax = axes[i]
            
            # Boxplot
            sns.boxplot(
                data=protein_data,
                x='Condition',
                y='Log2_Intensity',
                ax=ax,
                palette={'EV': '#95A5A6', 'WT': '#E74C3C', 'd2d3': '#3498DB'},
                linewidth=1.5
            )
            
            # Points
            sns.stripplot(
                data=protein_data,
                x='Condition',
                y='Log2_Intensity',
                ax=ax,
                color='black',
                alpha=0.6,
                size=6
            )
            
            ax.set_title(f'{gene}', fontsize=11, fontweight='bold')
            ax.set_xlabel('Condition', fontsize=10)
            ax.set_ylabel('Log2 Intensity', fontsize=10)
            ax.grid(axis='y', alpha=0.3)
        
        # Hide empty subplots
        for i in range(n_proteins, len(axes)):
            axes[i].set_visible(False)
        
        # Save
        boxplot_path = os.path.join(viz_dir, f'boxplot_individual_proteins_{stats_params["threshold_label"]}.pdf')
        plt.tight_layout()
        plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: boxplot_individual_proteins_{stats_params['threshold_label']}.pdf")
    
    else:
        print(f"\n  ⚠ Invalid group_by parameter: {group_by}")
        print(f"     Valid options: 'condition', 'category', 'protein'")
        return
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*80)
    print("BOXPLOT CREATION COMPLETE")
    print("="*80)
    
    print(f"\nPlots saved to: {viz_dir}")
    print(f"\nBoxplots:")
    if group_by == 'condition':
        print(f"  • boxplot_enriched_proteins_{stats_params['threshold_label']}.pdf")
    elif group_by == 'category':
        print(f"  • boxplot_by_category_{stats_params['threshold_label']}.pdf")
        print(f"    Shows WT-only, Shared, and d2d3-only proteins side-by-side")
    else:
        print(f"  • boxplot_individual_proteins_{stats_params['threshold_label']}.pdf")
    
    print("\n" + "="*80 + "\n")


















    





# ============================================================================
# ============================================================================
# ============================================================================
#############################
# == SUMMARY IP FUNCTION == #
#############################


def summary_ip(data, output_format='txt'):
    """
    Generate analysis summary report.
    
    Creates:
    - Text summary of results
    - Tables of significant proteins
    - Statistical summaries
    
    Parameters:
    -----------
    data : dict
        Output from stat_ip()
    output_format : str
        Format for report: 'txt', 'html', or 'both'
        
    Returns:
    --------
    None (saves report to results/)
    """
    # TODO: We'll implement this together
    pass


























# ============================================================================
# ============================================================================
# ============================================================================
# ============================================================================
# ============================================================================
# ============================================================================
# ============================================================================
# ============================================================================
# ============================================================================
# ============================================================================
# ============================================================================
# ============================================================================
# HELPER FUNCTIONS (internal use)
# ============================================================================



# load config
# ============================================================================

def _load_config(config_path):
    """Load YAML config file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)







# Identify intensity columns
# ============================================================================

def _identify_intensity_columns(df, config):
    """Auto-detect intensity columns from config pattern"""
    intensity_prefix = config['data_columns']['intensity_prefix']
    
    intensity_cols = {}
    # Control
    control = config['conditions']['control']
    pattern = f"{intensity_prefix} {control}"
    intensity_cols[control] = [c for c in df.columns if c.startswith(pattern)]
    
    # Treatments
    for treatment in config['conditions']['treatments']:
        pattern = f"{intensity_prefix} {treatment}"
        intensity_cols[treatment] = [c for c in df.columns if c.startswith(pattern)]
    
    return intensity_cols




# Create output directories
# ============================================================================

def _create_output_dirs(base_dir):
    """Create organized output directory structure"""
    dirs = {
        'base': base_dir,
        'figures': f"{base_dir}/figures",
        'qc': f"{base_dir}/figures/qc",
        # 'stats': f"{base_dir}/figures/stats",
        'viz': f"{base_dir}/figures/viz",
        'tables': f"{base_dir}/tables"
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs








# Save data
# ============================================================================

def save_data(data, filename=None):
    """
    Save analysis data to pickle file for sequential workflow.
    
    Parameters:
    -----------
    data : dict
        Analysis data dictionary (output from prep_ip, norm_ip, etc.)
    filename : str, optional
        Custom filename. If None, uses default based on output_dir in config
        
    Returns:
    --------
    str : Path where data was saved
    
    Example:
    --------
    >>> data = prep_ip('config/experiment.yaml')
    >>> save_data(data)  # Saves to results/data_after_prep.pkl
    """
    import pickle
    
    # Determine save path
    if filename is None:
        output_dir = data['config']['data_paths']['output_dir']
        filename = os.path.join(output_dir, 'data_checkpoint.pkl')
    
    # Save
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    
    # Get file size for user feedback
    size_mb = os.path.getsize(filename) / (1024 * 1024)
    
    print(f"\n{'='*80}")
    print(f"DATA SAVED")
    print(f"{'='*80}")
    print(f"Location: {filename}")
    print(f"Size: {size_mb:.1f} MB")
    print(f"\nTo load this data later:")
    print(f"  from ipms.analysis import load_data")
    print(f"  data = load_data('{filename}')")
    print(f"{'='*80}\n")
    
    return filename





# Load data
# ============================================================================

def load_data(filepath):
    """
    Load analysis data from pickle file.
    
    Parameters:
    -----------
    filepath : str
        Path to saved pickle file
        
    Returns:
    --------
    dict : Analysis data dictionary
    
    Example:
    --------
    >>> from ipms.analysis import load_data
    >>> data = load_data('results/data_after_prep.pkl')
    >>> qc_ip(data)  # Continue from where you left off
    """
    import pickle
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    print(f"\n{'='*80}")
    print(f"LOADING DATA")
    print(f"{'='*80}")
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    # Get file size
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    
    print(f"Location: {filepath}")
    print(f"Size: {size_mb:.1f} MB")
    
    # Print what's in the data
    if 'metadata' in data:
        print(f"\nData contains:")
        print(f"  Proteins: {data['metadata']['n_proteins']}")
        print(f"  Samples: {data['metadata']['n_samples']}")
        print(f"  Conditions: {data['metadata']['conditions']}")
    
    print(f"{'='*80}\n")
    
    return data






