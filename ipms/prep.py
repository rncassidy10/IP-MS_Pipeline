"""
Data preparation functions for IP-MS pipeline.

Handles loading raw proteomics data, filtering, gene symbol mapping,
and contaminant removal.
"""

import os
import sys

import numpy as np
import pandas as pd
import yaml

from .utils import _create_output_dirs, save_data


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

    Parameters
    ----------
    config_path : str
        Path to YAML configuration file.

    Returns
    -------
    dict
        Dictionary containing:
        - 'df': pd.DataFrame with cleaned protein data
        - 'config': loaded configuration dictionary
        - 'intensity_cols': maps condition names to intensity column names
        - 'metadata': summary statistics about the data
        - 'output_dirs': paths to output directories

    Example
    -------
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

    print(f"\n> Configuration loaded")
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
    print(f"  > Loaded {df.shape[0]} proteins, {df.shape[1]} columns")

    # =========================================================================
    # 3. IDENTIFY INTENSITY COLUMNS
    # =========================================================================
    print(f"\n[2/8] Identifying intensity columns...")

    intensity_prefix = config['data_columns']['intensity_prefix']

    intensity_cols = {}

    control = config['conditions']['control']
    cols = [c for c in df.columns if intensity_prefix in c and control in c]
    intensity_cols[control] = sorted(cols)
    print(f"  {control}: {len(cols)} replicates")

    for treatment in config['conditions']['treatments']:
        cols = [c for c in df.columns if intensity_prefix in c and treatment in c]
        intensity_cols[treatment] = sorted(cols)
        print(f"  {treatment}: {len(cols)} replicates")

    all_intensity_cols = []
    for cols in intensity_cols.values():
        all_intensity_cols.extend(cols)

    print(f"  > Total intensity columns: {len(all_intensity_cols)}")

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
        print(f"  > Removed {removed} proteins with < {min_peptides} peptides")
        print(f"    Remaining: {len(df)} proteins")
    else:
        print(f"  Warning: Column '{peptide_col}' not found, skipping peptide filter")

    # =========================================================================
    # 5. FILTER LOW QUALITY PROTEINS (BY MISSINGNESS IN TREATMENTS)
    # =========================================================================
    print(f"\n[4/8] Filtering proteins by missingness in treatments...")

    before = len(df)

    treatment_conditions = config['conditions']['treatments']
    keep_protein = pd.Series(False, index=df.index)

    for treatment in treatment_conditions:
        treatment_cols = intensity_cols[treatment]
        n_replicates = len(treatment_cols)
        min_required = int(np.ceil(n_replicates * 0.5))

        valid_count = df[treatment_cols].notna().sum(axis=1)
        passes_threshold = valid_count >= min_required
        keep_protein = keep_protein | passes_threshold

        n_passing = passes_threshold.sum()
        print(f"  {treatment}: {n_passing} proteins have >={min_required}/{n_replicates} valid values")

    df = df[keep_protein].copy()
    removed = before - len(df)

    print(f"\n  > Removed {removed} low-quality proteins")
    print(f"    Remaining: {len(df)} proteins")
    print(f"    (Kept proteins present in >=50% of replicates in at least one treatment)")

    # =========================================================================
    # 6. ADD GENE SYMBOLS (if missing)
    # =========================================================================
    print(f"\n[5/8] Checking for gene symbols...")

    gene_col = config['data_columns']['gene_symbol']
    protein_col = config['data_columns']['protein_id']

    if gene_col not in df.columns or df[gene_col].isna().all():
        print(f"\n  Gene symbols missing - mapping from protein IDs using mygene...")

        try:
            import mygene
            mg = mygene.MyGeneInfo()

            protein_ids = df[protein_col].dropna().unique().tolist()
            print(f"  Querying mygene for {len(protein_ids)} proteins...")

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
                    print(f"    > Mapped {newly_mapped} proteins")

            original_to_gene = {}
            for orig_id, clean_id in cleaned_ids.items():
                if clean_id in protein_to_gene:
                    original_to_gene[orig_id] = protein_to_gene[clean_id]

            df[gene_col] = df[protein_col].map(original_to_gene)

            still_unmapped = df[gene_col].isna().sum()
            if still_unmapped > 0:
                df[gene_col] = df[gene_col].fillna(df[protein_col])

            successful = len(original_to_gene)
            print(f"\n  > Successfully mapped {successful}/{len(protein_ids)} proteins to gene symbols")
            if still_unmapped > 0:
                print(f"  Warning: {still_unmapped} proteins kept protein ID as identifier")

        except ImportError:
            print(f"  Error: mygene not installed. Run: pip install mygene")
            return None

        except Exception as e:
            print(f"  Warning: Could not map gene symbols: {e}")
            print(f"  Using protein IDs as gene names instead")
            df[gene_col] = df[protein_col]
    else:
        print(f"\n[5/8] Gene symbols found in data")
        valid_symbols = df[gene_col].notna().sum()
        print(f"  > {valid_symbols} proteins have gene symbols")

    # =========================================================================
    # 6. REMOVE MANUAL CONTAMINANTS (before CRAPome)
    # =========================================================================
    print(f"\n[6/8] Removing manual contaminants...")

    manual_contaminants = config.get('manual_contaminants', [
        'KRT', 'Keratin', 'keratin',
        'TRYP', 'Trypsin', 'trypsin',
        'ALB', 'Albumin', 'albumin',
        'IGG', 'Immunoglobulin', 'immunoglobulin'
    ])

    if len(manual_contaminants) > 0:
        desc_col = 'Description' if 'Description' in df.columns else None

        if desc_col:
            df['_search'] = df[protein_col].fillna('') + ' ' + df[desc_col].fillna('')
        else:
            df['_search'] = df[protein_col].fillna('')

        before = len(df)

        for contaminant in manual_contaminants:
            mask = df['_search'].str.contains(contaminant, case=False, na=False)
            removed = df[mask]

            if len(removed) > 0:
                print(f"    Found {len(removed)} proteins matching '{contaminant}'")
                df = df[~mask].copy()

        df = df.drop(columns=['_search'])

        removed = before - len(df)
        print(f"\n  > Removed {removed} manual contaminant proteins")
        print(f"    Remaining: {len(df)} proteins")
    else:
        print(f"  No manual contaminants specified, skipping")

    # =========================================================================
    # 7. REMOVE CONTAMINANTS USING CRAPOME DATABASE
    # =========================================================================
    print(f"\n[7/8] Filtering contaminants using CRAPome database...")

    crapome_file = config['data_paths'].get('crapome_list', 'data/raw/Crapome_list.xlsx')

    if os.path.exists(crapome_file):
        crapome_df = pd.read_excel(crapome_file)

        frequency_threshold = config.get('crapome_filtering', {}).get('freq_threshold', 0.30)
        spectral_count_threshold = config.get('crapome_filtering', {}).get('spectral_count_threshold', 50)

        contaminant_mask = (
            (crapome_df['FREQ'] > frequency_threshold) &
            (crapome_df['ALL_NUMSPECSTOT'] > spectral_count_threshold)
        )

        contaminant_ids = set(crapome_df[contaminant_mask]['PROTID'].tolist())
        contaminant_genes = set(crapome_df[contaminant_mask]['GENEID'].tolist())

        print(f"  CRAPome database loaded: {len(crapome_df)} proteins")
        print(f"  Identified {len(contaminant_ids)} contaminant protein IDs")
        print(f"  Identified {len(contaminant_genes)} contaminant gene symbols")
        print(f"  Thresholds: FREQ > {frequency_threshold:.0%}, Spectral Count > {spectral_count_threshold}")

        before = len(df)

        crapome_clean_ids = set([pid.split('.')[0] for pid in contaminant_ids])
        df_clean_ids = df[protein_col].str.split('.').str[0]

        remove_by_id = df_clean_ids.isin(crapome_clean_ids)

        if gene_col in df.columns:
            remove_by_gene = df[gene_col].isin(contaminant_genes)
            remove_mask = remove_by_id | remove_by_gene
        else:
            remove_mask = remove_by_id

        df = df[~remove_mask].copy()

        removed = before - len(df)
        print(f"\n  > Removed {removed} CRAPome contaminants")
        print(f"    Remaining: {len(df)} proteins")

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
        print(f"  Warning: CRAPome file not found at {crapome_file}")
        print(f"  Skipping CRAPome filtering")

    # =========================================================================
    # 8. SAVE FILTERED DATA AS CSV (for reference)
    # =========================================================================
    print(f"\n[8/8] Saving filtered data as CSV for reference...")

    csv_output_dir = os.path.join(config['data_paths']['output_dir'], 'tables')
    os.makedirs(csv_output_dir, exist_ok=True)

    csv_path = os.path.join(csv_output_dir, 'filtered_proteins_after_prep.csv')
    df.to_csv(csv_path, index=False)

    print(f"  > Saved: filtered_proteins_after_prep.csv")
    print(f"    Location: {csv_output_dir}")
    print(f"    {len(df)} proteins x {len(df.columns)} columns")

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

    print(f"  > Saved: filtered_proteins_summary.csv")
    print(f"    (Protein info + intensities only)")

    # =========================================================================
    # 9. CHECK DATA QUALITY
    # =========================================================================
    print(f"\nData quality summary...")

    print(f"\n  Missing values by condition:")
    for condition, cols in intensity_cols.items():
        total_values = len(df) * len(cols)
        missing = df[cols].isna().sum().sum()
        pct_missing = (missing / total_values) * 100
        print(f"    {condition}: {pct_missing:.1f}% missing")

    print(f"\n  Proteins detected per condition:")
    for condition, cols in intensity_cols.items():
        detected = (df[cols].notna().any(axis=1)).sum()
        print(f"    {condition}: {detected} proteins")

    # =========================================================================
    # 10. CREATE OUTPUT DIRECTORIES
    # =========================================================================
    output_dir = config['data_paths']['output_dir']
    output_dirs = _create_output_dirs(output_dir)

    print(f"\n> Output directories created at: {output_dir}")

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

    return_data = {
        'df': df,
        'config': config,
        'intensity_cols': intensity_cols,
        'metadata': metadata,
        'output_dirs': output_dirs
    }

    # Auto-save for sequential workflow
    save_path = os.path.join(output_dir, 'data_after_prep.pkl')
    save_data(return_data, save_path)

    return return_data
