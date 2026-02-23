"""
STRING Database Analysis for IP-MS Pipeline
============================================

Queries the STRING database API to perform:
- Protein-protein interaction (PPI) network analysis
- Functional enrichment (GO, KEGG, Reactome, Pfam)
- PPI enrichment testing (are your proteins more connected than expected?)
- Network image retrieval

Requires: requests (already in requirements.txt via pip)
Internet connection required for STRING API calls.

Author: Richard Cassidy
"""

import os
import sys
import time
import warnings
import pandas as pd
import numpy as np

# ============================================================================
# STRING API HELPERS
# ============================================================================

STRING_API_URL = "https://string-db.org/api"
CALLER_IDENTITY = "ipms_pipeline"  # Identifies your app to STRING


def _string_request(method, params, output_format='tsv', max_retries=3):
    """
    Make a request to the STRING API with retry logic.
    
    Parameters:
    -----------
    method : str
        API method (e.g., 'get_string_ids', 'enrichment', 'network', 'ppi_enrichment')
    params : dict
        Request parameters
    output_format : str
        Response format ('tsv', 'json', 'image')
    max_retries : int
        Number of retries on failure
        
    Returns:
    --------
    requests.Response object
    """
    try:
        import requests
    except ImportError:
        print("\n⚠ requests module not installed!")
        print("Install with: pip install requests")
        return None
    
    url = f"{STRING_API_URL}/{output_format}/{method}"
    params['caller_identity'] = CALLER_IDENTITY
    
    for attempt in range(max_retries):
        try:
            response = requests.post(url, data=params, timeout=60)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"  ⚠ API request failed (attempt {attempt + 1}/{max_retries}), "
                      f"retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"\n✗ STRING API request failed after {max_retries} attempts: {e}")
                return None


def _map_to_string_ids(gene_list, species=9606):
    """
    Map gene symbols to STRING identifiers.
    
    Parameters:
    -----------
    gene_list : list
        List of gene symbols
    species : int
        NCBI taxonomy ID (9606 = human)
        
    Returns:
    --------
    dict mapping gene_symbol -> string_id, or None on failure
    """
    params = {
        "identifiers": "\r".join(gene_list),
        "species": species,
        "limit": 1,
        "echo_query": 1
    }
    
    response = _string_request("get_string_ids", params)
    if response is None:
        return None
    
    import io
    df = pd.read_csv(io.StringIO(response.text), sep='\t')
    
    if df.empty:
        print("  ⚠ No STRING IDs found for input genes")
        return None
    
    # Create mapping: queryItem -> stringId
    mapping = dict(zip(df['queryItem'], df['stringId']))
    
    # Report unmapped
    mapped = set(df['queryItem'].tolist())
    unmapped = set(gene_list) - mapped
    if unmapped:
        print(f"  ⚠ {len(unmapped)} genes not found in STRING: {', '.join(sorted(unmapped))}")
    
    print(f"  ✓ Mapped {len(mapping)}/{len(gene_list)} genes to STRING IDs")
    
    return mapping


# ============================================================================
# MAIN FUNCTION: string_ip()
# ============================================================================

def string_ip(data, species=9606, score_threshold=400, 
              categories=None, fdr_threshold=0.05, top_n_terms=10,
              save_network_image=True, exclude_genes=None):
    """
    Perform STRING database analysis on significant proteins from IP-MS data.
    
    Queries the STRING API to:
    1. Map gene symbols to STRING identifiers
    2. Test PPI enrichment (are proteins more connected than expected?)
    3. Retrieve functional enrichment (GO, KEGG, Reactome)
    4. Get network interactions between proteins
    5. Optionally save network images
    
    Runs analysis on each Venn category separately (WT-only, d2d3-only, shared)
    AND on all enriched proteins combined.
    
    Parameters:
    -----------
    data : dict
        Output from stat_ip() — must contain 'stats_results' and 'significant_proteins'
    species : int, optional
        NCBI taxonomy ID (default: 9606 for human)
    score_threshold : int, optional
        Minimum STRING confidence score 0-1000 (default: 400 = medium confidence)
        Options: 150 (low), 400 (medium), 700 (high), 900 (highest)
    categories : list, optional
        Which Venn categories to analyze. Default: all available.
        Options: 'WT_vs_EV_only', 'd2d3_vs_EV_only', 'shared', 'all_enriched'
    fdr_threshold : float, optional
        FDR cutoff for reporting enriched terms (default: 0.05)
    top_n_terms : int, optional
        Number of top enrichment terms to display per category (default: 10)
    save_network_image : bool, optional
        Download STRING network images as PDF (default: True)
    exclude_genes : list, optional
        Gene symbols to exclude from analysis (e.g., cell-type contaminants).
        These are removed BEFORE querying STRING, and the exclusion is logged.
        Example: ['DSG1', 'DSC1', 'SPRR1B', 'SPRR2D', 'FLG2', 'PKP1']
        
    Returns:
    --------
    dict containing:
        'ppi_enrichment' : dict of PPI enrichment results per category
        'functional_enrichment' : dict of DataFrames per category
        'network' : dict of network edge DataFrames per category
        'string_ids' : dict of gene->STRING ID mappings per category
        'summary' : DataFrame summarizing key results
        'excluded_genes' : list of genes that were excluded (for documentation)
    
    Example:
    --------
    >>> data = stat_ip(data)
    >>> string_results = string_ip(data)
    >>> # Results saved to results/tables/string/
    >>> # Images saved to results/figures/viz/
    
    >>> # Exclude epithelial contaminants (e.g., in BEAS-2B cells)
    >>> epithelial_contaminants = ['DSG1', 'DSC1', 'PKP1', 'SPRR1B', 'SPRR2D', 
    ...                            'FLG2', 'ECM1', 'S100A8', 'LYZ']
    >>> string_results = string_ip(data, exclude_genes=epithelial_contaminants)
    
    >>> # High confidence, custom exclusions
    >>> string_results = string_ip(data, score_threshold=700, 
    ...                            exclude_genes=['DSG1', 'DSC1'])
    """
    
    print("\n" + "=" * 80)
    print("STRING DATABASE ANALYSIS")
    print("=" * 80)
    
    try:
        import requests
    except ImportError:
        print("\n⚠ requests module not installed!")
        print("Installing requests...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "requests", "--quiet"])
        print("✓ Installed! Please run string_ip() again.")
        return None
    
    # =========================================================================
    # 1. VALIDATE INPUT AND SETUP
    # =========================================================================
    print("\n[1/5] Validating input data...")
    
    # Check required keys
    required_keys = ['stats_results', 'significant_proteins', 'config', 'stats_params']
    for key in required_keys:
        if key not in data:
            print(f"\n✗ Missing required key: '{key}'")
            print("  Run stat_ip() first before string_ip()")
            return None
    
    config = data['config']
    stats_results = data['stats_results']
    sig_proteins = data['significant_proteins']
    stats_params = data['stats_params']
    
    # Create output directories
    string_table_dir = os.path.join(config['data_paths']['output_dir'], 'tables', 'string')
    viz_dir = os.path.join(config['data_paths']['output_dir'], 'figures', 'viz')
    os.makedirs(string_table_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)
    
    print(f"  Species: {species} ({'Homo sapiens' if species == 9606 else species})")
    print(f"  Score threshold: {score_threshold}")
    print(f"  FDR threshold: {fdr_threshold}")
    print(f"  Output: {string_table_dir}")
    
    # Handle gene exclusions
    exclude_accessions = set()
    excluded_gene_list = []
    
    if exclude_genes is not None and len(exclude_genes) > 0:
        exclude_genes = [g.strip() for g in exclude_genes if g and isinstance(g, str)]
        excluded_gene_list = list(exclude_genes)
        
        # Map gene symbols to accessions for filtering
        for gene in exclude_genes:
            matches = stats_results[stats_results['Gene_Symbol'] == gene]['Accession'].tolist()
            exclude_accessions.update(matches)
        
        print(f"\n  ⚠ EXCLUDING {len(exclude_genes)} genes (cell-type contaminants):")
        for g in exclude_genes:
            print(f"    ✗ {g}")
        
        if len(exclude_accessions) != len(exclude_genes):
            not_found = set(exclude_genes) - set(
                stats_results[stats_results['Accession'].isin(exclude_accessions)]['Gene_Symbol'].tolist()
            )
            if not_found:
                print(f"  Note: {len(not_found)} genes not found in data: {', '.join(not_found)}")
        
        # Save exclusion log
        exclusion_log_path = os.path.join(
            string_table_dir, 
            f'excluded_genes_{stats_params["threshold_label"]}.txt'
        )
        with open(exclusion_log_path, 'w') as f:
            f.write("# Genes excluded from STRING analysis\n")
            f.write(f"# Reason: Cell-type specific contaminants (BEAS-2B epithelial)\n")
            f.write(f"# Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
            f.write(f"# Threshold: {stats_params['threshold_label']}\n\n")
            for g in sorted(exclude_genes):
                f.write(f"{g}\n")
        print(f"  ✓ Exclusion list saved: excluded_genes_{stats_params['threshold_label']}.txt")
    
    # =========================================================================
    # 2. BUILD PROTEIN CATEGORIES (same logic as venn_ip)
    # =========================================================================
    print("\n[2/5] Building protein categories...")
    
    comparisons = list(sig_proteins.keys())
    
    # Get enriched protein sets per comparison (with exclusions applied)
    protein_sets = {}
    for comparison in comparisons:
        sig_mask = (stats_results[f'{comparison}_significant'] == 'significant') & \
                   (stats_results[f'{comparison}_log2FC'] > 0)
        raw_set = set(stats_results[sig_mask]['Accession'].tolist())
        protein_sets[comparison] = raw_set - exclude_accessions
        
        n_removed = len(raw_set) - len(protein_sets[comparison])
        removed_str = f" ({n_removed} excluded)" if n_removed > 0 else ""
        print(f"  {comparison}: {len(protein_sets[comparison])} enriched proteins{removed_str}")
    
    # Build Venn-style categories
    protein_categories = {}
    
    if len(comparisons) == 2:
        comp1, comp2 = comparisons[0], comparisons[1]
        set1, set2 = protein_sets[comp1], protein_sets[comp2]
        
        protein_categories[f'{comp1}_only'] = set1 - set2
        protein_categories[f'{comp2}_only'] = set2 - set1
        protein_categories['shared'] = set1 & set2
        protein_categories['all_enriched'] = set1 | set2
        
    elif len(comparisons) == 3:
        comp1, comp2, comp3 = comparisons[0], comparisons[1], comparisons[2]
        set1, set2, set3 = protein_sets[comp1], protein_sets[comp2], protein_sets[comp3]
        
        protein_categories[f'{comp1}_only'] = set1 - set2 - set3
        protein_categories[f'{comp2}_only'] = set2 - set1 - set3
        protein_categories[f'{comp3}_only'] = set3 - set1 - set2
        protein_categories['shared'] = set1 & set2  # For 3-way, this is all-three
        protein_categories['all_enriched'] = set1 | set2 | set3
    else:
        # Single comparison
        protein_categories['all_enriched'] = protein_sets[comparisons[0]]
    
    # Filter to requested categories
    if categories is not None:
        protein_categories = {k: v for k, v in protein_categories.items() if k in categories}
    
    # Remove empty categories
    protein_categories = {k: v for k, v in protein_categories.items() if len(v) > 0}
    
    print(f"\n  Categories to analyze:")
    for cat, proteins in protein_categories.items():
        print(f"    {cat}: {len(proteins)} proteins")
    
    # =========================================================================
    # 3. QUERY STRING FOR EACH CATEGORY
    # =========================================================================
    print("\n[3/5] Querying STRING database...")
    
    results = {
        'ppi_enrichment': {},
        'functional_enrichment': {},
        'network': {},
        'string_ids': {},
        'summary': []
    }
    
    for cat_name, accession_set in protein_categories.items():
        print(f"\n{'─' * 60}")
        print(f"  Category: {cat_name} ({len(accession_set)} proteins)")
        print(f"{'─' * 60}")
        
        # Get gene symbols for these accessions
        gene_df = stats_results[stats_results['Accession'].isin(accession_set)]
        gene_list = gene_df['Gene_Symbol'].dropna().unique().tolist()
        
        # Remove any empty strings or problematic entries
        gene_list = [g for g in gene_list if g and isinstance(g, str) and len(g) > 0]
        
        if len(gene_list) < 2:
            print(f"  ⚠ Need at least 2 genes for STRING analysis, got {len(gene_list)}")
            continue
        
        print(f"  Gene symbols: {', '.join(gene_list[:10])}"
              f"{'...' if len(gene_list) > 10 else ''}")
        
        # ----- Map to STRING IDs -----
        print(f"\n  Mapping to STRING IDs...")
        id_mapping = _map_to_string_ids(gene_list, species=species)
        
        if id_mapping is None or len(id_mapping) < 2:
            print(f"  ⚠ Could not map enough genes, skipping {cat_name}")
            continue
        
        results['string_ids'][cat_name] = id_mapping
        mapped_genes = list(id_mapping.keys())
        string_ids = list(id_mapping.values())
        
        # Small delay to be respectful to STRING servers
        time.sleep(1)
        
        # ----- PPI Enrichment Test -----
        print(f"\n  Testing PPI enrichment...")
        ppi_params = {
            "identifiers": "\r".join(mapped_genes),
            "species": species,
            "required_score": score_threshold
        }
        
        ppi_response = _string_request("ppi_enrichment", ppi_params, output_format='json')
        
        if ppi_response is not None:
            import json
            try:
                ppi_raw = ppi_response.json()
                
                # STRING API returns a list — grab first element if so
                if isinstance(ppi_raw, list) and len(ppi_raw) > 0:
                    ppi_data = ppi_raw[0]
                elif isinstance(ppi_raw, dict):
                    ppi_data = ppi_raw
                else:
                    print(f"    ⚠ Unexpected PPI response format: {type(ppi_raw)}")
                    ppi_data = {}
                
                results['ppi_enrichment'][cat_name] = ppi_data
                
                n_edges = ppi_data.get('number_of_edges', 'N/A')
                expected = ppi_data.get('expected_number_of_edges', 'N/A')
                ppi_pval = ppi_data.get('p_value', 'N/A')
                
                print(f"    Observed interactions: {n_edges}")
                print(f"    Expected (random):     {expected}")
                print(f"    PPI enrichment p-value: {ppi_pval}")
                
                if isinstance(ppi_pval, (int, float)) and ppi_pval < 0.05:
                    print(f"    → ✓ Significant! Your proteins interact more than expected")
                elif isinstance(ppi_pval, (int, float)):
                    print(f"    → Not significant (proteins may not form a tight network)")
                    
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                print(f"    ⚠ Could not parse PPI enrichment response: {e}")
                results['ppi_enrichment'][cat_name] = None
        
        time.sleep(1)
        
        # ----- Functional Enrichment -----
        print(f"\n  Running functional enrichment...")
        enrich_params = {
            "identifiers": "\r".join(mapped_genes),
            "species": species
        }
        
        enrich_response = _string_request("enrichment", enrich_params)
        
        if enrich_response is not None and enrich_response.text.strip():
            import io
            try:
                enrich_df = pd.read_csv(io.StringIO(enrich_response.text), sep='\t')
                
                if not enrich_df.empty and 'fdr' in enrich_df.columns:
                    # Filter by FDR
                    sig_enrichment = enrich_df[enrich_df['fdr'] < fdr_threshold].copy()
                    
                    results['functional_enrichment'][cat_name] = enrich_df
                    
                    # Print summary by category
                    if 'category' in sig_enrichment.columns:
                        print(f"\n    Significant terms (FDR < {fdr_threshold}):")
                        for term_cat in sig_enrichment['category'].unique():
                            cat_terms = sig_enrichment[sig_enrichment['category'] == term_cat]
                            print(f"      {term_cat}: {len(cat_terms)} terms")
                        
                        # Show top terms
                        print(f"\n    Top {top_n_terms} enriched terms:")
                        top_terms = sig_enrichment.head(top_n_terms)
                        for _, row in top_terms.iterrows():
                            term_desc = row.get('description', row.get('term', 'N/A'))
                            fdr_val = row.get('fdr', 'N/A')
                            category = row.get('category', '')
                            
                            # Truncate long descriptions
                            if len(str(term_desc)) > 60:
                                term_desc = str(term_desc)[:57] + '...'
                            
                            print(f"      [{category}] {term_desc} (FDR={fdr_val:.2e})")
                    
                    if len(sig_enrichment) == 0:
                        print(f"    ⚠ No terms below FDR {fdr_threshold}")
                        print(f"    Total terms tested: {len(enrich_df)}")
                        if len(enrich_df) > 0 and 'fdr' in enrich_df.columns:
                            print(f"    Best FDR: {enrich_df['fdr'].min():.2e}")
                else:
                    print(f"    ⚠ No enrichment results returned")
                    results['functional_enrichment'][cat_name] = pd.DataFrame()
                    
            except Exception as e:
                print(f"    ⚠ Could not parse enrichment results: {e}")
                results['functional_enrichment'][cat_name] = pd.DataFrame()
        else:
            print(f"    ⚠ No enrichment response from STRING")
            results['functional_enrichment'][cat_name] = pd.DataFrame()
        
        time.sleep(1)
        
        # ----- Network Interactions -----
        print(f"\n  Retrieving network interactions...")
        network_params = {
            "identifiers": "\r".join(mapped_genes),
            "species": species,
            "required_score": score_threshold
        }
        
        network_response = _string_request("network", network_params)
        
        if network_response is not None and network_response.text.strip():
            import io
            try:
                network_df = pd.read_csv(io.StringIO(network_response.text), sep='\t')
                results['network'][cat_name] = network_df
                print(f"    Interactions found: {len(network_df)}")
                
                if not network_df.empty and 'preferredName_A' in network_df.columns:
                    # Show top interactions by score
                    if 'score' in network_df.columns:
                        top_interactions = network_df.nlargest(5, 'score')
                        print(f"    Top interactions:")
                        for _, row in top_interactions.iterrows():
                            print(f"      {row['preferredName_A']} ↔ {row['preferredName_B']} "
                                  f"(score: {row['score']:.3f})")
            except Exception as e:
                print(f"    ⚠ Could not parse network results: {e}")
                results['network'][cat_name] = pd.DataFrame()
        else:
            results['network'][cat_name] = pd.DataFrame()
        
        time.sleep(1)
        
        # ----- Network Image -----
        if save_network_image and len(mapped_genes) >= 2:
            print(f"\n  Downloading network image...")
            image_params = {
                "identifiers": "\r".join(mapped_genes),
                "species": species,
                "required_score": score_threshold,
                "network_flavor": "confidence"
            }
            
            image_response = _string_request("network", image_params, output_format='image')
            
            if image_response is not None:
                # Save as PDF (STRING only serves PNG, so we convert)
                from io import BytesIO
                from matplotlib import pyplot as plt
                from matplotlib.image import imread
                
                img = imread(BytesIO(image_response.content))
                
                fig, ax = plt.subplots(figsize=(img.shape[1] / 100, img.shape[0] / 100), dpi=300)
                ax.imshow(img)
                ax.axis('off')
                
                pdf_filename = f"string_network_{cat_name}_{stats_params['threshold_label']}.pdf"
                pdf_path = os.path.join(viz_dir, pdf_filename)
                fig.savefig(pdf_path, format='pdf', bbox_inches='tight', pad_inches=0, dpi=300)
                plt.close(fig)
                
                print(f"    ✓ Saved: {pdf_filename}")
            
            time.sleep(1)
        
        # ----- Build summary row -----
        ppi_pval_val = None
        if cat_name in results['ppi_enrichment'] and results['ppi_enrichment'][cat_name]:
            ppi_pval_val = results['ppi_enrichment'][cat_name].get('p_value', None)
        
        n_sig_terms = 0
        if cat_name in results['functional_enrichment']:
            edf = results['functional_enrichment'][cat_name]
            if not edf.empty and 'fdr' in edf.columns:
                n_sig_terms = (edf['fdr'] < fdr_threshold).sum()
        
        n_interactions = 0
        if cat_name in results['network']:
            n_interactions = len(results['network'][cat_name])
        
        results['summary'].append({
            'category': cat_name,
            'n_proteins': len(accession_set),
            'n_mapped': len(id_mapping) if id_mapping else 0,
            'n_interactions': n_interactions,
            'ppi_enrichment_pvalue': ppi_pval_val,
            'n_enriched_terms': n_sig_terms,
            'ppi_significant': ppi_pval_val < 0.05 if isinstance(ppi_pval_val, (int, float)) else None
        })
    
    # =========================================================================
    # 4. SAVE RESULTS
    # =========================================================================
    print(f"\n[4/5] Saving results...")
    
    threshold_label = stats_params['threshold_label']
    
    # Save summary table
    summary_df = pd.DataFrame(results['summary'])
    summary_path = os.path.join(string_table_dir, f'string_summary_{threshold_label}.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"  ✓ Saved: string_summary_{threshold_label}.csv")
    results['summary'] = summary_df
    
    # Save enrichment results per category
    for cat_name, enrich_df in results['functional_enrichment'].items():
        if isinstance(enrich_df, pd.DataFrame) and not enrich_df.empty:
            # Full enrichment
            enrich_path = os.path.join(
                string_table_dir, 
                f'string_enrichment_{cat_name}_{threshold_label}.csv'
            )
            enrich_df.to_csv(enrich_path, index=False)
            print(f"  ✓ Saved: string_enrichment_{cat_name}_{threshold_label}.csv")
            
            # Significant only
            if 'fdr' in enrich_df.columns:
                sig_df = enrich_df[enrich_df['fdr'] < fdr_threshold]
                if not sig_df.empty:
                    sig_path = os.path.join(
                        string_table_dir, 
                        f'string_enrichment_{cat_name}_significant_{threshold_label}.csv'
                    )
                    sig_df.to_csv(sig_path, index=False)
    
    # Save network edges per category
    for cat_name, network_df in results['network'].items():
        if isinstance(network_df, pd.DataFrame) and not network_df.empty:
            network_path = os.path.join(
                string_table_dir, 
                f'string_network_{cat_name}_{threshold_label}.csv'
            )
            network_df.to_csv(network_path, index=False)
            print(f"  ✓ Saved: string_network_{cat_name}_{threshold_label}.csv")
    
    # =========================================================================
    # 5. PRINT FINAL SUMMARY
    # =========================================================================
    print(f"\n[5/5] Analysis complete!")
    
    print(f"\n{'=' * 80}")
    print("STRING ANALYSIS SUMMARY")
    print(f"{'=' * 80}")
    
    print(f"\n{'Category':<25} {'Proteins':<10} {'Mapped':<10} {'Edges':<10} "
          f"{'PPI p-val':<15} {'Enriched Terms':<15}")
    print(f"{'─' * 85}")
    
    for _, row in summary_df.iterrows():
        ppi_str = f"{row['ppi_enrichment_pvalue']:.2e}" if pd.notna(row['ppi_enrichment_pvalue']) else 'N/A'
        sig_marker = " ✓" if row.get('ppi_significant') else ""
        print(f"{row['category']:<25} {row['n_proteins']:<10} {row['n_mapped']:<10} "
              f"{row['n_interactions']:<10} {ppi_str:<15} {row['n_enriched_terms']:<15}{sig_marker}")
    
    print(f"\nOutput files:")
    print(f"  Tables:  {string_table_dir}/")
    print(f"  Images:  {viz_dir}/")
    
    print(f"\n{'=' * 80}")
    print("STRING ANALYSIS COMPLETE")
    print(f"{'=' * 80}\n")
    
    # Track exclusions in results for documentation/reproducibility
    results['excluded_genes'] = excluded_gene_list
    
    # Store in data dict for downstream use
    data['string_results'] = results
    
    return results


# ============================================================================
# TARGETED QUERY: string_query()
# ============================================================================

def string_query(genes, add_genes=None, add_nodes=0, species=9606,
                 score_threshold=400, fdr_threshold=0.05, top_n_terms=15,
                 save_to=None, label=None):
    """
    Run a targeted STRING query on a custom list of genes.
    
    Unlike string_ip() which works from your full statistical results, this
    function takes any list of gene symbols directly. Use it for:
    - Hypothesis-driven queries (e.g., "do my hits connect through p53?")
    - Cross-category analysis (mixing proteins from different Venn groups)
    - Adding pathway anchors to see if your proteins connect to known biology
    
    Parameters:
    -----------
    genes : list
        Gene symbols from your data (e.g., your IP-MS hits)
    add_genes : list, optional
        Additional gene symbols to include that are NOT in your data — 
        pathway anchors or hypothesis proteins (e.g., ['TP53', 'MDM2']).
        These are added to the query but flagged separately in the output.
    add_nodes : int, optional
        Number of STRING interactors to add automatically (default: 0).
        STRING picks the best connectors from its database.
        Use 5-10 to let STRING fill in bridging proteins.
    species : int, optional
        NCBI taxonomy ID (default: 9606 for human)
    score_threshold : int, optional
        Minimum STRING confidence score (default: 400)
    fdr_threshold : float, optional
        FDR cutoff for enrichment (default: 0.05)
    top_n_terms : int, optional
        Number of top enrichment terms to show (default: 15)
    save_to : str, optional
        Directory to save output files. If None, only prints results.
    label : str, optional
        Label for output filenames (e.g., 'p53_hypothesis').
        Default: 'custom_query'
        
    Returns:
    --------
    dict containing:
        'input_genes' : list of your IP-MS genes
        'added_genes' : list of hypothesis genes you added
        'all_genes' : combined list sent to STRING
        'ppi_enrichment' : PPI enrichment result (dict)
        'functional_enrichment' : DataFrame of enrichment terms
        'network' : DataFrame of network edges
        'string_ids' : gene -> STRING ID mapping
        
    Example:
    --------
    >>> # Test p53 hypothesis — do your hits connect through p53/MDM2?
    >>> dna_damage_hits = ['USP7', 'ZFP36L1', 'TOPBP1', 'CDC45', 'SMC5']
    >>> result = string_query(
    ...     genes=dna_damage_hits,
    ...     add_genes=['TP53', 'MDM2', 'CHEK1', 'ATR'],
    ...     label='p53_hypothesis'
    ... )
    
    >>> # Let STRING find bridging proteins automatically
    >>> result = string_query(
    ...     genes=['USP7', 'TOPBP1', 'CDC45', 'SMC5', 'ZFP36L1'],
    ...     add_nodes=5,
    ...     label='auto_bridge'
    ... )
    
    >>> # RNA biology hypothesis
    >>> rna_hits = ['ZFP36L1', 'DDX49', 'PUS1', 'RBPMS']
    >>> result = string_query(
    ...     genes=rna_hits,
    ...     add_genes=['DIS3', 'XRN1', 'EXOSC10'],
    ...     label='rna_metabolism'
    ... )
    """
    
    import io
    
    try:
        import requests
    except ImportError:
        print("⚠ requests module not installed. Run: pip install requests")
        return None
    
    if label is None:
        label = 'custom_query'
    
    if add_genes is None:
        add_genes = []
    
    # Clean inputs
    genes = [g.strip() for g in genes if g and isinstance(g, str)]
    add_genes = [g.strip() for g in add_genes if g and isinstance(g, str)]
    all_genes = genes + [g for g in add_genes if g not in genes]
    
    print("\n" + "=" * 80)
    print(f"STRING TARGETED QUERY: {label}")
    print("=" * 80)
    
    print(f"\n  Your IP-MS hits ({len(genes)}): {', '.join(genes)}")
    if add_genes:
        print(f"  Hypothesis genes ({len(add_genes)}): {', '.join(add_genes)}")
    if add_nodes > 0:
        print(f"  STRING auto-add: {add_nodes} best connectors")
    print(f"  Total query: {len(all_genes)} genes")
    
    results = {
        'input_genes': genes,
        'added_genes': add_genes,
        'all_genes': all_genes,
        'label': label,
    }
    
    # =========================================================================
    # 1. MAP TO STRING IDs
    # =========================================================================
    print(f"\n[1/4] Mapping to STRING IDs...")
    id_mapping = _map_to_string_ids(all_genes, species=species)
    
    if id_mapping is None or len(id_mapping) < 2:
        print("  ✗ Could not map enough genes. Check gene symbols.")
        return None
    
    results['string_ids'] = id_mapping
    mapped_genes = list(id_mapping.keys())
    
    time.sleep(1)
    
    # =========================================================================
    # 2. PPI ENRICHMENT
    # =========================================================================
    print(f"\n[2/4] Testing PPI enrichment...")
    
    # Only test enrichment on YOUR genes (not the added hypothesis genes)
    # This is the scientifically correct approach — you're testing whether
    # your experimental hits are enriched, not whether known pathway members
    # interact with each other (which would be circular)
    your_mapped = [g for g in mapped_genes if g in genes]
    
    if len(your_mapped) >= 2:
        ppi_params = {
            "identifiers": "\r".join(your_mapped),
            "species": species,
            "required_score": score_threshold
        }
        
        ppi_response = _string_request("ppi_enrichment", ppi_params, output_format='json')
        
        if ppi_response is not None:
            import json
            try:
                ppi_raw = ppi_response.json()
                ppi_data = ppi_raw[0] if isinstance(ppi_raw, list) and len(ppi_raw) > 0 else ppi_raw
                results['ppi_enrichment'] = ppi_data
                
                n_edges = ppi_data.get('number_of_edges', 'N/A')
                expected = ppi_data.get('expected_number_of_edges', 'N/A')
                ppi_pval = ppi_data.get('p_value', 'N/A')
                
                print(f"    (tested on YOUR {len(your_mapped)} genes only, not added genes)")
                print(f"    Observed interactions: {n_edges}")
                print(f"    Expected (random):     {expected}")
                print(f"    PPI enrichment p-value: {ppi_pval}")
                
                if isinstance(ppi_pval, (int, float)) and ppi_pval < 0.05:
                    print(f"    → ✓ Significant!")
                    
            except Exception as e:
                print(f"    ⚠ Could not parse: {e}")
                results['ppi_enrichment'] = None
    
    time.sleep(1)
    
    # =========================================================================
    # 3. NETWORK (with all genes including hypothesis genes + auto-add)
    # =========================================================================
    print(f"\n[3/4] Retrieving network...")
    
    network_params = {
        "identifiers": "\r".join(mapped_genes),
        "species": species,
        "required_score": score_threshold,
        "add_nodes": add_nodes
    }
    
    network_response = _string_request("network", network_params)
    
    if network_response is not None and network_response.text.strip():
        try:
            network_df = pd.read_csv(io.StringIO(network_response.text), sep='\t')
            results['network'] = network_df
            
            if not network_df.empty and 'preferredName_A' in network_df.columns:
                print(f"    Interactions found: {len(network_df)}")
                
                # Flag which interactions involve your data vs added genes
                if add_genes or add_nodes > 0:
                    input_set = set(genes)
                    added_set = set(add_genes)
                    
                    # Categorize each edge
                    edge_types = []
                    for _, row in network_df.iterrows():
                        a = row['preferredName_A']
                        b = row['preferredName_B']
                        a_source = 'data' if a in input_set else ('hypothesis' if a in added_set else 'STRING-added')
                        b_source = 'data' if b in input_set else ('hypothesis' if b in added_set else 'STRING-added')
                        edge_types.append(f"{a_source}↔{b_source}")
                    
                    network_df['edge_type'] = edge_types
                    results['network'] = network_df
                    
                    # Print edges grouped by type
                    print(f"\n    --- Edges connecting YOUR hits to hypothesis genes ---")
                    bridge_edges = network_df[network_df['edge_type'].str.contains('data.*hypothesis|hypothesis.*data')]
                    if not bridge_edges.empty:
                        for _, row in bridge_edges.iterrows():
                            print(f"    ★ {row['preferredName_A']} ↔ {row['preferredName_B']} "
                                  f"(score: {row['score']:.3f})")
                    else:
                        print(f"      (none found at score ≥ {score_threshold})")
                    
                    print(f"\n    --- Edges among YOUR hits ---")
                    data_edges = network_df[network_df['edge_type'] == 'data↔data']
                    if not data_edges.empty:
                        for _, row in data_edges.iterrows():
                            print(f"      {row['preferredName_A']} ↔ {row['preferredName_B']} "
                                  f"(score: {row['score']:.3f})")
                    else:
                        print(f"      (none)")
                    
                    print(f"\n    --- Edges among hypothesis genes ---")
                    hyp_edges = network_df[network_df['edge_type'] == 'hypothesis↔hypothesis']
                    if not hyp_edges.empty:
                        for _, row in hyp_edges.iterrows():
                            print(f"      {row['preferredName_A']} ↔ {row['preferredName_B']} "
                                  f"(score: {row['score']:.3f})")
                    else:
                        print(f"      (none)")
                    
                    if add_nodes > 0:
                        # Identify STRING-added bridge proteins
                        all_nodes = set(network_df['preferredName_A']) | set(network_df['preferredName_B'])
                        string_added = all_nodes - set(mapped_genes)
                        if string_added:
                            print(f"\n    --- STRING auto-added bridge proteins ---")
                            for protein in sorted(string_added):
                                connections = network_df[
                                    (network_df['preferredName_A'] == protein) | 
                                    (network_df['preferredName_B'] == protein)
                                ]
                                partners = []
                                for _, row in connections.iterrows():
                                    partner = row['preferredName_B'] if row['preferredName_A'] == protein else row['preferredName_A']
                                    partners.append(f"{partner}({row['score']:.2f})")
                                print(f"    ★ {protein} connects to: {', '.join(partners)}")
                
                else:
                    # No added genes — just show top interactions
                    print(f"\n    Top interactions:")
                    top = network_df.nlargest(min(10, len(network_df)), 'score')
                    for _, row in top.iterrows():
                        print(f"      {row['preferredName_A']} ↔ {row['preferredName_B']} "
                              f"(score: {row['score']:.3f})")
                              
        except Exception as e:
            print(f"    ⚠ Could not parse network: {e}")
            results['network'] = pd.DataFrame()
    else:
        results['network'] = pd.DataFrame()
    
    time.sleep(1)
    
    # =========================================================================
    # 4. FUNCTIONAL ENRICHMENT (on combined gene list)
    # =========================================================================
    print(f"\n[4/4] Running functional enrichment...")
    
    enrich_params = {
        "identifiers": "\r".join(mapped_genes),
        "species": species
    }
    
    enrich_response = _string_request("enrichment", enrich_params)
    
    if enrich_response is not None and enrich_response.text.strip():
        try:
            enrich_df = pd.read_csv(io.StringIO(enrich_response.text), sep='\t')
            results['functional_enrichment'] = enrich_df
            
            if not enrich_df.empty and 'fdr' in enrich_df.columns:
                sig = enrich_df[enrich_df['fdr'] < fdr_threshold]
                
                if not sig.empty and 'category' in sig.columns:
                    print(f"\n    Significant terms (FDR < {fdr_threshold}):")
                    for cat in sig['category'].unique():
                        n = len(sig[sig['category'] == cat])
                        print(f"      {cat}: {n} terms")
                    
                    print(f"\n    Top {top_n_terms} terms:")
                    for _, row in sig.head(top_n_terms).iterrows():
                        desc = str(row.get('description', row.get('term', 'N/A')))
                        if len(desc) > 55:
                            desc = desc[:52] + '...'
                        cat = row.get('category', '')
                        fdr = row.get('fdr', 0)
                        n = row.get('number_of_genes', 0)
                        names = row.get('preferredNames', '')
                        print(f"      [{cat}] {desc}")
                        print(f"        FDR={fdr:.2e} | genes: {names}")
                else:
                    print(f"    No terms below FDR {fdr_threshold}")
                    if len(enrich_df) > 0 and 'fdr' in enrich_df.columns:
                        print(f"    Best FDR: {enrich_df['fdr'].min():.2e}")
                        # Show top terms anyway
                        print(f"\n    Top {min(5, len(enrich_df))} terms (not significant):")
                        for _, row in enrich_df.head(5).iterrows():
                            desc = str(row.get('description', row.get('term', 'N/A')))
                            if len(desc) > 55:
                                desc = desc[:52] + '...'
                            print(f"      [{row.get('category','')}] {desc} (FDR={row['fdr']:.2e})")
            else:
                print(f"    No enrichment results returned")
        except Exception as e:
            print(f"    ⚠ Could not parse enrichment: {e}")
            results['functional_enrichment'] = pd.DataFrame()
    else:
        results['functional_enrichment'] = pd.DataFrame()
    
    time.sleep(1)
    
    # =========================================================================
    # NETWORK IMAGE
    # =========================================================================
    if save_to is not None:
        os.makedirs(save_to, exist_ok=True)
        
        print(f"\n  Downloading network image...")
        image_params = {
            "identifiers": "\r".join(mapped_genes),
            "species": species,
            "required_score": score_threshold,
            "network_flavor": "confidence",
            "add_nodes": add_nodes
        }
        
        image_response = _string_request("network", image_params, output_format='image')
        
        if image_response is not None:
            from io import BytesIO
            from matplotlib import pyplot as plt
            from matplotlib.image import imread
            
            img = imread(BytesIO(image_response.content))
            fig, ax = plt.subplots(figsize=(img.shape[1] / 100, img.shape[0] / 100), dpi=300)
            ax.imshow(img)
            ax.axis('off')
            
            pdf_path = os.path.join(save_to, f"string_{label}.pdf")
            fig.savefig(pdf_path, format='pdf', bbox_inches='tight', pad_inches=0, dpi=300)
            plt.close(fig)
            print(f"  ✓ Saved: string_{label}.pdf")
        
        # Save network CSV
        if isinstance(results.get('network'), pd.DataFrame) and not results['network'].empty:
            csv_path = os.path.join(save_to, f"string_{label}_network.csv")
            results['network'].to_csv(csv_path, index=False)
            print(f"  ✓ Saved: string_{label}_network.csv")
        
        # Save enrichment CSV
        if isinstance(results.get('functional_enrichment'), pd.DataFrame) and not results['functional_enrichment'].empty:
            csv_path = os.path.join(save_to, f"string_{label}_enrichment.csv")
            results['functional_enrichment'].to_csv(csv_path, index=False)
            print(f"  ✓ Saved: string_{label}_enrichment.csv")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print(f"\n{'=' * 80}")
    print(f"QUERY COMPLETE: {label}")
    print(f"{'=' * 80}")
    
    n_net = len(results.get('network', []))
    print(f"  Genes queried:  {len(mapped_genes)} ({len(genes)} yours + {len(add_genes)} hypothesis + {add_nodes} auto)")
    print(f"  Interactions:   {n_net}")
    
    ppi = results.get('ppi_enrichment', {})
    if ppi:
        print(f"  PPI enrichment: p = {ppi.get('p_value', 'N/A')} (your genes only)")
    
    enrich = results.get('functional_enrichment', pd.DataFrame())
    if isinstance(enrich, pd.DataFrame) and not enrich.empty and 'fdr' in enrich.columns:
        n_sig = (enrich['fdr'] < fdr_threshold).sum()
        print(f"  Enriched terms: {n_sig} (FDR < {fdr_threshold})")
    
    print(f"{'=' * 80}\n")
    
    return results