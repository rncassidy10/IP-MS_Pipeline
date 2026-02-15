"""
Statistical analysis functions for IP-MS pipeline.

Performs fold-change calculations, t-tests, and multiple testing
correction for treatment vs. control comparisons.
"""

import copy
import os

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

from .utils import save_data


def stat_ip(data, p_threshold=0.05, log2fc_threshold=1.0, correction='fdr_bh'):
    """
    Perform statistical analysis comparing treatments to control.

    For each comparison (treatment vs control):
    - Calculates log2 fold change
    - Performs t-test
    - Applies multiple testing correction
    - Identifies significant proteins

    Parameters
    ----------
    data : dict
        Output from norm_ip().
    p_threshold : float, optional
        P-value threshold for significance (default: 0.05).
    log2fc_threshold : float, optional
        Log2 fold change threshold for significance (default: 1.0).
    correction : str, optional
        Multiple testing correction method (default: 'fdr_bh').
        Options: 'fdr_bh' (Benjamini-Hochberg), 'bonferroni', 'none'.

    Returns
    -------
    dict
        Updated data dictionary with:
        - 'stats_results': DataFrame with all proteins and statistics
        - 'significant_proteins': Dict of significant proteins per comparison
        - 'stats_params': Parameters used for analysis

    Example
    -------
    >>> data = norm_ip(data)
    >>> data = stat_ip(data, p_threshold=0.05, log2fc_threshold=1.0)
    """

    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS")
    print("="*80)

    df = data['df'].copy()
    config = data['config']
    intensity_cols = data['intensity_cols']

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

    info_cols = [
        config['data_columns']['protein_id'],
        config['data_columns']['gene_symbol'],
        config['data_columns']['peptides']
    ]

    results_df = df[info_cols].copy()
    print(f"  > Starting with {len(results_df)} proteins")

    # =========================================================================
    # 2. CALCULATE STATISTICS FOR EACH COMPARISON
    # =========================================================================
    print(f"\n[2/4] Calculating statistics...")

    significant_proteins = {}

    for treatment in treatments:
        comparison_name = f"{treatment}_vs_{control}"
        print(f"\n  Analyzing: {comparison_name}")

        control_cols = intensity_cols[control]
        treatment_cols = intensity_cols[treatment]

        control_mean = df[control_cols].mean(axis=1)
        treatment_mean = df[treatment_cols].mean(axis=1)

        # Already log2 transformed, so just subtract
        log2fc = treatment_mean - control_mean

        pvalues = []
        for idx in df.index:
            control_vals = df.loc[idx, control_cols].dropna()
            treatment_vals = df.loc[idx, treatment_cols].dropna()

            if len(control_vals) >= 2 and len(treatment_vals) >= 2:
                _, pval = ttest_ind(treatment_vals, control_vals)
                pvalues.append(pval)
            else:
                pvalues.append(np.nan)

        pvalues = np.array(pvalues)

        # Apply multiple testing correction
        if correction in ('fdr_bh', 'bonferroni'):
            valid_mask = ~np.isnan(pvalues)
            adj_pvalues = np.full(len(pvalues), np.nan)
            if valid_mask.any():
                _, adj_p, _, _ = multipletests(pvalues[valid_mask], method=correction)
                adj_pvalues[valid_mask] = adj_p
        else:
            adj_pvalues = pvalues

        results_df[f'{comparison_name}_log2FC'] = log2fc
        results_df[f'{comparison_name}_pvalue'] = pvalues
        results_df[f'{comparison_name}_adj_pvalue'] = adj_pvalues
        results_df[f'{comparison_name}_control_mean'] = control_mean
        results_df[f'{comparison_name}_treatment_mean'] = treatment_mean

        sig_mask = (
            (adj_pvalues < p_threshold) &
            (np.abs(log2fc) > log2fc_threshold)
        )

        results_df[f'{comparison_name}_significant'] = np.where(
            sig_mask, 'significant', 'not_significant'
        )

        n_sig = sig_mask.sum()
        n_enriched = ((log2fc > log2fc_threshold) & (adj_pvalues < p_threshold)).sum()
        n_depleted = ((log2fc < -log2fc_threshold) & (adj_pvalues < p_threshold)).sum()

        print(f"    Total significant: {n_sig}")
        print(f"      Enriched in {treatment}: {n_enriched}")
        print(f"      Depleted in {treatment}: {n_depleted}")

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

    threshold_label = f"pval{str(p_threshold).replace('.', '')}_l2fc{str(log2fc_threshold).replace('.', '')}"

    output_dir = os.path.join(config['data_paths']['output_dir'], 'tables')
    os.makedirs(output_dir, exist_ok=True)

    results_path = os.path.join(output_dir, f'stats_results_{threshold_label}.csv')
    results_df.to_csv(results_path, index=False)
    print(f"  > Saved: stats_results_{threshold_label}.csv")
    print(f"    Location: {output_dir}")
    print(f"    {len(results_df)} proteins x {len(results_df.columns)} columns")

    for comparison_name in significant_proteins.keys():
        sig_df = results_df[results_df[f'{comparison_name}_significant'] == 'significant'].copy()

        if len(sig_df) > 0:
            sig_path = os.path.join(output_dir, f'{comparison_name}_significant_{threshold_label}.csv')
            sig_df.to_csv(sig_path, index=False)
            print(f"  > Saved: {comparison_name}_significant_{threshold_label}.csv ({len(sig_df)} proteins)")

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
    print(f"  > Saved: summary_{threshold_label}.csv")

    # =========================================================================
    # 5. UPDATE DATA DICTIONARY
    # =========================================================================
    data_updated = copy.copy(data)
    data_updated['df'] = df
    data_updated['stats_results'] = results_df
    data_updated['significant_proteins'] = significant_proteins
    data_updated['stats_params'] = {
        'p_threshold': p_threshold,
        'log2fc_threshold': log2fc_threshold,
        'correction': correction,
        'threshold_label': threshold_label
    }

    # Auto-save
    output_path = os.path.join(config['data_paths']['output_dir'], 'data_after_stat.pkl')
    save_data(data_updated, output_path)

    # =========================================================================
    # 6. PRINT SUMMARY
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
    print(f"  - stats_results_{threshold_label}.csv (all proteins)")
    print(f"  - [comparison]_significant_{threshold_label}.csv (sig proteins only)")
    print(f"  - summary_{threshold_label}.csv (overview)")

    print("\n" + "="*80)
    print("Next step: viz_ip() for volcano plots and visualizations")
    print("="*80 + "\n")

    return data_updated
