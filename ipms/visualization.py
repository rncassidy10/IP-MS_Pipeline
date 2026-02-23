"""
Visualization functions for IP-MS pipeline.

Generates volcano plots, heatmaps, Venn diagrams, and boxplots
from statistical analysis results.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Consistent color palette for an arbitrary number of conditions
_PALETTE = [
    '#1f77b4', '#2ca02c', '#d62728', '#ff7f0e', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
]


def _condition_color_map(intensity_cols):
    """Build a color map for an arbitrary number of conditions."""
    conditions = list(intensity_cols.keys())
    return {cond: _PALETTE[i % len(_PALETTE)] for i, cond in enumerate(conditions)}


def viz_ip(data, create_volcano=True, create_heatmap=True, top_n=100):
    """
    Create visualization plots for IP-MS results.

    Creates:
    - Volcano plots (labeled and unlabeled versions)
    - Clustered heatmap of top significant proteins (intensities)
    - Clustered heatmap of log2 fold changes (sample-level)

    Parameters
    ----------
    data : dict
        Output from stat_ip().
    create_volcano : bool, optional
        Create volcano plots (default: True).
    create_heatmap : bool, optional
        Create heatmap of top proteins (default: True).
    top_n : int, optional
        Number of top proteins to show in heatmap (default: 100).

    Returns
    -------
    None
        Saves plots to results/figures/viz/.

    Example
    -------
    >>> data = stat_ip(data)
    >>> viz_ip(data)
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

    protein_col = config['data_columns']['protein_id']
    gene_col = config['data_columns']['gene_symbol']

    viz_dir = os.path.join(config['data_paths']['output_dir'], 'figures', 'viz')
    os.makedirs(viz_dir, exist_ok=True)

    print(f"\nOutput directory: {viz_dir}")

    comparisons = list(sig_proteins.keys())

    # =========================================================================
    # 1. VOLCANO PLOTS
    # =========================================================================
    if create_volcano:
        print(f"\n[1/3] Creating volcano plots...")

        for comparison in comparisons:
            print(f"  Creating volcano plot for {comparison}...")

            log2fc = stats_results[f'{comparison}_log2FC']
            pval = stats_results[f'{comparison}_adj_pvalue']
            significant = stats_results[f'{comparison}_significant']

            neg_log10_pval = -np.log10(pval.replace(0, 1e-300))

            # Classify each point
            categories = []
            for i in range(len(stats_results)):
                if significant.iloc[i] == 'significant':
                    categories.append('enriched' if log2fc.iloc[i] > 0 else 'depleted')
                else:
                    categories.append('not_significant')

            for labeled in [True, False]:
                fig, ax = plt.subplots(figsize=(10, 8))

                for category, color, label in [
                    ('not_significant', '#CCCCCC', 'Not Significant'),
                    ('depleted', '#CCCCCC', 'Depleted'),
                    ('enriched', '#E74C3C', 'Enriched')
                ]:
                    mask = np.array(categories) == category
                    ax.scatter(
                        log2fc[mask],
                        neg_log10_pval[mask],
                        c=color,
                        label=label,
                        s=30,
                        alpha=0.6,
                        edgecolors='none'
                    )

                p_thresh = stats_params['p_threshold']
                fc_thresh = stats_params['log2fc_threshold']

                ax.axhline(-np.log10(p_thresh), color='black', linestyle='--',
                          linewidth=1, alpha=0.5, label=f'p = {p_thresh}')
                ax.axvline(fc_thresh, color='black', linestyle='--',
                          linewidth=1, alpha=0.5)
                ax.axvline(-fc_thresh, color='black', linestyle='--',
                          linewidth=1, alpha=0.5)

                if labeled:
                    enriched_mask = (np.array(categories) == 'enriched')
                    if enriched_mask.any():
                        enriched_data = stats_results[enriched_mask].copy()
                        enriched_data['neg_log10_pval'] = neg_log10_pval[enriched_mask]
                        top_enriched = enriched_data.nlargest(25, f'{comparison}_log2FC')

                        try:
                            from adjustText import adjust_text
                            texts = []
                            for idx, row in top_enriched.iterrows():
                                texts.append(ax.text(
                                    row[f'{comparison}_log2FC'],
                                    row['neg_log10_pval'],
                                    row[gene_col],
                                    fontsize=8,
                                    alpha=0.8
                                ))

                            adjust_text(
                                texts,
                                arrowprops=dict(arrowstyle='-', color='black', lw=0.5),
                                expand_points=(2.0, 2.0),
                                expand_text=(2.0, 4.0),
                                force_points=2.0,
                                force_text=2.0,
                                shrinkA=10,
                                shrinkB=10
                            )

                        except (ImportError, Exception):
                            for idx, row in top_enriched.iterrows():
                                ax.annotate(
                                    row[gene_col],
                                    xy=(row[f'{comparison}_log2FC'], row['neg_log10_pval']),
                                    xytext=(20, 20),
                                    textcoords='offset points',
                                    fontsize=8,
                                    alpha=0.8,
                                    arrowprops=dict(arrowstyle='->', lw=0.5, color='black')
                                )

                ax.set_xlabel('Log2 Fold Change', fontsize=12, fontweight='bold')
                ax.set_ylabel('-Log10 Adjusted P-value', fontsize=12, fontweight='bold')
                ax.set_title(f'Volcano Plot: {comparison}', fontsize=14, fontweight='bold')
                ax.legend(loc='lower right', fontsize=10)
                ax.grid(alpha=0.3)

                suffix = '' if labeled else '_clean'
                volcano_path = os.path.join(viz_dir, f'volcano_{comparison}_{stats_params["threshold_label"]}{suffix}.pdf')
                plt.tight_layout()
                plt.savefig(volcano_path, dpi=300, bbox_inches='tight')
                plt.close()

            print(f"    > Saved: volcano_{comparison}_{stats_params['threshold_label']}.pdf (labeled)")
            print(f"    > Saved: volcano_{comparison}_{stats_params['threshold_label']}_clean.pdf (unlabeled)")

    # =========================================================================
    # 2. HEATMAP OF TOP PROTEINS
    # =========================================================================
    if create_heatmap:
        print(f"\n[2/3] Creating heatmap of top {top_n} proteins...")

        all_sig_proteins = set()
        for comparison in comparisons:
            sig_mask = (stats_results[f'{comparison}_significant'] == 'significant') & \
                    (stats_results[f'{comparison}_log2FC'] > 0)
            sig_data = stats_results[sig_mask].copy()
            top_proteins = sig_data.nlargest(top_n, f'{comparison}_log2FC')
            all_sig_proteins.update(top_proteins[protein_col].tolist())

        print(f"  Total unique significant proteins: {len(all_sig_proteins)}")

        if len(all_sig_proteins) > 0:
            heatmap_df = stats_results[stats_results[protein_col].isin(all_sig_proteins)].copy()

            all_intensity_cols = []
            for cols in intensity_cols.values():
                all_intensity_cols.extend(cols)

            heatmap_data = df.loc[heatmap_df.index, all_intensity_cols]
            heatmap_data.index = heatmap_df[gene_col].values

            g = sns.clustermap(
                heatmap_data,
                cmap='RdBu_r',
                center=heatmap_data.max().max() / 2,
                cbar_kws={'label': 'Normalized Intensity (log2)'},
                yticklabels=True,
                xticklabels=True,
                figsize=(20, 16),
                row_cluster=True,
                col_cluster=False,
                method='average',
                metric='euclidean'
            )

            g.ax_heatmap.set_title(f'Top {len(heatmap_data)} Enriched Proteins',
                        fontsize=14, fontweight='bold', pad=20)
            g.ax_heatmap.set_xlabel('Samples', fontsize=12)
            g.ax_heatmap.set_ylabel('Proteins', fontsize=12)
            g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=45, ha='right', fontsize=8)
            g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=8)

            heatmap_path = os.path.join(viz_dir, f'heatmap_top_proteins_{stats_params["threshold_label"]}.pdf')
            g.savefig(heatmap_path, dpi=300)
            plt.close()

            print(f"  > Saved: heatmap_top_proteins_{stats_params['threshold_label']}.pdf")
        else:
            print(f"  Warning: No significant proteins to plot")

    # =========================================================================
    # 3. HEATMAP OF LOG2 FOLD CHANGES
    # =========================================================================
    if create_heatmap:
        print(f"\n[3/3] Creating log2 fold change heatmap...")

        all_sig_proteins = set()
        for comparison in comparisons:
            sig_mask = (stats_results[f'{comparison}_significant'] == 'significant') & \
                    (stats_results[f'{comparison}_log2FC'] > 0)
            sig_data = stats_results[sig_mask].copy()
            top_proteins = sig_data.nlargest(top_n, f'{comparison}_log2FC')
            all_sig_proteins.update(top_proteins[protein_col].tolist())

        if len(all_sig_proteins) > 0:
            heatmap_df = stats_results[stats_results[protein_col].isin(all_sig_proteins)].copy()

            control_condition = config['conditions']['control']
            control_cols = intensity_cols[control_condition]
            control_avg = df.loc[heatmap_df.index, control_cols].mean(axis=1)

            log2fc_data = pd.DataFrame(index=heatmap_df.index)

            for condition, cols in intensity_cols.items():
                if condition != control_condition:
                    for col in cols:
                        sample_name = col.split(',')[-1].strip()[:20]
                        col_name = f"{condition}_{sample_name}"
                        log2fc_data[col_name] = df.loc[heatmap_df.index, col] - control_avg

            log2fc_data.index = heatmap_df[gene_col].values

            g = sns.clustermap(
                log2fc_data,
                cmap='RdBu_r',
                center=0,
                cbar_kws={'label': 'Log2 Fold Change (vs Control Avg)'},
                yticklabels=True,
                xticklabels=True,
                figsize=(20, 16),
                row_cluster=True,
                col_cluster=False,
                method='average',
                metric='euclidean'
            )

            g.ax_heatmap.set_title(f'Log2 Fold Changes - Top {len(log2fc_data)} Enriched Proteins',
                        fontsize=14, fontweight='bold', pad=20)
            g.ax_heatmap.set_xlabel('Samples (vs Control Average)', fontsize=12)
            g.ax_heatmap.set_ylabel('Proteins', fontsize=12)
            g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=45, ha='right', fontsize=5)
            g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=8)

            fc_heatmap_path = os.path.join(viz_dir, f'heatmap_log2fc_{stats_params["threshold_label"]}.pdf')
            plt.savefig(fc_heatmap_path, dpi=300, bbox_inches=None, pad_inches=2.0)
            plt.close('all')

            print(f"  > Saved: heatmap_log2fc_{stats_params['threshold_label']}.pdf")
        else:
            print(f"  Warning: No significant proteins to plot")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)

    print(f"\nPlots saved to: {viz_dir}")

    if create_volcano:
        print(f"\nVolcano plots:")
        for comparison in comparisons:
            print(f"  - volcano_{comparison}_{stats_params['threshold_label']}.pdf (labeled)")
            print(f"  - volcano_{comparison}_{stats_params['threshold_label']}_clean.pdf (unlabeled)")

    if create_heatmap:
        print(f"\nHeatmaps:")
        print(f"  - heatmap_top_proteins_{stats_params['threshold_label']}.pdf (clustered intensities)")
        print(f"  - heatmap_log2fc_{stats_params['threshold_label']}.pdf (clustered fold changes)")

    print("\n" + "="*80 + "\n")



def boxplot_ip(data, top_n=20, group_by='category'):
    """
    Create boxplots showing log2 intensities of enriched proteins across samples.

    Parameters
    ----------
    data : dict
        Output from stat_ip().
    top_n : int, optional
        Number of top enriched proteins to plot per category (default: 20).
    group_by : str, optional
        How to group boxplots: 'condition', 'protein', or 'category' (default).

    Returns
    -------
    None
        Saves plots to results/figures/viz/.

    Example
    -------
    >>> data = stat_ip(data)
    >>> boxplot_ip(data, top_n=20, group_by='category')
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

    protein_col = config['data_columns']['protein_id']
    gene_col = config['data_columns']['gene_symbol']

    # Build dynamic color palette from conditions
    palette = _condition_color_map(intensity_cols)

    viz_dir = os.path.join(config['data_paths']['output_dir'], 'figures', 'viz')
    os.makedirs(viz_dir, exist_ok=True)

    print(f"\nOutput directory: {viz_dir}")

    comparisons = list(sig_proteins.keys())

    # =========================================================================
    # COLLECT ALL UNIQUE ENRICHED PROTEINS
    # =========================================================================
    print(f"\nCollecting enriched proteins...")

    all_enriched = set()
    for comparison in comparisons:
        sig_mask = (stats_results[f'{comparison}_significant'] == 'significant') & \
                   (stats_results[f'{comparison}_log2FC'] > 0)
        enriched = stats_results[sig_mask][protein_col].tolist()
        all_enriched.update(enriched)
        print(f"  {comparison}: {len(enriched)} enriched")

    print(f"\nTotal unique enriched proteins: {len(all_enriched)}")

    if len(all_enriched) == 0:
        print("  Warning: No enriched proteins found!")
        return

    enriched_df = stats_results[stats_results[protein_col].isin(all_enriched)].copy()

    fc_cols = [c for c in enriched_df.columns if '_log2FC' in c]
    enriched_df['avg_log2FC'] = enriched_df[fc_cols].mean(axis=1)
    enriched_df = enriched_df.sort_values('avg_log2FC', ascending=False)

    # =========================================================================
    # CATEGORIZE PROTEINS (dynamically based on comparisons)
    # =========================================================================
    if group_by == 'category':
        print(f"\nCategorizing proteins by enrichment pattern...")

        protein_categories = {}
        for idx in enriched_df.index:
            accession = enriched_df.loc[idx, protein_col]

            sig_in = []
            for comp in comparisons:
                sig_mask = (stats_results[f'{comp}_significant'] == 'significant') & \
                           (stats_results[f'{comp}_log2FC'] > 0)
                if accession in stats_results[sig_mask][protein_col].values:
                    sig_in.append(comp)

            if len(sig_in) == len(comparisons):
                category = 'Shared'
            elif len(sig_in) == 1:
                # Use the treatment name from the comparison (e.g. "WT" from "WT_vs_EV")
                treatment_name = sig_in[0].split('_vs_')[0]
                category = f'{treatment_name} only'
            else:
                category = 'Shared ({})'.format(', '.join(c.split('_vs_')[0] for c in sig_in))

            protein_categories[accession] = category

        enriched_df['category'] = enriched_df[protein_col].map(protein_categories)

        category_counts = enriched_df['category'].value_counts()
        for cat, count in category_counts.items():
            print(f"  {cat}: {count} proteins")

    # =========================================================================
    # PREPARE DATA FOR PLOTTING
    # =========================================================================
    plot_data = []

    for idx in enriched_df.index:
        gene = enriched_df.loc[idx, gene_col]
        accession = enriched_df.loc[idx, protein_col]
        category = enriched_df.loc[idx, 'category'] if 'category' in enriched_df.columns else None

        for condition, cols in intensity_cols.items():
            for col in cols:
                intensity = df.loc[idx, col]
                if pd.notna(intensity):
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

        if top_n is not None:
            proteins_to_plot = enriched_df.head(top_n)[gene_col].tolist()
            plot_df = plot_df[plot_df['Protein'].isin(proteins_to_plot)]
            print(f"  Plotting top {len(proteins_to_plot)} proteins")

        fig, ax = plt.subplots(figsize=(14, max(8, len(plot_df['Protein'].unique()) * 0.4)))

        sns.boxplot(data=plot_df, y='Protein', x='Log2_Intensity', hue='Condition',
                    ax=ax, palette=palette, linewidth=1.5)
        sns.stripplot(data=plot_df, y='Protein', x='Log2_Intensity', hue='Condition',
                      ax=ax, dodge=True, alpha=0.6, size=4, palette=palette, legend=False)

        ax.set_xlabel('Log2 Intensity', fontsize=12, fontweight='bold')
        ax.set_ylabel('Protein', fontsize=12, fontweight='bold')
        ax.set_title(f'Top {len(plot_df["Protein"].unique())} Enriched Proteins - Log2 Intensities',
                    fontsize=14, fontweight='bold')
        ax.legend(title='Condition', fontsize=10, title_fontsize=11)
        ax.grid(axis='x', alpha=0.3)

        boxplot_path = os.path.join(viz_dir, f'boxplot_enriched_proteins_{stats_params["threshold_label"]}.pdf')
        plt.tight_layout()
        plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  > Saved: boxplot_enriched_proteins_{stats_params['threshold_label']}.pdf")

    elif group_by == 'category':
        print(f"\n[1/1] Creating boxplots grouped by category...")

        # Get unique categories in a consistent order: individual first, then shared
        all_categories = enriched_df['category'].unique().tolist()
        individual_cats = sorted([c for c in all_categories if c != 'Shared'])
        categories = individual_cats + (['Shared'] if 'Shared' in all_categories else [])

        n_categories = len(categories)
        fig, axes = plt.subplots(1, n_categories, figsize=(8*n_categories, 10), sharey=True)
        if n_categories == 1:
            axes = [axes]

        for i, category in enumerate(categories):
            ax = axes[i]
            cat_df = plot_df[plot_df['Category'] == category].copy()

            category_proteins = enriched_df[enriched_df['category'] == category].head(top_n)[gene_col].tolist()
            cat_df = cat_df[cat_df['Protein'].isin(category_proteins)]

            n_proteins = len(cat_df['Protein'].unique())
            print(f"  {category}: plotting {n_proteins} proteins")

            if len(cat_df) == 0:
                ax.text(0.5, 0.5, f'No proteins in\n{category}',
                       ha='center', va='center', fontsize=12)
                ax.set_title(f'{category} (0 proteins)', fontsize=12, fontweight='bold')
                continue

            sns.boxplot(data=cat_df, y='Protein', x='Log2_Intensity', hue='Condition',
                        ax=ax, palette=palette, linewidth=1.5)
            sns.stripplot(data=cat_df, y='Protein', x='Log2_Intensity', hue='Condition',
                          ax=ax, dodge=True, alpha=0.6, size=4, palette=palette, legend=False)

            ax.set_xlabel('Log2 Intensity', fontsize=12, fontweight='bold')
            if i == 0:
                ax.set_ylabel('Protein', fontsize=12, fontweight='bold')
            else:
                ax.set_ylabel('')
            ax.set_title(f'{category}\n({n_proteins} proteins)', fontsize=12, fontweight='bold')
            ax.legend(title='Condition', fontsize=10, title_fontsize=11)
            ax.grid(axis='x', alpha=0.3)

        fig.suptitle('Enriched Proteins by Category', fontsize=14, fontweight='bold', y=0.98)

        boxplot_path = os.path.join(viz_dir, f'boxplot_by_category_{stats_params["threshold_label"]}.pdf')
        plt.tight_layout()
        plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  > Saved: boxplot_by_category_{stats_params['threshold_label']}.pdf")

    elif group_by == 'protein':
        print(f"\n[1/1] Creating individual boxplots per protein...")

        if top_n is not None:
            proteins_to_plot = enriched_df.head(top_n)
        else:
            proteins_to_plot = enriched_df

        plot_df = plot_df[plot_df['Protein'].isin(proteins_to_plot[gene_col])]

        n_proteins = len(proteins_to_plot)
        n_cols = min(3, n_proteins)
        n_rows = int(np.ceil(n_proteins / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
        if n_proteins == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i, (idx, row) in enumerate(proteins_to_plot.iterrows()):
            gene = row[gene_col]
            protein_data = plot_df[plot_df['Protein'] == gene]

            ax = axes[i]
            sns.boxplot(data=protein_data, x='Condition', y='Log2_Intensity',
                        ax=ax, palette=palette, linewidth=1.5)
            sns.stripplot(data=protein_data, x='Condition', y='Log2_Intensity',
                          ax=ax, color='black', alpha=0.6, size=6)

            ax.set_title(f'{gene}', fontsize=11, fontweight='bold')
            ax.set_xlabel('Condition', fontsize=10)
            ax.set_ylabel('Log2 Intensity', fontsize=10)
            ax.grid(axis='y', alpha=0.3)

        for i in range(n_proteins, len(axes)):
            axes[i].set_visible(False)

        boxplot_path = os.path.join(viz_dir, f'boxplot_individual_proteins_{stats_params["threshold_label"]}.pdf')
        plt.tight_layout()
        plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  > Saved: boxplot_individual_proteins_{stats_params['threshold_label']}.pdf")

    else:
        print(f"\n  Warning: Invalid group_by parameter: {group_by}")
        print(f"     Valid options: 'condition', 'category', 'protein'")
        return

    print("\n" + "="*80)
    print("BOXPLOT CREATION COMPLETE")
    print("="*80)
    print(f"\nPlots saved to: {viz_dir}")
    print("="*80 + "\n")


def summary_ip(data, output_format='txt'):
    """
    Generate analysis summary report.

    Parameters
    ----------
    data : dict
        Output from stat_ip().
    output_format : str
        Format for report: 'txt', 'html', or 'both'.

    Returns
    -------
    None
        Saves report to results/.
    """
    # TODO: Implement summary report generation
    pass
