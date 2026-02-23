"""
Venn visualization function for IP-MS pipeline.

Generates Venn diagrams
from statistical analysis results.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3




def venn_ip(data, show_names=False, top_n=10, labels_in_diagram=False, max_labels_per_region=5):
    """
    Create Venn diagrams showing overlap between significant proteins.

    Creates 2-way or 3-way Venn diagrams depending on the number of comparisons.
    Optionally displays protein names directly inside the Venn diagram regions.

    Parameters
    ----------
    data : dict
        Output from stat_ip().
    show_names : bool, optional
        Print protein names in each category to console (default: False).
    top_n : int, optional
        Number of top proteins to show per category in console (default: 10).
    labels_in_diagram : bool, optional
        Display protein names inside Venn diagram regions (default: False).
    max_labels_per_region : int, optional
        Maximum number of protein names to display per region (default: 5).

    Returns
    -------
    dict or None
        Dictionary with 'overlaps', 'protein_sets', 'n_comparisons',
        or None if matplotlib-venn is not installed.

    Example
    -------
    >>> data = stat_ip(data)
    >>> venn_results = venn_ip(data, show_names=True, labels_in_diagram=True)
    """

    print("\n" + "="*80)
    print("CREATING VENN DIAGRAMS")
    print("="*80)

    try:
        from matplotlib_venn import venn2, venn3
    except ImportError:
        print("\nError: matplotlib-venn not installed!")
        print("Run: pip install matplotlib-venn")
        return None

    config = data['config']
    stats_results = data['stats_results']
    sig_proteins = data['significant_proteins']
    stats_params = data['stats_params']

    protein_col = config['data_columns']['protein_id']
    gene_col = config['data_columns']['gene_symbol']

    viz_dir = os.path.join(config['data_paths']['output_dir'], 'figures', 'viz')
    os.makedirs(viz_dir, exist_ok=True)

    print(f"\nOutput directory: {viz_dir}")

    comparisons = list(sig_proteins.keys())
    protein_sets = {}

    for comparison in comparisons:
        sig_mask = (stats_results[f'{comparison}_significant'] == 'significant') & \
                   (stats_results[f'{comparison}_log2FC'] > 0)
        protein_sets[comparison] = set(stats_results[sig_mask][protein_col].tolist())
        print(f"\n{comparison}: {len(protein_sets[comparison])} enriched proteins")

    # =========================================================================
    # HELPER FUNCTION TO GET TOP PROTEIN NAMES
    # =========================================================================
    def get_top_protein_names(protein_set, max_count=5):
        """
        Get top protein gene symbols from a set of protein IDs.
        
        Parameters:
        -----------
        protein_set : set
            Set of protein IDs
        max_count : int
            Maximum number of names to return
            
        Returns:
        --------
        list : List of gene symbols
        """
        if len(protein_set) == 0:
            return []
        
        # Get protein info for this set
        protein_df = stats_results[stats_results[protein_col].isin(protein_set)].copy()
        
        # Get fold change columns to sort by average FC
        fc_cols = [c for c in protein_df.columns if '_log2FC' in c]
        if len(fc_cols) > 0:
            protein_df['avg_log2FC'] = protein_df[fc_cols].mean(axis=1)
            protein_df = protein_df.sort_values('avg_log2FC', ascending=False)
        
        # Get top gene symbols
        top_genes = protein_df[gene_col].head(max_count).tolist()
        
        # Clean up gene names (handle NaN, etc.)
        top_genes = [str(g) for g in top_genes if pd.notna(g)]
        
        return top_genes

    # =========================================================================
    # CREATE VENN DIAGRAM
    # =========================================================================
    n_comparisons = len(comparisons)

    if n_comparisons == 2:
        print(f"\nCreating 2-way Venn diagram...")

        comp1, comp2 = comparisons[0], comparisons[1]
        set1, set2 = protein_sets[comp1], protein_sets[comp2]

        # Create figure with appropriate size
        fig_height = 10 if labels_in_diagram else 8
        fig, ax = plt.subplots(figsize=(10, fig_height))
        v = venn2([set1, set2], set_labels=(comp1, comp2), ax=ax)

        # Color the regions
        for region_id, color in [('10', '#E74C3C'), ('01', '#3498DB'), ('11', '#9B59B6')]:
            if v.get_patch_by_id(region_id):
                v.get_patch_by_id(region_id).set_color(color)
                v.get_patch_by_id(region_id).set_alpha(0.6)

        # Add protein names to diagram if requested
        if labels_in_diagram:
            # Region '10': comp1 only
            region_10_proteins = set1 - set2
            region_10_names = get_top_protein_names(region_10_proteins, max_labels_per_region)
            
            # Region '01': comp2 only
            region_01_proteins = set2 - set1
            region_01_names = get_top_protein_names(region_01_proteins, max_labels_per_region)
            
            # Region '11': shared
            region_11_proteins = set1 & set2
            region_11_names = get_top_protein_names(region_11_proteins, max_labels_per_region)
            
            # Add text labels to regions
            if v.get_label_by_id('10') and len(region_10_names) > 0:
                # Get current label position
                current_text = v.get_label_by_id('10').get_text()
                x, y = v.get_label_by_id('10').get_position()
                
                # Create new text with protein names
                protein_text = '\n'.join(region_10_names)
                if len(region_10_proteins) > max_labels_per_region:
                    protein_text += f'\n... +{len(region_10_proteins) - max_labels_per_region} more'
                
                new_text = f"{current_text}\n\n{protein_text}"
                v.get_label_by_id('10').set_text(new_text)
                v.get_label_by_id('10').set_fontsize(8)
            
            if v.get_label_by_id('01') and len(region_01_names) > 0:
                current_text = v.get_label_by_id('01').get_text()
                x, y = v.get_label_by_id('01').get_position()
                
                protein_text = '\n'.join(region_01_names)
                if len(region_01_proteins) > max_labels_per_region:
                    protein_text += f'\n... +{len(region_01_proteins) - max_labels_per_region} more'
                
                new_text = f"{current_text}\n\n{protein_text}"
                v.get_label_by_id('01').set_text(new_text)
                v.get_label_by_id('01').set_fontsize(8)
            
            if v.get_label_by_id('11') and len(region_11_names) > 0:
                current_text = v.get_label_by_id('11').get_text()
                x, y = v.get_label_by_id('11').get_position()
                
                protein_text = '\n'.join(region_11_names)
                if len(region_11_proteins) > max_labels_per_region:
                    protein_text += f'\n... +{len(region_11_proteins) - max_labels_per_region} more'
                
                new_text = f"{current_text}\n\n{protein_text}"
                v.get_label_by_id('11').set_text(new_text)
                v.get_label_by_id('11').set_fontsize(8)

        ax.set_title(f'Enriched Protein Overlap\n({stats_params["threshold_label"]})',
                    fontsize=14, fontweight='bold')

        # Save with appropriate filename
        suffix = '_with_labels' if labels_in_diagram else ''
        venn_path = os.path.join(viz_dir, f'venn_diagram_{stats_params["threshold_label"]}{suffix}.pdf')
        plt.tight_layout()
        plt.savefig(venn_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"> Saved: venn_diagram_{stats_params['threshold_label']}{suffix}.pdf")

        overlaps = {
            f'{comp1}_only': set1 - set2,
            f'{comp2}_only': set2 - set1,
            'shared': set1 & set2
        }

    elif n_comparisons == 3:
        print(f"\nCreating 3-way Venn diagram...")

        comp1, comp2, comp3 = comparisons[0], comparisons[1], comparisons[2]
        set1, set2, set3 = protein_sets[comp1], protein_sets[comp2], protein_sets[comp3]

        # Create figure with appropriate size
        fig_height = 12 if labels_in_diagram else 10
        fig, ax = plt.subplots(figsize=(12, fig_height))
        v = venn3([set1, set2, set3], set_labels=(comp1, comp2, comp3), ax=ax)

        # Color the regions
        region_colors = {
            '100': '#E74C3C', '010': '#3498DB', '001': '#2ECC71',
            '110': '#F39C12', '101': '#9B59B6', '011': '#1ABC9C',
            '111': '#34495E'
        }
        for region, color in region_colors.items():
            if v.get_patch_by_id(region):
                v.get_patch_by_id(region).set_color(color)
                v.get_patch_by_id(region).set_alpha(0.6)

        # Add protein names to diagram if requested
        if labels_in_diagram:
            # Calculate all protein sets for each region
            region_proteins = {
                '100': set1 - set2 - set3,
                '010': set2 - set1 - set3,
                '001': set3 - set1 - set2,
                '110': (set1 & set2) - set3,
                '101': (set1 & set3) - set2,
                '011': (set2 & set3) - set1,
                '111': set1 & set2 & set3
            }
            
            # Add labels to each region
            for region_id, proteins in region_proteins.items():
                label = v.get_label_by_id(region_id)
                if label and len(proteins) > 0:
                    # Get top protein names
                    protein_names = get_top_protein_names(proteins, max_labels_per_region)
                    
                    if len(protein_names) > 0:
                        # Get current count
                        current_text = label.get_text()
                        
                        # Create protein name text
                        protein_text = '\n'.join(protein_names)
                        if len(proteins) > max_labels_per_region:
                            protein_text += f'\n... +{len(proteins) - max_labels_per_region} more'
                        
                        # Combine count and protein names
                        new_text = f"{current_text}\n\n{protein_text}"
                        label.set_text(new_text)
                        label.set_fontsize(7)

        ax.set_title(f'Enriched Protein Overlap\n({stats_params["threshold_label"]})',
                    fontsize=14, fontweight='bold')

        # Save with appropriate filename
        suffix = '_with_labels' if labels_in_diagram else ''
        venn_path = os.path.join(viz_dir, f'venn_diagram_{stats_params["threshold_label"]}{suffix}.pdf')
        plt.tight_layout()
        plt.savefig(venn_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"> Saved: venn_diagram_{stats_params['threshold_label']}{suffix}.pdf")

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
        print(f"\nWarning: Can only create Venn diagrams for 2 or 3 comparisons")
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
            protein_info = stats_results[stats_results[protein_col].isin(proteins)][
                [protein_col, gene_col]
            ].head(top_n)

            print(f"  Top {min(len(proteins), top_n)}:")
            for _, row in protein_info.iterrows():
                print(f"    - {row[gene_col]}")

            if len(proteins) > top_n:
                print(f"    ... and {len(proteins) - top_n} more")

    # =========================================================================
    # SAVE OVERLAP LISTS TO CSV
    # =========================================================================
    csv_output_dir = os.path.join(config['data_paths']['output_dir'], 'tables')

    for category, proteins in overlaps.items():
        if len(proteins) > 0:
            overlap_df = stats_results[stats_results[protein_col].isin(proteins)].copy()

            fc_cols = [c for c in overlap_df.columns if '_log2FC' in c]
            if len(fc_cols) > 0:
                overlap_df['avg_log2FC'] = overlap_df[fc_cols].mean(axis=1)
                overlap_df = overlap_df.sort_values('avg_log2FC', ascending=False)

            csv_path = os.path.join(csv_output_dir, f'venn_{category}_{stats_params["threshold_label"]}.csv')
            overlap_df.to_csv(csv_path, index=False)
            print(f"\n> Saved: venn_{category}_{stats_params['threshold_label']}.csv")

    print("\n" + "="*80)
    print("VENN ANALYSIS COMPLETE")
    print("="*80 + "\n")

    return {
        'overlaps': overlaps,
        'protein_sets': protein_sets,
        'n_comparisons': n_comparisons
    }