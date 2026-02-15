"""
Quality control functions for IP-MS pipeline.

Generates QC plots (missing values, correlations, PCA) and
handles sample dropping after QC review.
"""

import copy

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .utils import save_data

# Consistent color palette for an arbitrary number of conditions
_PALETTE = [
    '#1f77b4', '#2ca02c', '#d62728', '#ff7f0e', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
]


def _condition_color_map(intensity_cols):
    """Build a color map for an arbitrary number of conditions."""
    conditions = list(intensity_cols.keys())
    return {cond: _PALETTE[i % len(_PALETTE)] for i, cond in enumerate(conditions)}


def qc_ip(data, output_suffix=''):
    """
    Generate quality control plots and metrics.

    Creates:
    - Missing value heatmap
    - Sample correlation heatmap (0 to 1)
    - PCA plot (all samples)
    - PCA plot (treatments only, control excluded)

    Parameters
    ----------
    data : dict
        Output from prep_ip().
    output_suffix : str, optional
        Suffix to add to output filenames. Use this to distinguish QC runs.
        Example: output_suffix='_after_drop' creates '01_missing_values_after_drop.pdf'

    Returns
    -------
    None
        Saves plots to results/figures/qc/.

    Example
    -------
    >>> qc_ip(data)
    >>> data = drop_samples(data)
    >>> qc_ip(data, output_suffix='_after_drop')
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
    color_map = _condition_color_map(intensity_cols)

    all_intensity = []
    for cols in intensity_cols.values():
        all_intensity.extend(cols)

    print(f"\nGenerating QC plots...")
    print(f"  Output directory: {qc_dir}")

    # =========================================================================
    # 1. MISSING VALUES HEATMAP
    # =========================================================================
    print(f"\n[1/4] Creating missing values heatmap...")

    presence_data = df[all_intensity].notna().astype(int)

    fig, ax = plt.subplots(figsize=(12, 8))

    sns.heatmap(
        presence_data.T,
        cmap='RdYlGn',
        cbar_kws={'label': 'Present (1) vs Missing (0)'},
        yticklabels=all_intensity,
        xticklabels=False,
        ax=ax
    )

    ax.set_title('Missing Value Pattern Across Samples', fontsize=14, fontweight='bold')
    ax.set_xlabel('Proteins', fontsize=12)
    ax.set_ylabel('Samples', fontsize=12)

    ax.set_yticks(np.arange(len(all_intensity)) + 0.5)
    ax.set_yticklabels(all_intensity, fontsize=8, rotation=0)

    plt.tight_layout()
    plt.savefig(f"{qc_dir}/01_missing_values{output_suffix}.pdf", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  > Saved: 01_missing_values{output_suffix}.pdf")

    total_values = len(df) * len(all_intensity)
    total_present = presence_data.sum().sum()
    total_missing = total_values - total_present
    pct_missing = (total_missing / total_values) * 100
    print(f"    Overall: {pct_missing:.1f}% missing values")

    # =========================================================================
    # 2. CORRELATION HEATMAP
    # =========================================================================
    print(f"\n[2/4] Creating sample correlation heatmap...")

    corr_data = df[all_intensity].corr()

    fig, ax = plt.subplots(figsize=(12, 10))

    sns.heatmap(
        corr_data,
        annot=False,
        cmap='RdBu_r',
        vmin=0,
        vmax=1,
        center=0.5,
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

    print(f"  > Saved: 02_correlation_heatmap{output_suffix}.pdf")

    print(f"\n  Average correlations within conditions:")
    for condition, cols in intensity_cols.items():
        if len(cols) > 1:
            condition_corr = corr_data.loc[cols, cols]
            mask = np.triu(np.ones_like(condition_corr), k=1).astype(bool)
            avg_corr = condition_corr.where(mask).stack().mean()
            print(f"    {condition}: {avg_corr:.3f}")

    # =========================================================================
    # 3. PCA PLOT (all samples)
    # =========================================================================
    print(f"\n[3/4] Creating PCA plot...")

    _create_pca_plot(
        df, all_intensity, intensity_cols, color_map,
        title='PCA - Sample Clustering',
        save_path=f"{qc_dir}/03_pca_plot{output_suffix}.pdf",
    )

    # =========================================================================
    # 4. PCA PLOT (treatments only)
    # =========================================================================
    print(f"\n[4/4] Creating PCA plot (treatments only, no control)...")

    control_condition = config['conditions']['control']
    treatment_intensity = []
    treatment_cols_map = {}
    for condition, cols in intensity_cols.items():
        if condition != control_condition:
            treatment_intensity.extend(cols)
            treatment_cols_map[condition] = cols

    if treatment_intensity:
        _create_pca_plot(
            df, treatment_intensity, treatment_cols_map, color_map,
            title='PCA - Treatments Only (Control Excluded)',
            save_path=f"{qc_dir}/04_pca_treatments_only{output_suffix}.pdf",
        )
    else:
        print(f"  Warning: No treatment samples found (only control)")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*80)
    print("QC COMPLETE")
    print("="*80)
    print(f"\nPlots saved to: {qc_dir}")
    print(f"  - 01_missing_values{output_suffix}.pdf")
    print(f"  - 02_correlation_heatmap{output_suffix}.pdf")
    print(f"  - 03_pca_plot{output_suffix}.pdf (all samples)")
    print(f"  - 04_pca_treatments_only{output_suffix}.pdf (no control)")
    print("="*80 + "\n")


def _create_pca_plot(df, sample_cols, intensity_cols_subset, color_map, title, save_path):
    """Create and save a PCA scatter plot for the given samples."""
    pca_data = df[sample_cols].dropna()

    if len(pca_data) < 10:
        print(f"  Warning: Only {len(pca_data)} complete proteins")

    pca_data_t = pca_data.T
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(pca_data_t)

    pca = PCA(n_components=min(2, scaled_data.shape[1]))
    pca_coords = pca.fit_transform(scaled_data)

    # Build labels list matching sample order
    labels = []
    for condition, cols in intensity_cols_subset.items():
        for col in cols:
            labels.append(condition)

    fig, ax = plt.subplots(figsize=(12, 10))

    for condition in intensity_cols_subset.keys():
        mask = np.array(labels) == condition
        ax.scatter(
            pca_coords[mask, 0],
            pca_coords[mask, 1],
            c=color_map[condition],
            label=condition,
            s=200,
            alpha=0.7,
            edgecolors='black',
            linewidth=2
        )

    for i, col in enumerate(sample_cols):
        short_label = col.split(',')[-1].strip()[:15]
        x, y = pca_coords[i, 0], pca_coords[i, 1]
        x_range = pca_coords[:, 0].max() - pca_coords[:, 0].min()
        y_range = pca_coords[:, 1].max() - pca_coords[:, 1].min()
        offset_scale = 0.05
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
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=12, loc='best')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  > Saved: {save_path.split('/')[-1]}")
    print(f"    PC1 explains {pca.explained_variance_ratio_[0]*100:.1f}% of variance")
    print(f"    PC2 explains {pca.explained_variance_ratio_[1]*100:.1f}% of variance")


def drop_samples(data, samples_to_drop=None):
    """
    Remove problematic samples from the dataset after QC review.

    Use this after reviewing QC plots to exclude samples that:
    - Have poor correlation with replicates
    - Cluster away from replicates in PCA
    - Have unusual distributions
    - Failed during sample prep

    Parameters
    ----------
    data : dict
        Output from prep_ip().
    samples_to_drop : list of str or dict, optional
        Samples to remove. Can be:
        - List of full column names
        - Dict mapping conditions to replicate numbers: {'EV': [1, 2], 'WT': [3]}
        - None: Interactive mode (prints samples and asks which to drop)

    Returns
    -------
    dict
        Updated data dictionary with samples removed.

    Example
    -------
    >>> data = drop_samples(data, samples_to_drop=[
    ...     'Abundances (Normalized): F7: Sample, EV, Gel2',
    ... ])
    >>> data = drop_samples(data, samples_to_drop={'EV': [2], 'WT': [4]})
    """

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
            short_name = col.split(',')[-1].strip() if ',' in col else col
            print(f"  {len(sample_list)}. {short_name} [{col}]")

    # Determine which samples to drop
    if samples_to_drop is None:
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
        cols_to_drop = []
        for condition, replicate_nums in samples_to_drop.items():
            if condition in intensity_cols:
                for rep_num in replicate_nums:
                    if 0 < rep_num <= len(intensity_cols[condition]):
                        cols_to_drop.append(intensity_cols[condition][rep_num - 1])
                    else:
                        print(f"  Warning: {condition} replicate {rep_num} doesn't exist")
            else:
                print(f"  Warning: Condition '{condition}' not found")

    elif isinstance(samples_to_drop, list):
        cols_to_drop = samples_to_drop

    else:
        print("Invalid samples_to_drop format. No samples dropped.")
        return data

    # Verify columns exist
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]

    if not cols_to_drop:
        print("\nNo valid samples to drop.")
        return data

    print(f"\nDropping {len(cols_to_drop)} sample(s):")
    for col in cols_to_drop:
        print(f"  - {col}")

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
        if updated_cols:
            intensity_cols_updated[condition] = updated_cols
        else:
            print(f"  Warning: All {condition} samples dropped! Condition removed.")

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

    data_updated = {
        'df': df_updated,
        'config': copy.deepcopy(config),
        'intensity_cols': intensity_cols_updated,
        'metadata': metadata_updated,
        'output_dirs': data['output_dirs']
    }

    print("\n" + "="*80)
    print("SAMPLES DROPPED")
    print("="*80)
    print(f"\nRemaining samples: {metadata_updated['n_samples']}")
    print(f"Remaining conditions: {metadata_updated['conditions']}")
    print(f"\nReplicates per condition:")
    for condition, n_reps in metadata_updated['replicates_per_condition'].items():
        print(f"  {condition}: {n_reps} replicates")

    print("\n" + "="*80)
    print("TIP: Save this updated data!")
    print("  save_data(data, 'results/data_after_qc.pkl')")
    print("="*80 + "\n")

    return data_updated
