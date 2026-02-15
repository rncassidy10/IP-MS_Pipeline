"""
Normalization and imputation functions for IP-MS pipeline.

Supports log2, z-score, quantile, and median normalization,
with mindet, zero, median, and KNN imputation strategies.
"""

import copy
import os

import matplotlib.pyplot as plt
import numpy as np

from .utils import save_data


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

    Parameters
    ----------
    data : dict
        Output from prep_ip() or drop_samples().
    method : str, optional
        Normalization method (default: 'log2').
    imputation : str, optional
        Imputation method for missing values (default: 'mindet').

    Returns
    -------
    dict
        Updated data dictionary with normalized/imputed values.

    Example
    -------
    >>> data = prep_ip('config/experiment.yaml')
    >>> data = norm_ip(data, method='log2', imputation='mindet')
    """

    print("\n" + "="*80)
    print("NORMALIZATION AND IMPUTATION")
    print("="*80)

    df = data['df'].copy()
    config = data['config']
    intensity_cols = data['intensity_cols']

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
        df[all_intensity] = np.log2(df[all_intensity] + 1)
        print(f"  > Log2 transformation applied")

    elif method == 'zscore':
        for col in all_intensity:
            values = df[col].dropna()
            mean = values.mean()
            std = values.std()
            df[col] = (df[col] - mean) / std
        print(f"  > Z-score normalization applied")

    elif method == 'quantile':
        from sklearn.preprocessing import quantile_transform
        mask = df[all_intensity].notna()
        df.loc[:, all_intensity] = df[all_intensity].where(
            ~mask,
            quantile_transform(df[all_intensity].fillna(0))
        )
        print(f"  > Quantile normalization applied")

    elif method == 'median':
        for col in all_intensity:
            median = df[col].median()
            global_median = df[all_intensity].median().median()
            df[col] = df[col] - median + global_median
        print(f"  > Median normalization applied")

    else:
        print(f"  Warning: Unknown method '{method}', skipping normalization")

    # =========================================================================
    # 3. IMPUTATION
    # =========================================================================
    print(f"\n[3/3] Applying {imputation} imputation...")

    missing_before = df[all_intensity].isna().sum().sum()

    if imputation == 'mindet':
        for col in all_intensity:
            if df[col].isna().any():
                valid_values = df[col].dropna()
                if len(valid_values) > 0:
                    min_val = valid_values.min()
                    std_val = valid_values.std()
                    impute_val = min_val - 1.8 * std_val
                    df[col] = df[col].fillna(impute_val)
        print(f"  > MinDet imputation applied (min - 1.8*std per sample)")

    elif imputation == 'zero':
        df[all_intensity] = df[all_intensity].fillna(0)
        print(f"  > Zero imputation applied")

    elif imputation == 'median':
        for col in all_intensity:
            median = df[col].median()
            df[col] = df[col].fillna(median)
        print(f"  > Median imputation applied")

    elif imputation == 'knn':
        from sklearn.impute import KNNImputer
        imputer = KNNImputer(n_neighbors=5)
        df[all_intensity] = imputer.fit_transform(df[all_intensity])
        print(f"  > KNN imputation applied (k=5)")

    else:
        print(f"  Warning: Unknown imputation '{imputation}', skipping")

    missing_after = df[all_intensity].isna().sum().sum()
    print(f"    Missing values: {missing_before} -> {missing_after}")

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
    df_original = data['df'][all_intensity].copy()

    n_conditions = len(intensity_cols)
    fig, axes = plt.subplots(2, n_conditions, figsize=(5*n_conditions, 10))

    if n_conditions == 1:
        axes = axes.reshape(2, 1)

    for idx, (condition, cols) in enumerate(intensity_cols.items()):
        ax_before = axes[0, idx]
        for col in cols:
            values = df_original[col].dropna()
            ax_before.hist(values, bins=50, alpha=0.5, label=col.split(',')[-1].strip()[:10])
        ax_before.set_title(f'{condition} - Before Normalization', fontweight='bold')
        ax_before.set_xlabel('Intensity (Raw)', fontsize=10)
        ax_before.set_ylabel('Frequency', fontsize=10)
        ax_before.legend(fontsize=8, loc='upper right')
        ax_before.grid(alpha=0.3)

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

    print(f"  > Saved: normalization_comparison.pdf")

    # =========================================================================
    # 6. UPDATE DATA DICTIONARY
    # =========================================================================
    data_updated = copy.copy(data)
    data_updated['df'] = df
    data_updated['normalization'] = {
        'method': method,
        'imputation': imputation
    }

    # Auto-save for sequential workflow
    output_path = os.path.join(config['data_paths']['output_dir'], 'data_after_norm.pkl')
    save_data(data_updated, output_path)

    print("\n" + "="*80)
    print("NORMALIZATION COMPLETE")
    print("="*80)
    print(f"\nNext step: stat_ip() for statistical analysis")
    print("="*80 + "\n")

    return data_updated
