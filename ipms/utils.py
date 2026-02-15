"""
Utility functions for IP-MS pipeline.

Internal helpers for configuration loading, directory management,
and data serialization.
"""

import os
import pickle

import yaml


def _load_config(config_path):
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def _identify_intensity_columns(df, config):
    """Auto-detect intensity columns from config pattern."""
    intensity_prefix = config['data_columns']['intensity_prefix']

    intensity_cols = {}
    # Control
    control = config['conditions']['control']
    cols = [c for c in df.columns if intensity_prefix in c and control in c]
    intensity_cols[control] = sorted(cols)

    # Treatments
    for treatment in config['conditions']['treatments']:
        cols = [c for c in df.columns if intensity_prefix in c and treatment in c]
        intensity_cols[treatment] = sorted(cols)

    return intensity_cols


def _create_output_dirs(base_dir):
    """Create organized output directory structure."""
    dirs = {
        'base': base_dir,
        'figures': f"{base_dir}/figures",
        'qc': f"{base_dir}/figures/qc",
        'viz': f"{base_dir}/figures/viz",
        'tables': f"{base_dir}/tables"
    }

    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    return dirs


def save_data(data, filename=None):
    """
    Save analysis data to pickle file for sequential workflow.

    Parameters
    ----------
    data : dict
        Analysis data dictionary (output from prep_ip, norm_ip, etc.)
    filename : str, optional
        Custom filename. If None, uses default based on output_dir in config.

    Returns
    -------
    str
        Path where data was saved.

    Example
    -------
    >>> data = prep_ip('config/experiment.yaml')
    >>> save_data(data)  # Saves to results/data_after_prep.pkl
    """
    if filename is None:
        output_dir = data['config']['data_paths']['output_dir']
        filename = os.path.join(output_dir, 'data_checkpoint.pkl')

    with open(filename, 'wb') as f:
        pickle.dump(data, f)

    size_mb = os.path.getsize(filename) / (1024 * 1024)

    print(f"\n{'='*80}")
    print(f"DATA SAVED")
    print(f"{'='*80}")
    print(f"Location: {filename}")
    print(f"Size: {size_mb:.1f} MB")
    print(f"\nTo load this data later:")
    print(f"  from ipms import load_data")
    print(f"  data = load_data('{filename}')")
    print(f"{'='*80}\n")

    return filename


def load_data(filepath):
    """
    Load analysis data from pickle file.

    Parameters
    ----------
    filepath : str
        Path to saved pickle file.

    Returns
    -------
    dict
        Analysis data dictionary.

    Example
    -------
    >>> from ipms import load_data
    >>> data = load_data('results/data_after_prep.pkl')
    >>> qc_ip(data)  # Continue from where you left off
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")

    print(f"\n{'='*80}")
    print(f"LOADING DATA")
    print(f"{'='*80}")

    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    size_mb = os.path.getsize(filepath) / (1024 * 1024)

    print(f"Location: {filepath}")
    print(f"Size: {size_mb:.1f} MB")

    if 'metadata' in data:
        print(f"\nData contains:")
        print(f"  Proteins: {data['metadata']['n_proteins']}")
        print(f"  Samples: {data['metadata']['n_samples']}")
        print(f"  Conditions: {data['metadata']['conditions']}")

    print(f"{'='*80}\n")

    return data
