"""Shared test fixtures for IP-MS pipeline tests."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest
import yaml


@pytest.fixture
def sample_config(tmp_path):
    """Create a minimal YAML config and matching Excel data for testing."""
    np.random.seed(42)

    n_proteins = 50
    n_replicates = 3

    # Build intensity columns
    ev_cols = [f"Abundances (Normalized): F{i}: Sample, EV, Gel{i}" for i in range(1, n_replicates + 1)]
    wt_cols = [f"Abundances (Normalized): F{i+3}: Sample, WT, Gel{i}" for i in range(1, n_replicates + 1)]
    d2d3_cols = [f"Abundances (Normalized): F{i+6}: Sample, d2d3, Gel{i}" for i in range(1, n_replicates + 1)]

    data = {
        'Accession': [f'P{str(i).zfill(5)}' for i in range(n_proteins)],
        'Gene_Symbol': [f'GENE{i}' for i in range(n_proteins)],
        '# Peptides': np.random.randint(1, 20, n_proteins),
    }

    for col in ev_cols + wt_cols + d2d3_cols:
        values = np.random.lognormal(mean=10, sigma=2, size=n_proteins)
        # Introduce ~10% missing values
        mask = np.random.random(n_proteins) < 0.1
        values[mask] = np.nan
        data[col] = values

    # Make some proteins clearly enriched in WT
    for col in wt_cols:
        data[col][:5] = np.random.lognormal(mean=14, sigma=0.5, size=5)

    # Ensure peptide filter will remove some proteins
    data['# Peptides'][-5:] = 1

    # Add a known contaminant
    data['Accession'][45] = 'KRT10_CONTAM'
    data['Gene_Symbol'][45] = 'KRT10'

    df = pd.DataFrame(data)
    excel_path = str(tmp_path / 'test_data.xlsx')
    df.to_excel(excel_path, index=False)

    config = {
        'experiment': {
            'name': 'Test_Experiment',
            'description': 'Unit test experiment',
            'date': '2026-01-01',
            'investigator': 'Test',
        },
        'conditions': {
            'control': 'EV',
            'treatments': ['WT', 'd2d3'],
        },
        'data_columns': {
            'protein_id': 'Accession',
            'gene_symbol': 'Gene_Symbol',
            'peptides': '# Peptides',
            'intensity_prefix': 'Abundances (Normalized):',
        },
        'data_paths': {
            'input_file': excel_path,
            'output_dir': str(tmp_path / 'results'),
        },
        'qc_parameters': {
            'min_peptides': 2,
            'min_valid_per_protein': 0.5,
            'min_valid_per_sample': 0.3,
        },
        'manual_contaminants': ['KRT'],
        'normalization': {
            'method': 'log2',
            'imputation': 'mindet',
            'bait_normalize': False,
            'bait_protein': None,
        },
        'statistics': {
            'alpha': 0.05,
            'log2fc_threshold': 1.0,
            'correction': 'fdr_bh',
        },
    }

    config_path = str(tmp_path / 'test_config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    return config_path, tmp_path


@pytest.fixture
def prepped_data(sample_config):
    """Run prep_ip and return the result for downstream tests."""
    from ipms import prep_ip

    config_path, tmp_path = sample_config
    data = prep_ip(config_path)
    return data
