"""Tests for ipms.prep module."""

import numpy as np
import pytest


class TestPrepIp:
    def test_returns_required_keys(self, prepped_data):
        assert 'df' in prepped_data
        assert 'config' in prepped_data
        assert 'intensity_cols' in prepped_data
        assert 'metadata' in prepped_data
        assert 'output_dirs' in prepped_data

    def test_filters_low_peptide_proteins(self, prepped_data):
        config = prepped_data['config']
        peptide_col = config['data_columns']['peptides']
        min_peptides = config['qc_parameters']['min_peptides']

        df = prepped_data['df']
        assert (df[peptide_col] >= min_peptides).all()

    def test_removes_contaminants(self, prepped_data):
        df = prepped_data['df']
        protein_col = prepped_data['config']['data_columns']['protein_id']

        # KRT10_CONTAM should have been removed
        assert 'KRT10_CONTAM' not in df[protein_col].values

    def test_intensity_cols_match_conditions(self, prepped_data):
        intensity_cols = prepped_data['intensity_cols']
        config = prepped_data['config']

        assert config['conditions']['control'] in intensity_cols
        for treatment in config['conditions']['treatments']:
            assert treatment in intensity_cols

    def test_metadata_counts_are_consistent(self, prepped_data):
        metadata = prepped_data['metadata']
        df = prepped_data['df']
        intensity_cols = prepped_data['intensity_cols']

        assert metadata['n_proteins'] == len(df)

        total_samples = sum(len(v) for v in intensity_cols.values())
        assert metadata['n_samples'] == total_samples

    def test_all_intensity_cols_exist_in_df(self, prepped_data):
        df = prepped_data['df']
        for condition, cols in prepped_data['intensity_cols'].items():
            for col in cols:
                assert col in df.columns, f"Column {col} missing from dataframe"
