"""Tests for ipms.statistics module."""

import numpy as np
import pytest

from ipms import norm_ip, stat_ip


@pytest.fixture
def normed_data(prepped_data):
    """Normalized data ready for statistical testing."""
    return norm_ip(prepped_data, method='log2', imputation='mindet')


class TestStatIp:
    def test_returns_stats_results(self, normed_data):
        result = stat_ip(normed_data)

        assert 'stats_results' in result
        assert 'significant_proteins' in result
        assert 'stats_params' in result

    def test_stats_results_has_expected_columns(self, normed_data):
        result = stat_ip(normed_data)
        stats_df = result['stats_results']

        config = normed_data['config']
        control = config['conditions']['control']

        for treatment in config['conditions']['treatments']:
            comp = f"{treatment}_vs_{control}"
            assert f'{comp}_log2FC' in stats_df.columns
            assert f'{comp}_pvalue' in stats_df.columns
            assert f'{comp}_adj_pvalue' in stats_df.columns
            assert f'{comp}_significant' in stats_df.columns

    def test_significant_classification_is_valid(self, normed_data):
        result = stat_ip(normed_data)
        stats_df = result['stats_results']

        config = normed_data['config']
        control = config['conditions']['control']

        for treatment in config['conditions']['treatments']:
            comp = f"{treatment}_vs_{control}"
            valid_values = {'significant', 'not_significant'}
            assert set(stats_df[f'{comp}_significant'].unique()).issubset(valid_values)

    def test_params_stored_correctly(self, normed_data):
        result = stat_ip(normed_data, p_threshold=0.01, log2fc_threshold=2.0, correction='bonferroni')

        params = result['stats_params']
        assert params['p_threshold'] == 0.01
        assert params['log2fc_threshold'] == 2.0
        assert params['correction'] == 'bonferroni'

    def test_stricter_thresholds_give_fewer_hits(self, normed_data):
        lenient = stat_ip(normed_data, p_threshold=0.1, log2fc_threshold=0.5)
        strict = stat_ip(normed_data, p_threshold=0.01, log2fc_threshold=2.0)

        for comp in lenient['significant_proteins']:
            assert strict['significant_proteins'][comp]['total'] <= lenient['significant_proteins'][comp]['total']

    def test_no_correction_option(self, normed_data):
        result = stat_ip(normed_data, correction='none')

        stats_df = result['stats_results']
        config = normed_data['config']
        control = config['conditions']['control']
        treatment = config['conditions']['treatments'][0]
        comp = f"{treatment}_vs_{control}"

        # With no correction, adj_pvalue should equal raw pvalue
        np.testing.assert_array_equal(
            stats_df[f'{comp}_adj_pvalue'].values,
            stats_df[f'{comp}_pvalue'].values
        )
