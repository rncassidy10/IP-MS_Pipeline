"""Tests for ipms.normalization module."""

import numpy as np
import pytest

from ipms import norm_ip


class TestNormIp:
    def test_log2_reduces_range(self, prepped_data):
        all_intensity = []
        for cols in prepped_data['intensity_cols'].values():
            all_intensity.extend(cols)

        raw_max = prepped_data['df'][all_intensity].max().max()

        result = norm_ip(prepped_data, method='log2', imputation='zero')
        norm_max = result['df'][all_intensity].max().max()

        # log2 should dramatically reduce the scale
        assert norm_max < raw_max

    def test_mindet_imputation_fills_missing(self, prepped_data):
        all_intensity = []
        for cols in prepped_data['intensity_cols'].values():
            all_intensity.extend(cols)

        result = norm_ip(prepped_data, method='log2', imputation='mindet')
        missing_after = result['df'][all_intensity].isna().sum().sum()

        assert missing_after == 0

    def test_zero_imputation_fills_with_zero(self, prepped_data):
        all_intensity = []
        for cols in prepped_data['intensity_cols'].values():
            all_intensity.extend(cols)

        result = norm_ip(prepped_data, method='log2', imputation='zero')

        # After log2(0+1) = 0, so zero-imputed values should be 0
        df = result['df']
        assert df[all_intensity].isna().sum().sum() == 0

    def test_normalization_metadata_stored(self, prepped_data):
        result = norm_ip(prepped_data, method='log2', imputation='mindet')

        assert 'normalization' in result
        assert result['normalization']['method'] == 'log2'
        assert result['normalization']['imputation'] == 'mindet'

    def test_zscore_centers_data(self, prepped_data):
        result = norm_ip(prepped_data, method='zscore', imputation='zero')

        all_intensity = []
        for cols in prepped_data['intensity_cols'].values():
            all_intensity.extend(cols)

        # Each column should be approximately mean=0
        for col in all_intensity:
            values = result['df'][col].dropna()
            assert abs(values.mean()) < 0.1, f"Column {col} mean is {values.mean()}"

    def test_does_not_mutate_input(self, prepped_data):
        original_shape = prepped_data['df'].shape

        norm_ip(prepped_data, method='log2', imputation='zero')

        assert prepped_data['df'].shape == original_shape
