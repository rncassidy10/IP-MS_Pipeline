"""Tests for ipms.utils module."""

import os
import pickle

import pytest
import yaml

from ipms.utils import _load_config, _create_output_dirs, save_data, load_data


class TestLoadConfig:
    def test_loads_valid_yaml(self, tmp_path):
        config = {'experiment': {'name': 'test'}, 'conditions': {'control': 'EV'}}
        path = str(tmp_path / 'config.yaml')
        with open(path, 'w') as f:
            yaml.dump(config, f)

        result = _load_config(path)
        assert result['experiment']['name'] == 'test'
        assert result['conditions']['control'] == 'EV'

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            _load_config('/nonexistent/config.yaml')


class TestCreateOutputDirs:
    def test_creates_all_directories(self, tmp_path):
        base = str(tmp_path / 'output')
        dirs = _create_output_dirs(base)

        assert os.path.isdir(dirs['base'])
        assert os.path.isdir(dirs['figures'])
        assert os.path.isdir(dirs['qc'])
        assert os.path.isdir(dirs['viz'])
        assert os.path.isdir(dirs['tables'])

    def test_idempotent(self, tmp_path):
        base = str(tmp_path / 'output')
        dirs1 = _create_output_dirs(base)
        dirs2 = _create_output_dirs(base)
        assert dirs1 == dirs2


class TestSaveLoadData:
    def test_roundtrip(self, tmp_path):
        data = {
            'config': {'data_paths': {'output_dir': str(tmp_path)}},
            'metadata': {'n_proteins': 100, 'n_samples': 9, 'conditions': ['EV', 'WT']},
            'df': 'placeholder',
        }
        path = str(tmp_path / 'test.pkl')
        save_data(data, path)

        loaded = load_data(path)
        assert loaded['metadata']['n_proteins'] == 100
        assert loaded['df'] == 'placeholder'

    def test_load_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_data('/nonexistent/data.pkl')

    def test_save_default_path(self, tmp_path):
        data = {
            'config': {'data_paths': {'output_dir': str(tmp_path)}},
            'metadata': {'n_proteins': 50, 'n_samples': 6, 'conditions': ['EV']},
        }
        result_path = save_data(data)
        assert os.path.exists(result_path)
        assert 'data_checkpoint.pkl' in result_path
