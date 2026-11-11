"""Training-anchor fitting, leakage protection, serialization and input layouts."""
import copy

import numpy as np
import pandas as pd
import pytest

from preprocessing import (AnchorPreprocessor, company_table, fit_preprocessor,
                           transform_features, transform_histories)
from sequences import create_histories, flatten_histories, valid_step_mask
from validation import FEATURE_COLUMNS


@pytest.fixture
def panel():
    rows = []
    for company, years in [('C_1', range(2000, 2007)), ('C_2', [2001, 2004]),
                           ('C_3', [2008]), ('C_4', [2009])]:
        for year in years:
            row = {'company_name': company, 'status_label': 'alive', 'year': year}
            for col in FEATURE_COLUMNS:
                row[col] = float(year - 2000)
            rows.append(row)
    data = pd.DataFrame(rows)
    splits = company_table(data)
    splits['split'] = ['train', 'train', 'validation', 'test']
    return data, splits


def test_fits_only_training_anchors_and_constant_features(panel):
    data, splits = panel
    data['X18'] = 7.
    fitted = fit_preprocessor(data, splits)
    assert fitted.record['mean'][:17] == [5.] * 17
    assert fitted.record['scale'] == [1.] * 18
    assert fitted.record['n_training_companies'] == 2
    assert fitted.record['fitting_observations'] == [
        {'company_name': 'C_1', 'anchor_year': 2006},
        {'company_name': 'C_2', 'anchor_year': 2004}]
    # Older training data and all validation/test features cannot influence fit.
    modified = data.copy()
    is_anchor = ((modified.company_name == 'C_1') & (modified.year == 2006)) | ((modified.company_name == 'C_2') & (modified.year == 2004))
    modified.loc[~is_anchor, FEATURE_COLUMNS] = 1e12
    assert fit_preprocessor(modified, splits).record == fitted.record


@pytest.mark.parametrize('model', ['logistic_regression', 'xgboost', 'mlp', 'gru'])
@pytest.mark.parametrize('condition', ['latest_year', 'five_year'])
def test_all_eight_paths_preserve_ids_padding_and_offsets(panel, model, condition):
    data, splits = panel
    fitted = fit_preprocessor(data, splits)
    raw = create_histories(data, condition=condition)
    before = raw.features.copy()
    result = transform_histories(raw, fitted, model=model)
    np.testing.assert_array_equal(raw.features, before)
    mask = valid_step_mask(result)
    expected = raw.features[mask, :18]
    if model != 'xgboost':
        expected = expected - 5
    np.testing.assert_allclose(result.features[mask, :18], expected)
    assert (result.features[~mask] == 0).all()
    np.testing.assert_array_equal(result.lengths, raw.lengths)
    pd.testing.assert_frame_equal(result.metadata, raw.metadata)
    if condition == 'five_year':
        assert result.lengths.tolist() == [5, 2, 1, 1]
        assert result.metadata.loc[0, 'years'] == '2002,2003,2004,2005,2006'
        np.testing.assert_array_equal(result.features[:, :, 18], raw.features[:, :, 18])
        flat, _ = flatten_histories(result)
        assert flat.shape == (4, 100)
        assert flat.loc[1, 'present_r2'] == 0
        assert flat.loc[1, 'years_before_anchor_r1'] == 3
        assert flat.loc[1, 'X1_r2'] == 0
        assert flat.loc[1, 'X1_r0'] == (4 if model == 'xgboost' else -1)


def test_save_load_and_schema_rejection(panel, tmp_path):
    data, splits = panel
    fitted = fit_preprocessor(data, splits)
    path = tmp_path / 'transform.json'
    fitted.save(path)
    loaded = AnchorPreprocessor.load(path)
    values = data[FEATURE_COLUMNS]
    np.testing.assert_array_equal(transform_features(values, fitted, model='mlp'),
                                  transform_features(values, loaded, model='mlp'))
    with pytest.raises(ValueError):
        loaded.transform(values[FEATURE_COLUMNS[::-1]], model='mlp')
    record = copy.deepcopy(fitted.record)
    record['schemas']['five_year']['sequence'].reverse()
    with pytest.raises(ValueError):
        AnchorPreprocessor(record)


def test_missing_values_not_padding(panel):
    data, splits = panel
    fitted = fit_preprocessor(data, splits)
    history = create_histories(data, condition='five_year')
    history.features[~valid_step_mask(history)] = np.nan
    assert np.isfinite(transform_histories(history, fitted, model='gru').features).all()
    history.features[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        transform_histories(history, fitted, model='gru')
    data.loc[(data.company_name == 'C_1') & (data.year == 2006), 'X1'] = np.nan
    with pytest.raises(ValueError):
        fit_preprocessor(data, splits)


def test_changed_split_anchor_is_rejected(panel):
    data, splits = panel
    splits.loc[0, 'anchor_year'] = 2005
    with pytest.raises(ValueError):
        fit_preprocessor(data, splits)


def test_preparation_script_roundtrip(panel, tmp_path, monkeypatch):
    import importlib.util
    import json
    import sys
    from pathlib import Path
    import yaml
    from preprocessing import file_sha256, save_splits
    data, splits = panel
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('prepare_preprocessing', root / 'scripts/prepare_preprocessing.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    csv = tmp_path / 'data.csv'
    data.to_csv(csv, index=False)
    hashes = save_splits(splits, tmp_path / 'splits')
    config = yaml.safe_load((root / 'configs/config.yaml').read_text())
    config['dataset']['sha256'] = file_sha256(csv)
    config['split'] = {'directory': str(tmp_path / 'splits'), 'sha256': hashes}
    cfg = tmp_path / 'config.yaml'
    cfg.write_text(yaml.safe_dump(config))
    output, report = tmp_path / 'transform.json', tmp_path / 'report.json'
    monkeypatch.setattr(sys, 'argv', ['prepare_preprocessing', '--csv', str(csv),
                        '--config', str(cfg), '--output', str(output), '--report', str(report)])
    module.main()
    loaded = AnchorPreprocessor.load(output)
    assert loaded.record['mean'] == [5.] * 18
    assert loaded.record['provenance']['split_sha256'] == hashes
    assert json.loads(report.read_text())['n_companies'] == 2
    config['preprocessing']['scaling_strategy'] = 'robust'
    cfg.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match='policy'):
        module.main()
