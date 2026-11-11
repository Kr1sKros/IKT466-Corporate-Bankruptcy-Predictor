"""Per-condition training fits, leakage protection, serialization and input layouts."""
import copy

import numpy as np
import pandas as pd
import pytest

from preprocessing import (FittedPreprocessor, company_table, fit_preprocessor,
                           training_fit_rows, transform_features, transform_histories)
from sequences import create_histories, flatten_histories, valid_step_mask, valid_steps
from validation import FEATURE_COLUMNS

# Training values per condition (every feature = year - 2000):
# latest_year: C_1 2006 -> 6, C_2 2004 -> 4.
# five_year:   C_1 2002-2006 -> 2..6, C_2 2001, 2004 -> 1, 4. C_1 2000-2001 excluded.
LATEST_FIT = np.array([6., 4.])
FIVE_FIT = np.array([2., 3., 4., 5., 6., 1., 4.])


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


def test_each_condition_is_fitted_on_its_own_training_rows(panel):
    data, splits = panel
    data['X18'] = 7.
    fitted = fit_preprocessor(data, splits)
    latest, five = fitted.record['conditions']['latest_year'], fitted.record['conditions']['five_year']
    assert latest['mean'][:17] == [LATEST_FIT.mean()] * 17
    assert latest['scale'][:17] == [LATEST_FIT.std()] * 17
    np.testing.assert_allclose(five['mean'][:17], FIVE_FIT.mean())
    np.testing.assert_allclose(five['scale'][:17], FIVE_FIT.std())
    assert latest['scale'][17] == five['scale'][17] == 1.  # constant column
    assert latest['fitting_years'] == {'C_1': [2006], 'C_2': [2004]}
    assert five['fitting_years'] == {'C_1': [2002, 2003, 2004, 2005, 2006], 'C_2': [2001, 2004]}
    assert (latest['n_fitting_observations'], five['n_fitting_observations']) == (2, 7)
    assert latest['n_training_companies'] == five['n_training_companies'] == 2


def test_fit_ignores_older_years_and_held_out_companies(panel):
    data, splits = panel
    fitted = fit_preprocessor(data, splits)
    # Validation/test companies and training years before the window: no effect.
    modified = data.copy()
    outside = data.company_name.isin(['C_3', 'C_4']) | ((data.company_name == 'C_1') & (data.year < 2002))
    modified.loc[outside, FEATURE_COLUMNS] = 1e12
    assert fit_preprocessor(modified, splits).record == fitted.record
    # An older year inside the five-year window changes only the five_year fit.
    modified = data.copy()
    modified.loc[(data.company_name == 'C_1') & (data.year == 2003), FEATURE_COLUMNS] = 100.
    changed = fit_preprocessor(modified, splits).record['conditions']
    assert changed['latest_year'] == fitted.record['conditions']['latest_year']
    assert changed['five_year']['mean'] != fitted.record['conditions']['five_year']['mean']


@pytest.mark.parametrize('condition', ['latest_year', 'five_year'])
def test_fit_rows_are_the_real_training_steps_of_the_inputs(panel, condition):
    data, splits = panel
    rows = training_fit_rows(data, splits, condition=condition)
    train = splits.loc[splits.split.eq('train'), 'company_name']
    steps = valid_steps(create_histories(data, condition=condition, companies=train))
    np.testing.assert_array_equal(rows[FEATURE_COLUMNS].to_numpy(np.float32), steps[:, :18])


@pytest.mark.parametrize('model', ['logistic_regression', 'xgboost', 'mlp', 'gru'])
@pytest.mark.parametrize('condition', ['latest_year', 'five_year'])
def test_all_eight_paths_preserve_ids_padding_and_offsets(panel, model, condition):
    data, splits = panel
    fitted = fit_preprocessor(data, splits)
    fit = LATEST_FIT if condition == 'latest_year' else FIVE_FIT
    raw = create_histories(data, condition=condition)
    before = raw.features.copy()
    result = transform_histories(raw, fitted, model=model)
    np.testing.assert_array_equal(raw.features, before)
    mask = valid_step_mask(result)
    expected = raw.features[mask, :18]
    if model != 'xgboost':
        expected = (expected - fit.mean()) / fit.std()
    np.testing.assert_allclose(result.features[mask, :18], expected, rtol=1e-6)
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
        scaled_r0 = 4 if model == 'xgboost' else (4 - fit.mean()) / fit.std()
        assert flat.loc[1, 'X1_r0'] == pytest.approx(scaled_r0)


def test_save_load_and_schema_rejection(panel, tmp_path):
    data, splits = panel
    fitted = fit_preprocessor(data, splits)
    path = tmp_path / 'transform.json'
    fitted.save(path)
    loaded = FittedPreprocessor.load(path)
    values = data[FEATURE_COLUMNS]
    for condition in ('latest_year', 'five_year'):
        np.testing.assert_array_equal(
            transform_features(values, fitted, model='mlp', condition=condition),
            transform_features(values, loaded, model='mlp', condition=condition))
    with pytest.raises(ValueError):
        loaded.transform(values[FEATURE_COLUMNS[::-1]], model='mlp', condition='latest_year')
    with pytest.raises(ValueError, match='supports'):
        loaded.transform(values, model='mlp', condition='full_history')
    record = copy.deepcopy(fitted.record)
    record['conditions']['five_year']['schema']['sequence'].reverse()
    with pytest.raises(ValueError):
        FittedPreprocessor(record)
    record = copy.deepcopy(fitted.record)
    del record['conditions']['latest_year']
    with pytest.raises(ValueError):
        FittedPreprocessor(record)


def test_missing_values_not_padding(panel):
    data, splits = panel
    fitted = fit_preprocessor(data, splits)
    history = create_histories(data, condition='five_year')
    history.features[~valid_step_mask(history)] = np.nan
    assert np.isfinite(transform_histories(history, fitted, model='gru').features).all()
    history.features[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        transform_histories(history, fitted, model='gru')
    data.loc[(data.company_name == 'C_1') & (data.year == 2003), 'X1'] = np.nan
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
    loaded = FittedPreprocessor.load(output)
    assert loaded.record['conditions']['latest_year']['mean'] == [LATEST_FIT.mean()] * 18
    assert loaded.record['provenance']['split_sha256'] == hashes
    conditions = json.loads(report.read_text())['conditions']
    assert conditions['latest_year']['n_observations'] == 2
    assert conditions['five_year']['n_observations'] == 7
    config['preprocessing']['scaling_strategy'] = 'robust'
    cfg.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match='policy'):
        module.main()
