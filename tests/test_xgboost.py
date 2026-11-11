"""Real tiny XGBoost fits, failure handling, and test-isolated paired studies."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from xgboost.core import XGBoostError

from models import xgboost_model as xgb
from preprocessing import company_table, file_sha256, fit_preprocessor, save_splits, transform_histories
from sequences import create_histories, flatten_histories
from validation import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('train_xgboost_script', ROOT / 'scripts/train_xgboost.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.fixture
def panel():
    rows = []
    for n in range(1, 65):
        years = [2000 + 2 * j for j in range(1 + n % 8)]
        for year in years:
            row = {'company_name': f'C_{n}', 'year': year,
                   'status_label': 'failed' if n % 4 == 0 else 'alive'}
            row.update({f: float((n % 4 == 0) * 4 + (year - 2000) / 8 + j / 10)
                        for j, f in enumerate(FEATURE_COLUMNS)})
            rows.append(row)
    data = pd.DataFrame(rows)
    companies = company_table(data)
    companies['split'] = ['train' if int(c[2:]) <= 40 else 'validation' if int(c[2:]) <= 52 else 'test'
                          for c in companies.company_name]
    return data, companies


@pytest.fixture
def search():
    config = yaml.safe_load((ROOT / 'configs/config.yaml').read_text())
    space = config['models']['xgboost']['search']
    space['n_estimators'] = {'low': 2, 'high': 5}
    space['max_depth'] = {'low': 1, 'high': 2}
    space['min_child_weight'] = {'low': 1., 'high': 1.}
    return space


def features(panel, condition='latest_year', partition='train'):
    data, companies = panel
    histories = create_histories(data, condition=condition,
        companies=companies.loc[companies.split.eq(partition), 'company_name'])
    return flatten_histories(transform_histories(histories, fit_preprocessor(data, companies), model='xgboost'))


@pytest.mark.parametrize('condition,width', [('latest_year', 18), ('five_year', 100)])
def test_fit_identity_inputs_and_reload(panel, tmp_path, condition, width):
    data, companies = panel
    x, meta = features(panel, condition)
    assert x.shape == (40, width)
    raw, _ = flatten_histories(create_histories(data, condition=condition, companies=meta.company_name))
    pd.testing.assert_frame_equal(x, raw)
    model = xgb.train_model(xgb.build_model(hyperparameters={'n_estimators': 5}), x, meta.target)
    pipeline = xgb.XGBoostPipeline(condition, fit_preprocessor(data, companies), model, .4)
    pipeline.save(tmp_path)
    before = pipeline.predict(data, companies.company_name)
    after = xgb.XGBoostPipeline.load(tmp_path).predict(data, companies.company_name)
    pd.testing.assert_frame_equal(before, after)
    assert before.probability.between(0, 1).all()
    assert len(before) == len(companies)


@pytest.mark.parametrize('bad', [{'n_jobs': 4}, {'objective': 'reg:squarederror'}, {'device': 'cuda'},
                               {'random_state': 4}, {'scale_pos_weight': 0}])
def test_protected_settings(bad):
    with pytest.raises(ValueError):
        xgb.build_model(hyperparameters=bad)


def test_rejects_wrong_schema_nan_and_bad_labels(panel):
    x, meta = features(panel)
    for bad in [x[x.columns[::-1]], x.assign(company_name='C_1'), x.assign(X1=np.nan)]:
        with pytest.raises(ValueError):
            xgb.train_model(xgb.build_model(), bad, meta.target)
    for bad in [np.zeros(len(meta)), ['failed'] * len(meta), meta.target.iloc[:-1]]:
        with pytest.raises(ValueError):
            xgb.train_model(xgb.build_model(), x, bad)


def test_seeded_search_and_training_weight_bounds(panel, search, tmp_path):
    train, tm = features(panel)
    val, vm = features(panel, partition='validation')
    selections, predictions, records = [], [], []
    # Cross the ten-trial random startup to exercise TPE itself.
    for n in range(2):
        model, selected = xgb.tune_condition(train, tm.target, val, vm.target,
            directory=tmp_path / str(n), study_name=f'seed_{n}', seed=42,
            sampler_seed=42, trials=11, search=search)
        selections.append(selected)
        predictions.append(model.predict_proba(val))
        records.append(json.loads((tmp_path / str(n) / 'trials.json').read_text()))
    assert selections[0]['parameters'] == selections[1]['parameters']
    assert selections[0]['threshold'] == selections[1]['threshold']
    np.testing.assert_array_equal(*predictions)
    assert [r['parameters'] for r in records[0]] == [r['parameters'] for r in records[1]]
    assert selections[0]['class_balance']['negative_to_positive_ratio'] == 3.
    weights = [r['attributes']['effective_parameters']['scale_pos_weight'] for r in records[0]]
    assert weights[0] == 1.
    assert all(1 <= w <= 3 for w in weights)
    assert len(records[0]) == 11


def test_failed_trials_recorded_and_never_selected(panel, search, tmp_path, monkeypatch):
    train, tm = features(panel)
    val, vm = features(panel, partition='validation')
    def fail(*args):
        raise XGBoostError('injected training failure')
    monkeypatch.setattr(xgb, 'train_model', fail)
    destination = tmp_path / 'failed'
    with pytest.raises(RuntimeError, match='wholly failed'):
        xgb.tune_condition(train, tm.target, val, vm.target, directory=destination,
            study_name='failed', seed=42, sampler_seed=42, trials=2, search=search)
    summary = json.loads((destination / 'study_summary.json').read_text())
    assert summary['attempted'] == summary['failed'] == 2
    assert summary['completed'] == 0
    assert not (destination / 'selection.json').exists()
    assert all('injected' in t['attributes']['failure']
               for t in json.loads((destination / 'trials.json').read_text()))


def prepare_run(panel, tmp_path, search):
    data, companies = panel
    csv = tmp_path / 'synthetic.csv'
    data.to_csv(csv, index=False)
    config = yaml.safe_load((ROOT / 'configs/config.yaml').read_text())
    config['dataset']['sha256'] = file_sha256(csv)
    config['split']['directory'] = str(tmp_path / 'splits')
    config['split']['sha256'] = save_splits(companies, tmp_path / 'splits')
    config['study_plan']['tuning']['attempted_trials_per_study'] = 1
    config['models']['xgboost']['search'] = search
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(config))
    return csv, path


def test_paired_runner_freezes_before_test_and_writes_aligned_artifacts(panel, search, tmp_path, monkeypatch):
    csv, config = prepare_run(panel, tmp_path, search)
    output = tmp_path / 'run'
    original = xgb.XGBoostPipeline.predict
    test_ids = set(panel[1].loc[panel[1].split.eq('test'), 'company_name'])
    def guarded_predict(self, data, ids):
        if set(ids) & test_ids:
            frozen = json.loads((output / 'frozen_selections.json').read_text())
            assert set(frozen) == {'latest_year', 'five_year'}
        return original(self, data, ids)
    monkeypatch.setattr(xgb.XGBoostPipeline, 'predict', guarded_predict)
    report = runner.run(csv, config, output)
    assert report['test_count'] == 12
    assert set(report['results']) == {'latest_year', 'five_year'}
    manifest = json.loads((output / 'manifest.json').read_text())
    assert len(manifest['runs']) == 2
    for condition in ['latest_year', 'five_year']:
        frame = pd.read_csv(output / condition / 'test_predictions.csv')
        assert set(frame.company_name) == test_ids
        assert frame.condition.eq(condition).all()
        metadata = json.loads((output / condition / 'metadata.json').read_text())
        assert metadata['target_verified'] is False
        assert metadata['parameters']['threshold_source'] == 'validation'
        assert metadata['software']['packages']['optuna']
    with pytest.raises(FileExistsError):
        runner.run(csv, config, output)


def test_test_data_cannot_change_selections(panel, search, tmp_path):
    csv, config_path = prepare_run(panel, tmp_path, search)
    runner.run(csv, config_path, tmp_path / 'original')
    data = pd.read_csv(csv)
    test_ids = set(panel[1].loc[panel[1].split.eq('test'), 'company_name'])
    data.loc[data.company_name.isin(test_ids), FEATURE_COLUMNS] = 1e6
    data.to_csv(csv, index=False)
    config = yaml.safe_load(config_path.read_text())
    config['dataset']['sha256'] = file_sha256(csv)
    config_path.write_text(yaml.safe_dump(config))
    runner.run(csv, config_path, tmp_path / 'altered')
    for condition in ['latest_year', 'five_year']:
        original = json.loads((tmp_path / 'original' / condition / 'selection.json').read_text())
        altered = json.loads((tmp_path / 'altered' / condition / 'selection.json').read_text())
        for key in ['parameters', 'threshold', 'validation_average_precision', 'class_balance']:
            assert original[key] == altered[key]
