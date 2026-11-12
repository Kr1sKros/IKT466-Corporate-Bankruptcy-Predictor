"""LR convergence, repeatable selection, reload and held-out evaluation gates."""
import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml

from models.logistic_regression import build_model, train_model, tune_condition, ConvergenceError
from preprocessing import company_table, save_splits, file_sha256
from validation import FEATURE_COLUMNS


def test_convergence_and_reload(tmp_path):
    rng = np.random.default_rng(5)
    x = rng.normal(size=(150, 5))
    y = (x[:, 0] + x[:, 1] > 0).astype(int)
    model = train_model(build_model(hyperparameters={'C': 0.1}), x, y)
    path = tmp_path / 'lr.joblib'
    joblib.dump(model, path)
    np.testing.assert_array_equal(model.predict_proba(x), joblib.load(path).predict_proba(x))
    with pytest.raises(ConvergenceError):
        train_model(build_model(hyperparameters={'max_iter': 1, 'tol': 1e-12}), x, y)


def test_independent_seeded_studies_and_weights(tmp_path):
    rng = np.random.default_rng(2)
    x = pd.DataFrame(rng.normal(size=(120, 3)))
    y = np.array([0] * 80 + [1] * 40)
    selections = []
    for name in ('a', 'b'):
        _, selected = tune_condition(x.iloc[:100], y[:100], x.iloc[::2], y[::2],
            directory=tmp_path / name, study_name=name, seed=42, sampler_seed=42, trials=3,
            search={'C_low': .0001, 'C_high': 10000})
        selections.append(selected)
    assert selections[0]['parameters'] == selections[1]['parameters']
    assert selections[0]['threshold'] == selections[1]['threshold']
    assert selections[0]['study_name'] != selections[1]['study_name']
    records = json.loads((tmp_path / 'a/trials.json').read_text())
    for t in records:
        weight = t['parameters']['positive_class_weight']
        assert 1 <= weight <= 4
        assert t['attributes']['effective_training_class_weights'] == {'0': 1., '1': weight}
        distribution = json.loads(t['distributions']['positive_class_weight'])['attributes']
        assert distribution == {'low': 1., 'high': 4., 'log': True, 'step': None}
        estimator = joblib.load(tmp_path / 'a' / f"trial_{t['number']}.joblib")
        assert estimator.class_weight == {0: 1., 1: weight}
    assert len(records) == 3


def test_all_failed_study_has_no_selection(tmp_path, monkeypatch):
    import models.logistic_regression as lr
    def fail(*args):
        raise ConvergenceError('deliberate test failure')
    monkeypatch.setattr(lr, 'train_model', fail)
    with pytest.raises(RuntimeError, match='wholly failed'):
        tune_condition(np.ones((4, 2)), [0, 0, 1, 1], np.ones((2, 2)), [0, 1],
            directory=tmp_path / 'failed', study_name='failed', seed=42, sampler_seed=42,
            trials=2, search={'C_low': .01, 'C_high': 1})
    summary = json.loads((tmp_path / 'failed/study_summary.json').read_text())
    assert summary['failed'] == summary['attempted'] == 2
    assert not (tmp_path / 'failed/selection.json').exists()


def test_runner_is_validation_only_and_loads_shared_preprocessor(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('train_logistic', root / 'scripts/train_logistic.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    rng = np.random.default_rng(7)
    rows = []
    for i in range(60):
        for year in range(2010, 2011 + i % 7):
            row = {'company_name': f'C_{i+1}', 'year': year, 'status_label': 'failed' if i % 3 == 0 else 'alive'}
            row.update(dict(zip(FEATURE_COLUMNS, rng.normal(size=18) + (i % 3 == 0))))
            rows.append(row)
    panel = pd.DataFrame(rows)
    csv = tmp_path / 'data.csv'
    panel.to_csv(csv, index=False)
    companies = company_table(panel)
    companies['split'] = ['train'] * 36 + ['validation'] * 12 + ['test'] * 12
    split_dir = tmp_path / 'splits'
    hashes = save_splits(companies, split_dir)
    config = yaml.safe_load((root / 'configs/config.yaml').read_text(encoding='utf-8'))
    config['dataset']['sha256'] = file_sha256(csv)
    config['split']['directory'] = str(split_dir)
    config['split']['sha256'] = hashes
    config['study_plan']['tuning']['attempted_trials_per_study'] = 2
    config_path = tmp_path / 'config.yaml'
    config_path.write_text(yaml.safe_dump(config))
    from preprocessing import fit_preprocessor
    shared = fit_preprocessor(panel, companies)
    shared.record['provenance'] = {'dataset_sha256': file_sha256(csv), 'split_sha256': hashes}
    shared_path = tmp_path / 'preprocessor.json'
    shared.save(shared_path)
    output = tmp_path / 'run'
    original_predict = runner.LogisticPipeline.predict
    test_ids = set(companies.loc[companies.split.eq('test'), 'company_name'])
    def guarded_predict(self, data, ids):
        assert not test_ids & set(ids), 'LR must never predict on test companies'
        assert self.preprocessor.record == shared.record
        return original_predict(self, data, ids)
    monkeypatch.setattr(runner.LogisticPipeline, 'predict', guarded_predict)
    original_histories = runner.create_histories
    def guarded_histories(data, *, condition, companies):
        assert not test_ids & set(companies), 'Runner must not construct test inputs'
        return original_histories(data, condition=condition, companies=companies)
    monkeypatch.setattr(runner, 'create_histories', guarded_histories)
    result = runner.run(csv, config_path, output, shared_path)
    assert result['status'] == 'validation_only'
    tables = [pd.read_csv(output / c / 'validation_predictions.csv') for c in ('latest_year', 'five_year')]
    pd.testing.assert_frame_equal(tables[0].iloc[:, :3], tables[1].iloc[:, :3])
    assert not list(output.rglob('test*'))
    assert not (output / 'comparison.json').exists()
    assert not (output / 'manifest.json').exists()
    with pytest.raises(FileExistsError):
        runner.run(csv, config_path, output, shared_path)
    shared.record['provenance']['dataset_sha256'] = 'wrong'
    shared.save(shared_path)
    with pytest.raises(ValueError, match='provenance'):
        runner.run(csv, config_path, tmp_path / 'wrong', shared_path)
