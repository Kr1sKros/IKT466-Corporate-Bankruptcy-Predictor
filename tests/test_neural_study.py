"""MLP/GRU studies, runner and the eight-study orchestrator share one protocol (#56)."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import pytest
import yaml

from models import neural_study as ns
from preprocessing import company_table, file_sha256, fit_preprocessor, save_splits, transform_histories
from sequences import create_histories
from validation import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / 'configs' / 'config.yaml').read_text())
FAST = {**CONFIG['training'], 'max_epochs': 4, 'early_stopping_patience': 2}


@pytest.fixture(scope='module')
def panel():
    """Failed companies have a falling X1; history lengths 1-7; 60/20/20 companies."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(1, 101):
        failed = i % 3 == 0
        n_years = int(rng.integers(1, 8))
        for k, year in enumerate(range(2010 - n_years + 1, 2011)):
            values = rng.lognormal(size=len(FEATURE_COLUMNS)) * 1e3
            values[0] = (-k if failed else k) + rng.normal(scale=0.1)
            rows.append({'company_name': f'C_{i}', 'status_label': 'failed' if failed else 'alive',
                         'year': year, **dict(zip(FEATURE_COLUMNS, values))})
    data = pd.DataFrame(rows)
    companies = company_table(data)
    number = companies.company_name.str[2:].astype(int)
    companies['split'] = np.where(number <= 60, 'train', np.where(number <= 80, 'validation', 'test'))
    return data, companies


def histories(panel, model_name, condition):
    data, companies = panel
    preprocessor = fit_preprocessor(data, companies)
    return [transform_histories(create_histories(data, condition=condition,
                                                 companies=companies.loc[companies.split.eq(p), 'company_name']),
                                preprocessor, model=model_name)
            for p in ('train', 'validation')]


def tune(model_name, train, validation, directory, name, trials=3):
    return ns.tune_condition(model_name, train, validation, directory=directory, study_name=name,
                             seed=42, sampler_seed=42, trials=trials,
                             space=CONFIG['models'][model_name]['search_space'], training=FAST)


@pytest.mark.parametrize('model_name,condition', [('mlp', 'five_year'), ('gru', 'five_year'), ('gru', 'latest_year')])
def test_seeded_isolated_studies_and_training_weight_bounds(panel, tmp_path, model_name, condition):
    train, validation = histories(panel, model_name, condition)
    runs = [tune(model_name, train, validation, tmp_path / n, f'study_{n}') for n in ('a', 'b')]
    (model, first), (_, second) = runs
    assert first['parameters'] == second['parameters'] and first['threshold'] == second['threshold']
    assert first['study_name'] != second['study_name']
    records = [json.loads((tmp_path / n / 'trials.json').read_text()) for n in ('a', 'b')]
    assert [r['parameters'] for r in records[0]] == [r['parameters'] for r in records[1]]
    assert [r['value'] for r in records[0]] == [r['value'] for r in records[1]]
    ratio = first['class_balance']['negative_to_positive_ratio']
    assert first['class_balance']['source'] == 'train' and ratio == 2.
    for record in records[0]:
        distribution = json.loads(record['distributions']['pos_weight'])['attributes']
        assert (distribution['low'], distribution['high'], distribution['log']) == (1., ratio, True)
        assert len(record['attributes']['training']['validation_ap_by_epoch']) <= FAST['max_epochs']
    # Only the selected checkpoint is kept, and it reproduces the selected validation AP.
    assert [p.name for p in (tmp_path / 'a').glob('trial_*.pt')] == [f"trial_{first['trial_number']}.pt"]
    assert first['validation_average_precision'] == max(r['value'] for r in records[0])
    assert first['threshold_source'] == 'validation'
    np.testing.assert_array_equal(ns.predict(model_name, model, validation),
                                  ns.predict(model_name, runs[1][0], validation))
    protocol = optuna.load_study(study_name='study_a', storage='sqlite:///' + str(tmp_path / 'a/study.sqlite3')).user_attrs['protocol']
    assert protocol['training']['max_epochs'] == 4 and protocol['pruning'] == 'none' and protocol['n_jobs'] == 1
    with pytest.raises(FileExistsError):
        tune(model_name, train, validation, tmp_path / 'a', 'again')


def test_failed_trials_are_recorded_and_never_selected(panel, tmp_path, monkeypatch):
    train, validation = histories(panel, 'mlp', 'latest_year')

    def fail(*args, **kwargs):
        raise FloatingPointError('injected non-finite loss')
    monkeypatch.setattr(ns, '_train', fail)
    with pytest.raises(RuntimeError, match='wholly failed'):
        tune('mlp', train, validation, tmp_path / 'failed', 'failed', trials=2)
    summary = json.loads((tmp_path / 'failed/study_summary.json').read_text())
    assert summary['attempted'] == summary['failed'] == 2 and summary['completed'] == 0
    assert not (tmp_path / 'failed/selection.json').exists()
    assert not list((tmp_path / 'failed').glob('*.pt'))
    assert all('injected' in t['attributes']['failure']
               for t in json.loads((tmp_path / 'failed/trials.json').read_text()))


def test_rejects_mixed_conditions_and_changed_training_rules(panel, tmp_path):
    train, _ = histories(panel, 'gru', 'five_year')
    _, validation = histories(panel, 'gru', 'latest_year')
    with pytest.raises(ValueError, match='one condition'):
        tune('gru', train, validation, tmp_path / 'mixed', 'mixed')
    with pytest.raises(ValueError, match='rules'):
        ns.check_training_rules({**FAST, 'optimizer': 'sgd'})
    with pytest.raises(ValueError, match='positive'):
        ns.check_training_rules({**FAST, 'max_epochs': 0})
    with pytest.raises(ValueError, match='Expected one of'):
        ns.tune_condition('lstm', train, train, directory=tmp_path / 'other', study_name='other',
                          seed=42, sampler_seed=42, trials=1, space={}, training=FAST)
    assert not (tmp_path / 'other').exists()


def prepare_run(panel, tmp_path, trials=1):
    data, companies = panel
    csv = tmp_path / 'synthetic.csv'
    data.to_csv(csv, index=False)
    config = yaml.safe_load((ROOT / 'configs/config.yaml').read_text())
    config['dataset']['sha256'] = file_sha256(csv)
    config['split']['directory'] = str(tmp_path / 'splits')
    config['split']['sha256'] = save_splits(companies, tmp_path / 'splits')
    config['study_plan']['tuning']['attempted_trials_per_study'] = trials
    config['training'] = FAST
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(config))
    preprocessor = fit_preprocessor(data, companies)
    preprocessor.record['provenance'] = {'dataset_sha256': file_sha256(csv),
                                         'split_sha256': config['split']['sha256']}
    preprocessor_path = tmp_path / 'shared_preprocessor.json'
    preprocessor.save(preprocessor_path)
    return csv, path, preprocessor_path


@pytest.mark.parametrize('model_name', ['mlp', 'gru'])
def test_runner_is_validation_only_and_reloads(panel, tmp_path, monkeypatch, model_name):
    data, companies = panel
    csv, config, preprocessor = prepare_run(panel, tmp_path)
    test_ids = set(companies.loc[companies.split.eq('test'), 'company_name'])
    original = ns.create_histories

    def guarded(data, *, condition, companies):
        assert not test_ids & set(companies), 'Runner must not construct test inputs'
        return original(data, condition=condition, companies=companies)
    monkeypatch.setattr(ns, 'create_histories', guarded)
    output = tmp_path / 'run'
    report = ns.run(model_name, csv, config, output, preprocessor)
    monkeypatch.undo()
    assert report['status'] == 'validation_only'
    assert report['test_evaluation'] == 'deferred_until_all_eight_studies_are_frozen'
    validation_ids = set(companies.loc[companies.split.eq('validation'), 'company_name'])
    shared = json.loads(preprocessor.read_text())
    for condition in ('latest_year', 'five_year'):
        directory = output / condition
        frame = pd.read_csv(directory / 'validation_predictions.csv')
        assert set(frame.company_name) == validation_ids
        assert list(frame.columns) == ['company_name', 'anchor_year', 'target', 'probability']
        assert json.loads((directory / 'preprocessor.json').read_text()) == shared
        loaded = ns.NeuralPipeline.load(directory)
        pd.testing.assert_frame_equal(loaded.predict(data, frame.company_name), frame,
                                      check_dtype=False)
        metadata = json.loads((directory / 'metadata.json').read_text())
        assert metadata['model'] == model_name and metadata['target_verified'] is False
        assert metadata['parameters']['threshold_source'] == 'validation'
        assert not list(directory.glob('test*'))
    assert set(json.loads((output / 'frozen_selections.json').read_text())) == {'latest_year', 'five_year'}
    with pytest.raises(FileExistsError):
        ns.run(model_name, csv, config, output, preprocessor)


def test_saved_pipeline_refuses_legacy_preprocessor_and_foreign_condition(panel, tmp_path):
    data, companies = panel
    train, validation = histories(panel, 'mlp', 'latest_year')
    model, selection = tune('mlp', train, validation, tmp_path / 'study', 'study', trials=1)
    pipeline = ns.NeuralPipeline('mlp', 'latest_year', fit_preprocessor(data, companies), model, selection['threshold'])
    pipeline.save(tmp_path / 'saved')
    record = json.loads((tmp_path / 'saved/pipeline.json').read_text())
    (tmp_path / 'saved/pipeline.json').write_text(json.dumps({**record, 'condition': 'five_year'}))
    with pytest.raises(ValueError):
        ns.NeuralPipeline.load(tmp_path / 'saved')
    (tmp_path / 'saved/pipeline.json').write_text(json.dumps(record))
    legacy = json.loads((tmp_path / 'saved/preprocessor.json').read_text())
    legacy.update(schema_version=2, strategy='standard',
                  model_strategies={'logistic_regression': 'standard', 'mlp': 'standard',
                                    'gru': 'standard', 'xgboost': 'identity'})
    (tmp_path / 'saved/preprocessor.json').write_text(json.dumps(legacy))
    with pytest.raises(ValueError, match='current preprocessor'):
        ns.NeuralPipeline.load(tmp_path / 'saved')


def test_all_eight_studies_share_one_protocol_and_freeze_together(panel, tmp_path):
    spec = importlib.util.spec_from_file_location('run_studies', ROOT / 'scripts/run_studies.py')
    orchestrator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(orchestrator)
    csv, config, preprocessor = prepare_run(panel, tmp_path, trials=2)
    output = tmp_path / 'all'
    orchestrator.main(['--csv', str(csv), '--config', str(config), '--output', str(output),
                       '--preprocessor', str(preprocessor), '--parallel', '8'])
    frozen = json.loads((output / 'frozen_studies.json').read_text())
    assert frozen['status'] == 'all_eight_frozen' and not frozen['missing']
    assert {(m, c) for m, s in frozen['studies'].items() for c in s} == {
        (m, c) for m in ('logistic_regression', 'xgboost', 'mlp', 'gru') for c in ('latest_year', 'five_year')}
    common = ['objective', 'attempted_trials', 'sampler_seed', 'n_jobs', 'pruning', 'training_seed']
    protocols, names = [], set()
    for model, conditions in frozen['studies'].items():
        for condition, study in conditions.items():
            assert study['attempted'] == study['completed'] == 2 and study['threshold_source'] == 'validation'
            names.add(study['study_name'])
            loaded = optuna.load_study(study_name=study['study_name'],
                                       storage='sqlite:///' + str(output / study['directory'] / 'study.sqlite3'))
            protocols.append({k: loaded.user_attrs['protocol'][k] for k in common})
    assert len(names) == 8 and all(p == protocols[0] for p in protocols)
    assert not list(output.rglob('test_*'))
    with pytest.raises(FileExistsError):
        orchestrator.main(['--csv', str(csv), '--config', str(config), '--output', str(output)])


def test_runner_can_study_one_condition(panel, tmp_path):
    from studies import select_conditions
    assert select_conditions(None) == ('latest_year', 'five_year')
    assert select_conditions(['five_year', 'latest_year']) == ('latest_year', 'five_year')
    for bad in ([], ['five_year', 'five_year'], ['all_history']):
        with pytest.raises(ValueError):
            select_conditions(bad)
    csv, config, preprocessor = prepare_run(panel, tmp_path)
    report = ns.run('mlp', csv, config, tmp_path / 'one', preprocessor, conditions=['five_year'])
    assert set(report['selections']) == {'five_year'}
    assert not (tmp_path / 'one' / 'latest_year').exists()
