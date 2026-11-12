"""Validation-only Optuna studies and runner for the MLP and GRU (#56).

Same contract as the LR (#53) and XGBoost (#54) studies, through ``studies.py``:
seeded TPE, validation AP, the common trial budget, n_jobs=1, no pruning, every
trial recorded, no selection from an incomplete study. Per trial, Optuna picks
the settings (``suggest_params``) and ``train_model`` learns the network weights
with the shared rules in ``configs/config.yaml`` ``training`` (Adam, early
stopping on validation AP, best epoch restored). ``pos_weight`` bounds come from
training labels only; validation is never weighted or resampled.

The threshold is chosen on validation and the selected checkpoint is frozen.
The runner builds only training and validation inputs: test evaluation waits
until all eight studies are frozen (#60/#62).

To keep disk use bounded, only the best checkpoint so far is kept on disk
(``trial_<n>.pt``); ties keep the earlier trial, as Optuna's ``best_trial`` does.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score

from data_loader import load_data
from leakage import check_partitions, check_preprocessor, check_study_plan, class_weight_bounds
from metrics import select_validation_threshold
from models import gru_model, mlp_model
from preprocessing import (MODEL_STRATEGIES, FittedPreprocessor, file_sha256, load_splits,
                           split_paths, transform_histories)
from reproducibility import configure_reproducibility, file_provenance, write_run_metadata
from sequences import CONDITIONS, create_histories, flatten_histories, step_columns, tabular_columns
from studies import (THRESHOLD_RULE, create_study, record_failure, run_study, select_conditions,
                     write_json)

MODULES = {'mlp': mlp_model, 'gru': gru_model}
TRAINING_RULES = {'optimizer': 'adam', 'selection_metric': 'validation_average_precision',
                  'restore_best_checkpoint': True}


def _module(model_name: str):
    if model_name not in MODULES:
        raise ValueError(f'Expected one of {tuple(MODULES)}, got {model_name!r}')
    return MODULES[model_name]


def check_training_rules(training: dict[str, Any]) -> dict[str, int]:
    """The shared neural rules; returns the epoch cap and patience."""
    wrong = {k: training.get(k) for k, v in TRAINING_RULES.items() if training.get(k) != v}
    if wrong:
        raise ValueError(f'Neural training rules differ from the protocol: {wrong}')
    rules = {'max_epochs': training.get('max_epochs'), 'patience': training.get('early_stopping_patience')}
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in rules.values()):
        raise ValueError('max_epochs and early_stopping_patience must be positive integers')
    return rules


def input_columns(model_name: str, condition: str) -> list[str]:
    return tabular_columns(condition) if model_name == 'mlp' else step_columns(condition)


def predict(model_name: str, model, histories) -> np.ndarray:
    """Probabilities in the row order of ``histories`` (already transformed)."""
    if model_name == 'mlp':
        return mlp_model.predict_proba(model, flatten_histories(histories)[0])
    return _module(model_name).predict_proba(model, histories)


def _train(model_name, model, train, validation, params, rules, seed):
    settings = {k: params[k] for k in ('learning_rate', 'weight_decay', 'pos_weight', 'batch_size')}
    settings.update(max_epochs=rules['max_epochs'], patience=rules['patience'], seed=seed)
    if model_name == 'mlp':
        (x_train, m_train), (x_val, m_val) = flatten_histories(train), flatten_histories(validation)
        return mlp_model.train_model(model, x_train, m_train.target, x_val, m_val.target, **settings)
    return gru_model.train_model(model, train, validation, **settings)


@dataclass
class NeuralPipeline:
    """Raw panel -> shared preprocessing -> MLP/GRU probabilities. No fitting at inference."""
    model_name: str
    condition: str
    preprocessor: FittedPreprocessor
    model: Any
    threshold: float

    def predict(self, data, companies) -> pd.DataFrame:
        histories = transform_histories(
            create_histories(data, condition=self.condition, companies=companies),
            self.preprocessor, model=self.model_name)
        result = histories.metadata[['company_name', 'anchor_year', 'target']].copy()
        result['probability'] = predict(self.model_name, self.model, histories)
        return result

    def save(self, directory) -> None:
        directory = Path(directory)
        _module(self.model_name).save_checkpoint(
            self.model, directory / 'model.pt', extra={'model': self.model_name, 'threshold': self.threshold})
        self.preprocessor.save(directory / 'preprocessor.json')
        write_json(directory / 'pipeline.json', {
            'schema_version': 1, 'model': self.model_name, 'condition': self.condition,
            'threshold': self.threshold, 'input_columns': input_columns(self.model_name, self.condition),
            'financial_transform': MODEL_STRATEGIES[self.model_name]})

    @classmethod
    def load(cls, directory) -> 'NeuralPipeline':
        directory = Path(directory)
        record = json.loads((directory / 'pipeline.json').read_text(encoding='utf-8'))
        name, condition = record['model'], record['condition']
        if (record['schema_version'] != 1 or name not in MODULES or condition not in CONDITIONS
                or record['input_columns'] != input_columns(name, condition)
                or record['financial_transform'] != MODEL_STRATEGIES[name]
                or not 0 <= record['threshold'] <= 1):
            raise ValueError('Invalid saved MLP/GRU pipeline contract')
        model, extra = _module(name).load_checkpoint(directory / 'model.pt', condition=condition)
        if extra != {'model': name, 'threshold': record['threshold']}:
            raise ValueError('Checkpoint and pipeline record differ')
        preprocessor = FittedPreprocessor.load(directory / 'preprocessor.json')
        if preprocessor.is_legacy:
            raise ValueError('MLP/GRU need the current preprocessor (schema 3, #87)')
        return cls(name, condition, preprocessor, model, record['threshold'])


def tune_condition(model_name, train, validation, *, directory, study_name, seed, sampler_seed,
                   trials, space, training):
    """One study on transformed train/validation histories. Test data cannot enter."""
    module = _module(model_name)
    rules = check_training_rules(training)
    if train.condition != validation.condition:
        raise ValueError('Training and validation histories must share one condition')
    condition = train.condition
    for name, targets in (('Training', train.targets), ('Validation', validation.targets)):
        if set(np.unique(targets)) != {0, 1}:
            raise ValueError(f'{name} labels must contain both classes')
    balance = class_weight_bounds(pd.DataFrame({'target': np.asarray(train.targets), 'split': 'train'}))
    ratio = balance['negative_to_positive_ratio']
    if ratio < 1:
        raise ValueError('Training negative/positive ratio must be at least 1')
    directory = Path(directory)
    study = create_study(directory, study_name=study_name, sampler_seed=sampler_seed, trials=trials,
                         protocol={'model': model_name, 'condition': condition, 'training_seed': seed,
                                   'search_space': space, 'training': {**TRAINING_RULES, **rules},
                                   'class_balance': balance, 'pos_weight_bounds': [1., ratio],
                                   'device': 'cpu'})
    kept = {'number': None, 'value': -np.inf}

    def objective(trial):
        try:
            params = module.suggest_params(trial, space, max_pos_weight=ratio)
            configure_reproducibility(seed)
            model = module.build_model(params, condition)
            result = _train(model_name, model, train, validation, params, rules, seed)
            trial.set_user_attr('training', {
                'best_epoch': result.best_epoch, 'epochs_run': result.epochs_run,
                'stopped_early': result.stopped_early, 'seconds': result.seconds,
                'validation_ap_by_epoch': [h['validation_ap'] for h in result.history],
                'train_loss_by_epoch': [h['train_loss'] for h in result.history]})
            if result.best_validation_ap > kept['value']:
                module.save_checkpoint(model, directory / f'trial_{trial.number}.pt',
                                       extra={'parameters': params})
                if kept['number'] is not None:
                    (directory / f"trial_{kept['number']}.pt").unlink()
                kept.update(number=trial.number, value=result.best_validation_ap)
            return result.best_validation_ap
        except Exception as error:
            record_failure(trial, error)
            raise

    run_study(study, objective, directory=directory, trials=trials, catch=(FloatingPointError,))
    best = study.best_trial
    if best.number != kept['number']:
        raise RuntimeError('Saved checkpoint is not the selected trial')
    model, extra = module.load_checkpoint(directory / f'trial_{best.number}.pt', condition=condition)
    probabilities = predict(model_name, model, validation)
    if not np.isclose(average_precision_score(validation.targets, probabilities), best.value):
        raise RuntimeError('Reloaded checkpoint does not reproduce the selected validation AP')
    threshold = select_validation_threshold(validation.targets, probabilities)
    selection = {
        'study_name': study_name, 'model': model_name, 'condition': condition,
        'trial_number': best.number, 'parameters': extra['parameters'],
        'training': best.user_attrs['training'], 'class_balance': balance,
        'validation_average_precision': best.value, 'threshold': threshold,
        'threshold_source': 'validation', 'threshold_rule': THRESHOLD_RULE,
        'training_seed': seed, 'sampler_seed': sampler_seed, 'device': 'cpu'}
    write_json(directory / 'selection.json', selection)
    return model, selection


def run(model_name, csv, config_path, output, preprocessor_path=None, conditions=None):
    """One neural model's studies (default both conditions), validation only; test is deferred."""
    _module(model_name)
    conditions = select_conditions(conditions)
    csv, config_path, output = Path(csv).resolve(), Path(config_path).resolve(), Path(output).resolve()
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    check_study_plan(config)
    if config['preprocessing']['model_scaling'] != MODEL_STRATEGIES:
        raise ValueError('Use the fixed preprocessing policy for every model')
    if config['evaluation'].get('threshold_rule') != THRESHOLD_RULE:
        raise ValueError('Declare the validation F1 threshold rule')
    space = config['models'][model_name]['search_space']
    check_training_rules(config['training'])
    if file_sha256(csv) != config['dataset']['sha256']:
        raise ValueError('Dataset checksum differs from the audited CSV')
    split_directory = Path(config['split']['directory'])
    if not split_directory.is_absolute():
        split_directory = config_path.parent.parent / split_directory
    if output.exists():
        raise FileExistsError('Choose a new output directory; existing studies are never overwritten')
    companies = load_splits(split_directory, expected_sha256=config['split']['sha256'])
    data = load_data(csv)
    check_partitions(companies, data)
    preprocessor_path = (Path(preprocessor_path).resolve() if preprocessor_path is not None
                         else config_path.parent.parent / 'results/models/preprocessor.json')
    preprocessor = FittedPreprocessor.load(preprocessor_path)
    expected_provenance = {'dataset_sha256': file_sha256(csv), 'split_sha256': config['split']['sha256']}
    if preprocessor.record.get('provenance') != expected_provenance:
        raise ValueError('Shared preprocessor provenance differs from dataset/splits')
    check_preprocessor(preprocessor, data, companies)
    seed_settings = configure_reproducibility(config['random_seed'])
    output.mkdir(parents=True)
    write_json(output / 'run_config.json', config)
    packages = {p: version(p) for p in ('numpy', 'pandas', 'scikit-learn', 'torch', 'optuna')}
    write_json(output / 'run_context.json', {
        'model': model_name, 'dataset': file_provenance(csv), 'configuration': file_provenance(config_path),
        'shared_preprocessor': file_provenance(preprocessor_path),
        'splits': {k: file_provenance(v) for k, v in split_paths(split_directory).items()},
        'packages': packages, 'seed_settings': seed_settings, 'device': 'cpu', 'n_jobs': 1,
        'target_verified': False})
    tuning = config['study_plan']['tuning']
    started = perf_counter()
    selections = {}
    for condition in conditions:
        histories = {}
        for partition in ('train', 'validation'):
            ids = companies.loc[companies['split'].eq(partition), 'company_name']
            histories[partition] = transform_histories(
                create_histories(data, condition=condition, companies=ids), preprocessor, model=model_name)
        directory = output / condition
        model, selection = tune_condition(
            model_name, histories['train'], histories['validation'], directory=directory,
            study_name=f'{model_name}__{condition}__{output.name}', seed=config['random_seed'],
            sampler_seed=tuning['sampler_seed'], trials=tuning['attempted_trials_per_study'],
            space=space, training=config['training'])
        NeuralPipeline(model_name, condition, preprocessor, model, selection['threshold']).save(directory)
        validation_ids = histories['validation'].metadata.company_name
        predictions = NeuralPipeline.load(directory).predict(data, validation_ids)
        np.testing.assert_array_equal(predictions.probability,
                                      predict(model_name, model, histories['validation']))
        predictions.to_csv(directory / 'validation_predictions.csv', index=False)
        selection['input_columns'] = input_columns(model_name, condition)
        write_json(directory / 'selection.json', selection)
        metadata = write_run_metadata(
            directory / 'metadata.json', model=model_name, parameters=selection,
            seed_settings=seed_settings, dataset=csv, splits=split_paths(split_directory),
            configuration=config_path, checkpoint=directory / 'model.pt',
            repository=Path(__file__).resolve().parents[2],
            dataset_version=str(config['dataset']['kaggle_version']))
        metadata['software']['packages']['optuna'] = version('optuna')
        metadata['preprocessing'] = file_provenance(directory / 'preprocessor.json')
        metadata['pipeline'] = file_provenance(directory / 'pipeline.json')
        metadata['target_verified'] = False
        metadata['hardware']['training_device'] = 'cpu'
        write_json(directory / 'metadata.json', metadata)
        selections[condition] = selection
    write_json(output / 'frozen_selections.json', selections)
    summary = {'status': 'validation_only', 'model': model_name, 'selections': selections,
               'elapsed_seconds': perf_counter() - started,
               'test_evaluation': 'deferred_until_all_eight_studies_are_frozen'}
    write_json(output / 'validation_summary.json', summary)
    return summary
