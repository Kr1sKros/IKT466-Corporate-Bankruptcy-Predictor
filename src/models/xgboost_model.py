"""CPU XGBoost and validation-only Optuna studies (#54)."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from xgboost.core import XGBoostError

from metrics import compute_metrics, select_validation_threshold
from leakage import class_weight_bounds
from preprocessing import FittedPreprocessor, transform_histories
from reproducibility import estimator_seed_parameters
from sequences import CONDITIONS, create_histories, flatten_histories, tabular_columns
from studies import THRESHOLD_RULE, check_budget, create_study, record_failure, run_study

PARAMETERS = {'n_estimators', 'max_depth', 'learning_rate', 'min_child_weight',
              'subsample', 'colsample_bytree', 'reg_lambda', 'reg_alpha'}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')


def validate_features(features):
    if not isinstance(features, pd.DataFrame) or features.empty:
        raise ValueError('A nonempty DataFrame with the exact input schema is required')
    if not any(list(features.columns) == tabular_columns(c) for c in CONDITIONS):
        raise ValueError('Features differ from the latest_year/five_year schema')
    if any(not pd.api.types.is_numeric_dtype(d) for d in features.dtypes):
        raise ValueError('Features must be numeric')
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError('Real financial values must be finite; padding is already explicit')


def validate_targets(targets, size):
    values = np.asarray(targets)
    if (values.shape != (size,) or values.dtype.kind not in 'biuf'
            or set(np.unique(values)) != {0, 1}):
        raise ValueError('Aligned numeric labels containing both classes 0 and 1 are required')
    return values.astype(int)


def build_model(*, seed=42, hyperparameters=None):
    """Fixed CPU histogram learner; thread/seed/objective settings cannot be tuned."""
    params = dict(hyperparameters or {})
    if set(params) - (PARAMETERS | {'scale_pos_weight'}):
        raise ValueError('Only declared tree settings and scale_pos_weight are configurable')
    if 'scale_pos_weight' in params and (not np.isfinite(params['scale_pos_weight'])
                                       or params['scale_pos_weight'] <= 0):
        raise ValueError('scale_pos_weight must be finite and positive')
    return XGBClassifier(objective='binary:logistic', tree_method='hist', device='cpu',
                         eval_metric='logloss', verbosity=0,
                         **estimator_seed_parameters('xgboost', seed), **params)


def train_model(model, features, targets):
    """Fit training rows only; shared sklearn AP, not XGBoost aucpr, selects trials."""
    validate_features(features)
    model.fit(features, validate_targets(targets, len(features)))
    return model


@dataclass
class XGBoostPipeline:
    condition: str
    preprocessor: FittedPreprocessor
    model: XGBClassifier
    threshold: float

    def predict(self, data, companies):
        histories = create_histories(data, condition=self.condition, companies=companies)
        features, metadata = flatten_histories(
            transform_histories(histories, self.preprocessor, model='xgboost'))
        if list(features.columns) != tabular_columns(self.condition):
            raise ValueError('Prediction schema differs')
        validate_features(features)
        result = metadata[['company_name', 'anchor_year', 'target']].copy()
        result['probability'] = self.model.predict_proba(features)[:, 1]
        if not np.isfinite(result.probability).all() or not result.probability.between(0, 1).all():
            raise ValueError('Invalid prediction probabilities')
        return result

    def save(self, directory):
        directory = Path(directory)
        self.model.save_model(directory / 'model.ubj')
        self.preprocessor.save(directory / 'preprocessor.json')
        write_json(directory / 'pipeline.json', {
            'schema_version': 1, 'model': 'xgboost', 'condition': self.condition,
            'threshold': self.threshold, 'feature_columns': tabular_columns(self.condition),
            'financial_transform': 'identity'})

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        record = json.loads((directory / 'pipeline.json').read_text())
        if (record['schema_version'] != 1 or record['model'] != 'xgboost'
                or record['financial_transform'] != 'identity'
                or record['feature_columns'] != tabular_columns(record['condition'])
                or not 0 <= record['threshold'] <= 1):
            raise ValueError('Invalid saved XGBoost pipeline contract')
        model = build_model()
        model.load_model(directory / 'model.ubj')
        if list(model.feature_names_in_) != record['feature_columns']:
            raise ValueError('Model and saved input schema differ')
        return cls(record['condition'], FittedPreprocessor.load(directory / 'preprocessor.json'),
                   model, record['threshold'])


def validate_search(search):
    if set(search) != PARAMETERS:
        raise ValueError('Declare exactly the supported XGBoost search parameters')
    for name, spec in search.items():
        if set(spec) - {'low', 'high', 'log'} or not {'low', 'high'} <= set(spec):
            raise ValueError(f'Invalid search specification for {name}')
        low, high = spec['low'], spec['high']
        if (isinstance(low, bool) or isinstance(high, bool)
                or not np.isfinite([low, high]).all() or low > high
                or (spec.get('log', False) and low <= 0)):
            raise ValueError(f'Invalid bounds for {name}')
        if name in {'n_estimators', 'max_depth'}:
            if not all(isinstance(v, int) and v >= 1 for v in [low, high]):
                raise ValueError(f'{name} requires positive integer bounds')
        elif name in {'subsample', 'colsample_bytree', 'learning_rate'}:
            if not 0 < low <= high <= 1:
                raise ValueError(f'{name} bounds must be in (0, 1]')
        elif low < 0:
            raise ValueError(f'{name} cannot be negative')


def tune_condition(x_train, y_train, x_validation, y_validation, *, directory,
                   study_name, seed, sampler_seed, trials, search):
    """Only training/validation enter this API. Test data cannot enter selection."""
    validate_features(x_train)
    validate_features(x_validation)
    if list(x_train.columns) != list(x_validation.columns):
        raise ValueError('Training and validation schemas must match')
    y_train = validate_targets(y_train, len(x_train))
    y_validation = validate_targets(y_validation, len(x_validation))
    validate_search(search)
    check_budget(trials)
    balance = class_weight_bounds(pd.DataFrame({'target': y_train, 'split': 'train'}))
    ratio = balance['negative_to_positive_ratio']
    if ratio < 1:
        raise ValueError('Training negative/positive ratio must be at least 1')
    directory = Path(directory)
    study = create_study(directory, study_name=study_name, sampler_seed=sampler_seed, trials=trials,
                         protocol={'model': 'xgboost', 'training_seed': seed, 'search': search,
                                   'class_balance': balance, 'scale_pos_weight_bounds': [1., ratio]})

    def objective(trial):
        try:
            params = {}
            for name, spec in search.items():
                suggest = trial.suggest_int if name in {'n_estimators', 'max_depth'} else trial.suggest_float
                params[name] = suggest(name, spec['low'], spec['high'], log=spec.get('log', False))
            params['scale_pos_weight'] = trial.suggest_float('scale_pos_weight', 1., ratio, log=True)
            trial.set_user_attr('effective_parameters', params)
            model = train_model(build_model(seed=seed, hyperparameters=params), x_train, y_train)
            score = compute_metrics(y_validation, model.predict_proba(x_validation)[:, 1])['average_precision']
            model.save_model(directory / f'trial_{trial.number}.ubj')
            return score
        except Exception as error:
            record_failure(trial, error)
            raise

    run_study(study, objective, directory=directory, trials=trials, catch=(XGBoostError,))
    best = study.best_trial
    model = build_model(seed=seed)
    model.load_model(directory / f'trial_{best.number}.ubj')
    threshold = select_validation_threshold(y_validation, model.predict_proba(x_validation)[:, 1])
    selection = {
        'study_name': study_name, 'trial_number': best.number,
        'parameters': best.user_attrs['effective_parameters'], 'class_balance': balance,
        'validation_average_precision': best.value, 'threshold': threshold,
        'threshold_source': 'validation', 'threshold_rule': THRESHOLD_RULE,
        'tree_method': 'hist', 'device': 'cpu', 'n_jobs': 1,
        'training_seed': seed, 'sampler_seed': sampler_seed}
    write_json(directory / 'selection.json', selection)
    return model, selection
