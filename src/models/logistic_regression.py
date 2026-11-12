"""Fixed L2 Logistic Regression and validation-only Optuna selection (#53)."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import warnings

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from leakage import class_weight_bounds
import pandas as pd
from threadpoolctl import threadpool_limits

from metrics import compute_metrics, select_validation_threshold
from preprocessing import transform_histories
from reproducibility import estimator_seed_parameters
from sequences import create_histories, flatten_histories, tabular_columns
from studies import THRESHOLD_RULE, check_budget, create_study, record_failure, run_study


class ConvergenceError(RuntimeError):
    """A trial must not be selected if its solver failed to converge."""


def build_model(*, seed=42, hyperparameters=None):
    params = dict(hyperparameters or {})
    if set(params) - {'C', 'class_weight', 'max_iter', 'tol'}:
        raise ValueError('Only C, class_weight, max_iter and tol are configurable')
    return LogisticRegression(solver='lbfgs', **estimator_seed_parameters('logistic_regression', seed),
                              **{'max_iter': 5000, 'tol': 1e-4, **params})


def train_model(model, features, targets):
    """Fit only training rows. lbfgs uses L2 regularization by default."""
    if set(np.unique(targets)) != {0, 1}:
        raise ValueError('Training labels must contain both numeric classes 0 and 1')
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
        warnings.simplefilter('always', ConvergenceWarning)
        model.fit(features, targets)
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        raise ConvergenceError('LR solver did not converge within max_iter')
    return model


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')


@dataclass
class LogisticPipeline:
    """Reloadable raw-panel -> shared preprocessing -> LR prediction path.

    Load joblib artifacts only from trusted sources. No fit occurs at inference.
    """
    condition: str
    preprocessor: object
    model: object
    threshold: float

    def predict(self, data, companies):
        histories = create_histories(data, condition=self.condition, companies=companies)
        features, metadata = flatten_histories(transform_histories(histories, self.preprocessor, model='logistic_regression'))
        if list(features.columns) != tabular_columns(self.condition):
            raise ValueError('Prediction feature schema differs')
        result = metadata[['company_name', 'anchor_year', 'target']].copy()
        result['probability'] = self.model.predict_proba(features)[:, 1]
        return result


def tune_condition(x_train, y_train, x_validation, y_validation, *, directory,
                   study_name, seed, sampler_seed, trials, search, max_iter=5000, tol=1e-4):
    """Only train/validation inputs accepted: test data cannot enter selection."""
    if set(np.unique(y_validation)) != {0, 1}:
        raise ValueError('Validation labels must contain both classes')
    check_budget(trials)
    if set(np.unique(y_train)) != {0, 1}:
        raise ValueError('Training labels must contain both classes')
    bounds = class_weight_bounds(pd.DataFrame({'target': np.asarray(y_train), 'split': 'train'}))
    max_weight = bounds['negative_to_positive_ratio']
    if max_weight < 1:
        raise ValueError('Training negative/positive ratio must be at least 1')
    directory = Path(directory)
    study = create_study(directory, study_name=study_name, sampler_seed=sampler_seed, trials=trials,
                         protocol={'model': 'logistic_regression', 'training_seed': seed,
                                   'search': search, 'class_weight_bounds': bounds})

    def objective(trial):
        params = {'C': trial.suggest_float('C', search['C_low'], search['C_high'], log=True),
                  'positive_class_weight': trial.suggest_float('positive_class_weight', 1.0, max_weight, log=True)}
        weights = [1., params['positive_class_weight']]
        model = build_model(seed=seed, hyperparameters={'C': params['C'], 'class_weight': {0: 1., 1: weights[1]},
                                                      'max_iter': max_iter, 'tol': tol})
        trial.set_user_attr('effective_training_class_weights', {str(i): float(w) for i, w in enumerate(weights)})
        try:
            train_model(model, x_train, y_train)
            score = compute_metrics(y_validation, model.predict_proba(x_validation)[:, 1])['average_precision']
            trial.set_user_attr('n_iter', model.n_iter_.tolist())
            joblib.dump(model, directory / f'trial_{trial.number}.joblib')
            return score
        except Exception as error:
            record_failure(trial, error)
            raise

    with threadpool_limits(limits=1):
        run_study(study, objective, directory=directory, trials=trials, catch=(ConvergenceError,))
    best = study.best_trial
    model = joblib.load(directory / f'trial_{best.number}.joblib')
    validation_scores = model.predict_proba(x_validation)[:, 1]
    threshold = select_validation_threshold(y_validation, validation_scores)
    selection = {'study_name': study_name, 'trial_number': best.number, 'parameters': best.params,
                 'effective_training_class_weights': best.user_attrs['effective_training_class_weights'],
                 'validation_average_precision': best.value, 'threshold': threshold,
                 'threshold_source': 'validation', 'threshold_rule': THRESHOLD_RULE,
                 'n_iter': model.n_iter_.tolist(), 'max_iter': max_iter, 'tol': tol,
                 'solver': 'lbfgs', 'regularization': 'L2', 'class_weight_bounds': bounds}
    write_json(directory / 'selection.json', selection)
    return model, selection
