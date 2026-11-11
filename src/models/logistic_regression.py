"""Fixed L2 Logistic Regression and validation-only Optuna selection (#53)."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter
import warnings

import joblib
import numpy as np
import optuna
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from threadpoolctl import threadpool_limits

from metrics import compute_metrics, select_validation_threshold
from preprocessing import transform_histories
from reproducibility import estimator_seed_parameters
from sequences import create_histories, flatten_histories, tabular_columns


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
    if isinstance(trials, bool) or not isinstance(trials, int) or trials < 1:
        raise ValueError('A positive attempted-trial budget is required')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    start = perf_counter()
    study = optuna.create_study(study_name=study_name, direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=sampler_seed), pruner=optuna.pruners.NopPruner(),
        storage='sqlite:///' + (directory / 'study.sqlite3').resolve().as_posix())
    study.set_user_attr('protocol', {'objective': 'validation_average_precision', 'attempted_trials': trials,
                                   'sampler_seed': sampler_seed, 'training_seed': seed,
                                   'n_jobs': 1, 'pruning': 'none', 'search': search})

    def objective(trial):
        params = {'C': trial.suggest_float('C', search['C_low'], search['C_high'], log=True),
                  'class_weight': trial.suggest_categorical('class_weight', search['class_weight'])}
        model = build_model(seed=seed, hyperparameters={**params, 'max_iter': max_iter, 'tol': tol})
        weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=np.asarray(y_train)) if params['class_weight'] == 'balanced' else [1., 1.]
        trial.set_user_attr('effective_training_class_weights', {str(i): float(w) for i, w in enumerate(weights)})
        try:
            train_model(model, x_train, y_train)
            score = compute_metrics(y_validation, model.predict_proba(x_validation)[:, 1])['average_precision']
            trial.set_user_attr('n_iter', model.n_iter_.tolist())
            joblib.dump(model, directory / f'trial_{trial.number}.joblib')
            return score
        except Exception as error:
            trial.set_user_attr('failure', f'{type(error).__name__}: {error}')
            raise

    def save_records():
        records = [{'number': t.number, 'state': t.state.name, 'value': t.value,
                    'parameters': t.params, 'distributions': {k: optuna.distributions.distribution_to_json(v) for k, v in t.distributions.items()},
                    'duration_seconds': t.duration.total_seconds() if t.duration else None,
                    'attributes': t.user_attrs} for t in study.trials]
        write_json(directory / 'trials.json', records)
        return records

    try:
        with threadpool_limits(limits=1):
            study.optimize(objective, n_trials=trials, n_jobs=1, catch=(ConvergenceError,),
                           callbacks=[lambda _s, _t: save_records()])
    finally:
        records = save_records()
        complete = sum(t['state'] == 'COMPLETE' for t in records)
        write_json(directory / 'study_summary.json', {
            'study_name': study_name, 'attempted': len(records), 'completed': complete,
            'failed': sum(t['state'] == 'FAIL' for t in records), 'budget': trials,
            'budget_complete': len(records) == trials, 'elapsed_seconds': perf_counter() - start})
    if len(records) != trials or not complete:
        raise RuntimeError('Study incomplete or no converged trials; no test evaluation permitted')
    best = study.best_trial
    model = joblib.load(directory / f'trial_{best.number}.joblib')
    validation_scores = model.predict_proba(x_validation)[:, 1]
    threshold = select_validation_threshold(y_validation, validation_scores)
    selection = {'study_name': study_name, 'trial_number': best.number, 'parameters': best.params,
                 'effective_training_class_weights': best.user_attrs['effective_training_class_weights'],
                 'validation_average_precision': best.value, 'threshold': threshold,
                 'threshold_source': 'validation', 'threshold_rule': 'max_f1_highest_threshold_on_tie',
                 'n_iter': model.n_iter_.tolist(), 'max_iter': max_iter, 'tol': tol,
                 'solver': 'lbfgs', 'regularization': 'L2'}
    write_json(directory / 'selection.json', selection)
    return model, selection
