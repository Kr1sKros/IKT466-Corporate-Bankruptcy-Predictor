"""Shared Optuna study contract for all eight studies (#56).

Every model x input condition gets its own study through these helpers, so LR,
XGBoost, MLP and GRU are tuned the same way:

- a new output directory per study (an existing one is never resumed or overwritten);
- seeded ``TPESampler``, no pruning, ``n_jobs=1``, maximise validation AP;
- the same attempted-trial budget, stored with the protocol in the study;
- every trial (state, parameters, distributions, value, timing, failure) written to
  ``trials.json`` after each trial, and a ``study_summary.json`` with attempted /
  completed / failed counts and elapsed time, also when the study stops early;
- an incomplete or wholly failed study raises: there is no fallback to defaults and
  no selection, so no test evaluation can follow.

The model modules supply only the objective (search space + training of one trial)
and what to save for the selected trial.
"""
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import optuna

OBJECTIVE = 'validation_average_precision'
THRESHOLD_RULE = 'max_f1_highest_threshold_on_tie'


def write_json(path, value) -> None:
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n',
                          encoding='utf-8')


def check_budget(trials) -> int:
    if isinstance(trials, bool) or not isinstance(trials, int) or trials < 1:
        raise ValueError('A positive attempted-trial budget is required')
    return trials


def create_study(directory, *, study_name: str, sampler_seed: int, trials: int,
                 protocol: dict[str, Any]) -> optuna.Study:
    """New, isolated study in a new directory, with the common protocol recorded."""
    check_budget(trials)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    study = optuna.create_study(
        study_name=study_name, direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=sampler_seed), pruner=optuna.pruners.NopPruner(),
        storage='sqlite:///' + (directory / 'study.sqlite3').resolve().as_posix())
    study.set_user_attr('protocol', {'objective': OBJECTIVE, 'attempted_trials': trials,
                                     'sampler_seed': sampler_seed, 'n_jobs': 1, 'pruning': 'none',
                                     **protocol})
    return study


def trial_records(study: optuna.Study) -> list[dict[str, Any]]:
    return [{
        'number': t.number, 'state': t.state.name, 'value': t.value,
        'parameters': t.params, 'attributes': t.user_attrs,
        'distributions': {k: optuna.distributions.distribution_to_json(v)
                          for k, v in t.distributions.items()},
        'duration_seconds': t.duration.total_seconds() if t.duration else None,
    } for t in study.trials]


def run_study(study: optuna.Study, objective: Callable[[optuna.Trial], float], *, directory,
              trials: int, catch: tuple[type[BaseException], ...]) -> list[dict[str, Any]]:
    """Run the full budget sequentially; record everything; fail unless complete.

    Exceptions in ``catch`` mark that trial FAILED and the study continues; any
    other exception stops the study (still recorded). Objectives should store the
    error text in the trial attribute ``failure`` before re-raising.
    """
    directory = Path(directory)
    started = perf_counter()

    def save_records():
        records = trial_records(study)
        write_json(directory / 'trials.json', records)
        return records

    try:
        study.optimize(objective, n_trials=trials, n_jobs=1, catch=catch,
                       callbacks=[lambda _s, _t: save_records()])
    finally:
        records = save_records()
        complete = sum(t['state'] == 'COMPLETE' for t in records)
        write_json(directory / 'study_summary.json', {
            'study_name': study.study_name, 'budget': trials, 'attempted': len(records),
            'completed': complete, 'failed': sum(t['state'] == 'FAIL' for t in records),
            'budget_complete': len(records) == trials, 'elapsed_seconds': perf_counter() - started})
    if len(records) != trials or not complete:
        raise RuntimeError('Incomplete or wholly failed study; no test evaluation permitted')
    return records


def select_conditions(conditions=None) -> tuple[str, ...]:
    """Conditions a runner should study (default both), in the fixed order.

    Running one condition per process lets all eight studies run side by side;
    each study is unchanged by this, it is still seeded and single-threaded.
    """
    from sequences import CONDITIONS
    if conditions is None:
        return CONDITIONS
    chosen = set(conditions)
    if not chosen or not chosen <= set(CONDITIONS) or len(chosen) != len(list(conditions)):
        raise ValueError(f'Conditions must be distinct values from {CONDITIONS}')
    return tuple(c for c in CONDITIONS if c in chosen)


def add_conditions_argument(parser) -> None:
    from sequences import CONDITIONS
    parser.add_argument('--conditions', nargs='+', choices=CONDITIONS,
                        help='Run only these conditions (default: both)')


def record_failure(trial: optuna.Trial, error: BaseException) -> None:
    trial.set_user_attr('failure', f'{type(error).__name__}: {error}')
