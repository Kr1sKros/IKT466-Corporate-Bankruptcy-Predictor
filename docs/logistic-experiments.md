# Logistic Regression experiments (#53)

Run from the repository root with project dependencies installed:

```powershell
python scripts/train_logistic.py --csv PATH_TO_AUDITED_CSV --output results/models/lr_run_01
```

The output directory must be new. Existing studies are never silently resumed or
replaced. The command loads the shared persisted train/validation/test split and
checks the approved split and CSV hashes, company disjointness and anchors.
It reuses shared data loading, input construction, per-condition preprocessing,
metrics, seed settings and provenance helpers.

## Predeclared selection

Two independent Optuna studies: logistic_regression x latest_year/five_year.
Each uses sequential seeded TPE (42), no pruning, and the shared budget of
20 attempted trials. Both use the same declared search: log-uniform C in
[0.0001, 10000] and class_weight in [None, balanced]. lbfgs fits L2-regularized
coefficients inside each trial. The fixed solver cap is 5,000 iterations and
its tolerance is 0.0001; these are convergence controls, not search parameters.
BLAS is restricted to one thread during fitting and study evaluation.

Class weights for balanced trials are computed from training labels only;
validation and test scoring are unweighted and unresampled. Preprocessing is
fixed, outside Optuna: latest_year uses training anchors; five_year uses the
up to five training records per company. Shared preprocessing preserves offsets
and presence and excludes padding. Models receive 18/100 columns respectively.

The objective is validation Average Precision. Non-converged fits fail their
trial and consume budget. An unexpected error aborts the run; an incomplete or
wholly failed study cannot fall back to a default model. Trial states, failures,
parameters/distributions, actual weights, iteration counts and elapsed durations
are persisted in SQLite and JSON, including when selection fails.

For the selected converged trial, maximize validation F1 to choose its threshold;
on a tie use the highest threshold (shared metric helper). Do not refit on
train+validation. Freeze both selected models and thresholds before generating
any held-out test predictions. Test scores never select trials or thresholds.
No hyperparameter changes or repeat selection should be based on test results.

## Artifacts

Each condition directory contains the persistent Optuna study, trials JSON,
study summary, selection JSON, individual converged trial estimators and the
selected `pipeline.joblib`. The pipeline bundles condition, fitted preprocessing,
model and threshold; its `predict(data, company_ids)` produces the aligned
company_name/anchor_year/target/probability table without fitting anything.
Only load trusted joblib files. Software versions are recorded; cross-version
pickle compatibility is not promised.

Validation and test predictions are saved separately. Metadata records selected
parameters, class weights, model/input/study/trial IDs, feature order, solver
settings, thresholds, software/device, seeds, source commit and dataset/split/
config/checkpoint/preprocessor hashes. `run_context.json` and `run_config.json`
exist before tuning so failed runs still have input/settings provenance.

At the top level, `frozen_selections.json` marks completion of selection before
test access. `comparison.json` verifies identical test IDs/anchors/targets and
records AP, precision, recall, F1, other shared metrics and the paired AP change.
`manifest.json` can be supplied to `scripts/compare_models.py`. The two LR
artifacts are not a substitute for the complete eight-configuration comparison.
The target remains publisher-dependent company status, with target_verified=false.

Tests cover converged reload equivalence, non-converged/all-failed trials,
repeatable independent seeded searches, training-only class-weight calculation,
full file-to-prediction execution, aligned output schemas and the test-access gate.

References: [scikit-learn LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html),
[Optuna study optimization](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html).

## Recorded execution: lr_issue53_run01

Executed on 11 November 2026, source commit `bb2240d`, seed 42. Both studies
completed all 20 attempted trials without convergence failures. Each selected
trial used class_weight=None. Validation selected C=10.3531 (latest_year) and
C=110.2182 (five_year); respective thresholds were 0.082974 and 0.114728.

| Input | Validation AP | Test AP | Test precision | Test recall | Test F1 |
|---|---:|---:|---:|---:|---:|
| latest_year | 0.256833 | 0.291786 | 0.338710 | 0.456522 | 0.388889 |
| five_year | 0.302716 | 0.320200 | 0.322581 | 0.543478 | 0.404858 |

Both test files cover the same 1,346 companies, including 92 failures. Five-year
inputs found 50 of those failures versus 42, while false positives increased
from 82 to 105 at the separately validation-selected thresholds. This is one
seeded run, not an uncertainty estimate or evidence of a causal history effect.
Measured study durations were 1.86 and 2.44 seconds in this environment; these
are study wall times, not total setup time or portable performance guarantees.

The compact machine-readable record is [logistic-run-53.json](logistic-run-53.json).
Full local artifacts are under `results/models/lr_issue53_run01/` (gitignored),
including SQLite studies, pipelines, predictions and provenance. Reproduce them
with the command above using a new output directory. Validation: 145 tests passed.
