# XGBoost experiments (#54)

Run from the repository root after installing the project (`python -m pip install
-e .`). This implements the two XGBoost cells, not all eight experiments. It
uses the persisted company split and the shared input/preprocessing code.
XGBoost's financial transform is **identity** in both conditions. latest_year
has 18 financial features; five_year has 100 flattened financial/offset/presence
fields, retains short histories, and excludes records older than the latest five.

```sh
python scripts/prepare_data.py --csv PATH_TO_AUDITED_CSV
python scripts/prepare_preprocessing.py --csv PATH_TO_AUDITED_CSV
python scripts/train_xgboost.py --csv PATH_TO_AUDITED_CSV --output results/models/xgboost/run_01
```

The runner verifies the CSV and persisted split hashes from `configs/config.yaml`
and refuses to overwrite an existing output directory. It never regenerates a
split. The configuration's relative split directory is resolved against the
repository containing `configs/`; CLI CSV and output paths use the current
working directory. `--config` supports an explicit alternative configuration.
`--preprocessor` selects an existing shared preparation artifact; its default is
`results/models/preprocessor.json` in the configuration's repository. The runner
loads this artifact, checks its dataset/split provenance, audits its fitting
statistics, and uses its existing identity path for XGBoost. It does not create
a separate preprocessing policy. The reference audit refits for verification
only; actual predictions use the loaded artifact.

On macOS, XGBoost may need the OpenMP runtime (`brew install libomp`); see the
[official installation guide](https://xgboost.readthedocs.io/en/stable/install.html).
The temporary verification environment used PyTorch's bundled libomp through
`DYLD_LIBRARY_PATH`; this is an environment workaround, not a project requirement.

## Selection protocol

Both independent studies use the same declared search in `models.xgboost.search`
and the shared `study_plan.tuning` settings: seeded TPE, validation Average
Precision, 20 attempted trials, sequential execution and no pruning. Seeds and
one CPU thread are fixed; objective is binary logistic and tree method is `hist`.

The post-pilot space is 100–1000 trees, depth 2–10, learning rate 0.01–0.3 (log),
minimum child weight 1–10 (log), row/column subsampling 0.5–1, L2 regularization
0.1–10 (log), and L1 regularization 0.001–10 (log). Both conditions use this
single configuration. These ranges do not claim to contain optimal settings.

### Validation-only pilot review — 11 November 2026

The protocol permits one documented pilot review before the main searches.
[The PR reviewer](https://github.com/Kr1sKros/IKT466-Corporate-Bankruptcy-Predictor/pull/85#pullrequestreview-5858019974)
reported running both original studies on the audited real data, using validation
only and no test data. Reported best validation AP was 0.350 for latest_year and
0.457 for five_year. The selected five_year trial had learning rate 0.194,
subsample 0.706 and L1 regularization 0.855; depth 6 occurred in three of the top
five trials. These observations motivate exploring beyond the original bounds,
but 20 trials and near-boundary selections do not prove that the bounds limited
performance or that a wider search will improve it.

The reviewer-reported pilot was not independently rerun as part of this change;
its original study/trial artifacts should be retained as pilot evidence, separate
from main-search outputs. The accepted changes apply to **both** input conditions:

| Parameter | Pilot range | Main-search range |
|---|---|---|
| `n_estimators` | 100–400 | 100–1000 |
| `max_depth` | 2–6 | 2–10 |
| `learning_rate` | 0.02–0.2, log | 0.01–0.3, log |
| `subsample` | 0.7–1 | 0.5–1 |
| `colsample_bytree` | 0.7–1 | 0.5–1 |
| `reg_alpha` | 0–1, linear | 0.001–10, log |

The log-scaled L1 search excludes exact zero; its lower endpoint represents weak
positive L1 regularization. Minimum child weight, L2 bounds, preprocessing,
training-derived class-weight bounds, seeds and the common **20-attempt** budget
remain unchanged. Start new output directories for the main searches; do not
resume pilot studies with different ranges or select between pilot and main
models using test outcomes. Any later search redesign must be disclosed rather
than treated as another unrecorded pilot review.

Class-weight selection is a continuous log search from 1 to the training
negative/positive ratio returned by `leakage.class_weight_bounds`, matching the
merged LR convention. The unweighted value is in the domain, but sampling does
not guarantee the exact endpoint. The ratio and source counts are saved.
Validation/test labels never determine bounds and are not weighted or resampled.

The shared sklearn-based AP implementation scores validation predictions. This
avoids substituting XGBoost's own PR integration convention. Tree count is tuned
directly; this runner does not use an evaluation set or early stopping. Each
successful trial saves its trained model, and selection reloads the best trial
rather than refitting it. Threshold selection maximizes validation F1, breaking
exact ties toward the highest threshold.

Both selected configurations and thresholds are saved in `frozen_selections.json`
and `validation_summary.json`. **The runner never predicts or scores test
companies.** Final test evaluation is deferred until all eight studies are
frozen (#56/#60/#62), matching the LR runner. Test outcomes never select model
settings. Failed XGBoost trials consume attempts and save their errors; unexpected
errors stop the run with partial records. An incomplete/wholly failed study
cannot fall back to default settings or proceed to test evaluation.

## Saved artifacts and inference

Each condition directory contains `study.sqlite3`, `trials.json`,
`study_summary.json`, per-successful-trial `.ubj` models, the selected `model.ubj`,
`preprocessor.json`, `pipeline.json`, `selection.json`, provenance `metadata.json`,
validation prediction CSVs. Trial records include actual
parameters, distributions, states, validation AP, errors and elapsed times.

The root contains the run/config snapshot, frozen selections and a validation
summary with the same fields as the LR runner. Saved validation predictions
retain company IDs, anchors, labels and probabilities; selection/provenance
records identify the model, condition, study and trial. Generating final test
predictions and the comparison manifest is part of #60/#62, not training.

```python
from models.xgboost_model import XGBoostPipeline

pipeline = XGBoostPipeline.load('results/models/xgboost/run_01/five_year')
predictions = pipeline.predict(data, companies=validation_company_ids)
threshold = pipeline.threshold
```

Inference reloads the model, exact feature order and preprocessing contract,
without refitting. `target_verified=false` remains recorded: the publisher's t+1
convention and alive-company follow-up cannot be independently verified.

## Verification and remaining integration

Tests perform actual small CPU fits for both schemas, verify identity transforms,
exact model/pipeline reload, deterministic seeded TPE suggestions, training-only
class-weight bounds, failure records, and the validation-only boundary. A
test changes test financial values and verifies unchanged selected parameters,
validation AP and thresholds. End-to-end synthetic runs verify shared-artifact
provenance and saved IDs and forbid any test prediction. A cross-model check
compares LR/XGBoost trial protocol, weight bounds and threshold rules. These
tests are not real-data performance
results; no final real-data test scores are claimed here.

Issue #56 still owns consolidating the common runner/study interfaces across all
four families. This XGBoost implementation consumes the existing shared tuning
configuration and validation-only output convention as the LR work; it does not
implement other model families or claim #60/#62/#69 are complete.
