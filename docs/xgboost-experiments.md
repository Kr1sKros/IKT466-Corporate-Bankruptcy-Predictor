# XGBoost experiments (#54)

Run from the repository root after installing the project (`python -m pip install
-e .`). This implements the two XGBoost cells, not all eight experiments. It
uses the persisted company split and the shared input/preprocessing code.
XGBoost's financial transform is **identity** in both conditions. latest_year
has 18 financial features; five_year has 100 flattened financial/offset/presence
fields, retains short histories, and excludes records older than the latest five.

```sh
python scripts/prepare_data.py --csv PATH_TO_AUDITED_CSV
python scripts/train_xgboost.py --csv PATH_TO_AUDITED_CSV --output results/models/xgboost/run_01
```

The runner verifies the CSV and persisted split hashes from `configs/config.yaml`
and refuses to overwrite an existing output directory. It never regenerates a
split. The configuration's relative split directory is resolved against the
repository containing `configs/`; CLI CSV and output paths use the current
working directory. `--config` supports an explicit alternative configuration.

On macOS, XGBoost may need the OpenMP runtime (`brew install libomp`); see the
[official installation guide](https://xgboost.readthedocs.io/en/stable/install.html).
The temporary verification environment used PyTorch's bundled libomp through
`DYLD_LIBRARY_PATH`; this is an environment workaround, not a project requirement.

## Selection protocol

Both independent studies use the same declared search in `models.xgboost.search`
and the shared `study_plan.tuning` settings: seeded TPE, validation Average
Precision, 20 attempted trials, sequential execution and no pruning. Seeds and
one CPU thread are fixed; objective is binary logistic and tree method is `hist`.

The initial space is 100–400 trees, depth 2–6, learning rate 0.02–0.2 (log),
minimum child weight 1–10 (log), row/column subsampling 0.7–1, L2 regularization
0.1–10 (log), and L1 regularization 0–1. These are declared starting ranges, not
claims of optimal settings. A pilot review may change them before the main
experiments; paired conditions must receive the same changes.

Class-weight selection includes unweighted training (`scale_pos_weight=1`), the
actual training negative/positive ratio, and a log search between 1 and that
ratio. The ratio and its source counts are saved. Validation/test labels never
determine these bounds and are not weighted or resampled. The first attempt is
an unweighted reference; the remaining settings are suggested normally.

The shared sklearn-based AP implementation scores validation predictions. This
avoids substituting XGBoost's own PR integration convention. Tree count is tuned
directly; this runner does not use an evaluation set or early stopping. Each
successful trial saves its trained model, and selection reloads the best trial
rather than refitting it. Threshold selection maximizes validation F1, breaking
exact ties toward the highest threshold.

Both selected configurations and thresholds are saved in `frozen_selections.json`
**before either test prediction is computed**. Test outcomes never select model
settings. Failed XGBoost trials consume attempts and save their errors; unexpected
errors stop the run with partial records. An incomplete/wholly failed study
cannot fall back to default settings or proceed to test evaluation.

## Saved artifacts and inference

Each condition directory contains `study.sqlite3`, `trials.json`,
`study_summary.json`, per-successful-trial `.ubj` models, the selected `model.ubj`,
`preprocessor.json`, `pipeline.json`, `selection.json`, provenance `metadata.json`,
validation/test prediction CSVs and test metrics. Trial records include actual
parameters, distributions, states, validation AP, errors and elapsed times.

The root contains the run/config snapshot, frozen selections, a comparison of
the two conditions, and a `manifest.json` consumable by `scripts/compare_models.py`.
Saved test predictions include company IDs, anchors, labels, probabilities and
model/condition/study/trial identifiers. The comparison rejects cohort omissions
or changed anchors/labels against the persisted test partition.

```python
from models.xgboost_model import XGBoostPipeline

pipeline = XGBoostPipeline.load('results/models/xgboost/run_01/five_year')
predictions = pipeline.predict(data, companies=test_company_ids)
threshold = pipeline.threshold
```

Inference reloads the model, exact feature order and preprocessing contract,
without refitting. `target_verified=false` remains recorded: the publisher's t+1
convention and alive-company follow-up cannot be independently verified.

## Verification and remaining integration

Tests perform actual small CPU fits for both schemas, verify identity transforms,
exact model/pipeline reload, deterministic seeded TPE suggestions, training-only
class-weight bounds, failure records, and the freeze-before-test boundary. A
test changes test financial values and verifies unchanged selected parameters,
validation AP and thresholds. End-to-end synthetic runs verify saved IDs and
the existing comparison interface. These tests are not real-data performance
results; no final real-data test scores are claimed here.

Issue #56 still owns consolidating the common runner/study interfaces across all
four families. This XGBoost implementation consumes the existing shared tuning
configuration and uses the same manifest convention as the LR work; it does not
implement other model families or claim #60/#62/#69 are complete.
