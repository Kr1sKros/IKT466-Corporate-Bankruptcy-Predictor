# Comparing saved test predictions (#60)

Run from the repository root with the project installed (`pip install -e .`).
Export one CSV per model/run (including separate GRU k=1,...,5 files):

```csv
company_name,anchor_year,target,probability
C_1,2018,0,0.12
C_2,2017,1,0.81
```

These rows are illustrative, not experiment results. Use the company ID,
final recorded year and target from the persisted test split. `target` is
numeric alive=0 / failed=1; `probability` is the predicted probability of failed.
Every file must contain every test company exactly once, even for short GRU
histories (the approved padding policy applies). Extra columns are preserved.

Create a JSON manifest listing the files and previously validation-selected
thresholds. File paths are relative to the manifest. Run names must be unique;
use descriptive names such as `logistic_regression`, `xgboost`, `mlp`, `gru_k1`
through `gru_k5`. The following thresholds are examples only:

```json
{
  "runs": [
    {"name": "logistic_regression", "csv": "lr.csv", "threshold": 0.3, "threshold_source": "validation"},
    {"name": "gru_k5", "csv": "gru_k5.csv", "threshold": 0.4, "threshold_source": "validation"}
  ]
}
```

```powershell
python scripts/compare_models.py --manifest results/predictions/runs.json --output results/metrics/comparison.json
```

The runner loads all three persisted split files and checks their configured
checksums and company disjointness. Split paths in the config are relative to
the working directory, as in prepare_data.py. It checks each prediction file
against the saved test company/anchor/target table, then sorts company IDs
lexicographically. It never silently intersects files: missing/extra companies,
duplicate IDs, wrong years, conflicting labels, invalid probabilities or empty
inputs fail before a report is written. Passing only matching subsets cannot
hide missing test companies. The lower-level `align_test_observations` can also
align tables without a reference, but only the explicit reference proves full
test coverage; `compare_predictions` always requires it.

Main comparison uses all test companies for every model and lookback. For a
separate history sub-analysis, add `--min-history 5` to retain companies with
at least five recorded years for **every** listed run. Use the largest lookback
being compared for a full-history comparison across several lookbacks. Do not
compare metrics from separately filtered cohorts as if only lookback changed.
The runner never infers a history cutoff from run names.

The JSON report includes common company/anchor/target rows, original and
retained counts and bankruptcy prevalence, excluded IDs/count, and each run's
AP, precision, recall, F1, accuracy, ROC-AUC and confusion matrix. It records
input CSV hashes, split hashes and thresholds. Main comparison excludes zero
companies; sub-analysis exclusions are based solely on persisted history length.

Thresholds are mandatory, never fitted on test data. The manifest declares
validation provenance; the runner cannot independently prove how a supplied
threshold or prediction was produced. Keep training/run metadata alongside the
exports. The shared metric contract defines single-class/undefined values.
This implements comparison infrastructure, not model training or measured
results from the real dataset.
