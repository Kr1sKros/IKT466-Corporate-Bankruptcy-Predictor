# Shared metrics and reproducibility contract

These modules implement the common contracts for issues #58 and #59. The
four experiment runners on `main` are still placeholders; they must call these
helpers when implemented. No real-dataset experiment is claimed here. The
issues' references to an already working local LR/metric pipeline could not
be verified in the published repository, so this change implements the checked
interfaces and adds regression tests rather than claiming reuse of absent code.

## Metrics

Use numeric `alive=0`, `failed=1`. `compute_metrics` requires equally sized,
nonempty one-dimensional vectors and finite probabilities in [0, 1]. Its
`average_precision` is scikit-learn's non-interpolated AP, not trapezoidal area.
AP/ROC-AUC use probabilities and do not change when the decision threshold
changes. Discrete predictions use `probability >= threshold`; confusion matrix
rows are true labels and columns predicted labels, both ordered `[0, 1]`.

Undefined precision, recall, and F1 use zero. With no positive labels, AP is
`None` (JSON `null`), deliberately avoiding a misleading numerical score. With
only positives, AP is 1. ROC-AUC is `None` for either single-class case. Always
report sample count and prevalence, and do not silently compare undefined runs.

Call `select_validation_threshold(y_validation, p_validation)` only on the
validation partition; it maximizes F1 and breaks exact ties toward the highest
threshold. This is an explicit available selection rule, not a retrospectively
chosen claim about the study. Record the chosen rule and threshold in metadata;
any alternative recall/cost policy needs agreement before test evaluation.
Never call it on test labels. Pass the persisted result into
`evaluate_predictions(observations, y_test, p_test, threshold=threshold)`.
Observation rows must contain unique company IDs, non-null anchor years, and
the same row order as the vectors. An optional numeric `label` column is checked
against `y_test`. When predictions come from separate tables, join by company
and anchor year first: arrays alone cannot prove identity alignment.

## Reproducibility

Before model, tensor, or loader creation:

```python
from reproducibility import (
    configure_reproducibility, estimator_seed_parameters, make_data_loader,
    write_run_metadata,
)

settings = configure_reproducibility(42)
# LogisticRegression(**estimator_seed_parameters('logistic_regression', 42), ...)
# XGBClassifier(**estimator_seed_parameters('xgboost', 42), ...)
# MLP/GRU: construct after configuration; use the shared seeded DataLoader.
# loader = make_data_loader(training_dataset, seed=42, batch_size=32,
#                           shuffle=True, num_workers=2)
```

Python, NumPy's legacy RNG, and PyTorch CPU/CUDA RNGs are seeded. New NumPy
Generators need their own explicit `np.random.default_rng(42)`. LR and XGBoost
constructors need explicit `random_state`; XGBoost additionally uses one CPU
thread. PyTorch uses one CPU thread, deterministic algorithms, disabled cuDNN
benchmarking, and deterministic cuDNN. Unsupported deterministic operations fail
instead of silently falling back. Set `PYTHONHASHSEED=42` **before launching
Python** when hash iteration order matters; setting it at runtime cannot change
the running interpreter. The helper records its current environment value.

The loader owns its generator and seeds Python/NumPy inside each worker from
PyTorch's worker seed. This works with Windows-spawn workers; keep custom dataset
classes at module scope and application entry points under an `if __name__ ==
'__main__'` guard. Recreate the loader to repeat a run; reuse it to advance epoch
shuffle order. Do not share a generator between training and evaluation loaders.

After saving the validation-selected checkpoint, call `write_run_metadata` with
the actual model name, all hyperparameters/threshold rules, returned seed
settings, dataset CSV, saved train/validation/test ID files, configuration file,
and checkpoint path. The JSON records SHA-256 and byte size for every input and
checkpoint, publisher version when actually known, installed software versions,
device/CUDA/cuDNN details, source commit, and dirty-worktree status. A missing
file is an error. An unknown publisher version stays null; a checksum does not
establish publisher provenance. Do not put credentials in parameters or config.

Timestamps differ between repeated runs. Seeds are not a complete resume state:
resuming training also requires saved optimizer, scheduler, RNG and loader state.
CPU/CUDA, package releases, GPU kernels/drivers and distributed execution can
produce different results even with the same seed. This change tests repeated
small CPU runs, not CUDA or cross-platform numerical equivalence. Keep device
and software records, compare predictions, and document measured tolerances.

References:

- [Scikit-learn Average Precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)
- [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html)
