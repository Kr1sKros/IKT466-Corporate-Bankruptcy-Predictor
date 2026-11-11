# Logistic Regression selection (#53): validation only

```powershell
python scripts/prepare_preprocessing.py --csv PATH_TO_AUDITED_CSV
python scripts/train_logistic.py --csv PATH_TO_AUDITED_CSV --preprocessor results/models/preprocessor.json --output results/models/lr_validation_01
```

Run from the repository root with dependencies installed. The output directory
must be new. The runner loads the shared preprocessor, verifies its dataset/split
provenance and calls the #52 audit. That audit fits a temporary training-only
reference to verify statistics; inference uses the loaded artifact, not a newly
fitted replacement. A snapshot is saved with the run. Per-condition scaling stays
fixed, outside Optuna, as specified in docs/preprocessing.md.

## Search and selection

Two independent studies use seeded sequential TPE (42), no pruning, and the same
20-attempt budget. lbfgs learns L2-regularized coefficients with max_iter=5000
and tol=0.0001. Both input conditions search C logarithmically in [0.0001, 10000].

The positive class weight w is sampled continuously on a log scale in
[1, training negatives / training positives], using leakage.class_weight_bounds.
The estimator receives class_weight={0: 1, 1: w}. Bounds use training labels only;
validation scores are unweighted and unresampled. The domain contains unweighted
w=1, but continuous sampling does not guarantee drawing that exact endpoint.
This follows the same range as GRU/MLP pos_weight and XGBoost scale_pos_weight.

Validation Average Precision selects the converged trial. Non-converged trials
are failed attempts; unexpected errors abort. No incomplete/all-failed study
falls back to defaults. Validation F1 selects the threshold, breaking ties toward
the highest threshold. No train+validation refit occurs. SQLite/JSON persist
trial parameters, distributions, states, failures, weights, iterations and timing.

## Held-out test gate

This runner produces **no test inputs, test predictions, test metrics or test
comparison manifest**. It saves LR selections only. Final test evaluation belongs
to #56/#62 and must wait for all eight model/input studies and thresholds to be
frozen. Finishing both LR studies does not open that gate. Shared dataset/split
integrity checks still validate company/anchor/target metadata across partitions;
held-out values are not used for tuning or scoring.

An earlier version evaluated the LR test partition prematurely. That historical
exposure cannot be undone. The previous run is superseded, its test numbers are
removed from the current documentation/PR description, and no further test
selection/evaluation is performed here. Prior exposure remains a study limitation;
removing current files does not erase Git history or previously viewed results.

## Outputs and reload

Each condition contains study.sqlite3, trials.json, study_summary.json,
selection.json, validation_predictions.csv, metadata.json and pipeline.joblib.
The pipeline bundles condition, shared preprocessing, estimator and threshold.
Its predict method is available for eventual centrally gated inference. Load
joblib artifacts only from trusted sources; cross-version compatibility is not
promised. Trial checkpoints are also saved.

run_context.json and run_config.json record input hashes, seeds and software even
for failed studies. Selected metadata adds feature order, source commit and
checkpoint/preprocessor hashes. frozen_selections.json and validation_summary.json
record the two LR selections; they do not certify that all eight studies are ready.
No old test-result summary is committed. Generated artifacts remain gitignored.
The target retains the audited publisher-dependent limitations (target_verified=false).

Tests cover continuous training-derived weight bounds, repeatable studies,
convergence/failure handling, reload, shared-preprocessor verification and guards
that reject test-company input construction or prediction in the LR runner.

References: [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html),
[Optuna study optimization](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html).
