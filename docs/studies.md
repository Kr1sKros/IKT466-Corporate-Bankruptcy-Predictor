# Running the eight studies (#56)

All four models are tuned the same way, once per input condition, through the
shared study contract in `src/studies.py`:

| Rule | Value |
|---|---|
| Studies | 4 models × 2 conditions = 8, each in its own new directory and Optuna study |
| Sampler | `TPESampler(seed=42)`, no pruning, `n_jobs=1` |
| Objective | validation Average Precision (unweighted) |
| Budget | 100 attempted trials per study, the same for all eight |
| Class weight | log scale from 1 to the training negative/positive ratio (training labels only) |
| Threshold | chosen on validation: highest threshold with maximum F1 |
| Failures | recorded per trial; a study with fewer trials than the budget or no completed trial raises, and nothing is selected |

What each model tunes is model-specific (`configs/config.yaml` `models`); the
ranges are the same for latest_year and five_year. Per trial, Optuna picks the
settings and the model learns its own weights:

- **LR:** `C` and the positive class weight; lbfgs fits the coefficients.
- **XGBoost:** 8 tree settings and `scale_pos_weight`; the trees are fitted per trial.
- **MLP / GRU:** width, depth, dropout, learning rate, weight decay, batch size and
  `pos_weight`. Training follows `training` in the config: Adam, at most 200
  epochs, stop after 20 epochs without a better validation AP, restore the best
  epoch. Only the best checkpoint so far is kept on disk.

## Budget: 100 trials

The 20-trial pilot (#53, #54) is too small for the larger spaces. Optuna's TPE
samples the first 10 trials at random, so 20 trials leave about 10 guided trials.
That covers LR's 2 parameters well, but not XGBoost's 9 or the MLP/GRU's 7, and it
would favour LR. The budget was raised to 100 for all eight studies before the main
searches.

## Running

Regenerate the preprocessor first if it predates #87, then:

```powershell
python scripts/run_studies.py --csv PATH_TO_AUDITED_CSV --output results/studies/main --parallel 4
```

`--parallel 4` runs the four model runners at the same time as separate
processes. Each study is still single-threaded and seeded, so the results are the
same as with `--parallel 1`; it only saves wall-clock time. One model can also be
run on its own with `scripts/train_logistic.py`, `train_xgboost.py`,
`train_mlp.py` or `train_gru.py` (same arguments, no `--parallel`).

The output directory must not exist; studies are never resumed or overwritten.

## Outputs

Per model and condition (`OUTPUT/<model>/<condition>/`):

- `study.sqlite3`, `trials.json`: every trial with state, parameters,
  distributions, value, duration and failure text; MLP/GRU trials also store the
  best epoch and the validation AP per epoch.
- `study_summary.json`: budget, attempted / completed / failed, elapsed seconds.
- `selection.json`: selected trial and settings, validation AP, threshold.
- the frozen model (`pipeline.joblib`, `model.ubj` or `model.pt`) with its
  `preprocessor.json`, and `validation_predictions.csv` (company, anchor year,
  target, probability) from the reloaded model.
- `metadata.json`: seeds, data/split/config/checkpoint hashes, software and hardware.

`OUTPUT/frozen_studies.json` is written only when all eight studies are complete.
It lists every selection, its validation AP, threshold, trial counts and compute
time. If anything fails, `run_report.json` and the per-model logs say what.

No runner builds test inputs. The held-out test evaluation (#60/#62) starts from
`frozen_studies.json`. Validation AP is not a fair score to compare models on: the
MLP and GRU also choose their best epoch on validation, so their validation AP
is a bit optimistic. Compare models on test only.
