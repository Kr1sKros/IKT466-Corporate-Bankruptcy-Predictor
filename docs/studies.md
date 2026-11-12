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

```bash
python scripts/run_studies.py --csv PATH_TO_AUDITED_CSV --output results/studies/main --parallel 8
```

Each of the eight studies runs as its own process; `--parallel N` runs N at the
same time. Every study is single-threaded, CPU-only and seeded, so the results
are identical for any N; only the wall-clock time changes. Give each parallel
study one free CPU core. With 8 cores the run takes about as long as the slowest
study (GRU, roughly 1.5 h); with 4 cores about twice that. One study can also be
run on its own, e.g. `python scripts/train_gru.py --csv ... --output ... --conditions five_year`.

The output directory must not exist; studies are never resumed or overwritten.

### On a many-core machine (e.g. Coder)

```bash
git clone https://github.com/Kr1sKros/IKT466-Corporate-Bankruptcy-Predictor.git
cd IKT466-Corporate-Bankruptcy-Predictor
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -e .
python scripts/inspect_data.py        # downloads the dataset into the KaggleHub cache
CSV=~/.cache/kagglehub/datasets/utkarshx27/american-companies-bankruptcy-prediction-dataset/versions/3/american_bankruptcy.csv
python scripts/prepare_data.py --csv "$CSV"            # recreates the split; fails if checksums differ
python scripts/prepare_preprocessing.py --csv "$CSV"
python -m pytest -q
nohup python scripts/run_studies.py --csv "$CSV" --output results/studies/main --parallel 8 > studies.log 2>&1 &
```

`nohup` keeps the run alive if the terminal or connection closes. Progress is in
`results/studies/main/<model>__<condition>.log` and each study's `trials.json`.

**No GPU needed.** The networks and data are small (at most 256 units, about 6,300
training companies, sequences of 1-5 years), so a GPU spends more time moving
data and launching work than computing, and it would make runs harder to
reproduce exactly. The studies are pinned to CPU; more cores is what makes the
run faster.

## Outputs

Per model and condition (`OUTPUT/<model>__<condition>/<condition>/`):

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
