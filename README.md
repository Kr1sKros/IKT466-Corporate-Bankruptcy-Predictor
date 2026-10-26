# IKT466 – Corporate Bankruptcy Predictor

Project for the Introduction to Machine Learning course (IKT466) at UiA.

We predict whether a US public company goes bankrupt from its yearly accounting data, and test whether a **GRU over multi-year financial history** beats models that only see the **latest year**.

## 1. Task, data and research question

**Task:** binary classification. Given a company's accounting data, did it go bankrupt (`failed`) or not (`alive`)?

**Dataset:** US public companies bankruptcy dataset (Lombardo et al., *Future Internet* 14(8):244, 2022).
- 8,971 companies listed on NYSE/NASDAQ, 1999–2018
- 78,682 firm-year rows with 18 accounting variables each
- Label: `status_label` (`alive` / `failed`), used as provided
- 609 failed companies (≈ 6.8%)

**Research question:**

> Does modelling a company's multi-year financial history with a GRU predict bankruptcy better than using only its latest year? How many years of history actually help?

## 2. Baselines

- Logistic regression
- XGBoost
- MLP

All baselines use only the company's latest year, so comparing them with the GRU shows the value of history.

## 3. Neural model

A GRU that reads a company's last *k* years of accounting data, starting from the latest year and adding years backwards in time.

## 4. Investigations

1. **Baseline comparison:** logistic regression vs XGBoost vs MLP vs GRU.
2. **History length:** the GRU with *k* = 1, 2, 3, 4 and 5 years.

## 5. Evaluation

- Separate training, validation and test sets, split by company
- **Main metric: PR-AUC**, because bankruptcies are rare and accuracy would be misleading
- Precision and recall reported together
- Error analysis of missed bankruptcies and false alarms

## 6. Reproducibility

The repository will contain the code, configurations, split definitions and saved results needed to reproduce the main comparisons.

## 7. Repository structure

The project uses a simple `src/` layout without an additional project-package folder.

```text
.
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── main.py                       # Shortcut for dataset inspection
├── configs/config.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── src/
│   ├── data_loader.py            # Existing Kaggle download and CSV loading
│   ├── preprocessing.py          # Label audit, company splits, imputation/scaling
│   ├── sequences.py              # Latest-year selection and GRU histories
│   ├── metrics.py
│   ├── evaluation.py
│   ├── visualization.py
│   └── models/
│       ├── __init__.py
│       ├── logistic_regression.py
│       ├── xgboost_model.py
│       ├── mlp_model.py
│       └── gru_model.py
├── scripts/
│   ├── inspect_data.py
│   ├── prepare_data.py
│   ├── train_logistic.py
│   ├── train_xgboost.py
│   ├── train_mlp.py
│   ├── train_gru.py
│   ├── compare_models.py
│   └── compare_lookback.py         # GRU only, k = 1–5
├── tests/
│   ├── test_preprocessing.py
│   ├── test_sequences.py
│   └── test_splits.py
└── results/
    ├── figures/
    ├── metrics/
    └── models/
```

Data and results directories contain tracked `.gitkeep` markers. Generated data,
models, and results are ignored by Git.

## 8. Setup and implementation status

Use Python 3.10+ (3.12 recommended). From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python scripts/inspect_data.py
python -m pytest
```

Editable installation makes the modules importable without the former
`bankruptcy_ml` prefix:

```python
from data_loader import load_data
from models.logistic_regression import build_model
```

`main.py` delegates to the basic dataset inspection script. Inspection downloads
into KaggleHub's cache and prints a preview; detailed auditing is still a TODO.
The local CSV path in the config is separate from that cache. Run scripts from
the repository root; future config loading should resolve paths with `pathlib`.

Only downloading/loading is implemented. Model, preprocessing, evaluation, and
plotting interfaces raise `NotImplementedError`; tests are explicitly skipped.
No trained models or experimental results are included. Settings marked `null`
are unresolved choices, not runnable defaults.

## 9. Planned implementation order

1. Audit company IDs, years, and provided labels in `data_loader.py`,
   `preprocessing.py`, and `inspect_data.py`.
2. Implement reproducible company-disjoint splits and training-only preprocessing.
3. Implement latest-year baseline inputs and aligned GRU histories; agree short
   history and missing-year policies and add meaningful tests.
4. Implement shared metrics/evaluation, then each model and its training entry
   point step by step.
5. Run the two planned experiments, inspect errors, and document measured results.

All baselines receive only latest-year features. The GRU history experiment holds
model family fixed; the baseline-versus-GRU comparison changes both architecture
and history, which should be discussed when interpreting improvements.
Use feature branches and teammate-reviewed pull requests; record split decisions,
experiment settings, dataset version, and contributions in tracked documentation.
