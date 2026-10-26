# Bankruptcy Prediction Using Machine Learning

IKT466 Introduction to Machine Learning final project at UiA, developed by three students.

**Research question:** Does using multiple years of financial data improve bankruptcy
prediction, and how do Logistic Regression, XGBoost, and GRU compare?

This repository is a project scaffold. The existing Kaggle download/CSV loader is
preserved; preprocessing, model training, evaluation, and plotting are documented
interfaces that raise `NotImplementedError`. No models have been trained or results generated.

## Dataset and target warning

[American Companies Bankruptcy Prediction dataset on Kaggle](https://www.kaggle.com/datasets/utkarshx27/american-companies-bankruptcy-prediction-dataset).

**TODO: Verify label semantics first.** The dataset may assign the same
failed/alive label to every historical observation of a company. These labels
must not be assumed to indicate bankruptcy in the following year. Verify event
timing and target feasibility before preparing any next-year prediction labels.
If failure dates cannot be established, revisit the prediction task explicitly.
Record the dataset version and source evidence in the report.

## Models and experiments

- **Logistic Regression:** an interpretable linear baseline using flattened histories.
- **XGBoost:** boosted decision trees using the same flattened financial histories.
- **GRU:** a PyTorch recurrent neural network using ordered annual histories.

[Experiment 01](experiments/01_model_comparison.md) compares the three models on
identical test observations at an agreed lookback.
[Experiment 02](experiments/02_historical_data.md) investigates 1–5 years of
financial history, using a common eligible test cohort for the primary comparison.
Primary metrics are precision, recall, F1, and average precision (our planned
PR-AUC summary). Accuracy, ROC-AUC, and confusion matrices provide context.

## Repository structure

The current Git repository is the project root; no nested repository is required.

```text
bankruptcy-prediction/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── main.py                       # Preserved compatibility entry point
├── data/
│   ├── raw/.gitkeep
│   ├── processed/.gitkeep
│   └── splits/.gitkeep
├── notebooks/
│   └── 01_eda.ipynb
├── src/
│   └── bankruptcy_ml/
│       ├── __init__.py
│       ├── data_loader.py
│       ├── preprocessing.py
│       ├── sequences.py
│       ├── metrics.py
│       ├── evaluation.py
│       ├── visualization.py
│       └── models/
│           ├── __init__.py
│           ├── logistic_regression.py
│           ├── xgboost_model.py
│           └── gru_model.py
├── scripts/
│   ├── inspect_data.py
│   ├── prepare_data.py
│   ├── train_logistic.py
│   ├── train_xgboost.py
│   ├── train_gru.py
│   ├── compare_models.py
│   └── compare_lookback.py
├── configs/
│   └── config.yaml
├── experiments/
│   ├── 01_model_comparison.md
│   └── 02_historical_data.md
├── results/
│   ├── figures/.gitkeep
│   ├── metrics/.gitkeep
│   └── models/.gitkeep
├── tests/
│   ├── test_preprocessing.py
│   ├── test_sequences.py
│   └── test_splits.py
├── report/
│   └── notes.md
└── presentation/
    └── notes.md
```

## Setup

Use Python 3.10+ (Python 3.12 is recommended for this project).
From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python scripts/inspect_data.py
python -m pytest
```

Editable installation makes `bankruptcy_ml` importable by scripts and notebooks.
Dependencies come from `requirements.txt`; `kagglehub` is retained for the existing
download integration. `python main.py` runs the same basic inspection entry point.
The inspection command downloads into KaggleHub's cache and prints a preview;
it does not yet perform the planned detailed audit. `load_data(Path(...))` loads a
local CSV without downloading. The configured `data/raw/` path is a planned local
input location, separate from that cache. Detailed auditing and config loading
are still TODOs. Run scripts from the repository root; config paths are intended
to be resolved from that root using `pathlib`.

Other scripts deliberately raise `NotImplementedError`. Tests are explicitly
skipped until their implementation contracts are agreed. Configuration values
set to `null` are unresolved choices, not runnable training defaults.

## Planned workflow and team collaboration

1. Extend `data_loader.py`, `preprocessing.verify_target_labels`, and
   `scripts/inspect_data.py`; use the EDA notebook to record schema and label findings.
2. Agree the target, one-year horizon feasibility, temporal partitions, company
   overlap rules, and leakage controls. Document decisions before coding splits.
3. Implement preprocessing and tests, fitting all learned transforms on training data only.
4. Implement consecutive-year histories and shared observation IDs for 1–5 years;
   test temporal alignment and split boundaries, then implement `prepare_data.py`.
5. Agree shared metrics/evaluation contracts before implementing model training
   step by step. Tune thresholds and hyperparameters using validation data only.
6. Run both experiments on aligned test observations, save real outputs, and
   develop the report and presentation from verified findings.

Use feature branches and pull requests reviewed by a teammate. Assign owners for
small tasks while reviewing target/splitting decisions together. Keep reusable
logic in `src/bankruptcy_ml/`; scripts should only orchestrate it. Do not commit
raw data, processed datasets, trained models, credentials, or generated results.
Record experiment settings and team contributions in the tracked documentation.
