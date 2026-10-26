# Experiment 01: Baseline and GRU comparison

Status: planned; no models trained and no results available.

Compare Logistic Regression, XGBoost, and MLP using each company's latest year
with a GRU using its last k years of financial data.

## Planned protocol

- Audit the provided alive/failed company labels; do not redefine them as
  next-year events. Use one latest-year anchor per company for all models.
- Split by company, keeping all rows of a company in exactly one partition.
- Give every baseline only the latest year's 18 financial variables.
- Choose the GRU comparison lookback using validation data, never test scores.
- Fit preprocessing on training data only; tune models/thresholds on validation.
- Compare on identical eligible test companies and report history exclusions.
- Record seed 42, saved company splits, hyperparameters, and imbalance handling.

## Evaluation and interpretation

Main metric: average precision, our chosen PR-AUC summary. Report precision and
recall together, plus F1; accuracy, ROC-AUC, and confusion matrices are secondary.
Inspect missed bankruptcies and false alarms. Architecture and history both
change in the baseline-vs-GRU comparison; the GRU k=1–5 experiment helps examine
history length while holding the model family fixed.

## Results

Pending actual experiments. No estimated or synthetic scores.
