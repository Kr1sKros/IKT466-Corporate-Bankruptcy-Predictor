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
