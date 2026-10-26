# Experiment 02: Historical financial data

Status: planned; no results available.

Investigate whether multiple years improve prediction using 1, 2, 3, 4, and 5
years of financial data for Logistic Regression, XGBoost, and GRU.

## Protocol to agree

- Verify next-year target feasibility before constructing financial histories.
- Require consecutive company years and align histories to the prediction year.
- Keep the prediction horizon fixed at one year, subject to verified event timing.
- Flatten histories for baseline models and retain temporal tensors for the GRU.
- Use identical eligible test company/prediction-year observations across lookbacks
  and models for the primary comparison; report exclusions and sample coverage.
- Agree whether training cohorts are held constant; document this choice and any
  secondary comparison using all eligible histories.
- Keep split boundaries, preprocessing policy, tuning budgets, and threshold
  selection rules consistent. Never use test scores for model selection.

## Evaluation plan

Primary metrics: precision, recall, F1, and average precision (the chosen PR-AUC
summary). Report additional accuracy, ROC-AUC, and confusion matrices as useful.
Plot measured performance versus lookback and discuss coverage limitations.

## Results

Pending implementation and actual experiments. No estimated or synthetic scores.
