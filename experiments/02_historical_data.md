# Experiment 02: GRU history length

Status: planned; no results available.

Compare only the GRU at k = 1, 2, 3, 4, and 5 years. Logistic Regression, XGBoost,
and MLP remain latest-year baselines in Experiment 01.

## Planned protocol

- Anchor each company's history at its latest observed year and add older years
  backwards in time, as specified in the README. Document tensor order.
- Use the provided company-level status label, not an assumed next-year outcome.
- Agree handling of missing years and companies with insufficient history.
- Reuse company-disjoint training, validation, and test split definitions.
- Use identical eligible test companies across all five lookbacks for the main
  comparison; report excluded companies, class balance, and sample coverage.
- Agree whether training cohorts are held constant and document this choice.
- Keep preprocessing, tuning budgets, and threshold selection policies consistent.
- Never choose the preferred lookback using test performance.

## Evaluation

Main metric: average precision (the chosen PR-AUC summary). Report precision,
recall, and F1; include accuracy, ROC-AUC, and confusion matrices as context.
Plot measured GRU performance versus history length and inspect false negatives
and false positives.

## Results

Pending actual experiments. No estimated or synthetic scores.
