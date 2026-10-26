# Experiment 01: Model comparison

Status: planned; no models trained and no results available.

Compare Logistic Regression (linear baseline), XGBoost (boosted decision trees),
and GRU (recurrent neural network) on bankruptcy prediction.

## Protocol to agree

- Verify label semantics and feasibility of a one-year prediction horizon first.
- Choose a fixed lookback, shared features, partitions, and prediction-year IDs.
- Give baseline models flattened versions of the same histories used by the GRU.
- Fit preprocessing on training data only. Tune models and thresholds on validation data.
- Evaluate every model on identical test observations and report sample coverage.
- Record seed 42, hyperparameters, class imbalance handling, and selection rules.

## Evaluation plan

Primary metrics: precision, recall, F1, and average precision (the chosen PR-AUC
summary). Additional metrics: accuracy, ROC-AUC, and confusion matrix.
Use consistent positive labels and shared metric/evaluation functions.

## Results

Pending implementation and actual experiments. No estimated or synthetic scores.
