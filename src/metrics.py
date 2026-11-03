"""Shared classification metric interface."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)


def _validated_predictions(y_true, y_score) -> tuple[np.ndarray, np.ndarray]:
    """Accept nonempty, aligned vectors of binary labels and probabilities."""
    labels = np.asarray(y_true)
    scores = np.asarray(y_score)
    if labels.ndim != 1 or scores.ndim != 1 or len(labels) == 0 or labels.shape != scores.shape:
        raise ValueError("Labels and probabilities must be nonempty, aligned 1-D vectors")
    if labels.dtype.kind not in 'biuf' or not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must be numeric 0 (alive) or 1 (failed)")
    if scores.dtype.kind not in 'iuf' or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Scores must be finite probabilities in [0, 1]")
    return labels.astype(int), scores.astype(float)

def compute_metrics(
    y_true: np.ndarray, y_score: np.ndarray, *, threshold: float = 0.5
) -> dict[str, object]:
    """Evaluate failed=1 using scores >= a previously fixed threshold.

    AP is sklearn's non-interpolated average precision, not trapezoidal PR area.
    Undefined precision/recall/F1 are reported as 0. ROC-AUC is None with one
    observed class; AP is None with no positives (1 with only positives).
    Outputs are JSON-compatible, with confusion rows=true/columns=predicted,
    always ordered [alive=0, failed=1]. This function never tunes a threshold.
    """
    labels, scores = _validated_predictions(y_true, y_score)
    if isinstance(threshold, (bool, np.bool_)) or not np.isscalar(threshold):
        raise ValueError("Threshold must be a finite number in [0, 1]")
    try:
        threshold = float(threshold)
    except (TypeError, ValueError) as error:
        raise ValueError("Threshold must be a finite number in [0, 1]") from error
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be a finite number in [0, 1]")
    predicted = (scores >= threshold).astype(int)
    matrix = confusion_matrix(labels, predicted, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    return {
        'threshold': threshold,
        'positive_label': 1,
        'n_observations': int(len(labels)),
        'positive_prevalence': float(labels.mean()),
        'precision': float(precision_score(labels, predicted, zero_division=0)),
        'recall': float(recall_score(labels, predicted, zero_division=0)),
        'f1': float(f1_score(labels, predicted, zero_division=0)),
        'accuracy': float(accuracy_score(labels, predicted)),
        'average_precision': float(average_precision_score(labels, scores)) if tp + fn else None,
        'roc_auc': float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None,
        'confusion_matrix': matrix.tolist(),
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
    }


def select_validation_threshold(y_validation, validation_probabilities) -> float:
    """Maximize validation F1; choose the highest threshold on exact ties.

    Call only on validation observations, then persist and reuse the result on
    test observations. Both classes are required for meaningful selection.
    """
    labels, scores = _validated_predictions(y_validation, validation_probabilities)
    if len(np.unique(labels)) != 2:
        raise ValueError("Threshold selection requires both validation classes")
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    denominator = precision[:-1] + recall[:-1]
    f1 = np.divide(2 * precision[:-1] * recall[:-1], denominator,
                   out=np.zeros_like(denominator), where=denominator != 0)
    return float(thresholds[np.flatnonzero(f1 == f1.max())[-1]])
