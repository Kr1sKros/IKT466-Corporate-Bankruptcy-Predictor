"""Shared classification metric interface."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    import numpy as np

def compute_metrics(
    y_true: np.ndarray, y_score: np.ndarray, *, threshold: float = 0.5
) -> dict[str, object]:
    """Calculate precision, recall, F1, accuracy, ROC-AUC, AP, and confusion matrix."""
    # TODO: Define positive label and undefined-metric behavior.
    # Average precision is the chosen PR-AUC summary; distinguish it from
    # trapezoidal integration. Apply threshold only to discrete predictions.
    raise NotImplementedError("Metric computation is pending.")
