"""Plotting interfaces for measured experiment outputs only."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pathlib import Path

if TYPE_CHECKING:
    import pandas as pd
    import numpy as np

def plot_confusion_matrix(matrix: np.ndarray, output_path: Path) -> None:
    """Save a confusion matrix plot with explicit label ordering."""
    # TODO: Plot actual computed counts and save the figure.
    raise NotImplementedError("Confusion matrix plotting is pending.")


def plot_precision_recall(
    y_true: np.ndarray, scores: dict[str, np.ndarray], output_path: Path
) -> None:
    """Save precision-recall curves for aligned model predictions."""
    # TODO: Use consistent positive labels and include prevalence as a reference.
    raise NotImplementedError("Precision-recall plotting is pending.")


def plot_lookback_performance(results: pd.DataFrame, output_path: Path) -> None:
    """Plot measured performance against 1–5 years of financial history."""
    # TODO: Show the metric, model, sample coverage, and lookback explicitly.
    raise NotImplementedError("Historical-data plotting is pending.")


def plot_model_comparison(results: pd.DataFrame, output_path: Path) -> None:
    """Save a comparison of measured model metrics on identical test observations."""
    # TODO: Plot real experiment records only.
    raise NotImplementedError("Model comparison plotting is pending.")
