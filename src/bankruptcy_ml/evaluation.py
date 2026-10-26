"""Model-independent evaluation and observation alignment interfaces."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    import numpy as np

def evaluate_predictions(
    observations: pd.DataFrame,
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    threshold: float,
) -> dict[str, object]:
    """Evaluate aligned predictions using the shared metrics module."""
    # TODO: Validate lengths/IDs and delegate all metric calculations to metrics.py.
    # Choose thresholds on validation data and keep them fixed for test evaluation.
    raise NotImplementedError("Shared evaluation is pending.")


def align_test_observations(
    predictions: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Align model predictions to identical company/prediction-year test IDs."""
    # TODO: Validate uniqueness and matching labels; document intersection coverage.
    raise NotImplementedError("Test-observation alignment is pending.")
