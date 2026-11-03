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
    """Evaluate vectors in observation row order, with a fixed threshold.

    Each company must occur once at its final-year anchor. If a label column is
    included, it must match y_true exactly. This checks positional alignment;
    callers must join separately sourced predictions by company/anchor first.
    """
    import numpy as np
    from metrics import compute_metrics

    required = ['company_name', 'year']
    if any(column not in observations for column in required):
        raise ValueError("Observations require company_name and year columns")
    if observations.columns.duplicated().any():
        raise ValueError("Observation column names must be unique")
    if observations[required].isna().any().any() or observations['company_name'].duplicated().any():
        raise ValueError("Observation IDs must be non-null and companies unique")
    if len(observations) != len(y_true):
        raise ValueError("Observation IDs and labels must have matching lengths")
    if 'label' in observations and not np.array_equal(observations['label'].to_numpy(), y_true):
        raise ValueError("Observation labels do not match y_true")
    return compute_metrics(y_true, y_score, threshold=threshold)


def align_test_observations(
    predictions: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Align model predictions to identical company/anchor-year test IDs."""
    # TODO: Validate uniqueness and matching labels; document intersection coverage.
    raise NotImplementedError("Test-observation alignment is pending.")
