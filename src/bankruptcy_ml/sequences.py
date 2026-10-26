"""Interfaces for aligned financial histories with 1–5 annual observations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    import numpy as np

def create_histories(
    data: pd.DataFrame, *, lookback: int, prediction_horizon: int = 1
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Return histories, verified targets, and company/prediction-year identifiers."""
    # TODO: Support lookback 1–5; sort by company/year, require consecutive years,
    # exclude future features, and handle insufficient histories explicitly.
    # Target construction is blocked until event timing is verified.
    raise NotImplementedError("Financial-history construction is pending.")


def flatten_histories(histories: np.ndarray) -> np.ndarray:
    """Convert [observations, years, features] to baseline model inputs."""
    # TODO: Document time/feature order and retain alignment with observation IDs.
    raise NotImplementedError("History flattening is pending.")
