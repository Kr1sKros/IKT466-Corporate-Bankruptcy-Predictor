"""Interfaces for latest-year baselines and 1–5-year GRU histories."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


def select_latest_year(data: pd.DataFrame) -> pd.DataFrame:
    """Select one latest-year observation per company for all baselines."""
    # TODO: Check unique company/year pairs and preserve company IDs and labels.
    # Use the same latest-year anchor for baseline and GRU observations.
    raise NotImplementedError("Latest-year selection is pending.")


def create_histories(
    data: pd.DataFrame, *, lookback: int
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Return GRU histories, provided company labels, and company/anchor-year IDs."""
    # TODO: Support lookback 1–5, anchor at each company's latest year, then add
    # older years backwards in time as described in the README.
    # Define missing-year/short-history handling and tensor order explicitly.
    # Respect company partitions and align evaluation cohorts across lookbacks.
    raise NotImplementedError("Financial-history construction is pending.")
