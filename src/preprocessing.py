"""Preprocessing interfaces; implementation is pending.

Dataset validation (schema, duplicates, labels, history) lives in
``validation.py``; this module only splits, imputes and scales.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    import numpy as np

def split_dataset(
    data: pd.DataFrame, *, train_fraction: float, validation_fraction: float, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Partition by company with no company shared across train/validation/test."""
    # TODO: Split unique company IDs, preserve every company's rows together,
    # validate fractions, and agree class stratification. Save IDs for all models.
    raise NotImplementedError("Company-level splitting is pending.")


def fit_preprocessor(training_features: pd.DataFrame) -> object:
    """Fit missing-value handling and scaling using training features only."""
    # TODO: Choose imputation/scaling and exclude identifiers and target columns.
    raise NotImplementedError("Training-only preprocessing is pending.")


def transform_features(features: pd.DataFrame, preprocessor: object) -> np.ndarray:
    """Transform features using an already fitted training preprocessor."""
    # TODO: Preserve feature order and never refit on validation/test observations.
    raise NotImplementedError("Feature transformation is pending.")
