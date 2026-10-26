"""Preprocessing interfaces; implementation is pending.

TODO: Verify whether failed/alive labels repeat across a company's entire
history BEFORE defining any next-year bankruptcy target. Company-level failure
status alone does not establish the failure year.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    import numpy as np

def verify_target_labels(data: pd.DataFrame) -> dict[str, object]:
    """Audit provided company labels and consistency across annual observations."""
    # TODO: Check company/year uniqueness, label transitions, and source documentation.
    raise NotImplementedError("Provided-label audit is pending.")


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
