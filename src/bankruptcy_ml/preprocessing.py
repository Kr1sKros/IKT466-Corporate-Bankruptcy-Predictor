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
    """Audit label meanings, within-company consistency, and event timing."""
    # TODO: Check company/year uniqueness, label transitions, and source documentation.
    raise NotImplementedError("Verify labels and prediction-target feasibility first.")


def split_dataset(
    data: pd.DataFrame, *, strategy: str, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return train, validation, and test partitions under an agreed protocol."""
    # TODO: Define cutoffs, company overlap policy, and horizon-aware boundaries.
    # Prevent future information and overlapping target windows leaking across splits.
    raise NotImplementedError("Splitting strategy has not been agreed yet.")


def fit_preprocessor(training_features: pd.DataFrame) -> object:
    """Fit missing-value handling and scaling using training features only."""
    # TODO: Choose imputation/scaling and exclude identifiers and target columns.
    raise NotImplementedError("Training-only preprocessing is pending.")


def transform_features(features: pd.DataFrame, preprocessor: object) -> np.ndarray:
    """Transform features using an already fitted training preprocessor."""
    # TODO: Preserve feature order and never refit on validation/test observations.
    raise NotImplementedError("Feature transformation is pending.")
