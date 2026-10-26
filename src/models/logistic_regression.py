"""LogisticRegression construction and training interfaces."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from sklearn.linear_model import LogisticRegression

def build_model(*, seed: int = 42, hyperparameters: dict[str, object] | None = None) -> LogisticRegression:
    """Build an untrained model from explicitly supplied settings."""
    # TODO: Choose hyperparameters using training/validation data only.
    raise NotImplementedError("Model construction is pending.")


def train_model(model: LogisticRegression, features: np.ndarray, targets: np.ndarray) -> LogisticRegression:
    """Fit the model on latest-year training features only."""
    # TODO: Implement training after target, splitting, and preprocessing are verified.
    raise NotImplementedError("Model training is pending.")
