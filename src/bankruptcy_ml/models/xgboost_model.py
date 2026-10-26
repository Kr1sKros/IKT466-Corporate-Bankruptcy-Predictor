"""XGBClassifier construction and training interfaces."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from xgboost import XGBClassifier

def build_model(*, seed: int = 42, hyperparameters: dict[str, object] | None = None) -> XGBClassifier:
    """Build an untrained model from explicitly supplied settings."""
    # TODO: Choose hyperparameters using training/validation data only.
    raise NotImplementedError("Model construction is pending.")


def train_model(model: XGBClassifier, features: np.ndarray, targets: np.ndarray) -> XGBClassifier:
    """Fit the model on prepared training observations."""
    # TODO: Implement training after target, splitting, and preprocessing are verified.
    raise NotImplementedError("Model training is pending.")
