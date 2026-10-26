"""Latest-year MLP baseline construction and training placeholders."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader


class MLPClassifier(nn.Module):
    """Planned binary classifier accepting [batch, financial_features]."""

    def __init__(self, input_size: int, hidden_sizes: tuple[int, ...]) -> None:
        """Declare layer settings; the network is not implemented."""
        super().__init__()
        # TODO: Define fully connected layers, activations, and binary logits.
        raise NotImplementedError("MLP architecture is pending.")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return bankruptcy logits from latest-year features only."""
        raise NotImplementedError("MLP forward pass is pending.")


def train_model(
    model: MLPClassifier,
    training_loader: DataLoader,
    validation_loader: DataLoader,
    *,
    epochs: int,
    learning_rate: float,
    seed: int = 42,
) -> MLPClassifier:
    """Train on latest-year observations with validation-based model selection."""
    # TODO: Define optimizer, loss, device, and early stopping after data checks.
    raise NotImplementedError("MLP training is pending.")
