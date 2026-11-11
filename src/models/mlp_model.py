"""MLP classifier, training loop and Optuna search space (#70).

Inputs are the scaled tabular features from ``sequences.flatten_histories`` +
``preprocessing.transform_histories(..., model='mlp')``: latest_year has 18
columns (X1-X18 at t); five_year has 100 (5 recency slots x X1-X18,
years_before_anchor and present). Column order is checked against
``sequences.tabular_columns``.

Architecture: ``num_layers`` hidden layers of equal ``hidden_size``, each
Linear -> ReLU -> Dropout, then Linear(hidden, 1) -> one logit;
``sigmoid(logit)`` is the bankruptcy probability.

Training follows the same rules as the GRU (``configs/config.yaml`` training):
Adam, BCE with ``pos_weight`` on the training loss only, early stopping on
unweighted validation AP (max 200 epochs, patience 20), best weights restored.
Optuna (#56) chooses the settings in ``suggest_params``.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import TensorDataset

from reproducibility import make_data_loader
from sequences import validate_tabular_columns

if TYPE_CHECKING:
    import optuna

CHECKPOINT_VERSION = 1
PREDICT_BATCH_SIZE = 4096


class MLPClassifier(nn.Module):
    """Binary MLP over one row of tabular features per company; returns logits."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 dropout: float = 0.0) -> None:
        super().__init__()
        if input_size < 1 or hidden_size < 1 or num_layers < 1 or not 0 <= dropout < 1:
            raise ValueError("Invalid MLP architecture settings")
        self.config = {"input_size": input_size, "hidden_size": hidden_size,
                       "num_layers": num_layers, "dropout": dropout}
        layers: list[nn.Module] = []
        width = input_size
        for _ in range(num_layers):
            layers += [nn.Linear(width, hidden_size), nn.ReLU(), nn.Dropout(dropout)]
            width = hidden_size
        self.hidden = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: (batch, input_size) -> logits (batch,)."""
        if features.dim() != 2 or features.shape[1] != self.config["input_size"]:
            raise ValueError(f"Expected (batch, {self.config['input_size']}), got {tuple(features.shape)}")
        return self.head(self.hidden(features)).squeeze(-1)


def _features(features: pd.DataFrame | np.ndarray) -> torch.Tensor:
    tensor = torch.as_tensor(np.asarray(features, dtype=np.float32))
    if tensor.dim() != 2 or not torch.isfinite(tensor).all():
        raise ValueError("Features must be a finite 2-D table")
    return tensor


def _targets(targets, n_rows: int) -> torch.Tensor:
    tensor = torch.as_tensor(np.asarray(targets, dtype=np.float32))
    if tensor.shape != (n_rows,) or not torch.isin(tensor, torch.tensor([0.0, 1.0])).all():
        raise ValueError("Targets must be one 0/1 value per row")
    return tensor


@torch.no_grad()
def predict_proba(model: MLPClassifier, features: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Bankruptcy probabilities in row order (eval mode)."""
    x = _features(features)
    was_training = model.training
    model.eval()
    try:
        logits = [model(x[i:i + PREDICT_BATCH_SIZE]) for i in range(0, len(x), PREDICT_BATCH_SIZE)]
    finally:
        model.train(was_training)
    probabilities = torch.sigmoid(torch.cat(logits)).numpy()
    if not np.isfinite(probabilities).all():
        raise FloatingPointError("Non-finite probabilities")
    return probabilities


def validation_average_precision(model: MLPClassifier, features, targets) -> float:
    """Unweighted AP on validation rows (the selection metric)."""
    targets = np.asarray(targets)
    if len(np.unique(targets)) != 2:
        raise ValueError("Validation AP needs both classes")
    return float(average_precision_score(targets, predict_proba(model, features)))


@dataclass
class TrainingResult:
    best_epoch: int
    best_validation_ap: float
    epochs_run: int
    stopped_early: bool
    seconds: float
    history: list[dict[str, float]] = field(default_factory=list)


def train_model(
    model: MLPClassifier,
    x_train: pd.DataFrame | np.ndarray,
    y_train,
    x_validation: pd.DataFrame | np.ndarray,
    y_validation,
    *,
    learning_rate: float,
    weight_decay: float,
    pos_weight: float,
    batch_size: int,
    max_epochs: int = 200,
    patience: int = 20,
    seed: int = 42,
) -> TrainingResult:
    """Train with Adam and weighted BCE; keep the epoch with the best validation AP.

    ``pos_weight`` weights only the training loss; validation AP is unweighted.
    Stops after ``patience`` epochs without improvement or at ``max_epochs`` and
    restores the best weights into ``model``. Call
    ``reproducibility.configure_reproducibility(seed)`` before creating the model.
    """
    if isinstance(x_train, pd.DataFrame) and isinstance(x_validation, pd.DataFrame) \
            and list(x_train.columns) != list(x_validation.columns):
        raise ValueError("Train and validation use different feature columns")
    if pos_weight <= 0 or batch_size < 1 or max_epochs < 1 or patience < 1:
        raise ValueError("Invalid training settings")
    features = _features(x_train)
    targets = _targets(y_train, len(features))
    loader = make_data_loader(TensorDataset(features, targets), seed=seed,
                              batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(pos_weight)))

    start = time.perf_counter()
    best_ap, best_epoch, best_state, history = -1.0, 0, None, []
    for epoch in range(1, max_epochs + 1):
        model.train()
        total, count = 0.0, 0
        for batch_features, batch_targets in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch_features), batch_targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss in epoch {epoch}")
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch_targets)
            count += len(batch_targets)
        ap = validation_average_precision(model, x_validation, y_validation)
        history.append({"epoch": epoch, "train_loss": total / count, "validation_ap": ap})
        if ap > best_ap:
            best_ap, best_epoch, best_state = ap, epoch, copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= patience:
            break
    model.load_state_dict(best_state)
    return TrainingResult(best_epoch=best_epoch, best_validation_ap=best_ap,
                          epochs_run=len(history), stopped_early=len(history) < max_epochs,
                          seconds=time.perf_counter() - start, history=history)


# --- Optuna search space ---------------------------------------------------------

def suggest_params(trial: "optuna.Trial", space: dict[str, Any], *, max_pos_weight: float) -> dict[str, Any]:
    """Suggest one MLP configuration from the declared ``space``
    (``configs/config.yaml`` models.mlp.search_space, same ranges as the GRU).

    ``max_pos_weight`` must be the training negative/positive ratio
    (``leakage.class_weight_bounds``); the range starts at 1 (unweighted).
    """
    if max_pos_weight < 1:
        raise ValueError("max_pos_weight must be at least 1")
    low_lr, high_lr = space["learning_rate"]
    low_wd, high_wd = space["weight_decay"]
    low_do, high_do = space["dropout"]
    return {
        "hidden_size": trial.suggest_categorical("hidden_size", space["hidden_size"]),
        "num_layers": trial.suggest_categorical("num_layers", space["num_layers"]),
        "dropout": trial.suggest_float("dropout", low_do, high_do),
        "learning_rate": trial.suggest_float("learning_rate", low_lr, high_lr, log=True),
        "weight_decay": trial.suggest_float("weight_decay", low_wd, high_wd, log=True),
        "pos_weight": trial.suggest_float("pos_weight", 1.0, float(max_pos_weight), log=True),
        "batch_size": trial.suggest_categorical("batch_size", space["batch_size"]),
    }


def build_model(params: dict[str, Any], input_size: int) -> MLPClassifier:
    return MLPClassifier(input_size, params["hidden_size"], params["num_layers"], params["dropout"])


# --- Checkpoints ------------------------------------------------------------------

def save_checkpoint(model: MLPClassifier, path: str | Path, *, condition: str,
                    feature_columns: list[str], extra: dict[str, Any] | None = None) -> None:
    """Save weights, architecture, condition and exact column order."""
    validate_tabular_columns(feature_columns, condition)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"version": CHECKPOINT_VERSION, "config": model.config, "condition": condition,
                "feature_columns": list(feature_columns), "state_dict": model.state_dict(),
                "extra": extra or {}}, path)


def load_checkpoint(path: str | Path, *, condition: str, feature_columns: list[str]) -> tuple[MLPClassifier, dict]:
    """Load a checkpoint; refuse one saved for another condition or column order."""
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if checkpoint["version"] != CHECKPOINT_VERSION:
        raise ValueError("Unsupported MLP checkpoint version")
    if checkpoint["condition"] != condition or checkpoint["feature_columns"] != list(feature_columns):
        raise ValueError("Checkpoint was saved for another condition or column order")
    model = MLPClassifier(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint["extra"]
