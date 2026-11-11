"""GRU classifier, training loop and Optuna search space (#55).

Inputs are the scaled histories from ``sequences.create_histories`` +
``preprocessing.transform_histories``: ``features`` (batch, steps, inputs) with
real steps oldest to newest followed by zero padding, and ``lengths`` (real
steps per company). latest_year has 1 step of 18 inputs; five_year up to 5
steps of 19 inputs (X1-X18 + years_before_anchor).

The GRU reads packed sequences, so padding is never seen and the final hidden
state is the state after the record at the anchor year t. Head: dropout ->
Linear(hidden, 1) -> one logit; ``sigmoid(logit)`` is the bankruptcy probability.

The model is bound to one condition: training and prediction refuse histories
of another condition, step-column order or step count, also after reload.

Optuna (#56) chooses the settings in ``suggest_params``; the network weights
are learned by ``train_model`` inside each trial (Adam, early stopping on
validation AP, best checkpoint restored).
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import TensorDataset

from reproducibility import make_data_loader
from sequences import FIVE_YEAR_WINDOW, pack_histories, step_columns

if TYPE_CHECKING:
    import optuna

    from sequences import Histories

CHECKPOINT_VERSION = 1
PREDICT_BATCH_SIZE = 4096


class GRUClassifier(nn.Module):
    """Binary GRU over packed annual histories; returns one logit per company.

    ``condition`` fixes the exact step columns (``step_columns``) and step count.
    """

    def __init__(self, condition: str, hidden_size: int, num_layers: int = 1,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.step_columns = step_columns(condition)
        self.max_steps = 1 if condition == "latest_year" else FIVE_YEAR_WINDOW
        input_size = len(self.step_columns)
        if hidden_size < 1 or num_layers < 1 or not 0 <= dropout < 1:
            raise ValueError("Invalid GRU architecture settings")
        self.condition = condition
        self.config = {"condition": condition, "hidden_size": hidden_size,
                       "num_layers": num_layers, "dropout": dropout}
        # PyTorch applies GRU dropout only between stacked layers.
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """features: (batch, steps, inputs) in ``step_columns`` order; lengths: (batch,) -> logits.

        Use ``predict_proba``/``train_model`` with ``Histories`` so the condition
        and column order are checked; a raw tensor carries no names.
        """
        width = len(self.step_columns)
        if features.dim() != 3 or features.shape[2] != width:
            raise ValueError(f"Expected (batch, steps, {width}), got {tuple(features.shape)}")
        if lengths.shape != (features.shape[0],):
            raise ValueError("lengths must have one entry per company")
        _, h_n = self.gru(pack_histories(features, lengths))
        return self.head(self.dropout(h_n[-1])).squeeze(-1)


def _tensors(model: GRUClassifier, histories: "Histories") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Check condition, step columns and step count against the model, then convert."""
    if histories.condition != model.condition:
        raise ValueError(f"Model is for {model.condition}, histories are {histories.condition}")
    if list(histories.step_columns) != model.step_columns:
        raise ValueError("Histories step columns differ from the model's input order")
    if histories.features.shape[1:] != (model.max_steps, len(model.step_columns)):
        raise ValueError(f"Expected histories of shape (n, {model.max_steps}, {len(model.step_columns)})")
    features = torch.as_tensor(histories.features, dtype=torch.float32)
    lengths = torch.as_tensor(histories.lengths, dtype=torch.int64)
    targets = torch.as_tensor(histories.targets, dtype=torch.float32)
    if not torch.isfinite(features).all():
        raise ValueError("Histories contain non-finite values")
    return features, lengths, targets


@torch.no_grad()
def predict_proba(model: GRUClassifier, histories: "Histories") -> np.ndarray:
    """Bankruptcy probabilities in the row order of ``histories`` (eval mode)."""
    features, lengths, _ = _tensors(model, histories)
    was_training = model.training
    model.eval()
    try:
        logits = [model(features[i:i + PREDICT_BATCH_SIZE], lengths[i:i + PREDICT_BATCH_SIZE])
                  for i in range(0, len(lengths), PREDICT_BATCH_SIZE)]
    finally:
        model.train(was_training)
    probabilities = torch.sigmoid(torch.cat(logits)).numpy()
    if not np.isfinite(probabilities).all():
        raise FloatingPointError("Non-finite probabilities")
    return probabilities


def validation_average_precision(model: GRUClassifier, histories: "Histories") -> float:
    """Unweighted AP on the validation histories (the selection metric)."""
    if len(np.unique(histories.targets)) != 2:
        raise ValueError("Validation AP needs both classes")
    return float(average_precision_score(histories.targets, predict_proba(model, histories)))


@dataclass
class TrainingResult:
    best_epoch: int
    best_validation_ap: float
    epochs_run: int
    stopped_early: bool
    seconds: float
    history: list[dict[str, float]] = field(default_factory=list)


def train_model(
    model: GRUClassifier,
    train: "Histories",
    validation: "Histories",
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
    if pos_weight <= 0 or batch_size < 1 or max_epochs < 1 or patience < 1:
        raise ValueError("Invalid training settings")
    features, lengths, targets = _tensors(model, train)
    _tensors(model, validation)  # same schema check before any training
    loader = make_data_loader(TensorDataset(features, lengths, targets), seed=seed,
                              batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(pos_weight)))

    start = time.perf_counter()
    best_ap, best_epoch, best_state, history = -1.0, 0, None, []
    for epoch in range(1, max_epochs + 1):
        model.train()
        total, count = 0.0, 0
        for batch_features, batch_lengths, batch_targets in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch_features, batch_lengths), batch_targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss in epoch {epoch}")
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch_targets)
            count += len(batch_targets)
        ap = validation_average_precision(model, validation)
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
    """Suggest one GRU configuration from the declared ``space``
    (``configs/config.yaml`` models.gru.search_space).

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


def build_model(params: dict[str, Any], condition: str) -> GRUClassifier:
    return GRUClassifier(condition, params["hidden_size"], params["num_layers"], params["dropout"])


# --- Checkpoints ------------------------------------------------------------------

def save_checkpoint(model: GRUClassifier, path: str | Path, *,
                    extra: dict[str, Any] | None = None) -> None:
    """Save weights, architecture and the model's own condition/input order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"version": CHECKPOINT_VERSION, "config": model.config,
                "step_columns": list(model.step_columns), "state_dict": model.state_dict(),
                "extra": extra or {}}, path)


def load_checkpoint(path: str | Path, *, condition: str) -> tuple[GRUClassifier, dict]:
    """Load a checkpoint; refuse one saved for another condition or input order."""
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if checkpoint["version"] != CHECKPOINT_VERSION:
        raise ValueError("Unsupported GRU checkpoint version")
    if (checkpoint["config"]["condition"] != condition
            or checkpoint["step_columns"] != step_columns(condition)):
        raise ValueError("Checkpoint was saved for another condition or input order")
    model = GRUClassifier(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint["extra"]
