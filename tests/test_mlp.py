"""MLP shapes, training, search space and checkpoints for both input schemas (#70)."""
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import pytest
import torch
import yaml
from sklearn.metrics import average_precision_score

from models.mlp_model import (
    MLPClassifier,
    build_model,
    load_checkpoint,
    predict_proba,
    save_checkpoint,
    suggest_params,
    train_model,
)
from reproducibility import configure_reproducibility
from sequences import tabular_inputs
from validation import FEATURE_COLUMNS

CONFIG = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "config.yaml").read_text())
SPACE = CONFIG["models"]["mlp"]["search_space"]


def make_panel(n_companies=60, seed=0):
    """Failed companies have a falling X1 over their last years; lengths 1-7."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(1, n_companies + 1):
        failed = i % 3 == 0
        n_years = int(rng.integers(1, 8))
        for k, year in enumerate(range(2010 - n_years + 1, 2011)):
            values = rng.normal(size=len(FEATURE_COLUMNS))
            values[0] = (-k if failed else k) + rng.normal(scale=0.1)
            rows.append({"company_name": f"C_{i}", "status_label": "failed" if failed else "alive",
                         "year": year, **dict(zip(FEATURE_COLUMNS, values))})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def tables():
    data = make_panel()
    ids = sorted(data.company_name.unique(), key=lambda c: int(c[2:]))
    out = {}
    for condition in ("latest_year", "five_year"):
        x_train, m_train = tabular_inputs(data, condition=condition, companies=ids[:40])
        x_val, m_val = tabular_inputs(data, condition=condition, companies=ids[40:])
        out[condition] = (x_train, m_train["target"], x_val, m_val["target"])
    return out


@pytest.mark.parametrize("condition, columns", [("latest_year", 18), ("five_year", 100)])
def test_shapes_and_finite_probabilities(tables, condition, columns):
    x_train, *_ = tables[condition]
    assert x_train.shape[1] == columns
    torch.manual_seed(0)
    model = MLPClassifier(columns, 8, num_layers=3, dropout=0.2)
    assert sum(isinstance(m, torch.nn.Linear) for m in model.modules()) == 4  # 3 hidden + head
    logits = model(torch.tensor(x_train.to_numpy(np.float32)))
    assert logits.shape == (len(x_train),) and torch.isfinite(logits).all()
    proba = predict_proba(model, x_train)
    assert proba.shape == (len(x_train),) and ((proba > 0) & (proba < 1)).all()
    with pytest.raises(ValueError, match="Expected"):
        model(torch.zeros(2, columns + 1))


def test_training_learns_restores_best_epoch_and_is_seeded(tables):
    x_train, y_train, x_val, y_val = tables["five_year"]
    settings = dict(learning_rate=1e-2, weight_decay=1e-6, pos_weight=2.0, batch_size=16,
                    max_epochs=30, patience=5, seed=42)
    runs = []
    for _ in range(2):
        configure_reproducibility(42)
        model = MLPClassifier(100, 16)
        result = train_model(model, x_train, y_train, x_val, y_val, **settings)
        runs.append((result, predict_proba(model, x_val)))
    result, proba = runs[0]
    assert result.best_validation_ap > 0.9  # the falling-X1 trend is learnable
    assert result.best_epoch == max(result.history, key=lambda h: h["validation_ap"])["epoch"]
    assert result.epochs_run <= 30 and (not result.stopped_early or
                                        result.epochs_run - result.best_epoch == 5)
    assert average_precision_score(y_val, proba) == pytest.approx(result.best_validation_ap)
    np.testing.assert_array_equal(proba, runs[1][1])  # same seed, same model


def test_training_rejects_mismatched_columns_and_bad_settings(tables):
    x_train, y_train, _, _ = tables["five_year"]
    x_val, y_val = tables["latest_year"][2:]  # 18 columns instead of 100
    model = MLPClassifier(100, 4)
    with pytest.raises(ValueError, match="different feature columns"):
        train_model(model, x_train, y_train, x_val, y_val, learning_rate=1e-3,
                    weight_decay=0, pos_weight=1, batch_size=8)
    with pytest.raises(ValueError, match="Invalid"):
        train_model(model, *tables["five_year"], learning_rate=1e-3, weight_decay=0,
                    pos_weight=0, batch_size=8)
    with pytest.raises(ValueError, match="0/1"):
        train_model(model, x_train, y_train * 2, *tables["five_year"][2:], learning_rate=1e-3,
                    weight_decay=0, pos_weight=1, batch_size=8)


def test_search_space_matches_gru_is_seeded_and_within_ranges():
    assert SPACE == CONFIG["models"]["gru"]["search_space"]  # equal opportunity

    def sample(seed):
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
        out = []
        for _ in range(12):
            trial = study.ask()
            params = suggest_params(trial, SPACE, max_pos_weight=13.74)
            study.tell(trial, params["learning_rate"])
            out.append(params)
        return out

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    first = sample(42)
    assert first == sample(42)
    for p in first:
        assert p["hidden_size"] in SPACE["hidden_size"] and p["num_layers"] in SPACE["num_layers"]
        assert p["batch_size"] in SPACE["batch_size"]
        assert 0 <= p["dropout"] <= 0.5 and 1e-4 <= p["learning_rate"] <= 1e-2
        assert 1e-6 <= p["weight_decay"] <= 1e-3 and 1 <= p["pos_weight"] <= 13.74
    model = build_model(first[0], input_size=100)
    assert model.config["hidden_size"] == first[0]["hidden_size"]
    with pytest.raises(ValueError):
        suggest_params(optuna.trial.FixedTrial({}), SPACE, max_pos_weight=0.5)


def test_checkpoint_roundtrip_and_schema_guard(tables, tmp_path):
    x_train, _, x_val, _ = tables["five_year"]
    columns = list(x_train.columns)
    torch.manual_seed(0)
    model = MLPClassifier(100, 8, num_layers=2, dropout=0.3)
    path = tmp_path / "mlp.pt"
    save_checkpoint(model, path, condition="five_year", feature_columns=columns, extra={"threshold": 0.4})
    loaded, extra = load_checkpoint(path, condition="five_year", feature_columns=columns)
    assert extra == {"threshold": 0.4} and loaded.config == model.config
    np.testing.assert_array_equal(predict_proba(model, x_val), predict_proba(loaded, x_val))
    with pytest.raises(ValueError, match="another condition"):
        load_checkpoint(path, condition="five_year", feature_columns=columns[::-1])
    with pytest.raises(Exception):
        save_checkpoint(model, path, condition="latest_year", feature_columns=columns)
