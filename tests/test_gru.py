"""GRU shapes, packed padding invariance, training, search space and checkpoints (#55)."""
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import pytest
import torch
import yaml

from models.gru_model import (
    GRUClassifier,
    build_model,
    load_checkpoint,
    predict_proba,
    save_checkpoint,
    suggest_params,
    train_model,
)
from reproducibility import configure_reproducibility
from sequences import create_histories, valid_step_mask
from validation import FEATURE_COLUMNS

CONFIG = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "config.yaml").read_text())
SPACE = CONFIG["models"]["gru"]["search_space"]


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
def histories():
    data = make_panel()
    ids = sorted(data.company_name.unique(), key=lambda c: int(c[2:]))
    return {c: (create_histories(data, condition=c, companies=ids[:40]),
                create_histories(data, condition=c, companies=ids[40:]))
            for c in ("latest_year", "five_year")}


@pytest.mark.parametrize("condition, steps, inputs", [("latest_year", 1, 18), ("five_year", 5, 19)])
def test_shapes_and_finite_probabilities(histories, condition, steps, inputs):
    train, _ = histories[condition]
    assert train.features.shape[1:] == (steps, inputs)
    torch.manual_seed(0)
    model = GRUClassifier(inputs, 8, num_layers=2, dropout=0.2)
    logits = model(torch.from_numpy(train.features), torch.from_numpy(train.lengths))
    assert logits.shape == (len(train.lengths),) and torch.isfinite(logits).all()
    proba = predict_proba(model, train)
    assert proba.shape == (len(train.lengths),) and ((proba > 0) & (proba < 1)).all()
    with pytest.raises(ValueError, match="Expected"):
        model(torch.zeros(2, steps, inputs + 1), torch.ones(2, dtype=torch.int64))


def test_padding_never_changes_the_output(histories):
    train, _ = histories["five_year"]
    torch.manual_seed(0)
    model = GRUClassifier(19, 8, num_layers=2).eval()
    features, lengths = torch.from_numpy(train.features), torch.from_numpy(train.lengths)
    noisy = features.clone()
    noisy[torch.from_numpy(~valid_step_mask(train))] = 1e3
    with torch.no_grad():
        torch.testing.assert_close(model(features, lengths), model(noisy, lengths))
        # Final state is the state after the record at t: same as running each
        # company alone on exactly its real steps.
        for i in range(5):
            alone = model(features[i:i + 1, :lengths[i]], lengths[i:i + 1])
            torch.testing.assert_close(model(features, lengths)[i:i + 1], alone)


def test_training_learns_restores_best_epoch_and_is_seeded(histories):
    train, validation = histories["five_year"]
    settings = dict(learning_rate=1e-2, weight_decay=1e-6, pos_weight=2.0, batch_size=16,
                    max_epochs=30, patience=5, seed=42)
    runs = []
    for _ in range(2):
        configure_reproducibility(42)
        model = GRUClassifier(19, 8)
        result = train_model(model, train, validation, **settings)
        runs.append((result, predict_proba(model, validation)))
    result, proba = runs[0]
    assert result.best_validation_ap > 0.9  # the falling-X1 trend is learnable
    assert result.best_epoch == max(result.history, key=lambda h: h["validation_ap"])["epoch"]
    assert result.epochs_run <= 30 and (not result.stopped_early or
                                        result.epochs_run - result.best_epoch == 5)
    from sklearn.metrics import average_precision_score
    assert average_precision_score(validation.targets, proba) == pytest.approx(result.best_validation_ap)
    np.testing.assert_array_equal(proba, runs[1][1])  # same seed, same model


def test_training_rejects_mixed_conditions_and_bad_settings(histories):
    model = GRUClassifier(19, 4)
    with pytest.raises(ValueError, match="conditions"):
        train_model(model, histories["five_year"][0], histories["latest_year"][1],
                    learning_rate=1e-3, weight_decay=0, pos_weight=1, batch_size=8)
    with pytest.raises(ValueError, match="Invalid"):
        train_model(model, *histories["five_year"], learning_rate=1e-3, weight_decay=0,
                    pos_weight=0, batch_size=8)


def test_search_space_is_seeded_and_within_declared_ranges():
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
    model = build_model(first[0], input_size=19)
    assert model.config["hidden_size"] == first[0]["hidden_size"]
    with pytest.raises(ValueError):
        suggest_params(optuna.trial.FixedTrial({}), SPACE, max_pos_weight=0.5)


def test_checkpoint_roundtrip_and_condition_guard(histories, tmp_path):
    train, validation = histories["five_year"]
    torch.manual_seed(0)
    model = GRUClassifier(19, 8, num_layers=2, dropout=0.3)
    path = tmp_path / "gru.pt"
    save_checkpoint(model, path, condition="five_year", step_columns=train.step_columns,
                    extra={"threshold": 0.4})
    loaded, extra = load_checkpoint(path, condition="five_year", step_columns=train.step_columns)
    assert extra == {"threshold": 0.4} and loaded.config == model.config
    np.testing.assert_array_equal(predict_proba(model, validation), predict_proba(loaded, validation))
    with pytest.raises(ValueError, match="another condition"):
        load_checkpoint(path, condition="latest_year", step_columns=train.step_columns)
