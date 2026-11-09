"""Inputs for latest_year and full_history: GRU sequences and flattened tables."""

import numpy as np
import pandas as pd
import pytest
import torch

from sequences import (
    FULL_HISTORY_CAPACITY,
    OFFSET_COLUMN,
    create_histories,
    flatten_histories,
    history_coverage,
    pack_histories,
    step_columns,
    tabular_columns,
    tabular_inputs,
    valid_step_mask,
    valid_steps,
    validate_tabular_columns,
)
from validation import FEATURE_COLUMNS, DataValidationError

N_FEATURES = len(FEATURE_COLUMNS)


def make_panel(years: dict[str, list[int]], failed=("C_2",)) -> pd.DataFrame:
    """X1 = year, X18 = year + 0.17, so positions are easy to check."""
    rows = [(c, "failed" if c in failed else "alive", y) for c, ys in years.items() for y in ys]
    frame = pd.DataFrame(rows, columns=["company_name", "status_label", "year"])
    for j, column in enumerate(FEATURE_COLUMNS):
        frame[column] = frame["year"].astype(float) + j / 100
    # Shuffle so nothing depends on file order.
    return frame.sample(frac=1, random_state=0).reset_index(drop=True)


@pytest.fixture
def panel() -> pd.DataFrame:
    """C_1: 7 consecutive years; C_2: gap (2005, 2006, 2008), failed;
    C_10: a single year."""
    return make_panel({"C_1": list(range(2009, 2016)), "C_2": [2005, 2006, 2008], "C_10": [2012]})


def gru_column(h, company, column):
    row = h.metadata.index[h.metadata["company_name"].eq(company)][0]
    return h.features[row, :, h.step_columns.index(column)].tolist()


# --- GRU layout -------------------------------------------------------------

def test_full_history_keeps_every_record_oldest_first(panel):
    h = create_histories(panel, condition="full_history")
    assert h.metadata["company_name"].tolist() == ["C_1", "C_2", "C_10"]
    assert h.features.shape == (3, FULL_HISTORY_CAPACITY, N_FEATURES + 1)
    assert h.lengths.tolist() == [7, 3, 1]  # more than five records, nothing cut
    pad = FULL_HISTORY_CAPACITY
    assert gru_column(h, "C_1", "X1") == list(range(2009, 2016)) + [0] * (pad - 7)
    # The 2007 gap is skipped, not filled, and visible through the offsets.
    assert gru_column(h, "C_2", "X1")[:3] == [2005, 2006, 2008]
    assert gru_column(h, "C_2", OFFSET_COLUMN)[:3] == [3, 2, 0]
    assert gru_column(h, "C_10", "X1") == [2012] + [0] * (pad - 1)
    assert h.metadata["has_gap"].tolist() == [False, True, False]


def test_latest_year_is_one_step_without_temporal_fields(panel):
    h = create_histories(panel, condition="latest_year")
    assert h.features.shape == (3, 1, N_FEATURES)
    assert h.step_columns == FEATURE_COLUMNS
    assert h.features[:, 0, 0].tolist() == [2015, 2008, 2012]
    assert h.lengths.tolist() == [1, 1, 1]


def test_anchor_target_and_last_real_step_are_the_same_in_both_conditions(panel):
    latest = create_histories(panel, condition="latest_year")
    full = create_histories(panel, condition="full_history")
    for h in (latest, full):
        assert h.metadata["anchor_year"].tolist() == [2015, 2008, 2012]
        assert h.targets.tolist() == [0, 1, 0]
    last_real = full.features[np.arange(3), full.lengths - 1, :N_FEATURES]
    np.testing.assert_array_equal(last_real, latest.features[:, 0, :])
    assert (full.features[np.arange(3), full.lengths - 1, -1] == 0).all()  # offset 0 at t


def test_capacity_overflow_is_rejected_not_truncated():
    long = make_panel({"C_1": list(range(1990, 2011))}, failed=())  # 21 records
    with pytest.raises(DataValidationError, match="refusing to truncate"):
        create_histories(long, condition="full_history")
    exactly = make_panel({"C_1": list(range(1991, 2011))}, failed=())  # 20 records
    assert create_histories(exactly, condition="full_history").lengths.tolist() == [20]
    # Latest year never depends on the capacity.
    assert create_histories(long, condition="latest_year").lengths.tolist() == [1]


def test_padding_is_excluded_from_fitting_data(panel):
    h = create_histories(panel, condition="full_history")
    mask = valid_step_mask(h)
    assert mask.sum() == h.lengths.sum() == 7 + 3 + 1
    assert (h.features[~mask] == 0).all()
    assert len(valid_steps(h)) == 11 and (valid_steps(h)[:, 0] > 0).all()


def test_restricting_to_partition_companies(panel):
    h = create_histories(panel, condition="full_history", companies=["C_10", "C_2"])
    assert h.metadata["company_name"].tolist() == ["C_2", "C_10"]
    with pytest.raises(ValueError, match="Unknown"):
        create_histories(panel, condition="full_history", companies=["C_404"])


def test_unknown_condition(panel):
    with pytest.raises(ValueError, match="condition"):
        create_histories(panel, condition="last_five")


def test_packed_gru_ignores_padding_and_ends_at_anchor(panel):
    torch.manual_seed(0)
    h = create_histories(panel, condition="full_history")
    gru = torch.nn.GRU(len(h.step_columns), 4, batch_first=True)
    features = torch.from_numpy(h.features) / 2000

    _, h_n = gru(pack_histories(features, h.lengths))
    noisy = features.clone()
    noisy[torch.from_numpy(~valid_step_mask(h))] = 99.0
    _, h_noisy = gru(pack_histories(noisy, h.lengths))
    torch.testing.assert_close(h_n, h_noisy)
    for i, length in enumerate(h.lengths):
        _, alone = gru(features[i : i + 1, :length])
        torch.testing.assert_close(h_n[:, i], alone[:, 0])


def test_pack_rejects_bad_lengths():
    with pytest.raises(ValueError, match="lengths"):
        pack_histories(torch.zeros(2, 3, 1), np.array([0, 3]))


# --- Flattened layout ---------------------------------------------------------

def test_latest_year_table_is_x1_to_x18_at_t(panel):
    features, meta = tabular_inputs(panel, condition="latest_year")
    assert list(features.columns) == FEATURE_COLUMNS
    assert features["X1"].tolist() == [2015, 2008, 2012]
    assert features["X18"].tolist() == pytest.approx([2015.17, 2008.17, 2012.17])
    assert meta["company_name"].tolist() == ["C_1", "C_2", "C_10"]


def test_full_history_table_is_recency_aligned_with_presence(panel):
    features, _ = tabular_inputs(panel, condition="full_history")
    assert features.shape == (3, FULL_HISTORY_CAPACITY * (N_FEATURES + 2))
    # r0 is always the anchor year t.
    assert features["X1_r0"].tolist() == [2015, 2008, 2012]
    assert features["years_before_anchor_r0"].tolist() == [0, 0, 0]
    # C_2: r1 = 2006 (offset 2, the gap is visible), r2 = 2005, r3 empty.
    c2 = features.iloc[1]
    assert (c2["X1_r1"], c2["years_before_anchor_r1"], c2["present_r1"]) == (2006, 2, 1)
    assert (c2["X1_r2"], c2["years_before_anchor_r2"], c2["present_r2"]) == (2005, 3, 1)
    assert (c2["X1_r3"], c2["years_before_anchor_r3"], c2["present_r3"]) == (0, 0, 0)
    presence = features[[f"present_r{r}" for r in range(FULL_HISTORY_CAPACITY)]].sum(axis=1)
    assert presence.tolist() == [7, 3, 1]
    # Empty slots come first (oldest end) and are all zero.
    assert features.columns[0] == "X1_r19" and features.columns[-1] == "present_r0"
    assert (features.iloc[2, : -(N_FEATURES + 2)] == 0).all()


def test_flattened_and_gru_hold_the_same_information(panel):
    h = create_histories(panel, condition="full_history")
    features, _ = flatten_histories(h)
    for i, length in enumerate(h.lengths):
        for step in range(length):
            r = length - 1 - step
            slot = features.loc[i, [f"{c}_r{r}" for c in h.step_columns]].to_numpy()
            np.testing.assert_array_equal(slot, h.features[i, step])


def test_single_year_company_has_same_financials_in_both_conditions(panel):
    latest, _ = tabular_inputs(panel, condition="latest_year")
    full, _ = tabular_inputs(panel, condition="full_history")
    np.testing.assert_array_equal(
        latest.iloc[2].to_numpy(), full.loc[2, [f"{c}_r0" for c in FEATURE_COLUMNS]].to_numpy()
    )


def test_columns_are_deterministic_and_validated(panel):
    assert tabular_columns("full_history") == tabular_columns("full_history")
    a, _ = tabular_inputs(panel, condition="full_history")
    b, _ = tabular_inputs(panel.sample(frac=1, random_state=3), condition="full_history")
    pd.testing.assert_frame_equal(a, b)
    with pytest.raises(DataValidationError, match="non-features"):
        validate_tabular_columns([*FEATURE_COLUMNS, "year"], "latest_year")
    with pytest.raises(DataValidationError, match="schema"):
        validate_tabular_columns(list(reversed(FEATURE_COLUMNS)), "latest_year")
    assert step_columns("full_history") == [*FEATURE_COLUMNS, OFFSET_COLUMN]


def test_history_coverage(panel):
    split = pd.DataFrame({"company_name": ["C_1", "C_2", "C_10"], "split": ["train", "test", "test"]})
    coverage = history_coverage(panel, split)
    assert coverage["capacity"] == FULL_HISTORY_CAPACITY
    assert coverage["all"]["records_total"] == 11
    assert coverage["all"]["companies_with_gaps"] == 1
    assert coverage["all"]["records_per_company"]["max"] == 7
    assert coverage["test"]["companies_per_length"] == {"1": 1, "3": 1}
    assert coverage["train"]["padded_slots_total"] == FULL_HISTORY_CAPACITY - 7
