"""GRU histories: window selection, order, padding, packing and cohorts."""

import numpy as np
import pandas as pd
import pytest
import torch

from sequences import (
    cohort_mask,
    create_histories,
    history_coverage,
    pack_histories,
    valid_step_mask,
    valid_steps,
)
from validation import FEATURE_COLUMNS


@pytest.fixture
def panel() -> pd.DataFrame:
    """C_1: 6 consecutive years; C_2: gap (2005, 2006, 2008), failed;
    C_10: a single year. Feature X1 = year, so positions are easy to check."""
    years = {"C_1": list(range(2010, 2016)), "C_2": [2005, 2006, 2008], "C_10": [2012]}
    rows = [
        (c, "failed" if c == "C_2" else "alive", y)
        for c, ys in years.items() for y in ys
    ]
    frame = pd.DataFrame(rows, columns=["company_name", "status_label", "year"])
    for j, column in enumerate(FEATURE_COLUMNS):
        frame[column] = frame["year"].astype(float) + j / 100
    # Shuffle so nothing depends on file order.
    return frame.sample(frac=1, random_state=0).reset_index(drop=True)


def x1(histories, company):
    row = histories.metadata.index[histories.metadata["company_name"].eq(company)][0]
    return histories.features[row, :, 0].tolist()


def test_windows_are_last_k_recorded_years_oldest_first(panel):
    h = create_histories(panel, lookback=3)
    assert h.metadata["company_name"].tolist() == ["C_1", "C_2", "C_10"]
    assert x1(h, "C_1") == [2013, 2014, 2015]
    # The 2007 gap is skipped, not filled.
    assert x1(h, "C_2") == [2005, 2006, 2008]
    # Short history: real step first, zero padding after it.
    assert x1(h, "C_10") == [2012, 0, 0]
    assert h.lengths.tolist() == [3, 3, 1]
    assert h.features.shape == (3, 3, len(FEATURE_COLUMNS))
    assert h.metadata["window_has_gap"].tolist() == [False, True, False]


def test_anchor_target_and_last_real_step_do_not_depend_on_k(panel):
    by_k = [create_histories(panel, lookback=k) for k in range(1, 6)]
    for h in by_k:
        assert h.metadata["anchor_year"].tolist() == [2015, 2008, 2012]
        assert h.targets.tolist() == [0, 1, 0]
        last_real = h.features[np.arange(3), h.lengths - 1, 0]
        assert last_real.tolist() == [2015, 2008, 2012]
    assert by_k[4].metadata["full_window"].tolist() == [True, False, False]


def test_feature_order_and_no_ids_in_inputs(panel):
    h = create_histories(panel, lookback=2)
    assert h.feature_columns == FEATURE_COLUMNS
    # X18 is the last column and carries year + 0.17.
    assert h.features[0, -1, -1] == pytest.approx(2015.17, abs=1e-3)


def test_padding_is_excluded_from_fitting_data(panel):
    h = create_histories(panel, lookback=5)
    mask = valid_step_mask(h)
    assert mask.sum() == h.lengths.sum() == 5 + 3 + 1
    assert (h.features[~mask] == 0).all()
    assert len(valid_steps(h)) == 9 and (valid_steps(h)[:, 0] > 0).all()


def test_cohorts(panel):
    h = create_histories(panel, lookback=3)
    assert cohort_mask(h, "all_companies").tolist() == [True, True, True]
    assert cohort_mask(h, "at_least_k_years").tolist() == [True, True, False]
    with pytest.raises(ValueError):
        cohort_mask(h, "something_else")


def test_restricting_to_split_companies(panel):
    h = create_histories(panel, lookback=2, companies=["C_10", "C_2"])
    assert h.metadata["company_name"].tolist() == ["C_2", "C_10"]
    with pytest.raises(ValueError, match="Unknown"):
        create_histories(panel, lookback=2, companies=["C_404"])


@pytest.mark.parametrize("bad", [0, 6, 2.5])
def test_invalid_lookback(panel, bad):
    with pytest.raises(ValueError, match="lookback"):
        create_histories(panel, lookback=bad)


def test_packed_gru_ignores_padding_and_ends_at_anchor(panel):
    torch.manual_seed(0)
    gru = torch.nn.GRU(len(FEATURE_COLUMNS), 4, batch_first=True)
    h = create_histories(panel, lookback=5)
    features = torch.from_numpy(h.features) / 2000

    _, h_n = gru(pack_histories(features, h.lengths))
    # Changing padding values must not change the final state.
    noisy = features.clone()
    noisy[torch.from_numpy(~valid_step_mask(h))] = 99.0
    _, h_noisy = gru(pack_histories(noisy, h.lengths))
    torch.testing.assert_close(h_n, h_noisy)
    # Final state equals running the GRU on only the real steps of each company.
    for i, length in enumerate(h.lengths):
        _, alone = gru(features[i : i + 1, :length])
        torch.testing.assert_close(h_n[:, i], alone[:, 0])


def test_pack_rejects_bad_lengths():
    with pytest.raises(ValueError, match="lengths"):
        pack_histories(torch.zeros(2, 3, 1), np.array([0, 3]))


def test_history_coverage_counts(panel):
    split = pd.DataFrame({"company_name": ["C_1", "C_2", "C_10"], "split": ["train", "test", "test"]})
    coverage = history_coverage(panel, split)
    assert coverage["3"]["all"] == {
        "n_companies": 3, "full_window": 2, "padded": 1, "window_has_gap": 1,
        "failed_full_window": 1, "failed_padded": 0,
    }
    assert coverage["5"]["test"]["padded"] == 2
