"""Company-level split: disjoint, stratified, persisted and never regenerated."""

import numpy as np
import pandas as pd
import pytest

from preprocessing import (
    check_split_matches_data,
    company_table,
    load_splits,
    save_splits,
    split_companies,
    split_summary,
)
from validation import FEATURE_COLUMNS


def make_panel(n_companies: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Synthetic panel: 1-7 years per company, ~15% failed, features irrelevant."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(1, n_companies + 1):
        n_years = int(rng.integers(1, 8))
        label = "failed" if rng.random() < 0.15 else "alive"
        start = int(rng.integers(1999, 2012))
        rows += [(f"C_{c}", label, start + y) for y in range(n_years)]
    frame = pd.DataFrame(rows, columns=["company_name", "status_label", "year"])
    for column in FEATURE_COLUMNS:
        frame[column] = rng.normal(size=len(frame))
    return frame


@pytest.fixture
def panel() -> pd.DataFrame:
    return make_panel()


@pytest.fixture
def split(panel) -> pd.DataFrame:
    return split_companies(panel, train_fraction=0.7, validation_fraction=0.15, seed=42)


def test_every_company_in_exactly_one_partition(panel, split):
    assert split["company_name"].is_unique
    assert set(split["company_name"]) == set(panel["company_name"])
    assert set(split["split"]) == {"train", "validation", "test"}


def test_fractions_and_strata_are_preserved(split):
    summary = split_summary(split)
    assert summary["train"]["company_share"] == pytest.approx(0.70, abs=0.01)
    assert summary["validation"]["company_share"] == pytest.approx(0.15, abs=0.01)
    for name in ("train", "validation", "test"):
        assert summary[name]["failed_share"] == pytest.approx(summary["all"]["failed_share"], abs=0.02)
        for group, share in summary["all"]["history_group_shares"].items():
            assert summary[name]["history_group_shares"][group] == pytest.approx(share, abs=0.03)


def test_same_seed_same_split_different_seed_different_split(panel, split):
    again = split_companies(panel, train_fraction=0.7, validation_fraction=0.15, seed=42)
    other = split_companies(panel, train_fraction=0.7, validation_fraction=0.15, seed=7)
    assert split["split"].tolist() == again["split"].tolist()
    assert split["split"].tolist() != other["split"].tolist()


@pytest.mark.parametrize("train, validation", [(0.9, 0.2), (0.0, 0.5), (0.7, 0.3)])
def test_invalid_fractions_rejected(panel, train, validation):
    with pytest.raises(ValueError, match="Fractions"):
        split_companies(panel, train_fraction=train, validation_fraction=validation, seed=42)


def explicit_panel(years_by_company: dict[str, list[int]], failed=()) -> pd.DataFrame:
    rows = [
        (c, "failed" if c in failed else "alive", y)
        for c, years in years_by_company.items() for y in years
    ]
    frame = pd.DataFrame(rows, columns=["company_name", "status_label", "year"])
    for j, column in enumerate(FEATURE_COLUMNS):
        frame[column] = np.arange(len(frame), dtype=float) + j
    return frame


def test_history_groups_count_recorded_years_including_gaps():
    table = company_table(explicit_panel({
        "C_1": [2005],
        "C_2": [2005, 2006, 2008],  # gap: still 3 recorded years
        "C_3": list(range(2000, 2009)),
    }, failed={"C_2"}))
    assert table["n_years"].tolist() == [1, 3, 9]
    assert table["history_group"].tolist() == ["1", "3", "5+"]
    assert table["anchor_year"].tolist() == [2005, 2008, 2008]
    assert table["target"].tolist() == [0, 1, 0]


def test_saved_split_round_trips_and_checksums_are_stable(tmp_path, panel, split):
    checksums = save_splits(split, tmp_path)
    assert save_splits(split, tmp_path / "again") == checksums
    loaded = load_splits(tmp_path, expected_sha256=checksums)
    check_split_matches_data(loaded, panel)
    merged = loaded.merge(split, on="company_name", suffixes=("", "_orig"))
    assert (merged["split"] == merged["split_orig"]).all()
    assert (merged["anchor_year"] == merged["anchor_year_orig"]).all()


def test_load_never_regenerates_and_detects_changes(tmp_path, panel, split):
    with pytest.raises(FileNotFoundError, match="prepare_data.py"):
        load_splits(tmp_path)
    checksums = save_splits(split, tmp_path)
    with pytest.raises(ValueError, match="differs"):
        load_splits(tmp_path, expected_sha256={**checksums, "test": "0" * 64})
    # A company moved into a second partition must be rejected.
    test_file = tmp_path / "test.csv"
    first_train = (tmp_path / "train.csv").read_text().splitlines()[1]
    test_file.write_text(test_file.read_text() + first_train + "\n")
    with pytest.raises(ValueError, match="more than one partition"):
        load_splits(tmp_path)


def test_split_must_match_data(tmp_path, panel, split):
    save_splits(split, tmp_path)
    loaded = load_splits(tmp_path)
    extra_company = explicit_panel({"C_9999": [2010]})
    with pytest.raises(ValueError, match="missing"):
        check_split_matches_data(loaded, pd.concat([panel, extra_company]))
    # Adding a later year to one company moves its anchor: the split is stale.
    company = loaded.loc[loaded["n_years"] < 7, "company_name"].iloc[0]
    later = panel[panel["company_name"].eq(company)].tail(1).assign(year=lambda f: f["year"] + 1)
    with pytest.raises(ValueError, match="differ"):
        check_split_matches_data(loaded, pd.concat([panel, later]))


def test_tiny_strata_give_a_clear_error():
    panel = explicit_panel({f"C_{i}": [2010, 2011] for i in range(1, 11)}, failed={"C_1"})
    with pytest.raises(ValueError, match="Strata too small"):
        split_companies(panel, train_fraction=0.7, validation_fraction=0.15, seed=42)
