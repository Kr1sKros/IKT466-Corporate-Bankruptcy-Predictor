"""Tests for the dataset validators, using small hand-written frames."""

import numpy as np
import pandas as pd
import pytest

from validation import (
    FEATURE_COLUMNS,
    DataValidationError,
    audit_dataset,
    check_duplicates,
    check_nonfinite,
    compare_with_publisher,
    compare_with_reference,
    history_summary,
    identical_feature_columns,
    validate_schema,
    verify_target_labels,
)


def make_frame(rows: list[tuple[str, int, str]]) -> pd.DataFrame:
    """Build a valid frame; feature values are distinct per row and column."""
    frame = pd.DataFrame(rows, columns=["company_name", "year", "status_label"])
    for j, column in enumerate(FEATURE_COLUMNS, start=1):
        frame[column] = np.arange(len(frame), dtype=float) + j * 100.5
    return frame[["company_name", "status_label", "year", *FEATURE_COLUMNS]]


@pytest.fixture
def clean() -> pd.DataFrame:
    return make_frame(
        [
            ("C_1", 2000, "alive"), ("C_1", 2001, "alive"), ("C_1", 2002, "alive"),
            ("C_2", 2000, "failed"), ("C_2", 2001, "failed"),
            ("C_3", 2000, "alive"), ("C_3", 2002, "alive"),  # gap in 2001
        ]
    )


def test_clean_frame_passes_full_audit(clean):
    report = audit_dataset(clean)
    assert report["measured"]["n_companies"] == 3
    assert report["labels"]["n_failed_companies"] == 1
    assert report["labels"]["failed_rows_before_terminal_year"] == 1
    assert report["feature_values"]["missing_values_total"] == 0


def test_schema_rejects_missing_and_extra_columns(clean):
    with pytest.raises(DataValidationError, match="missing"):
        validate_schema(clean.drop(columns="X5"))
    with pytest.raises(DataValidationError, match="extra"):
        validate_schema(clean.assign(Division="D"))


def test_schema_rejects_unknown_label_and_fractional_year(clean):
    with pytest.raises(DataValidationError, match="labels"):
        validate_schema(clean.replace({"status_label": {"alive": "bankrupt"}}))
    with pytest.raises(DataValidationError, match="non-integer"):
        validate_schema(clean.assign(year=clean["year"] + 0.5))


def test_duplicate_company_year_fails(clean):
    conflicting = pd.concat([clean, clean.iloc[[0]].assign(X1=-1.0)])
    with pytest.raises(DataValidationError, match="1 conflicting"):
        check_duplicates(conflicting)
    with pytest.raises(DataValidationError, match="1 exact"):
        check_duplicates(pd.concat([clean, clean.iloc[[0]]]))


def test_repeated_feature_rows_are_reported(clean):
    copied = clean.copy()
    copied.loc[1, FEATURE_COLUMNS] = copied.loc[0, FEATURE_COLUMNS].to_numpy()
    assert check_duplicates(copied)["repeated_feature_rows_within_company"] == 1


def test_nonfinite_counts_nan_and_rejects_inf(clean):
    with_nan = clean.copy()
    with_nan.loc[0, "X3"] = np.nan
    assert check_nonfinite(with_nan)["missing_values_per_column"]["X3"] == 1
    with_inf = clean.copy()
    with_inf.loc[0, "X3"] = np.inf
    with pytest.raises(DataValidationError, match="Infinite"):
        check_nonfinite(with_inf)


def test_label_change_within_company_fails(clean):
    changed = clean.copy()
    changed.loc[changed["company_name"].eq("C_2") & changed["year"].eq(2001), "status_label"] = "alive"
    with pytest.raises(DataValidationError, match="change label"):
        verify_target_labels(changed)


def test_history_counts_gaps_and_consecutive_years(clean):
    history = history_summary(clean, max_lookback=3)
    assert history["companies_with_year_gaps"] == 1
    assert history["missing_years_inside_company_spans"] == 1
    coverage = history["lookback_coverage"]
    # C_3 has two rows but only one consecutive year ending at its anchor (2002).
    assert coverage["2"]["companies_with_at_least_k_rows"] == 3
    assert coverage["2"]["companies_with_k_consecutive_years_ending_at_anchor"] == 2
    assert coverage["3"]["companies_with_k_consecutive_years_ending_at_anchor"] == 1
    assert history["alive_companies_ending_before_last_dataset_year"] == 0


def test_identical_feature_columns_detected(clean):
    assert identical_feature_columns(clean) == []
    assert identical_feature_columns(clean.assign(X16=clean["X9"])) == [["X9", "X16"]]


def test_compare_with_reference_lists_only_differences():
    measured = {"n_rows": 10, "failed_company_share": 0.0679, "first_year": 1999}
    reference = {"n_rows": 12, "failed_company_share": 0.068, "first_year": 1999, "source": "x"}
    assert compare_with_reference(measured, reference) == [
        {"statistic": "n_rows", "reference": 12, "measured": 10}
    ]


def test_compare_with_publisher_finds_swapped_and_scaled_columns(clean):
    publisher = clean.rename(columns={"year": "fyear", "X1": "X2", "X2": "X1"})
    publisher["X3"] = publisher["X3"] * 1000
    result = compare_with_publisher(clean, publisher)
    assert result["kaggle_to_publisher_columns"]["X1"] == ["X2"]
    assert result["kaggle_to_publisher_columns"]["X3"] == ["X3"]
    assert result["same_column_names"] is False
    assert result["matched_cells_publisher_1000x"] == len(clean)
    assert result["labels_identical"] is True
