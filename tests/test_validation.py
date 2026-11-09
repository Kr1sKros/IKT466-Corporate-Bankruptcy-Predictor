"""Tests for the dataset validators, using small hand-written frames."""

import numpy as np
import pandas as pd
import pytest

from validation import (
    FEATURE_COLUMNS,
    FEATURE_DICTIONARY,
    DataValidationError,
    anchor_targets,
    audit_dataset,
    check_accounting_identities,
    check_company_ids_and_order,
    check_duplicates,
    sort_panel,
    target_summary,
    validate_predictor_columns,
    check_nonfinite,
    compare_with_publisher,
    compare_with_reference,
    history_summary,
    identical_feature_columns,
    validate_schema,
    verify_target_labels,
)


def make_frame(rows: list[tuple[str, int, str]]) -> pd.DataFrame:
    """Build a valid frame whose features satisfy the accounting relations.

    Every column is a different linear function of the row number, so values are
    distinct per row and per column.
    """
    frame = pd.DataFrame(rows, columns=["company_name", "year", "status_label"])
    r = np.arange(len(frame), dtype=float)
    f = {
        "X9": 800 + 7 * r, "X2": 500 + 3 * r, "X18": 700 + 5 * r, "X3": 20 + 0.25 * r,
        "X10": 2000 + 11 * r, "X1": 900 + 6 * r, "X5": 100 + 1.5 * r, "X7": 150 + 2.5 * r,
        "X17": 1200 + 9 * r, "X14": 400 + 3.5 * r, "X11": 310 + 4.5 * r,
        "X6": 50 + 0.5 * r, "X8": 3000 + 13 * r, "X15": -200 + 8 * r,
    }
    f["X16"] = f["X9"] + 0.5
    f["X13"] = f["X9"] - f["X2"]
    f["X4"] = f["X16"] - f["X18"]
    f["X12"] = f["X4"] - f["X3"]
    for column in FEATURE_COLUMNS:
        frame[column] = f[column]
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


@pytest.mark.parametrize("blank", ["", "   ", "\t"])
def test_schema_rejects_blank_company_ids(clean, blank):
    frame = clean.copy()
    frame.loc[0, "company_name"] = blank
    with pytest.raises(DataValidationError, match="whitespace-only company ID"):
        validate_schema(frame)


def test_compare_with_publisher_reports_no_overlap(clean):
    publisher = clean.rename(columns={"year": "fyear"})
    publisher["fyear"] = publisher["fyear"] + 100
    result = compare_with_publisher(clean, publisher)
    assert result["rows_in_both"] == 0
    assert result["comparison_possible"] is False
    assert result["labels_identical"] is None
    assert result["same_column_names"] is None
    assert result["kaggle_to_publisher_columns"] == {}


def test_feature_dictionary_covers_x1_to_x18():
    assert list(FEATURE_DICTIONARY) == FEATURE_COLUMNS
    assert {v["kind"] for v in FEATURE_DICTIONARY.values()} == {"stock", "flow", "market"}


def test_accounting_identities_hold_and_reject_swapped_columns(clean):
    report = check_accounting_identities(clean)
    assert all(r["rows_violating"] == 0 for r in report.values())
    # Swapping two columns (as in the publisher's reordered file) must fail.
    swapped = clean.rename(columns={"X4": "X12", "X12": "X4"})
    with pytest.raises(DataValidationError, match="Accounting relations"):
        check_accounting_identities(swapped)


def test_predictor_guard_rejects_id_year_label_and_unknown_columns():
    assert validate_predictor_columns(FEATURE_COLUMNS) == FEATURE_COLUMNS
    for bad in ("company_name", "year", "status_label", "target"):
        with pytest.raises(DataValidationError, match="X1-X18 only"):
            validate_predictor_columns([*FEATURE_COLUMNS, bad])
    with pytest.raises(DataValidationError, match="Repeated"):
        validate_predictor_columns(["X1", "X1"])


def test_company_ids_must_match_pattern(clean):
    assert check_company_ids_and_order(clean)["companies_contiguous_in_file"] is True
    with pytest.raises(DataValidationError, match="C_<number>"):
        check_company_ids_and_order(clean.replace({"company_name": {"C_2": "company 2"}}))


def test_file_order_is_reported_and_sort_panel_fixes_it(clean):
    shuffled = clean.iloc[[4, 0, 6, 2, 1, 5, 3]].reset_index(drop=True)
    report = check_company_ids_and_order(shuffled)
    assert report["companies_contiguous_in_file"] is False
    assert report["years_increasing_within_company_in_file"] is False
    pd.testing.assert_frame_equal(sort_panel(shuffled), clean)
    # Numeric, not lexicographic, company order: C_10 comes after C_2.
    ids = sort_panel(make_frame([("C_10", 2000, "alive"), ("C_2", 2000, "alive")]))
    assert ids["company_name"].tolist() == ["C_2", "C_10"]


def test_anchor_targets_use_only_last_year_per_company(clean):
    targets = anchor_targets(clean)
    assert targets[["company_name", "anchor_year", "target"]].values.tolist() == [
        ["C_1", 2002, 0], ["C_2", 2001, 1], ["C_3", 2002, 0],
    ]
    # C_2's earlier failed row (2000) is not a target.
    assert len(targets) == clean["company_name"].nunique()


def test_target_summary_counts_dataset_boundary(clean):
    summary = target_summary(clean)
    assert summary["n_anchors"] == 3 and summary["n_positive"] == 1
    assert summary["negatives_at_last_dataset_year"] == 2  # C_1, C_3 end in 2002
    assert summary["negatives_leaving_panel_early"] == 0
    assert summary["positives_at_last_dataset_year"] == 0
    assert summary["target_verified"] is False
