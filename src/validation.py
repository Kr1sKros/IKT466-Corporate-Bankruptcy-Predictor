"""Dataset validation and audit for the American Companies Bankruptcy CSV.

All dataset validators live here. Other modules and scripts should call these
functions rather than re-implementing the checks (see issue #44).

Hard failures (ambiguous records) raise ``DataValidationError``:
schema mismatches, unknown labels, non-integer years, missing company IDs,
duplicate company/year rows, infinite feature values and companies whose label
changes over time. Everything else is reported in the audit dictionary.

Facts that come from the publisher's documentation rather than from the CSV are
kept in ``PUBLISHER_DOCUMENTATION`` so the audit can show them separately.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

COMPANY_COLUMN = "company_name"
YEAR_COLUMN = "year"
LABEL_COLUMN = "status_label"
FEATURE_COLUMNS = [f"X{i}" for i in range(1, 19)]
EXPECTED_COLUMNS = [COMPANY_COLUMN, LABEL_COLUMN, YEAR_COLUMN, *FEATURE_COLUMNS]
LABEL_VALUES = ("alive", "failed")
MAX_LOOKBACK = 5

# Statistics stated in README.md section 1; compared against measured values.
README_STATISTICS: dict[str, Any] = {
    "n_companies": 8971,
    "n_rows": 78682,
    "n_failed_companies": 609,
    "failed_company_share": 0.068,
    "first_year": 1999,
    "last_year": 2018,
}

# Claims made by the publisher (https://github.com/sowide/bankruptcy_dataset,
# README at commit 8bcd8db; Lombardo et al., Future Internet 14(8):244, 2022).
# These are documentation, not something the CSV alone can prove.
PUBLISHER_DOCUMENTATION: dict[str, Any] = {
    "source": "https://github.com/sowide/bankruptcy_dataset (commit 8bcd8db)",
    "n_companies": 8262,
    "n_rows": 78682,
    "first_year": 1999,
    "last_year": 2018,
    "label_convention": (
        "The fiscal year before a Chapter 7/11 filing is labelled bankrupt (1) "
        "'for the next year'; otherwise the company is alive (0)."
    ),
    "no_missing_synthetic_or_imputed_values": True,
}


class DataValidationError(ValueError):
    """Raised when the dataset contains records that make the audit ambiguous."""


def validate_schema(data: pd.DataFrame) -> dict[str, Any]:
    """Check columns, dtypes, label values, company IDs and integer years."""
    columns = list(data.columns)
    missing = [c for c in EXPECTED_COLUMNS if c not in columns]
    extra = [c for c in columns if c not in EXPECTED_COLUMNS]
    if missing or extra:
        raise DataValidationError(f"Schema mismatch: missing={missing}, extra={extra}")

    if data[COMPANY_COLUMN].isna().any():
        raise DataValidationError(
            f"{int(data[COMPANY_COLUMN].isna().sum())} rows have no company ID"
        )

    years = data[YEAR_COLUMN]
    if not pd.api.types.is_numeric_dtype(years) or years.isna().any():
        raise DataValidationError("Year column must be numeric with no missing values")
    if not np.all(np.equal(np.mod(years.to_numpy(dtype=float), 1), 0)):
        raise DataValidationError("Year column contains non-integer values")

    unknown = sorted(set(data[LABEL_COLUMN].dropna().unique()) - set(LABEL_VALUES))
    if unknown or data[LABEL_COLUMN].isna().any():
        raise DataValidationError(
            f"Unexpected or missing labels: unknown={unknown}, "
            f"missing={int(data[LABEL_COLUMN].isna().sum())}"
        )

    non_numeric = [
        c for c in FEATURE_COLUMNS if not pd.api.types.is_numeric_dtype(data[c])
    ]
    if non_numeric:
        raise DataValidationError(f"Non-numeric feature columns: {non_numeric}")

    return {"columns": columns, "n_rows": len(data), "n_columns": len(columns)}


def check_duplicates(data: pd.DataFrame) -> dict[str, int]:
    """Fail on any repeated company/year pair; report repeated feature vectors."""
    key = [COMPANY_COLUMN, YEAR_COLUMN]
    extra_rows = int(data.duplicated(key).sum())
    if extra_rows:
        exact = int(data.duplicated().sum())
        raise DataValidationError(
            f"Duplicate company/year rows: {extra_rows} extra rows "
            f"({exact} exact copies, {extra_rows - exact} conflicting)"
        )
    # Identical features in two different years of one company would hint at
    # copied/filled values; this is reported, not treated as a failure.
    repeated_features = int(data.duplicated([COMPANY_COLUMN, *FEATURE_COLUMNS]).sum())
    return {"duplicate_company_years": 0, "repeated_feature_rows_within_company": repeated_features}


def check_nonfinite(data: pd.DataFrame) -> dict[str, Any]:
    """Count missing, infinite and zero feature values; fail on infinities."""
    features = data[FEATURE_COLUMNS]
    values = features.to_numpy(dtype=float)
    inf_per_column = dict(zip(FEATURE_COLUMNS, np.isinf(values).sum(axis=0).tolist()))
    if any(inf_per_column.values()):
        raise DataValidationError(
            f"Infinite feature values: { {c: n for c, n in inf_per_column.items() if n} }"
        )
    missing_per_column = features.isna().sum().astype(int).to_dict()
    zeros_per_column = features.eq(0).sum().astype(int).to_dict()
    negatives_per_column = features.lt(0).sum().astype(int).to_dict()
    return {
        "missing_values_total": int(sum(missing_per_column.values())),
        "missing_values_per_column": missing_per_column,
        "infinite_values_total": 0,
        "zero_values_per_column": {c: n for c, n in zeros_per_column.items() if n},
        "negative_values_per_column": {c: n for c, n in negatives_per_column.items() if n},
        "all_zero_feature_rows": int(features.eq(0).all(axis=1).sum()),
    }


def identical_feature_columns(data: pd.DataFrame) -> list[list[str]]:
    """Return groups of feature columns whose values are identical in every row."""
    groups: list[list[str]] = []
    seen: set[str] = set()
    for i, first in enumerate(FEATURE_COLUMNS):
        if first in seen:
            continue
        group = [first] + [
            other
            for other in FEATURE_COLUMNS[i + 1 :]
            if data[first].equals(data[other])
        ]
        if len(group) > 1:
            groups.append(group)
            seen.update(group)
    return groups


def verify_target_labels(data: pd.DataFrame) -> dict[str, Any]:
    """Audit provided labels: per-company consistency and position of failed rows.

    Raises if a company's label changes over time, because the label could then
    not be assigned to the company's anchor year unambiguously.
    """
    labels_per_company = data.groupby(COMPANY_COLUMN)[LABEL_COLUMN].nunique()
    changing = labels_per_company[labels_per_company > 1]
    if len(changing):
        raise DataValidationError(
            f"{len(changing)} companies change label over time, e.g. "
            f"{list(changing.index[:5])}"
        )

    company_label = data.groupby(COMPANY_COLUMN)[LABEL_COLUMN].first()
    last_year = data.groupby(COMPANY_COLUMN)[YEAR_COLUMN].transform("max")
    failed_rows = data[LABEL_COLUMN].eq("failed")
    n_companies = len(company_label)
    n_failed = int(company_label.eq("failed").sum())
    return {
        "labels_constant_within_company": True,
        "label_granularity": "company",
        "row_label_counts": data[LABEL_COLUMN].value_counts().astype(int).to_dict(),
        "failed_row_share": round(float(failed_rows.mean()), 6),
        "n_failed_companies": n_failed,
        "n_alive_companies": n_companies - n_failed,
        "failed_company_share": round(n_failed / n_companies, 6),
        "failed_rows_at_terminal_year": int((failed_rows & data[YEAR_COLUMN].eq(last_year)).sum()),
        "failed_rows_before_terminal_year": int((failed_rows & data[YEAR_COLUMN].lt(last_year)).sum()),
    }


def _consecutive_run_ending_at_last(years: pd.Series) -> int:
    """Number of consecutive years ending at the company's last recorded year."""
    ordered = np.sort(years.to_numpy())
    breaks = np.nonzero(np.diff(ordered) != 1)[0]
    return int(len(ordered) - (breaks[-1] + 1)) if len(breaks) else int(len(ordered))


def history_summary(data: pd.DataFrame, max_lookback: int = MAX_LOOKBACK) -> dict[str, Any]:
    """Summarise years per company, gaps and k-year history coverage at the anchor."""
    grouped = data.groupby(COMPANY_COLUMN)[YEAR_COLUMN]
    n_years = grouped.size()
    first, last = grouped.min(), grouped.max()
    missing_inside_span = (last - first + 1) - n_years
    run_at_anchor = grouped.apply(_consecutive_run_ending_at_last)
    failed = data.groupby(COMPANY_COLUMN)[LABEL_COLUMN].first().eq("failed")

    coverage = {}
    for k in range(1, max_lookback + 1):
        has_rows = n_years >= k
        consecutive = run_at_anchor >= k
        coverage[str(k)] = {
            "companies_with_at_least_k_rows": int(has_rows.sum()),
            "companies_with_k_consecutive_years_ending_at_anchor": int(consecutive.sum()),
            "failed_companies_with_k_consecutive_years": int((consecutive & failed).sum()),
        }

    def counts(series: pd.Series) -> dict[str, int]:
        return {str(k): int(v) for k, v in series.value_counts().sort_index().items()}

    return {
        "first_year": int(data[YEAR_COLUMN].min()),
        "last_year": int(data[YEAR_COLUMN].max()),
        "rows_per_year": counts(data[YEAR_COLUMN]),
        "years_per_company": {
            "min": int(n_years.min()),
            "median": float(n_years.median()),
            "mean": round(float(n_years.mean()), 3),
            "max": int(n_years.max()),
        },
        "companies_with_year_gaps": int((missing_inside_span > 0).sum()),
        "missing_years_inside_company_spans": int(missing_inside_span.sum()),
        "anchor_year_distribution_failed": counts(last[failed]),
        "anchor_year_distribution_alive": counts(last[~failed]),
        "alive_companies_ending_before_last_dataset_year": int(
            ((last < data[YEAR_COLUMN].max()) & ~failed).sum()
        ),
        "failed_companies_ending_at_last_dataset_year": int(
            ((last == data[YEAR_COLUMN].max()) & failed).sum()
        ),
        "lookback_coverage": coverage,
    }


def compare_with_reference(
    measured: dict[str, Any], reference: dict[str, Any], *, tolerance: float = 0.001
) -> list[dict[str, Any]]:
    """List every reference statistic that differs from the measured value."""
    discrepancies = []
    for key, expected in reference.items():
        if key not in measured or not isinstance(expected, (int, float)) or isinstance(expected, bool):
            continue
        actual = measured[key]
        if isinstance(expected, float):
            differs = abs(actual - expected) > tolerance
        else:
            differs = actual != expected
        if differs:
            discrepancies.append({"statistic": key, "reference": expected, "measured": actual})
    return discrepancies


def compare_with_publisher(kaggle: pd.DataFrame, publisher: pd.DataFrame) -> dict[str, Any]:
    """Compare the Kaggle CSV with the publisher's GitHub CSV.

    Finds, for each Kaggle feature, the publisher column with the same values,
    allowing the publisher value to be exactly 1000x larger (lost decimal point).
    """
    publisher = publisher.rename(columns={"fyear": YEAR_COLUMN}).copy()
    publisher[YEAR_COLUMN] = publisher[YEAR_COLUMN].astype(int)
    merged = kaggle.merge(
        publisher, on=[COMPANY_COLUMN, YEAR_COLUMN], how="outer",
        suffixes=("_kaggle", "_publisher"), indicator=True,
    )
    both = merged[merged["_merge"] == "both"]
    labels_equal = bool(
        (both[f"{LABEL_COLUMN}_kaggle"] == both[f"{LABEL_COLUMN}_publisher"]).all()
    )

    column_mapping: dict[str, list[str]] = {}
    exact_cells = scaled_cells = total_cells = 0
    for feature in FEATURE_COLUMNS:
        k_values = both[f"{feature}_kaggle"].to_numpy(dtype=float)
        matches = []
        for candidate in FEATURE_COLUMNS:
            p_values = both[f"{candidate}_publisher"].to_numpy(dtype=float)
            exact = np.isclose(k_values, p_values, rtol=1e-9, atol=1e-12)
            scaled = ~exact & np.isclose(k_values * 1000, p_values, rtol=1e-6)
            if np.all(exact | scaled):
                matches.append(candidate)
                if len(matches) == 1:
                    exact_cells += int(exact.sum())
                    scaled_cells += int(scaled.sum())
        total_cells += len(k_values)
        column_mapping[feature] = matches

    extra_publisher_columns = [
        c for c in publisher.columns if c not in EXPECTED_COLUMNS
    ]
    return {
        "rows_only_in_kaggle": int((merged["_merge"] == "left_only").sum()),
        "rows_only_in_publisher": int((merged["_merge"] == "right_only").sum()),
        "rows_in_both": len(both),
        "labels_identical": labels_equal,
        "kaggle_to_publisher_columns": column_mapping,
        "same_column_names": all(column_mapping[f] == [f] for f in FEATURE_COLUMNS),
        "unmatched_kaggle_columns": [f for f, m in column_mapping.items() if not m],
        "matched_cells_equal": exact_cells,
        "matched_cells_publisher_1000x": scaled_cells,
        "total_cells": total_cells,
        "extra_publisher_columns": extra_publisher_columns,
        "publisher_identical_columns": identical_feature_columns(
            publisher.reindex(columns=FEATURE_COLUMNS)
        ),
    }


def audit_dataset(
    data: pd.DataFrame,
    *,
    provenance: dict[str, Any] | None = None,
    publisher_data: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Run every validator and return one JSON-serialisable audit report."""
    schema = validate_schema(data)
    duplicates = check_duplicates(data)
    values = check_nonfinite(data)
    labels = verify_target_labels(data)
    history = history_summary(data)

    measured = {
        "n_rows": schema["n_rows"],
        "n_companies": int(data[COMPANY_COLUMN].nunique()),
        "n_failed_companies": labels["n_failed_companies"],
        "failed_company_share": labels["failed_company_share"],
        "first_year": history["first_year"],
        "last_year": history["last_year"],
    }

    report: dict[str, Any] = {
        "provenance": provenance or {},
        "measured": measured,
        "schema": schema,
        "duplicates": duplicates,
        "feature_values": values,
        "identical_feature_columns": identical_feature_columns(data),
        "labels": labels,
        "history": history,
        "discrepancies_vs_readme": compare_with_reference(measured, README_STATISTICS),
        "discrepancies_vs_publisher_documentation": compare_with_reference(
            measured, PUBLISHER_DOCUMENTATION
        ),
        "publisher_documentation": PUBLISHER_DOCUMENTATION,
        "verifiable_from_csv": {
            "no_missing_values": values["missing_values_total"] == 0,
            "no_infinite_values": True,
            "no_duplicate_company_years": True,
            "no_repeated_feature_rows_within_company": (
                duplicates["repeated_feature_rows_within_company"] == 0
            ),
            "labels_follow_documented_year_level_convention": (
                labels["failed_rows_before_terminal_year"] == 0
            ),
            "not_verifiable_from_csv": [
                "whether values were synthetic or imputed by the publisher",
                "that the terminal year of a failed company precedes its filing",
                "the t+1 outcome of alive companies that leave the panel before the last year",
                "the t+1 outcome of companies observed in the last dataset year",
            ],
        },
    }
    if publisher_data is not None:
        report["publisher_csv_comparison"] = compare_with_publisher(data, publisher_data)
    return report
