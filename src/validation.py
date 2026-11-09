"""Dataset validation and audit for the American Companies Bankruptcy CSV.

All dataset validators live here. Other modules and scripts should call these
functions rather than re-implementing the checks (see issue #44).

Hard failures (ambiguous records) raise ``DataValidationError``:
schema mismatches, unknown labels, non-integer years, missing or malformed
company IDs, duplicate company/year rows, infinite feature values, companies
whose label changes over time and feature columns that break the accounting
relations of the feature dictionary. Everything else is reported in the audit
dictionary.

Facts that come from the publisher's documentation rather than from the CSV are
kept in ``PUBLISHER_DOCUMENTATION`` so the audit can show them separately.

The input contract (issue #45) also lives here: the feature dictionary, the
accounting identities that confirm it, the predictor-column guard and the single
definition of the prediction target (``anchor_targets``).
"""

from __future__ import annotations

import re
from typing import Any, Iterable

import numpy as np
import pandas as pd

COMPANY_COLUMN = "company_name"
YEAR_COLUMN = "year"
LABEL_COLUMN = "status_label"
FEATURE_COLUMNS = [f"X{i}" for i in range(1, 19)]
NON_FEATURE_COLUMNS = (COMPANY_COLUMN, YEAR_COLUMN, LABEL_COLUMN)
EXPECTED_COLUMNS = [COMPANY_COLUMN, LABEL_COLUMN, YEAR_COLUMN, *FEATURE_COLUMNS]
LABEL_VALUES = ("alive", "failed")
POSITIVE_LABEL = "failed"
COMPANY_ID_PATTERN = re.compile(r"C_\d+")
MAX_LOOKBACK = 5

# Feature names from the Kaggle dataset description (version 3, retrieved via the
# Kaggle API). They apply to the Kaggle CSV only: the publisher's GitHub CSV uses
# a different X1-X18 order (see README section 10). "flow" = fiscal-year amount
# from the income statement; "stock" = year-end balance; "market" = market data.
# Neither source states units; magnitudes are consistent with millions of USD.
FEATURE_DICTIONARY: dict[str, dict[str, str]] = {
    "X1": {"name": "Current assets", "kind": "stock"},
    "X2": {"name": "Cost of goods sold", "kind": "flow"},
    "X3": {"name": "Depreciation and amortization", "kind": "flow"},
    "X4": {"name": "EBITDA", "kind": "flow"},
    "X5": {"name": "Inventory", "kind": "stock"},
    "X6": {"name": "Net income", "kind": "flow"},
    "X7": {"name": "Total receivables", "kind": "stock"},
    "X8": {"name": "Market value (market capitalization)", "kind": "market"},
    "X9": {"name": "Net sales", "kind": "flow"},
    "X10": {"name": "Total assets", "kind": "stock"},
    "X11": {"name": "Total long-term debt", "kind": "stock"},
    "X12": {"name": "EBIT", "kind": "flow"},
    "X13": {"name": "Gross profit", "kind": "flow"},
    "X14": {"name": "Total current liabilities", "kind": "stock"},
    "X15": {"name": "Retained earnings", "kind": "stock"},
    "X16": {"name": "Total revenue", "kind": "flow"},
    "X17": {"name": "Total liabilities", "kind": "stock"},
    "X18": {"name": "Total operating expenses", "kind": "flow"},
}

# Accounting relations implied by the dictionary. If the names were attached to
# the wrong columns (as in the publisher's reordered CSV) these would fail.
# Equalities: (lhs, [terms added], [terms subtracted]).
ACCOUNTING_EQUALITIES: dict[str, tuple[str, list[str], list[str]]] = {
    "gross_profit = net_sales - cogs": ("X13", ["X9"], ["X2"]),
    "ebit = ebitda - depreciation_amortization": ("X12", ["X4"], ["X3"]),
    "ebitda = total_revenue - operating_expenses": ("X4", ["X16"], ["X18"]),
}
# Inequalities: (smaller, larger).
ACCOUNTING_INEQUALITIES: dict[str, tuple[str, str]] = {
    "current_assets <= total_assets": ("X1", "X10"),
    "inventory <= current_assets": ("X5", "X1"),
    "receivables <= current_assets": ("X7", "X1"),
    "current_liabilities <= total_liabilities": ("X14", "X17"),
    "long_term_debt <= total_liabilities": ("X11", "X17"),
    "cogs <= operating_expenses": ("X2", "X18"),
}
# Values are rounded to three decimals, so allow rounding error; a relation
# must hold in at least this share of rows (a few source-data errors remain).
IDENTITY_ATOL = 0.0015
IDENTITY_MIN_SHARE = 0.99

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
    blank_ids = data[COMPANY_COLUMN].astype(str).str.strip().eq("")
    if blank_ids.any():
        raise DataValidationError(
            f"{int(blank_ids.sum())} rows have an empty or whitespace-only company ID"
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


def validate_predictor_columns(columns: Iterable[str]) -> list[str]:
    """Return ``columns`` as a list if they are all X1-X18 predictors.

    Raises if an identifier, year or label column (or anything outside the
    feature dictionary) would be used as a model input.
    """
    columns = list(columns)
    leaked = [c for c in columns if c in NON_FEATURE_COLUMNS]
    unknown = [c for c in columns if c not in FEATURE_COLUMNS and c not in NON_FEATURE_COLUMNS]
    if leaked or unknown:
        raise DataValidationError(
            f"Predictor columns must be X1-X18 only: non-features={leaked}, unknown={unknown}"
        )
    if len(set(columns)) != len(columns):
        raise DataValidationError(f"Repeated predictor columns: {columns}")
    return columns


def check_company_ids_and_order(data: pd.DataFrame) -> dict[str, Any]:
    """Validate the anonymised company IDs and describe the annual ordering.

    IDs must look like ``C_<number>``. The IDs are anonymised, so the CSV cannot
    show that one ID is the same legal entity in every year; that is taken from
    the publisher. The file order is reported only: consumers must sort with
    ``sort_panel`` instead of relying on it.
    """
    ids = data[COMPANY_COLUMN].astype(str)
    bad = sorted(set(ids[~ids.map(lambda s: bool(COMPANY_ID_PATTERN.fullmatch(s)))]))
    if bad:
        raise DataValidationError(
            f"{len(bad)} company IDs do not match 'C_<number>', e.g. {bad[:5]}"
        )
    # A company is contiguous if its rows form one block in file order.
    blocks = int((ids != ids.shift()).sum())
    years_increasing = bool(
        data.groupby(COMPANY_COLUMN, sort=False)[YEAR_COLUMN]
        .apply(lambda y: bool(y.is_monotonic_increasing))
        .all()
    )
    return {
        "company_id_format": "C_<number>",
        "n_companies": int(ids.nunique()),
        "companies_contiguous_in_file": blocks == ids.nunique(),
        "years_increasing_within_company_in_file": years_increasing,
    }


def sort_panel(data: pd.DataFrame) -> pd.DataFrame:
    """Sort rows by numeric company ID, then year (oldest first)."""
    order = data[COMPANY_COLUMN].str.slice(2).astype(int)
    return (
        data.assign(_company_order=order)
        .sort_values(["_company_order", YEAR_COLUMN], kind="stable")
        .drop(columns="_company_order")
        .reset_index(drop=True)
    )


def check_accounting_identities(data: pd.DataFrame) -> dict[str, Any]:
    """Confirm the feature dictionary with accounting relations between columns.

    Each relation must hold in at least ``IDENTITY_MIN_SHARE`` of rows; otherwise
    the X1-X18 names are attached to the wrong columns and the audit fails.
    """
    def total(columns: list[str]) -> np.ndarray:
        return data[columns].to_numpy(dtype=float).sum(axis=1)

    results: dict[str, dict[str, Any]] = {}
    for name, (lhs, plus, minus) in ACCOUNTING_EQUALITIES.items():
        rhs = total(plus) - (total(minus) if minus else 0.0)
        holds = np.isclose(data[lhs].to_numpy(dtype=float), rhs, rtol=1e-3, atol=IDENTITY_ATOL)
        results[name] = holds
    for name, (smaller, larger) in ACCOUNTING_INEQUALITIES.items():
        results[name] = (
            data[smaller].to_numpy(dtype=float)
            <= data[larger].to_numpy(dtype=float) + IDENTITY_ATOL
        )

    report = {
        name: {"share_holding": round(float(h.mean()), 6), "rows_violating": int((~h).sum())}
        for name, h in results.items()
    }
    failing = [n for n, r in report.items() if r["share_holding"] < IDENTITY_MIN_SHARE]
    if failing:
        raise DataValidationError(
            f"Accounting relations fail, so feature names do not match columns: {failing}"
        )
    return report


def anchor_targets(data: pd.DataFrame) -> pd.DataFrame:
    """Return the prediction target: one row per company at its anchor year t.

    The anchor t is the company's last recorded year. ``target`` is 1 when the
    company's (company-level) status is ``failed`` and 0 otherwise. Under the
    publisher's convention a failed company's last year is the fiscal year
    before its Chapter 7/11 filing, so ``target`` is meant as "files in t+1".
    Earlier rows of a failed company are never targets.
    """
    verify_target_labels(data)
    last = data.groupby(COMPANY_COLUMN)[YEAR_COLUMN].transform("max")
    anchors = data.loc[data[YEAR_COLUMN].eq(last), [COMPANY_COLUMN, YEAR_COLUMN, LABEL_COLUMN]]
    anchors = sort_panel(anchors).rename(columns={YEAR_COLUMN: "anchor_year"})
    anchors["target"] = anchors[LABEL_COLUMN].eq(POSITIVE_LABEL).astype(int)
    return anchors


def target_summary(data: pd.DataFrame) -> dict[str, Any]:
    """Describe the anchor-year target and what the CSV can and cannot show."""
    anchors = anchor_targets(data)
    last_dataset_year = int(data[YEAR_COLUMN].max())
    at_boundary = anchors["anchor_year"].eq(last_dataset_year)
    positive = anchors["target"].eq(1)
    return {
        "definition": (
            "One observation per company at anchor t = last recorded year; "
            "target = 1 if status_label is 'failed', else 0. Intended meaning: "
            "Chapter 7/11 filing in t+1 (publisher convention)."
        ),
        "n_anchors": len(anchors),
        "n_positive": int(positive.sum()),
        "positive_share": round(float(positive.mean()), 6),
        "positives_at_last_dataset_year": int((positive & at_boundary).sum()),
        "negatives_at_last_dataset_year": int((~positive & at_boundary).sum()),
        "negatives_leaving_panel_early": int((~positive & ~at_boundary).sum()),
        "target_verified": False,
        "evidence": [
            "status_label is constant within every company (company-level label).",
            "Earlier rows of a failed company repeat the company label, so only "
            "the last recorded year can match the documented year-level convention.",
            "Publisher and Kaggle documentation: the fiscal year before a Chapter 7/11 "
            "filing is labelled bankrupt 'for the next year'.",
        ],
        "limitations": [
            "The filing date is not in the CSV, so it cannot show that a failed "
            "company's last year is the year before its filing.",
            "Alive companies that leave the panel before the last dataset year have "
            "no observed t+1 (merger, delisting or going private are possible); "
            "their 0 rests on the publisher's labelling.",
            "Companies whose anchor is the last dataset year have t+1 outside the "
            "data; their label rests on the publisher's follow-up.",
        ],
    }


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
    comparison_possible = len(both) > 0
    # With no shared company/year rows, "all equal" would be vacuously true;
    # report None instead so the result is not mistaken for a real match.
    labels_equal = (
        bool((both[f"{LABEL_COLUMN}_kaggle"] == both[f"{LABEL_COLUMN}_publisher"]).all())
        if comparison_possible
        else None
    )

    column_mapping: dict[str, list[str]] = {}
    exact_cells = scaled_cells = total_cells = 0
    for feature in FEATURE_COLUMNS if comparison_possible else []:
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
        "comparison_possible": comparison_possible,
        "labels_identical": labels_equal,
        "kaggle_to_publisher_columns": column_mapping,
        "same_column_names": (
            all(column_mapping[f] == [f] for f in FEATURE_COLUMNS)
            if comparison_possible
            else None
        ),
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
    identity = check_company_ids_and_order(data)
    duplicates = check_duplicates(data)
    values = check_nonfinite(data)
    identities = check_accounting_identities(data)
    labels = verify_target_labels(data)
    target = target_summary(data)
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
        "company_ids_and_order": identity,
        "duplicates": duplicates,
        "feature_values": values,
        "feature_dictionary": FEATURE_DICTIONARY,
        "accounting_identities": identities,
        "identical_feature_columns": identical_feature_columns(data),
        "labels": labels,
        "target": target,
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
            "feature_dictionary_consistent_with_accounting_identities": True,
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
