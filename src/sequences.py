"""Model inputs for the two input conditions (issues #48-#51, scope #16).

See docs/study-design.md. Every company has one anchor t (its last recorded
year) and one target (``validation.anchor_targets``); both are the same in
every condition and for every model.

``latest_year``
    Only the record at t: X1-X18. GRU: one real step. Tabular models: 18 columns.
    No history length or temporal fields are added.

``five_year``
    The ``FIVE_YEAR_WINDOW`` = 5 most recent recorded years up to t, or all of
    them when a company has fewer. Older records are excluded. With gaps, the
    five records can span more than five calendar years.

    * Missing calendar years are skipped, never filled with made-up values. Each
      record carries ``years_before_anchor`` = t - year, so gaps stay visible:
      2005, 2006, 2008 with t = 2008 gives offsets 3, 2, 0.
    * GRU (``create_histories``): real records oldest to newest, then zero
      padding; ``lengths`` holds the real length and ``pack_histories`` packs
      them so padding never changes the final hidden state (the state at t).
    * Tabular models (``flatten_histories``): 5 slots aligned by recency, so the
      record at t is always slot r0. Each slot holds X1-X18, the offset and
      ``present`` (1 = real record, 0 = empty). Empty slots are 0 and sit before
      the real ones; short histories are kept, not dropped.

    Both layouts hold the same financial values and offsets. Padding is not an
    observed zero: scaling (#46) must be fitted on real records only and padding
    set back to 0 afterwards, so 0 means "no record" together with present = 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

import numpy as np
import pandas as pd

from validation import (
    COMPANY_COLUMN,
    FEATURE_COLUMNS,
    NON_FEATURE_COLUMNS,
    YEAR_COLUMN,
    DataValidationError,
    anchor_targets,
    sort_panel,
    validate_predictor_columns,
)

if TYPE_CHECKING:
    import torch

CONDITIONS = ("latest_year", "five_year")
FIVE_YEAR_WINDOW = 5
OFFSET_COLUMN = "years_before_anchor"
PRESENCE_COLUMN = "present"
DERIVED_COLUMNS = (OFFSET_COLUMN, PRESENCE_COLUMN)


def _check_condition(condition: str) -> None:
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}, got {condition!r}")


def step_columns(condition: str) -> list[str]:
    """Inputs per GRU step: X1-X18, plus the offset for five_year."""
    _check_condition(condition)
    base = validate_predictor_columns(FEATURE_COLUMNS)
    return base if condition == "latest_year" else [*base, OFFSET_COLUMN]


def tabular_columns(condition: str) -> list[str]:
    """Exact, deterministic column order of the tabular inputs.

    five_year: slots from oldest (r4) to latest (r0); per slot X1..X18, offset,
    presence. Names are ``<field>_r<recency>``.
    """
    _check_condition(condition)
    if condition == "latest_year":
        return validate_predictor_columns(FEATURE_COLUMNS)
    return [
        f"{field}_r{r}"
        for r in range(FIVE_YEAR_WINDOW - 1, -1, -1)
        for field in [*FEATURE_COLUMNS, *DERIVED_COLUMNS]
    ]


def validate_tabular_columns(columns: Iterable[str], condition: str) -> list[str]:
    """Check a derived tabular schema separately from the raw X1-X18 contract:
    exactly the expected columns in the expected order, no ID/year/label."""
    columns = list(columns)
    leaked = [c for c in columns if c in NON_FEATURE_COLUMNS]
    expected = tabular_columns(condition)
    if leaked or columns != expected:
        extra = sorted(set(columns) - set(expected))
        missing = sorted(set(expected) - set(columns))
        raise DataValidationError(
            f"Tabular columns for {condition} do not match the schema: "
            f"non-features={leaked}, unexpected={extra[:5]}, missing={missing[:5]}, "
            f"order_ok={not extra and not missing}"
        )
    return columns


@dataclass
class Histories:
    """Padded GRU inputs for one input condition.

    features: (n_companies, max_steps, n_step_columns) float32, oldest real
        step first and zero padding after the real steps. ``max_steps`` is 1 for
        latest_year and 5 for five_year.
    lengths: (n_companies,) number of real steps.
    targets: (n_companies,) 0/1 outcome at the anchor year.
    metadata: one row per company in the same order: company, anchor_year,
        target, n_years (all recorded years), length (years used), first_year,
        years (comma-separated years used), has_gap (within the years used).
    """

    condition: str
    features: np.ndarray
    lengths: np.ndarray
    targets: np.ndarray
    metadata: pd.DataFrame
    step_columns: list[str]

    @property
    def max_steps(self) -> int:
        return self.features.shape[1]


def create_histories(
    data: pd.DataFrame,
    *,
    condition: str,
    companies: Iterable[str] | None = None,
) -> Histories:
    """Build GRU inputs for ``condition`` ending at each company's anchor year.

    ``companies`` restricts the output to these IDs (e.g. one partition); rows
    are always ordered by numeric company ID. five_year keeps the latest five
    records and drops older ones.
    """
    columns = step_columns(condition)
    anchors = anchor_targets(data)
    panel = sort_panel(data)
    if companies is not None:
        wanted = set(companies)
        unknown = wanted - set(anchors[COMPANY_COLUMN])
        if unknown:
            raise ValueError(f"Unknown companies: {sorted(unknown)[:5]}")
        anchors = anchors[anchors[COMPANY_COLUMN].isin(wanted)].reset_index(drop=True)
        panel = panel[panel[COMPANY_COLUMN].isin(wanted)].reset_index(drop=True)

    n_years = panel.groupby(COMPANY_COLUMN, sort=False)[YEAR_COLUMN].size()
    max_steps = 1 if condition == "latest_year" else FIVE_YEAR_WINDOW

    # Recency rank: 0 = anchor year t, 1 = the record before it, ...
    recency = panel.groupby(COMPANY_COLUMN, sort=False).cumcount(ascending=False)
    keep = recency < max_steps
    window, window_recency = panel[keep], recency[keep].to_numpy()

    row_of = pd.Series(np.arange(len(anchors)), index=anchors[COMPANY_COLUMN])
    rows = row_of.loc[window[COMPANY_COLUMN]].to_numpy()
    lengths = np.minimum(anchors[COMPANY_COLUMN].map(n_years).to_numpy(), max_steps)
    steps = lengths[rows] - 1 - window_recency  # oldest real step at 0, t at length-1

    anchor_of = anchors.set_index(COMPANY_COLUMN)["anchor_year"]
    offsets = anchor_of.loc[window[COMPANY_COLUMN]].to_numpy() - window[YEAR_COLUMN].to_numpy()
    if (offsets < 0).any():
        raise AssertionError("A history contains a year after its anchor")

    values = window[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    if condition == "five_year":
        values = np.column_stack([values, offsets.astype(np.float32)])
    features = np.zeros((len(anchors), max_steps, len(columns)), dtype=np.float32)
    features[rows, steps] = values

    years = window.groupby(COMPANY_COLUMN, sort=False)[YEAR_COLUMN].agg(list)
    years = anchors[COMPANY_COLUMN].map(years)
    metadata = anchors[[COMPANY_COLUMN, "anchor_year", "target"]].copy()
    metadata["n_years"] = metadata[COMPANY_COLUMN].map(n_years).astype(int)
    metadata["length"] = lengths.astype(int)
    metadata["first_year"] = years.map(min).astype(int)
    metadata["years"] = years.map(lambda y: ",".join(map(str, y)))
    metadata["has_gap"] = years.map(lambda y: y[-1] - y[0] + 1 != len(y))

    return Histories(
        condition=condition,
        features=features,
        lengths=lengths.astype(np.int64),
        targets=metadata["target"].to_numpy(dtype=np.int64),
        metadata=metadata.reset_index(drop=True),
        step_columns=columns,
    )


def flatten_histories(histories: Histories) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tabular inputs for LR/XGBoost/MLP from the same histories as the GRU.

    Returns ``(features, metadata)`` with rows in the same order. five_year
    is recency-aligned: real step i of a company with length L goes to slot
    r = L-1-i, so the record at t is always r0; empty slots are 0 with
    present = 0.
    """
    if histories.condition == "latest_year":
        features = pd.DataFrame(histories.features[:, 0, :], columns=histories.step_columns)
        return features[tabular_columns("latest_year")], histories.metadata.copy()

    capacity = histories.max_steps  # FIVE_YEAR_WINDOW
    n, n_fields = len(histories.lengths), len(FEATURE_COLUMNS) + len(DERIVED_COLUMNS)
    slots = np.zeros((n, capacity, n_fields), dtype=np.float32)  # [:, r, field]
    mask = valid_step_mask(histories)
    rows, steps = np.nonzero(mask)
    recency = histories.lengths[rows] - 1 - steps
    slots[rows, recency, :-1] = histories.features[rows, steps]
    slots[rows, recency, -1] = 1.0
    # Columns run from the oldest slot to r0, so reverse the recency axis.
    flat = slots[:, ::-1, :].reshape(n, capacity * n_fields)
    features = pd.DataFrame(flat, columns=tabular_columns("five_year"))
    validate_tabular_columns(features.columns, "five_year")
    return features, histories.metadata.copy()


def tabular_inputs(
    data: pd.DataFrame, *, condition: str, companies: Iterable[str] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shortcut: tabular features and metadata for one condition."""
    return flatten_histories(create_histories(data, condition=condition, companies=companies))


def valid_step_mask(histories: Histories) -> np.ndarray:
    """Boolean (n_companies, max_steps) mask: True for real steps, False for padding."""
    return np.arange(histories.max_steps)[None, :] < histories.lengths[:, None]


def valid_steps(histories: Histories) -> np.ndarray:
    """All real (unpadded) steps as rows, for fitting scalers (#46).

    Padding must never be part of a fitted transform.
    """
    return histories.features[valid_step_mask(histories)]


def pack_histories(features: "torch.Tensor", lengths: "torch.Tensor | np.ndarray"):
    """Pack padded histories for a GRU with ``pack_padded_sequence``.

    ``features`` is (batch, max_steps, n_inputs) with real steps first. Order
    does not need to be sorted by length. The GRU's final hidden state ``h_n``
    is the state after the last real step (the anchor year), in input order.
    """
    import torch
    from torch.nn.utils.rnn import pack_padded_sequence

    lengths = torch.as_tensor(lengths, dtype=torch.int64).cpu()
    if (lengths < 1).any() or (lengths > features.shape[1]).any():
        raise ValueError("lengths must be between 1 and the padded length")
    return pack_padded_sequence(features, lengths, batch_first=True, enforce_sorted=False)


def history_coverage(data: pd.DataFrame, companies: pd.DataFrame | None = None) -> dict:
    """five_year coverage overall and per partition (``companies`` is the
    loaded split table): records used and excluded, gaps and padding."""
    meta = create_histories(data, condition="five_year").metadata
    groups = {"all": meta}
    if companies is not None:
        meta = meta.merge(companies[[COMPANY_COLUMN, "split"]], on=COMPANY_COLUMN)
        groups = {"all": meta, **{s: g for s, g in meta.groupby("split")}}

    def describe(g: pd.DataFrame) -> dict:
        lengths = g["length"]
        excluded = g["n_years"] - lengths
        return {
            "n_companies": len(g),
            "n_failed": int(g["target"].sum()),
            "records_used": int(lengths.sum()),
            "records_excluded": int(excluded.sum()),
            "companies_with_excluded_records": int((excluded > 0).sum()),
            "records_per_company": {
                "min": int(lengths.min()), "median": float(lengths.median()),
                "mean": round(float(lengths.mean()), 3), "max": int(lengths.max()),
            },
            "companies_with_gaps_in_window": int(g["has_gap"].sum()),
            "padded_slots_total": int((FIVE_YEAR_WINDOW - lengths).sum()),
            "companies_per_length": {
                str(k): int(v) for k, v in lengths.value_counts().sort_index().items()
            },
        }

    return {"window": FIVE_YEAR_WINDOW, **{n: describe(g) for n, g in groups.items()}}
