"""GRU financial histories for k = 1-5 (issues #49, #50, #51).

Agreed policy (configs/config.yaml, ``inputs``):

* **Anchor.** Every company has one anchor t = its last recorded year and one
  target (``validation.anchor_targets``). Neither depends on k.
* **Window selection.** Take the company's last k *recorded* years, going
  backwards from t. Missing calendar years are skipped, not filled: years
  2005, 2006, 2008 with k=3 give a 3-step sequence.
* **Input order.** The window is fed oldest to newest, so the last real step is
  always the anchor year t.
* **Short histories.** A company with fewer than k recorded years keeps all of
  them; the sequence is zero-padded *after* its last real step and ``lengths``
  holds the number of real steps. ``pack_histories`` wraps
  ``torch.nn.utils.rnn.pack_padded_sequence`` so the GRU never reads padding
  and its final hidden state is the state at year t.
* **Cohorts.** Main results use all companies for every k; a sub-analysis uses
  only companies with at least k recorded years (``cohort_mask``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

import numpy as np
import pandas as pd

from validation import (
    COMPANY_COLUMN,
    FEATURE_COLUMNS,
    MAX_LOOKBACK,
    YEAR_COLUMN,
    anchor_targets,
    sort_panel,
    validate_predictor_columns,
)

if TYPE_CHECKING:
    import torch

COHORTS = ("all_companies", "at_least_k_years")


@dataclass
class Histories:
    """Padded GRU inputs for one lookback k.

    features: (n_companies, k, n_features) float32, oldest step first, real
        steps first and zero padding after them.
    lengths: (n_companies,) number of real steps, 1..k.
    targets: (n_companies,) 0/1 outcome at the anchor year.
    metadata: one row per company, same order: company, anchor_year, target,
        n_years (recorded years in total), length, window_years, window_has_gap,
        full_window (company has at least k recorded years).
    """

    lookback: int
    features: np.ndarray
    lengths: np.ndarray
    targets: np.ndarray
    metadata: pd.DataFrame
    feature_columns: list[str]


def create_histories(
    data: pd.DataFrame, *, lookback: int, companies: Iterable[str] | None = None
) -> Histories:
    """Build k-year histories ending at each company's anchor year.

    ``companies`` restricts the output to these IDs (e.g. one split partition);
    rows are always ordered by numeric company ID.
    """
    if not isinstance(lookback, (int, np.integer)) or not 1 <= lookback <= MAX_LOOKBACK:
        raise ValueError(f"lookback must be an integer in 1..{MAX_LOOKBACK}, got {lookback}")
    feature_columns = validate_predictor_columns(FEATURE_COLUMNS)

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
    # Rank 0 = anchor year, 1 = the recorded year before it, and so on.
    rank_from_end = panel.groupby(COMPANY_COLUMN, sort=False).cumcount(ascending=False)
    window = panel[rank_from_end < lookback].copy()
    window_rank = rank_from_end[rank_from_end < lookback]

    row_of = pd.Series(np.arange(len(anchors)), index=anchors[COMPANY_COLUMN])
    rows = row_of.loc[window[COMPANY_COLUMN]].to_numpy()
    lengths = np.minimum(anchors[COMPANY_COLUMN].map(n_years).to_numpy(), lookback)
    # Oldest real step at position 0, anchor at position length - 1.
    steps = lengths[rows] - 1 - window_rank.to_numpy()

    features = np.zeros((len(anchors), lookback, len(feature_columns)), dtype=np.float32)
    features[rows, steps] = window[feature_columns].to_numpy(dtype=np.float32)

    years = window.groupby(COMPANY_COLUMN, sort=False)[YEAR_COLUMN].agg(list)
    years = anchors[COMPANY_COLUMN].map(years)
    if any(max(y) > a for y, a in zip(years, anchors["anchor_year"])):
        raise AssertionError("A history contains a year after its anchor")

    metadata = anchors[[COMPANY_COLUMN, "anchor_year", "target"]].copy()
    metadata["n_years"] = metadata[COMPANY_COLUMN].map(n_years).astype(int)
    metadata["length"] = lengths.astype(int)
    metadata["window_years"] = years.map(lambda y: ",".join(map(str, y)))
    metadata["window_has_gap"] = years.map(lambda y: y[-1] - y[0] + 1 != len(y))
    metadata["full_window"] = metadata["n_years"] >= lookback

    return Histories(
        lookback=int(lookback),
        features=features,
        lengths=lengths.astype(np.int64),
        targets=metadata["target"].to_numpy(dtype=np.int64),
        metadata=metadata.reset_index(drop=True),
        feature_columns=feature_columns,
    )


def valid_step_mask(histories: Histories) -> np.ndarray:
    """Boolean (n_companies, k) mask: True for real steps, False for padding."""
    return np.arange(histories.lookback)[None, :] < histories.lengths[:, None]


def valid_steps(histories: Histories) -> np.ndarray:
    """All real (unpadded) timesteps as rows, for fitting scalers (#46).

    Padding must never be part of a fitted transform.
    """
    return histories.features[valid_step_mask(histories)]


def cohort_mask(histories: Histories, cohort: str) -> np.ndarray:
    """Select the evaluation cohort: all companies, or only full k-year windows."""
    if cohort == "all_companies":
        return np.ones(len(histories.lengths), dtype=bool)
    if cohort == "at_least_k_years":
        return histories.metadata["full_window"].to_numpy()
    raise ValueError(f"cohort must be one of {COHORTS}")


def pack_histories(features: "torch.Tensor", lengths: "torch.Tensor | np.ndarray"):
    """Pack padded histories for a GRU with ``pack_padded_sequence``.

    ``features`` is (batch, k, n_features) with real steps first. Order does not
    need to be sorted by length. The GRU's final hidden state ``h_n`` is then the
    state after the last real step (the anchor year), in the original order.
    """
    import torch
    from torch.nn.utils.rnn import pack_padded_sequence

    lengths = torch.as_tensor(lengths, dtype=torch.int64).cpu()
    if (lengths < 1).any() or (lengths > features.shape[1]).any():
        raise ValueError("lengths must be between 1 and the padded length")
    return pack_padded_sequence(features, lengths, batch_first=True, enforce_sorted=False)


def history_coverage(data: pd.DataFrame, companies: pd.DataFrame | None = None) -> dict:
    """Per k: companies with full windows, padded windows and gaps, overall and
    per split (``companies`` is the loaded split table)."""
    coverage: dict[str, dict] = {}
    for k in range(1, MAX_LOOKBACK + 1):
        meta = create_histories(data, lookback=k).metadata
        if companies is not None:
            meta = meta.merge(companies[[COMPANY_COLUMN, "split"]], on=COMPANY_COLUMN)
            groups = {"all": meta, **{s: g for s, g in meta.groupby("split")}}
        else:
            groups = {"all": meta}
        coverage[str(k)] = {
            name: {
                "n_companies": len(g),
                "full_window": int(g["full_window"].sum()),
                "padded": int((~g["full_window"]).sum()),
                "window_has_gap": int(g["window_has_gap"].sum()),
                "failed_full_window": int((g["full_window"] & g["target"].eq(1)).sum()),
                "failed_padded": int((~g["full_window"] & g["target"].eq(1)).sum()),
            }
            for name, g in groups.items()
        }
    return coverage


def select_latest_year(data: pd.DataFrame) -> pd.DataFrame:
    """Select one latest-year observation per company for all baselines."""
    # TODO (#48): Equivalent to create_histories(lookback=1); reuse its anchors.
    raise NotImplementedError("Latest-year selection is pending.")
