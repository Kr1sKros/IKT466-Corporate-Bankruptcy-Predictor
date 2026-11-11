"""Company-level train/validation/test splits (issue #47) and pending scaling.

Every company appears in exactly one partition, together with all of its years.
The split is stratified by outcome and by history length (number of recorded
years, grouped 1, 2, 3, 4, 5+), so every partition has about the same share of
failed companies and of short/long histories.

The split is saved once by ``scripts/prepare_data.py`` as three CSV files
(``train.csv``, ``validation.csv``, ``test.csv``) and then only *loaded* by the
model runners: ``load_splits`` never regenerates a missing or changed split.
Each file stores the company's anchor year and target, so the input condition
(latest_year or five_year) never changes a company's anchor, outcome or
partition.

Dataset validation (schema, duplicates, labels, history) lives in
``validation.py``.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from validation import COMPANY_COLUMN, YEAR_COLUMN, anchor_targets

SPLIT_NAMES = ("train", "validation", "test")
HISTORY_GROUPS = ("1", "2", "3", "4", "5+")
SPLIT_COLUMNS = [COMPANY_COLUMN, "anchor_year", "target", "n_years", "history_group"]


def history_group(n_years: pd.Series) -> pd.Series:
    """Map recorded years per company to the groups 1, 2, 3, 4 and 5+."""
    return n_years.clip(upper=5).astype(int).astype(str).replace({"5": "5+"})


def company_table(data: pd.DataFrame) -> pd.DataFrame:
    """One row per company: anchor year, target, recorded years and group."""
    companies = anchor_targets(data)[[COMPANY_COLUMN, "anchor_year", "target"]]
    n_years = data.groupby(COMPANY_COLUMN)[YEAR_COLUMN].size()
    companies["n_years"] = companies[COMPANY_COLUMN].map(n_years).astype(int)
    companies["history_group"] = history_group(companies["n_years"])
    return companies.reset_index(drop=True)


def _check_fractions(train_fraction: float, validation_fraction: float) -> float:
    test_fraction = round(1.0 - train_fraction - validation_fraction, 9)
    if min(train_fraction, validation_fraction, test_fraction) <= 0:
        raise ValueError(
            f"Fractions must be positive and sum to 1: train={train_fraction}, "
            f"validation={validation_fraction}, test={test_fraction:.3f}"
        )
    return test_fraction


def split_companies(
    data: pd.DataFrame, *, train_fraction: float, validation_fraction: float, seed: int
) -> pd.DataFrame:
    """Assign every company to train/validation/test, stratified by
    target x history group. Returns the company table with a ``split`` column.
    """
    test_fraction = _check_fractions(train_fraction, validation_fraction)
    companies = company_table(data)
    strata = companies["target"].astype(str) + "_" + companies["history_group"]
    sizes = strata.value_counts()
    if (sizes < len(SPLIT_NAMES)).any():
        raise ValueError(
            f"Strata too small to place in all three partitions: "
            f"{sizes[sizes < len(SPLIT_NAMES)].to_dict()}"
        )

    train_idx, rest_idx = train_test_split(
        companies.index, train_size=train_fraction, stratify=strata, random_state=seed
    )
    validation_idx, test_idx = train_test_split(
        rest_idx,
        train_size=validation_fraction / (validation_fraction + test_fraction),
        stratify=strata[rest_idx],
        random_state=seed,
    )
    companies["split"] = ""
    for name, idx in zip(SPLIT_NAMES, (train_idx, validation_idx, test_idx)):
        companies.loc[idx, "split"] = name
    return companies


def _company_number(companies: pd.DataFrame) -> np.ndarray:
    return companies[COMPANY_COLUMN].str.slice(2).astype(int).to_numpy()


def split_paths(directory: str | Path) -> dict[str, Path]:
    """Paths of the three persisted split files."""
    return {name: Path(directory) / f"{name}.csv" for name in SPLIT_NAMES}


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_splits(companies: pd.DataFrame, directory: str | Path) -> dict[str, str]:
    """Write train/validation/test CSVs (sorted, LF line endings) and return
    their SHA-256 checksums, so identical splits give identical files on any OS.
    """
    paths = split_paths(directory)
    Path(directory).mkdir(parents=True, exist_ok=True)
    checksums = {}
    for name, path in paths.items():
        part = companies[companies["split"] == name]
        part = part.iloc[np.argsort(_company_number(part), kind="stable")]
        part[SPLIT_COLUMNS].to_csv(path, index=False, lineterminator="\n")
        checksums[name] = file_sha256(path)
    return checksums


class SplitChecksumError(ValueError):
    """A newly generated split differs from the approved checksums."""


def write_verified_splits(
    companies: pd.DataFrame,
    directory: str | Path,
    *,
    expected_sha256: dict[str, str] | None = None,
) -> dict[str, str]:
    """Write the split only if it matches ``expected_sha256``.

    The files are first written to a temporary folder inside ``directory`` and
    checked there. On a mismatch nothing in ``directory`` is changed and
    ``SplitChecksumError`` is raised; otherwise the files replace the old ones.
    Without expected checksums (first run) the files are written as they are.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=directory, prefix=".tmp-") as staging:
        checksums = save_splits(companies, staging)
        expected = expected_sha256 or {}
        mismatched = [n for n, sha in checksums.items() if expected.get(n) not in (None, sha)]
        if mismatched:
            raise SplitChecksumError(
                f"Generated split files {mismatched} differ from the approved "
                "checksums in configs/config.yaml (dataset, seed, fractions or "
                "library versions changed). Existing files were left unchanged."
            )
        targets = split_paths(directory)
        for name, path in split_paths(staging).items():
            os.replace(path, targets[name])
    return checksums


def load_splits(
    directory: str | Path, *, expected_sha256: dict[str, str] | None = None
) -> pd.DataFrame:
    """Load the persisted split; never regenerate it.

    Raises if a file is missing, its checksum differs from ``expected_sha256``
    (the values recorded in configs/config.yaml), or a company is in more than
    one partition.
    """
    frames = []
    for name, path in split_paths(directory).items():
        if not path.exists():
            raise FileNotFoundError(
                f"Split file {path} is missing. Run `python scripts/prepare_data.py`."
            )
        if expected_sha256 and expected_sha256.get(name) not in (None, file_sha256(path)):
            raise ValueError(
                f"{path} differs from the split recorded in the config; "
                "rerun prepare_data.py and check the dataset and seed."
            )
        frames.append(pd.read_csv(path).assign(split=name))
    companies = pd.concat(frames, ignore_index=True)
    companies["history_group"] = companies["history_group"].astype(str)
    duplicated = companies[COMPANY_COLUMN].duplicated()
    if duplicated.any():
        raise ValueError(
            f"Companies in more than one partition: "
            f"{companies.loc[duplicated, COMPANY_COLUMN].tolist()[:5]}"
        )
    return companies


def check_split_matches_data(companies: pd.DataFrame, data: pd.DataFrame) -> None:
    """Fail if the split does not cover exactly the companies, anchors and
    targets of ``data`` (e.g. after a dataset change)."""
    expected = company_table(data).set_index(COMPANY_COLUMN)
    actual = companies.set_index(COMPANY_COLUMN)
    missing = expected.index.difference(actual.index)
    extra = actual.index.difference(expected.index)
    if len(missing) or len(extra):
        raise ValueError(
            f"Split does not match the data: {len(missing)} companies missing, "
            f"{len(extra)} unknown"
        )
    columns = ["anchor_year", "target", "n_years"]
    differs = (actual.loc[expected.index, columns] != expected[columns]).any(axis=1)
    if differs.any():
        raise ValueError(
            f"Anchor/target/history differ for {int(differs.sum())} companies, "
            f"e.g. {list(differs[differs].index[:5])}"
        )


def split_summary(companies: pd.DataFrame) -> dict[str, Any]:
    """Company counts, failed share and history-group shares per partition."""
    summary: dict[str, Any] = {}
    for name in [*SPLIT_NAMES, "all"]:
        part = companies if name == "all" else companies[companies["split"] == name]
        groups = part["history_group"].value_counts().reindex(HISTORY_GROUPS, fill_value=0)
        summary[name] = {
            "n_companies": len(part),
            "company_share": round(len(part) / len(companies), 4),
            "n_failed": int(part["target"].sum()),
            "failed_share": round(float(part["target"].mean()), 4),
            "history_group_counts": {g: int(n) for g, n in groups.items()},
            "history_group_shares": {g: round(n / len(part), 4) for g, n in groups.items()},
        }
    return summary


def fit_preprocessor(training_features: pd.DataFrame) -> object:
    """Fit missing-value handling and scaling using training features only."""
    # TODO (#46): Choose scaling; for GRU inputs fit only on valid (unpadded)
    # timesteps, e.g. sequences.valid_steps(histories).
    raise NotImplementedError("Training-only preprocessing is pending.")


def transform_features(features: pd.DataFrame, preprocessor: object) -> np.ndarray:
    """Transform features using an already fitted training preprocessor."""
    # TODO (#46): Preserve feature order and never refit on validation/test.
    raise NotImplementedError("Feature transformation is pending.")
