"""Download and load the original American Companies Bankruptcy dataset.

Label semantics, the feature dictionary and the t+1 target definition
(``anchor_targets``) live in ``validation.py`` (issues #44 and #45).
"""

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import kagglehub
import pandas as pd


DATASET_HANDLE = "utkarshx27/american-companies-bankruptcy-prediction-dataset"


def download_dataset() -> Path:
    """Download the Kaggle dataset and locate its CSV file."""
    dataset_dir = Path(kagglehub.dataset_download(DATASET_HANDLE))
    csv_files = sorted(dataset_dir.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {dataset_dir}")
    if len(csv_files) > 1:
        raise ValueError(
            f"Multiple CSV files found in {dataset_dir}. "
            "Pass the desired CSV path to load_data(path)."
        )
    return csv_files[0]


def load_data(path: Path | None = None) -> pd.DataFrame:
    """Load a local CSV, or download the Kaggle dataset when no path is given."""
    path = download_dataset() if path is None else Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    return pd.read_csv(path)


def dataset_provenance(path: Path) -> dict[str, object]:
    """Describe the CSV file: source handle, Kaggle version, size and SHA-256."""
    path = Path(path)
    version = path.parent.name if path.parent.parent.name == "versions" else None
    return {
        "handle": DATASET_HANDLE,
        "kaggle_version": int(version) if version and version.isdigit() else None,
        "file_name": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "audited_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
