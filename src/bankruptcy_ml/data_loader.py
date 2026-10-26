"""Download and load the original American Companies Bankruptcy dataset.

TODO: Verify whether failed/alive labels are constant across each company's
history. Do not interpret them as next-year outcomes or define a next-year
bankruptcy target until event timing and label semantics have been verified.
"""

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
