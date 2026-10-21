"""Starting point for the corporate bankruptcy prediction project."""

from pathlib import Path

import pandas as pd


DATA_PATH = Path(__file__).resolve().parent / "data.csv"


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the bankruptcy dataset from a CSV file."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    return pd.read_csv(path)


def main() -> None:
    data = load_data()
    print(f"Loaded {len(data):,} rows and {len(data.columns)} columns from {DATA_PATH.name}")
    print(data.head())


if __name__ == "__main__":
    main()
