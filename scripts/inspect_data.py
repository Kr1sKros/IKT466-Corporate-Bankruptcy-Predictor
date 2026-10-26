"""Inspect raw data without modifying it or interpreting labels as future events."""

from bankruptcy_ml.data_loader import download_dataset, load_data


def main() -> None:
    """Preserve the existing download/preview flow; detailed auditing is pending."""
    path = download_dataset()
    data = load_data(path)
    print(f"Path to dataset file: {path}")
    print(f"Loaded {len(data):,} rows and {len(data.columns)} columns")
    print(data.head())
    # TODO: Audit company IDs, duplicate company/year pairs, gaps, missing values,
    # label values and within-company label changes; call shared audit logic.


if __name__ == "__main__":
    main()
