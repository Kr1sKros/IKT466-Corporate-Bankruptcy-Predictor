"""Create and verify the company-level train/validation/test split (issue #47).

Usage (from the repository root):
    python scripts/prepare_data.py
    python scripts/prepare_data.py --csv data/raw/american_bankruptcy.csv

Writes data/splits/{train,validation,test}.csv (gitignored) and
results/metrics/split_summary.json. Exits with status 1 if the data fails
validation or the generated files differ from the checksums in
configs/config.yaml; in that case the existing files and summary are kept.
Model runners must load the split with ``preprocessing.load_splits``.
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

from data_loader import download_dataset, load_data
from preprocessing import (
    SplitChecksumError,
    check_split_matches_data,
    load_splits,
    split_companies,
    split_summary,
    write_verified_splits,
)
from sequences import history_coverage
from validation import (
    DataValidationError,
    check_accounting_identities,
    check_company_ids_and_order,
    check_duplicates,
    validate_schema,
    verify_target_labels,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "config.yaml"
DEFAULT_SUMMARY = REPO_ROOT / "results" / "metrics" / "split_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path, help="Local CSV (default: KaggleHub download)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    split_config = config["split"]
    data = load_data(args.csv or download_dataset())

    try:
        validate_schema(data)
        check_company_ids_and_order(data)
        check_duplicates(data)
        check_accounting_identities(data)
        verify_target_labels(data)
    except DataValidationError as error:
        print(f"Data validation failed: {error}", file=sys.stderr)
        return 1

    companies = split_companies(
        data,
        train_fraction=split_config["train_fraction"],
        validation_fraction=split_config["validation_fraction"],
        seed=config["random_seed"],
    )
    directory = REPO_ROOT / split_config["directory"]
    expected = split_config.get("sha256") or {}
    try:
        # Checked before anything is replaced: on a mismatch the approved
        # split files and summary stay as they are.
        checksums = write_verified_splits(companies, directory, expected_sha256=expected)
    except SplitChecksumError as error:
        print(error, file=sys.stderr)
        return 1
    loaded = load_splits(directory, expected_sha256=expected)
    check_split_matches_data(loaded, data)

    summary = {
        "seed": config["random_seed"],
        "fractions": {
            "train": split_config["train_fraction"],
            "validation": split_config["validation_fraction"],
            "test": split_config["test_fraction"],
        },
        "stratify_by": split_config["stratify_by"],
        "sha256": checksums,
        "partitions": split_summary(loaded),
        "history_coverage": history_coverage(data, loaded),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    for name, part in summary["partitions"].items():
        shares = " ".join(f"{g}:{s:.1%}" for g, s in part["history_group_shares"].items())
        print(
            f"{name:10s} {part['n_companies']:5,} companies ({part['company_share']:.1%}), "
            f"failed {part['n_failed']:3} ({part['failed_share']:.2%}) | years {shares}"
        )
    print(f"Saved split to {directory} and summary to {args.summary}")
    if not expected:
        print("No split checksums in the config yet; record these:", json.dumps(checksums))
    return 0


if __name__ == "__main__":
    sys.exit(main())
