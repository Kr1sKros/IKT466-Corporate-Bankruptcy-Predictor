"""Audit the real dataset and save the report (issue #44).

Usage (from the repository root):
    python scripts/inspect_data.py
    python scripts/inspect_data.py --publisher-csv data/raw/american_bankruptcy_dataset.csv

Exits with status 1 if the data contains ambiguous records or if the file's
checksum differs from the one recorded in configs/config.yaml.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

from data_loader import dataset_provenance, download_dataset, load_data
from validation import DataValidationError, audit_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "config.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "metrics" / "data_audit.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path, help="Local CSV (default: KaggleHub download)")
    parser.add_argument("--publisher-csv", type=Path, help="Publisher's GitHub CSV to compare")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def print_summary(report: dict) -> None:
    m, labels, history = report["measured"], report["labels"], report["history"]
    print(f"File: {report['provenance']['file_name']}  sha256={report['provenance']['sha256'][:12]}...")
    print(f"Rows {m['n_rows']:,} | companies {m['n_companies']:,} | years {m['first_year']}-{m['last_year']}")
    print(
        f"Failed companies {m['n_failed_companies']} ({m['failed_company_share']:.2%}); "
        f"failed rows {labels['row_label_counts'].get('failed', 0):,}, "
        f"{labels['failed_rows_before_terminal_year']:,} of them before the terminal year"
    )
    print(
        f"Missing values {report['feature_values']['missing_values_total']}; "
        f"companies with year gaps {history['companies_with_year_gaps']}; "
        f"identical feature columns {report['identical_feature_columns']}"
    )
    for k, cov in history["lookback_coverage"].items():
        print(
            f"  k={k}: {cov['companies_with_k_consecutive_years_ending_at_anchor']:,} companies "
            f"({cov['failed_companies_with_k_consecutive_years']} failed) with {k} consecutive years"
        )
    for name in ("discrepancies_vs_readme", "discrepancies_vs_publisher_documentation"):
        for d in report[name]:
            print(f"  [{name}] {d['statistic']}: reference {d['reference']} vs measured {d['measured']}")
    if "publisher_csv_comparison" in report:
        cmp = report["publisher_csv_comparison"]
        print(
            f"Publisher CSV: same column names={cmp['same_column_names']}, "
            f"labels identical={cmp['labels_identical']}, "
            f"cells 1000x larger={cmp['matched_cells_publisher_1000x']:,}/{cmp['total_cells']:,}"
        )


def main() -> int:
    args = parse_args()
    path = args.csv or download_dataset()
    data = load_data(path)
    provenance = dataset_provenance(path)
    publisher = pd.read_csv(args.publisher_csv) if args.publisher_csv else None

    try:
        report = audit_dataset(data, provenance=provenance, publisher_data=publisher)
    except DataValidationError as error:
        print(f"Audit failed: {error}", file=sys.stderr)
        return 1

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    expected_sha = config.get("dataset", {}).get("sha256")
    report["checksum_matches_config"] = expected_sha == provenance["sha256"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print_summary(report)
    print(f"Saved audit to {args.output}")

    if not report["checksum_matches_config"]:
        print(
            f"Checksum {provenance['sha256']} differs from config ({expected_sha}); "
            "the dataset version has changed.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
