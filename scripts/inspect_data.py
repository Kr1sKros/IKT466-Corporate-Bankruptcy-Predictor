"""Inspect raw data without modifying it or interpreting labels as future events."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from data_loader import DATASET_HANDLE, download_dataset, load_data
from preprocessing import audit_dataset


def main() -> None:
    """Save a repeatable audit or fail on ambiguous input without changing it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--path', type=Path, help='Local CSV; omitted uses the configured Kaggle download')
    parser.add_argument('--output', type=Path, default=Path('data/processed/dataset_audit.json'))
    parser.add_argument('--dataset-version', help='Explicit source version, if independently established')
    args = parser.parse_args()
    path = args.path if args.path is not None else download_dataset()
    if path.resolve() == args.output.resolve():
        parser.error('Audit output must not overwrite the CSV')
    data = load_data(path)
    report = audit_dataset(data)
    report['provenance'] = {
        'dataset_handle': DATASET_HANDLE if args.path is None or args.dataset_version is not None else None,
        'source_version_supplied_by_caller': args.dataset_version,
        'csv_filename': path.name,
        'csv_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'csv_size_bytes': path.stat().st_size,
        'audited_at_utc': datetime.now(timezone.utc).isoformat(),
    }
    report['readme_comparison'] = {
        'rows': {'expected': 78682, 'observed': report['rows']},
        'companies': {'expected': 8971, 'observed': report['companies']},
        'failed_companies': {'expected': 609, 'observed': report['company_label_counts']['failed']},
        'year_range': {'expected': [1999, 2018], 'observed': report['year_range']},
    }
    report['readme_discrepancies'] = [key for key, value in report['readme_comparison'].items() if value['expected'] != value['observed']]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
    print(f"Audited {report['rows']:,} rows / {report['companies']:,} companies: {args.output}")
    print('Target timing remains unverified; see audit limitations before modelling.')


if __name__ == "__main__":
    main()
