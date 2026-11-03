"""Synthetic history audits and CLI failure checks without downloading data."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from data_loader import load_data
from preprocessing import audit_dataset, verify_target_labels


@pytest.fixture
def records():
    return pd.DataFrame([
        {'company_name': company, 'year': year, 'status_label': label,
         **{f'X{i}': float(i) for i in range(1, 19)}}
        for company, year, label in [('a', 2000, 'failed'), ('a', 2002, 'failed'),
            ('a', 2003, 'failed'), ('b', 2001, 'alive'), ('b', 2002, 'alive'),
            ('c', 2004, 'alive')]
    ])


def test_history_audit_distinguishes_rows_companies_and_contiguous_tails(records):
    before = records.copy(deep=True)
    report = audit_dataset(records.sample(frac=1, random_state=4))
    assert report['rows'] == 6
    assert report['companies'] == 3
    assert report['row_label_counts'] == {'alive': 3, 'failed': 3}
    assert report['company_label_counts'] == {'alive': 2, 'failed': 1}
    assert report['gap_intervals'] == report['companies_with_year_gaps'] == 1
    assert report['missing_company_years_between_observations'] == 1
    assert report['companies_with_at_least_k_rows']['3'] == 1
    assert report['companies_with_k_consecutive_years_ending_at_anchor']['3'] == 0
    assert report['companies_with_k_consecutive_years_ending_at_anchor']['2'] == 2
    assert report['failed_preterminal_rows'] == 2
    assert report['target_timing_verified'] is False
    pd.testing.assert_frame_equal(records, before)
    json.dumps(report, allow_nan=False)


def test_missing_features_are_reported_without_imputation(records):
    records.loc[0, 'X1'] = np.nan
    assert verify_target_labels(records)['missing_values_by_column']['X1'] == 1
    assert pd.isna(records.loc[0, 'X1'])


@pytest.mark.parametrize('column,value', [('company_name', None), ('company_name', ' a '),
    ('company_name', ''), ('year', 2000.5), ('year', np.nan),
    ('status_label', 'unknown'), ('status_label', 'alive'), ('X1', np.inf),
    ('X1', 'not a number')])
def test_ambiguous_records_fail(records, column, value):
    records[column] = records[column].astype(object)
    records.loc[0, column] = value
    with pytest.raises(ValueError):
        audit_dataset(records)


def test_duplicate_keys_columns_and_missing_schema_fail(records):
    for invalid in [pd.concat([records, records.iloc[[0]]]),
                    records.drop(columns='X18'),
                    pd.concat([records, records[['X1']]], axis=1), records.iloc[:0]]:
        with pytest.raises(ValueError):
            audit_dataset(invalid)


def test_cli_saves_provenance_without_changing_csv(records, tmp_path):
    csv = tmp_path / 'data.csv'
    output = tmp_path / 'audit.json'
    records.to_csv(csv, index=False)
    original = csv.read_bytes()
    command = [sys.executable, str(Path(__file__).parents[1] / 'scripts/inspect_data.py'),
               '--path', str(csv), '--output', str(output)]
    subprocess.run(command, check=True, capture_output=True)
    report = json.loads(output.read_text())
    assert report['provenance']['csv_sha256'] == hashlib.sha256(original).hexdigest()
    assert report['provenance']['dataset_handle'] is None
    assert report['readme_discrepancies']
    assert csv.read_bytes() == original
    assert subprocess.run(command[:-1] + [str(csv)], capture_output=True).returncode != 0
    assert csv.read_bytes() == original
    # Malformed data must not produce a fresh success artifact.
    records.loc[0, 'status_label'] = 'unknown'
    records.to_csv(csv, index=False)
    output.unlink()
    assert subprocess.run(command, capture_output=True).returncode != 0
    assert not output.exists()


def test_duplicate_csv_headers_rejected(tmp_path):
    path = tmp_path / 'ambiguous.csv'
    path.write_text('X1,X1\n1,2\n')
    with pytest.raises(ValueError, match='unique'):
        load_data(path)
