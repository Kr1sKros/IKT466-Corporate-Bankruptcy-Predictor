# Real-data audit for issue #44

The saved [Kaggle v3 audit](kaggle-v3.json) was computed from the actual CSV on
3 November 2026. It contains summaries and provenance, not company financial
records or model results. `verify_target_labels` is the shared record validator;
`audit_dataset` reuses it. The issue described an existing local validator, but
published `main` and the discovered local checkout contained placeholders, so
the checked interface has been implemented here rather than claiming reuse of
unavailable code.

## Observed file

- Source handle: `utkarshx27/american-companies-bankruptcy-prediction-dataset`.
- Kaggle's metadata API reported current version **3**, last updated
  `2023-05-27T15:59:52.35Z` when downloaded. The CLI's version flag records caller
  information; it does not independently authenticate provenance.
- File: `american_bankruptcy.csv`, 11,254,588 bytes.
- SHA-256: `cff2c899a97ecd629415cb22f59186000e74e1c0a78cfae036c0a53025419b5e`.
- 78,682 rows, 8,971 companies, 21 columns, accounting years 1999–2018.
- 609 failed companies / 8,971 (about 6.79%); 5,220 failed-labeled rows / 78,682
  (about 6.63%). These are different denominators.
- No missing cells, duplicate company/year pairs, infinite features, unknown
  labels, or changing within-company status labels were found.
- 710 companies have year gaps: 810 gap intervals containing 1,851 missing
  company-years between observed rows. Missing years are different from missing
  cells. No gaps were filled and no histories were discarded.
- Companies with 1–5 consecutive years ending at their own final-year anchor:
  **8,971 / 8,078 / 7,101 / 6,298 / 5,600**. These counts measure coverage;
  short-history and gap-handling choices remain for #49.

The README's row/company/failure counts and year range match this Kaggle file.
The publisher's [dataset repository](https://github.com/sowide/bankruptcy_dataset)
and [2022 article](https://doi.org/10.3390/fi14080244) instead describe 8,262
companies. This is a real source discrepancy, not grounds for silently changing
the measured count or claiming the versions are equivalent.

## Source comparison and limits

The publisher repository was inspected at commit
`8bcd8db3b432b1fa6d65e26753b5c9fc6567438d`. Its raw
`american_bankruptcy_dataset.csv` has SHA-256
`563b0eb19959a7b21fff7f194bd06bd4294e360bb578299a7944041f4bdcb5ca`,
12,474,507 bytes, 78,682 rows and **8,971 distinct company IDs**, despite the
8,262-company prose claim. Its row/company label totals match the Kaggle
summary, but its schema uses `fyear`, includes industry columns, and its X-column
ordering/numerical values differ. It is not a byte-identical replacement.
Feature definitions, units and mapping must be reconciled in #45 before treating
the files as interchangeable. The cause of the publisher's prose/count
discrepancy cannot be established from these files alone.

The same source's `dataset_paper.zip` is a distinct prepared format: its three
financial tables contain 3,142 training, 233 validation and 2,815 test rows,
with `cik`, `fyear`, status and three years of named features. It is not the
78,682-row long-format Kaggle CSV; do not combine the two layouts or their
partitions without an explicit reconciliation.

The publisher documents a pre-filing-year convention and states that its data
has no missing, synthetic or imputed observations. The CSV confirms no missing
cells in this download. It cannot prove that observations were never imputed or
synthetically constructed upstream; that part remains a publisher assertion.

All histories have constant provided status. There are **4,611 failed-labeled
preterminal rows**; they must not automatically become positive next-year
examples. Failed-company terminal-year counts, shifted forward by one year,
match the article's annual bankruptcy counts (e.g. accounting 1999: 3, 2018: 36).
This is aggregate consistency with the documented convention, not independent
verification of individual filing dates or alive-company follow-up. Neither
filing dates nor censoring/follow-up dates exist in this CSV. The saved audit
therefore deliberately keeps `target_timing_verified: false`. #45 must resolve
this evidence before a next-year forecasting claim is accepted. No model was
trained and no labels were rewritten.

## Reproduce

Install the project and obtain the exact CSV from the
[configured Kaggle source](https://www.kaggle.com/datasets/utkarshx27/american-companies-bankruptcy-prediction-dataset).
Check its hash against the value above, then run from the repository root:

```bash
python -m pip install -e .
python -m pytest tests/test_data_audit.py
python scripts/inspect_data.py --path /path/to/american_bankruptcy.csv --dataset-version 3 --output data/processed/dataset_audit.json
```

With no `--path`, the script uses the existing KaggleHub downloader, which can
retrieve a newer version; record and compare its hash instead of assuming it
is version 3. `--dataset-version` is an explicit caller-supplied provenance
field, not a downloader pin. Invalid schema, duplicate IDs/years, non-numeric or
infinite features, fractional years and ambiguous labels fail with a nonzero
exit status. Missing feature cells are reported without imputation. Source data
is never changed. A schema-valid audit does not certify the prediction target.

Validation: 14 audit tests passed, with 3 unrelated original placeholders
skipped. Tests cover unsorted/gapped histories, coverage at the final anchor,
ambiguous records and headers, missingness, immutable CSV input, output hashes,
and CLI failure behavior. The real CSV audit completed successfully. No raw
dataset files are committed.
