"""Fit and inspect preprocessing using only persisted training anchor rows."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from preprocessing import (file_sha256, fit_preprocessor, load_splits,
                           training_anchor_rows)
from validation import FEATURE_COLUMNS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('configs/config.yaml'))
    parser.add_argument('--output', type=Path, default=Path('results/models/anchor_preprocessor.json'))
    parser.add_argument('--report', type=Path, default=Path('results/metrics/training_anchor_distribution.json'))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    policy = config['preprocessing']
    expected_policy = {
        'scaling_strategy': 'standard', 'missing_value_strategy': 'error',
        'fit_population': 'training_company_final_year',
        'reuse_across_input_conditions': True, 'fit_on_training_only': True,
        'model_scaling': {'logistic_regression': 'standard', 'xgboost': 'identity',
                          'mlp': 'standard', 'gru': 'standard'},
        'relative_offsets': 'unchanged', 'padding': 'zero_after_transform',
    }
    if any(policy.get(key) != value for key, value in expected_policy.items()):
        raise ValueError('Configuration differs from the agreed fixed preprocessing policy')
    if file_sha256(args.csv) != config['dataset']['sha256']:
        raise ValueError('Dataset differs from the audited CSV')
    split = config['split']
    companies = load_splits(split['directory'], expected_sha256=split['sha256'])
    data = pd.read_csv(args.csv)
    fitted = fit_preprocessor(data, companies)
    values = training_anchor_rows(data, companies)[FEATURE_COLUMNS]
    diagnostics = {}
    standardized = fitted.transform(values, model='gru')
    for name in FEATURE_COLUMNS:
        col = values[name]
        skew = float(col.skew()) if len(col) >= 3 and col.nunique() > 1 else None
        q1, median, q3 = col.quantile([0.25, 0.5, 0.75])
        iqr = q3 - q1
        diagnostics[name] = {
            'count': len(col), 'missing': int(col.isna().sum()),
            'min': float(col.min()), 'q01': float(col.quantile(.01)),
            'q25': q1, 'median': median, 'q75': q3,
            'q99': float(col.quantile(.99)), 'max': float(col.max()),
            'mean': float(col.mean()), 'std_population': float(col.std(ddof=0)),
            'skewness': skew if skew is not None and np.isfinite(skew) else None,
            'outside_1_5_iqr_fraction': float(((col < q1 - 1.5 * iqr) | (col > q3 + 1.5 * iqr)).mean()),
            'max_absolute_standardized': float(np.abs(standardized[:, FEATURE_COLUMNS.index(name)]).max()),
        }
    provenance = {'dataset_sha256': file_sha256(args.csv), 'split_sha256': split['sha256']}
    fitted.record['provenance'] = provenance
    report = {'fitting_population': 'one final-year row per training company',
              'n_companies': len(values), 'features': diagnostics, **provenance,
              'policy': 'fixed StandardScaler for LR/MLP/GRU; identity for XGBoost; no Optuna selection'}
    inputs = [args.csv, args.config, *(Path(split['directory']) / f'{p}.csv' for p in ('train', 'validation', 'test'))]
    if args.output.resolve() == args.report.resolve() or any(p.resolve() in {args.output.resolve(), args.report.resolve()} for p in inputs):
        raise ValueError('Outputs must be distinct and cannot overwrite inputs')
    report_text = json.dumps(report, indent=2, allow_nan=False) + '\n'
    fitted.save(args.output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report_text, encoding='utf-8')
    print(json.dumps({'training_companies': len(values), 'report': str(args.report), 'transform': str(args.output)}))


if __name__ == '__main__':
    main()
