"""Fit and inspect the per-condition preprocessing on persisted training rows only."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from preprocessing import (FIT_POPULATIONS, MODEL_STRATEGIES, file_sha256,
                           fit_preprocessor, load_splits, training_fit_rows)
from validation import FEATURE_COLUMNS

EXPECTED_POLICY = {
    'scaling_strategy': 'signed_log1p_standard', 'missing_value_strategy': 'error',
    'fit_population': FIT_POPULATIONS,
    'reuse_across_input_conditions': False, 'fit_on_training_only': True,
    'model_scaling': MODEL_STRATEGIES,
    'relative_offsets': 'unchanged', 'padding': 'zero_after_transform',
}


def describe(values: pd.DataFrame, standardized: np.ndarray) -> dict:
    """Per-feature distribution of the fitting rows (training data only).

    Raw statistics describe the dollar values; the ``transformed_*`` fields
    describe what LR/MLP/GRU receive (signed log1p, then standardized).
    """
    diagnostics = {}
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
            'transformed_skewness': float(pd.Series(standardized[:, FEATURE_COLUMNS.index(name)]).skew())
            if len(col) >= 3 and col.nunique() > 1 else None,
        }
    return diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('configs/config.yaml'))
    parser.add_argument('--output', type=Path, default=Path('results/models/preprocessor.json'))
    parser.add_argument('--report', type=Path, default=Path('results/metrics/preprocessing_fit_distribution.json'))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    policy = config['preprocessing']
    if any(policy.get(key) != value for key, value in EXPECTED_POLICY.items()):
        raise ValueError('Configuration differs from the agreed fixed preprocessing policy')
    if file_sha256(args.csv) != config['dataset']['sha256']:
        raise ValueError('Dataset differs from the audited CSV')
    split = config['split']
    inputs = [args.csv, args.config, *(Path(split['directory']) / f'{p}.csv' for p in ('train', 'validation', 'test'))]
    if args.output.resolve() == args.report.resolve() or any(
            p.resolve() in {args.output.resolve(), args.report.resolve()} for p in inputs):
        raise ValueError('Outputs must be distinct and cannot overwrite inputs')

    companies = load_splits(split['directory'], expected_sha256=split['sha256'])
    data = pd.read_csv(args.csv)
    fitted = fit_preprocessor(data, companies)
    provenance = {'dataset_sha256': file_sha256(args.csv), 'split_sha256': split['sha256']}
    fitted.record['provenance'] = provenance

    conditions = {}
    for condition, population in FIT_POPULATIONS.items():
        values = training_fit_rows(data, companies, condition=condition)[FEATURE_COLUMNS]
        standardized = fitted.transform(values, model='gru', condition=condition)
        conditions[condition] = {
            'fitting_population': population,
            'n_companies': fitted.record['conditions'][condition]['n_training_companies'],
            'n_observations': len(values),
            'features': describe(values, standardized),
        }
    report = {'conditions': conditions, **provenance,
              'policy': 'fixed signed log1p + StandardScaler per condition for LR/MLP/GRU; identity for XGBoost; no Optuna selection'}
    report_text = json.dumps(report, indent=2, allow_nan=False) + '\n'
    fitted.save(args.output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report_text, encoding='utf-8')
    print(json.dumps({c: {'companies': v['n_companies'], 'observations': v['n_observations']}
                      for c, v in conditions.items()}))
    print(json.dumps({'report': str(args.report), 'transform': str(args.output)}))


if __name__ == '__main__':
    main()
