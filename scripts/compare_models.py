"""Compare saved model/lookback CSVs against the persisted test partition."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml

from evaluation import compare_predictions
from preprocessing import load_splits


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('configs/config.yaml'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--min-history', type=int, choices=range(1, 6))
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    split_config = config['split']
    splits = load_splits(split_config['directory'], expected_sha256=split_config['sha256'])
    test = splits.loc[splits['split'].eq('test')].copy()
    runs = json.loads(args.manifest.read_text(encoding='utf-8'))['runs']
    predictions, thresholds, sources = {}, {}, {}
    for run in runs:
        name = run['name']
        if not isinstance(name, str) or not name.strip() or name in predictions:
            raise ValueError('Run names must be nonempty and unique')
        if run.get('threshold_source') != 'validation':
            raise ValueError(f'{name}: threshold_source must be validation')
        path = args.manifest.parent / run['csv']
        with path.open(encoding='utf-8-sig', newline='') as handle:
            header = next(csv.reader(handle), [])
        if not header or len(header) != len(set(header)):
            raise ValueError(f'{name}: CSV column names must be nonempty and unique')
        predictions[name] = pd.read_csv(path, dtype={'company_name': str})
        thresholds[name] = run['threshold']
        sources[name] = {
            'csv': str(path.resolve()),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'threshold_source': 'validation',
        }
    report = compare_predictions(predictions, expected=test, thresholds=thresholds, min_history=args.min_history)
    report['prediction_sources'] = sources
    report['split_sha256'] = split_config['sha256']
    protected = [args.config, args.manifest]
    protected.extend(args.manifest.parent / run['csv'] for run in runs)
    protected.extend(Path(split_config['directory']) / f'{name}.csv' for name in ['train', 'validation', 'test'])
    if args.output.resolve() in {path.resolve() for path in protected}:
        raise ValueError('Output must not overwrite an input file')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
