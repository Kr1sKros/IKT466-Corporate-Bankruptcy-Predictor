"""Score the eight frozen studies once on the held-out test companies (#62).

Reads ``frozen_studies.json`` from ``scripts/run_studies.py``. For every model x
condition it reloads the frozen pipeline (preprocessor + model + validation
threshold), first checks that it still reproduces its saved validation
predictions, then predicts every test company. Nothing is fitted or tuned here:
thresholds come from validation. The metrics come from ``compare_models.py``
(#60): AP, ROC-AUC, F1, precision, recall, accuracy and the confusion matrix on
the identical test companies for all eight runs.

Writes to ``OUTPUT`` (must not exist): ``predictions/<model>__<condition>.csv``,
``predictions/manifest.json``, ``test_comparison.json`` and ``test_summary.csv``.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from data_loader import load_data
from models.neural_study import NeuralPipeline
from models.xgboost_model import XGBoostPipeline
from preprocessing import file_sha256, load_splits

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('logistic_regression', 'xgboost', 'mlp', 'gru')
CONDITIONS = ('latest_year', 'five_year')
METRICS = ['average_precision', 'roc_auc', 'f1', 'precision', 'recall', 'accuracy', 'threshold',
           'tp', 'fp', 'fn', 'tn']


def load_pipeline(model, directory):
    if model == 'logistic_regression':
        return joblib.load(directory / 'pipeline.joblib')  # trusted: written by our own runner
    if model == 'xgboost':
        return XGBoostPipeline.load(directory)
    return NeuralPipeline.load(directory)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--studies', required=True, type=Path, help='Output folder of run_studies.py')
    parser.add_argument('--csv', required=True, type=Path)
    parser.add_argument('--config', default=ROOT / 'configs/config.yaml', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    studies_dir, output = args.studies.resolve(), args.output.resolve()
    frozen = json.loads((studies_dir / 'frozen_studies.json').read_text(encoding='utf-8'))
    if frozen.get('status') != 'all_eight_frozen':
        raise ValueError('Test evaluation needs all eight frozen studies')
    config = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    if file_sha256(args.csv) != config['dataset']['sha256']:
        raise ValueError('Dataset checksum differs from the audited CSV')
    if output.exists():
        raise FileExistsError('Choose a new output directory; test results are never overwritten')
    split = config['split']
    companies = load_splits(split['directory'], expected_sha256=split['sha256'])
    data = load_data(args.csv)
    test_ids = companies.loc[companies['split'].eq('test'), 'company_name']

    (output / 'predictions').mkdir(parents=True)
    runs = []
    for model in MODELS:
        for condition in CONDITIONS:
            study = frozen['studies'][model][condition]
            directory = studies_dir / study['directory']
            selection = json.loads((directory / 'selection.json').read_text(encoding='utf-8'))
            pipeline = load_pipeline(model, directory)
            if (pipeline.condition != condition
                    or not pipeline.threshold == selection['threshold'] == study['threshold']):
                raise ValueError(f'{model}/{condition}: frozen pipeline differs from its selection')
            saved = pd.read_csv(directory / 'validation_predictions.csv', dtype={'company_name': str})
            again = pipeline.predict(data, saved.company_name)
            if not np.allclose(again.probability.to_numpy(), saved.probability.to_numpy(), rtol=0, atol=1e-6):
                raise ValueError(f'{model}/{condition}: frozen model no longer reproduces its validation predictions')
            name = f'{model}__{condition}'
            pipeline.predict(data, test_ids).to_csv(output / 'predictions' / f'{name}.csv', index=False)
            runs.append({'name': name, 'csv': f'{name}.csv', 'threshold': selection['threshold'],
                         'threshold_source': 'validation'})
            print(f'{name}: test predictions written', flush=True)
    manifest = output / 'predictions' / 'manifest.json'
    manifest.write_text(json.dumps({'runs': runs}, indent=2) + '\n', encoding='utf-8')

    spec = importlib.util.spec_from_file_location('compare_models', ROOT / 'scripts/compare_models.py')
    compare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compare)
    report_path = output / 'test_comparison.json'
    compare.main(['--manifest', str(manifest), '--config', str(args.config), '--output', str(report_path)])
    report = json.loads(report_path.read_text(encoding='utf-8'))

    rows = []
    for model in MODELS:
        for condition in CONDITIONS:
            result = report['results'][f'{model}__{condition}']
            rows.append({'model': model, 'condition': condition,
                         'validation_ap': frozen['studies'][model][condition]['validation_average_precision'],
                         **{k: result[k] for k in METRICS}})
    table = pd.DataFrame(rows)
    table.to_csv(output / 'test_summary.csv', index=False)
    print(f"\nTest set: {report['test_count']} companies, {report['test_prevalence']:.2%} bankrupt "
          f"(AP of random guessing = {report['test_prevalence']:.3f})\n")
    with pd.option_context('display.width', 200, 'display.max_columns', 20):
        print(table.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
