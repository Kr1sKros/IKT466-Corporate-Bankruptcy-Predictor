"""Tune latest-year/five-year XGBoost separately, then evaluate frozen models."""
import argparse
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import yaml

from data_loader import load_data
from evaluation import compare_predictions
from leakage import check_partitions, check_study_plan
from metrics import compute_metrics
from models.xgboost_model import XGBoostPipeline, tune_condition, validate_search, write_json
from preprocessing import (MODEL_STRATEGIES, file_sha256, fit_preprocessor, load_splits,
                           split_paths, transform_histories)
from reproducibility import configure_reproducibility, file_provenance, write_run_metadata
from sequences import CONDITIONS, create_histories, flatten_histories


def run(csv, config_path, output):
    csv, config_path, output = Path(csv).resolve(), Path(config_path).resolve(), Path(output).resolve()
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    check_study_plan(config)
    if config['preprocessing']['model_scaling'] != MODEL_STRATEGIES:
        raise ValueError('Use the fixed preprocessing policy: identity for XGBoost')
    if config['evaluation'].get('threshold_rule') != 'max_f1_highest_threshold_on_tie':
        raise ValueError('Declare the validation F1 threshold rule')
    search = config['models']['xgboost']['search']
    validate_search(search)
    if file_sha256(csv) != config['dataset']['sha256']:
        raise ValueError('Dataset checksum differs from the audited CSV')
    split_directory = Path(config['split']['directory'])
    if not split_directory.is_absolute():
        split_directory = config_path.parent.parent / split_directory
    if output.exists():
        raise FileExistsError('Choose a new output directory; existing studies are never overwritten')
    companies = load_splits(split_directory, expected_sha256=config['split']['sha256'])
    data = load_data(csv)
    check_partitions(companies, data)
    preprocessor = fit_preprocessor(data, companies)
    preprocessor.record['provenance'] = {
        'dataset_sha256': file_sha256(csv), 'split_sha256': config['split']['sha256']}
    seed_settings = configure_reproducibility(config['random_seed'])
    output.mkdir(parents=True)
    write_json(output / 'run_config.json', config)
    write_json(output / 'run_context.json', {
        'dataset': file_provenance(csv), 'configuration': file_provenance(config_path),
        'splits': {k: file_provenance(v) for k, v in split_paths(split_directory).items()},
        'packages': {p: version(p) for p in ('numpy', 'pandas', 'scikit-learn', 'xgboost', 'optuna')},
        'seed_settings': seed_settings, 'device': 'cpu', 'n_jobs': 1, 'target_verified': False})
    tuning = config['study_plan']['tuning']
    started = perf_counter()
    selections = {}
    for condition in CONDITIONS:
        partitions = {}
        for partition in ('train', 'validation'):
            ids = companies.loc[companies['split'].eq(partition), 'company_name']
            histories = create_histories(data, condition=condition, companies=ids)
            features, metadata = flatten_histories(
                transform_histories(histories, preprocessor, model='xgboost'))
            partitions[partition] = (features, metadata)
        x_train, train_meta = partitions['train']
        x_val, val_meta = partitions['validation']
        directory = output / condition
        model, selection = tune_condition(
            x_train, train_meta.target, x_val, val_meta.target, directory=directory,
            study_name=f'xgboost__{condition}__{output.name}', seed=config['random_seed'],
            sampler_seed=tuning['sampler_seed'], trials=tuning['attempted_trials_per_study'], search=search)
        pipeline = XGBoostPipeline(condition, preprocessor, model, selection['threshold'])
        pipeline.save(directory)
        val_predictions = XGBoostPipeline.load(directory).predict(data, val_meta.company_name)
        np.testing.assert_array_equal(val_predictions.probability, model.predict_proba(x_val)[:, 1])
        val_predictions.to_csv(directory / 'validation_predictions.csv', index=False)
        selection.update({'model': 'xgboost', 'condition': condition, 'feature_columns': list(x_train.columns)})
        write_json(directory / 'selection.json', selection)
        metadata = write_run_metadata(
            directory / 'metadata.json', model='xgboost', parameters=selection,
            seed_settings=seed_settings, dataset=csv, splits=split_paths(split_directory),
            configuration=config_path, checkpoint=directory / 'model.ubj',
            repository=Path(__file__).resolve().parents[1],
            dataset_version=str(config['dataset']['kaggle_version']))
        metadata['software']['packages']['optuna'] = version('optuna')
        metadata['preprocessing'] = file_provenance(directory / 'preprocessor.json')
        metadata['pipeline'] = file_provenance(directory / 'pipeline.json')
        metadata['target_verified'] = False
        metadata['hardware']['training_device'] = 'cpu'
        write_json(directory / 'metadata.json', metadata)
        selections[condition] = selection
    # BOTH validation selections are frozen before any test input/prediction.
    write_json(output / 'frozen_selections.json', selections)
    test = companies.loc[companies['split'].eq('test')]
    predictions, thresholds, manifest = {}, {}, []
    for condition, selection in selections.items():
        directory = output / condition
        predicted = XGBoostPipeline.load(directory).predict(data, test.company_name)
        for name, value in [('model', 'xgboost'), ('condition', condition),
                            ('study_name', selection['study_name']), ('trial_number', selection['trial_number'])]:
            predicted[name] = value
        predicted.to_csv(directory / 'test_predictions.csv', index=False)
        predictions[condition] = predicted
        thresholds[condition] = selection['threshold']
        write_json(directory / 'test_metrics.json', compute_metrics(
            predicted.target, predicted.probability, threshold=selection['threshold']))
        manifest.append({
            'name': f'xgboost__{condition}', 'model': 'xgboost', 'condition': condition,
            'csv': f'{condition}/test_predictions.csv', 'threshold': selection['threshold'],
            'threshold_source': 'validation', 'study_name': selection['study_name'],
            'trial_number': selection['trial_number'], 'selection': f'{condition}/selection.json'})
    comparison = compare_predictions(predictions, expected=test, thresholds=thresholds)
    ap5 = comparison['results']['five_year']['average_precision']
    ap1 = comparison['results']['latest_year']['average_precision']
    comparison['five_year_minus_latest_year_ap'] = None if ap5 is None or ap1 is None else ap5 - ap1
    comparison['elapsed_seconds'] = perf_counter() - started
    write_json(output / 'comparison.json', comparison)
    write_json(output / 'manifest.json', {'runs': manifest})
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', required=True, type=Path)
    parser.add_argument('--config', default=Path(__file__).resolve().parents[1] / 'configs/config.yaml', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = run(args.csv, args.config, args.output)
    print({name: metrics['average_precision'] for name, metrics in result['results'].items()})


if __name__ == '__main__':
    main()
