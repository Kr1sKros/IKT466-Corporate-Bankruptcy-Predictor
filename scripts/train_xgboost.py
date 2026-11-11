"""Tune latest-year/five-year XGBoost on validation only; defer final test evaluation."""
import argparse
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import yaml

from data_loader import load_data
from leakage import check_partitions, check_study_plan, check_preprocessor
from models.xgboost_model import XGBoostPipeline, tune_condition, validate_search, write_json
from preprocessing import (MODEL_STRATEGIES, FittedPreprocessor, file_sha256, load_splits,
                           split_paths, transform_histories)
from reproducibility import configure_reproducibility, file_provenance, write_run_metadata
from sequences import CONDITIONS, create_histories, flatten_histories


def run(csv, config_path, output, preprocessor_path=None):
    csv, config_path, output = Path(csv).resolve(), Path(config_path).resolve(), Path(output).resolve()
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    check_study_plan(config)
    if config['preprocessing']['model_scaling'] != MODEL_STRATEGIES:
        raise ValueError('Use the fixed preprocessing policy: identity for XGBoost')
    if config['evaluation'].get('threshold_rule') != 'max_f1_highest_threshold_on_tie':
        raise ValueError('Declare the validation F1 threshold rule')
    search = config['models']['xgboost']['search']
    if config['models']['xgboost'].get('positive_class_weight') != 'log_1_to_training_ratio':
        raise ValueError('Declare continuous training-derived positive class weights')
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
    preprocessor_path = (Path(preprocessor_path).resolve() if preprocessor_path is not None
                         else config_path.parent.parent / 'results/models/preprocessor.json')
    preprocessor = FittedPreprocessor.load(preprocessor_path)
    expected_provenance = {'dataset_sha256': file_sha256(csv), 'split_sha256': config['split']['sha256']}
    if preprocessor.record.get('provenance') != expected_provenance:
        raise ValueError('Shared preprocessor provenance differs from dataset/splits')
    # Reference fitting is for the audit only; inference uses the loaded artifact.
    check_preprocessor(preprocessor, data, companies)
    seed_settings = configure_reproducibility(config['random_seed'])
    output.mkdir(parents=True)
    write_json(output / 'run_config.json', config)
    write_json(output / 'run_context.json', {
        'dataset': file_provenance(csv), 'configuration': file_provenance(config_path),
        'shared_preprocessor': file_provenance(preprocessor_path),
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
    # Only validation outputs are written. Final held-out testing belongs to #60/#62.
    write_json(output / 'frozen_selections.json', selections)
    summary = {'status': 'validation_only', 'selections': selections,
               'elapsed_seconds': perf_counter() - started,
               'test_evaluation': 'deferred_until_all_eight_studies_are_frozen'}
    write_json(output / 'validation_summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', required=True, type=Path)
    parser.add_argument('--config', default=Path(__file__).resolve().parents[1] / 'configs/config.yaml', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--preprocessor', type=Path,
                        help='Shared #46 artifact (default: results/models/preprocessor.json in the config repository)')
    args = parser.parse_args()
    result = run(args.csv, args.config, args.output, args.preprocessor)
    print({name: selection['validation_average_precision'] for name, selection in result['selections'].items()})


if __name__ == '__main__':
    main()
