"""Run the two agreed Logistic Regression studies, validation only; final test evaluation is deferred."""
import argparse
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import yaml

from data_loader import load_data
from leakage import check_partitions, check_study_plan, check_preprocessor
from models.logistic_regression import LogisticPipeline, tune_condition, write_json
from preprocessing import FittedPreprocessor, load_splits, split_paths, file_sha256, transform_histories
from reproducibility import configure_reproducibility, file_provenance, write_run_metadata
from sequences import create_histories, flatten_histories


def run(csv, config_path, output, preprocessor_path=Path("results/models/preprocessor.json")):
    csv, config_path, output = Path(csv), Path(config_path), Path(output)
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    check_study_plan(config)
    lr = config['models']['logistic_regression']
    if lr.get('solver') != 'lbfgs' or lr.get('regularization') != 'L2':
        raise ValueError('The implemented LR protocol requires lbfgs with L2 regularization')
    if lr['search'].get('positive_class_weight') != 'log_1_to_training_ratio':
        raise ValueError('Declare continuous training-derived positive class weights')
    if file_sha256(csv) != config['dataset']['sha256']:
        raise ValueError('Dataset checksum differs from audited data')
    if config['evaluation'].get('threshold_rule') != 'max_f1_highest_threshold_on_tie':
        raise ValueError('Declare the agreed validation F1 threshold rule')
    # A new directory prevents overwriting or silently resuming completed studies.
    if output.exists():
        raise FileExistsError('Choose a new output directory; existing studies are never overwritten')
    companies = load_splits(config['split']['directory'], expected_sha256=config['split']['sha256'])
    data = load_data(csv)
    check_partitions(companies, data)
    preprocessor_path = Path(preprocessor_path)
    preprocessor = FittedPreprocessor.load(preprocessor_path)
    expected_provenance = {'dataset_sha256': file_sha256(csv), 'split_sha256': config['split']['sha256']}
    if preprocessor.record.get('provenance') != expected_provenance:
        raise ValueError('Shared preprocessor provenance differs from dataset/splits')
    # Audit refits a reference for verification only; the loaded artifact is used.
    check_preprocessor(preprocessor, data, companies)
    seed_settings = configure_reproducibility(config['random_seed'])
    output.mkdir(parents=True)
    preprocessor.save(output / 'preprocessor.json')
    write_json(output / 'run_config.json', config)
    write_json(output / 'run_context.json', {
        'shared_preprocessor': file_provenance(preprocessor_path),
        'dataset': file_provenance(csv), 'configuration': file_provenance(config_path),
        'splits': {k: file_provenance(v) for k, v in split_paths(config['split']['directory']).items()},
        'seed_settings': seed_settings, 'device': 'cpu', 'blas_threads': 1,
        'packages': {p: version(p) for p in ('numpy', 'pandas', 'scikit-learn', 'optuna', 'joblib', 'threadpoolctl')},
    })
    tuning = config['study_plan']['tuning']
    lr = config['models']['logistic_regression']
    start = perf_counter()
    selections = {}
    for condition in config['study_plan']['input_conditions']:
        partitions = {}
        for split in ('train', 'validation'):
            ids = companies.loc[companies['split'].eq(split), 'company_name']
            h = create_histories(data, condition=condition, companies=ids)
            x, meta = flatten_histories(transform_histories(h, preprocessor, model='logistic_regression'))
            partitions[split] = (x, meta)
        x_train, train_meta = partitions['train']
        x_val, val_meta = partitions['validation']
        directory = output / condition
        study_name = f'logistic_regression__{condition}__{output.name}'
        model, selection = tune_condition(x_train, train_meta.target, x_val, val_meta.target,
            directory=directory, study_name=study_name, seed=config['random_seed'],
            sampler_seed=tuning['sampler_seed'], trials=tuning['attempted_trials_per_study'],
            search=lr['search'], max_iter=lr['max_iter'], tol=lr['tol'])
        pipeline = LogisticPipeline(condition, preprocessor, model, selection['threshold'])
        checkpoint = directory / 'pipeline.joblib'
        joblib.dump(pipeline, checkpoint)
        loaded = joblib.load(checkpoint)
        val_predictions = loaded.predict(data, val_meta.company_name)
        np.testing.assert_array_equal(val_predictions.probability, model.predict_proba(x_val)[:, 1])
        val_predictions.to_csv(directory / 'validation_predictions.csv', index=False)
        selection.update({'model': 'logistic_regression', 'condition': condition,
                          'feature_columns': list(x_train.columns)})
        write_json(directory / 'selection.json', selection)
        metadata = write_run_metadata(directory / 'metadata.json', model='logistic_regression',
            parameters=selection, seed_settings=seed_settings, dataset=csv,
            splits=split_paths(config['split']['directory']), configuration=config_path,
            checkpoint=checkpoint, dataset_version=str(config['dataset']['kaggle_version']))
        metadata['software']['packages'].update({p: version(p) for p in ('optuna', 'joblib', 'threadpoolctl')})
        metadata['preprocessing'] = file_provenance(preprocessor_path)
        metadata['target_verified'] = False
        metadata['hardware']['training_device'] = 'cpu'
        metadata['blas_threads'] = 1
        write_json(directory / 'metadata.json', metadata)
        selections[condition] = selection
    # LR is selected, but the eight-study test gate belongs to #56/#62.
    write_json(output / 'frozen_selections.json', selections)
    summary = {'status': 'validation_only', 'selections': selections,
               'elapsed_seconds': perf_counter() - start,
               'test_evaluation': 'deferred_until_all_eight_studies_are_frozen'}
    write_json(output / 'validation_summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', required=True, type=Path)
    parser.add_argument('--config', default=Path('configs/config.yaml'), type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--preprocessor', default=Path('results/models/preprocessor.json'), type=Path)
    args = parser.parse_args()
    result = run(args.csv, args.config, args.output, args.preprocessor)
    print({name: selection['validation_average_precision'] for name, selection in result['selections'].items()})


if __name__ == '__main__':
    main()
