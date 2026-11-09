import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from evaluation import align_test_observations, compare_predictions


@pytest.fixture
def expected():
    return pd.DataFrame({'company_name': ['C_2', 'C_1', 'C_3'],
                         'anchor_year': [2018, 2017, 2016], 'target': [1, 0, 0],
                         'n_years': [5, 2, 5]})


def predictions(expected):
    a = expected.drop(columns='n_years').assign(probability=[0.8, 0.1, 0.3])
    return {'lr': a, 'gru_k5': a.iloc[::-1].copy()}


def test_alignment_is_deterministic_and_does_not_mutate(expected):
    frames = predictions(expected)
    before = frames['gru_k5'].copy(deep=True)
    aligned = align_test_observations(frames, expected=expected)
    assert list(aligned) == ['gru_k5', 'lr']
    assert aligned['lr'].company_name.tolist() == ['C_1', 'C_2', 'C_3']
    pd.testing.assert_frame_equal(aligned['lr'], aligned['gru_k5'])
    pd.testing.assert_frame_equal(frames['gru_k5'], before)


@pytest.mark.parametrize('column,value', [('anchor_year', 2000), ('target', 0),
                                           ('company_name', 'unknown'),
                                           ('anchor_year', np.nan), ('anchor_year', 2018.5),
                                           ('target', 2), ('company_name', '')])
def test_rejects_wrong_identity_or_label(expected, column, value):
    frames = predictions(expected)
    frames['lr'][column] = frames['lr'][column].astype(object if column == 'company_name' else float)
    frames['lr'].loc[0, column] = value
    with pytest.raises(ValueError):
        align_test_observations(frames, expected=expected)


def test_rejects_duplicates_and_shared_omissions(expected):
    frames = predictions(expected)
    frames['lr'] = pd.concat([frames['lr'], frames['lr'].iloc[:1]])
    with pytest.raises(ValueError, match='unique'):
        align_test_observations(frames, expected=expected)
    frames = {name: frame.loc[frame.company_name != 'C_1'] for name, frame in predictions(expected).items()}
    with pytest.raises(ValueError, match='company set'):
        align_test_observations(frames, expected=expected)


def test_main_and_common_history_subset(expected):
    frames = predictions(expected)
    thresholds = {'lr': 0.5, 'gru_k5': 0.9}
    main = compare_predictions(frames, expected=expected, thresholds=thresholds)
    assert main['retained_count'] == 3
    assert main['excluded_count'] == 0
    assert main['results']['lr']['recall'] == 1
    assert main['results']['gru_k5']['recall'] == 0
    assert main['results']['lr']['average_precision'] == 1
    subset = compare_predictions(frames, expected=expected, thresholds=thresholds, min_history=5)
    assert subset['excluded_companies'] == ['C_1']
    assert subset['retained_prevalence'] == 0.5
    assert all(result['n_observations'] == 2 for result in subset['results'].values())


@pytest.mark.parametrize('score', [np.nan, np.inf, -0.1, 1.1])
def test_invalid_scores_rejected_even_on_excluded_rows(expected, score):
    frames = predictions(expected)
    frames['lr'].loc[1, 'probability'] = score
    with pytest.raises(ValueError):
        compare_predictions(frames, expected=expected, thresholds={'lr': 0.5, 'gru_k5': 0.5}, min_history=5)


def test_empty_cohort_and_threshold_contract(expected):
    frames = predictions(expected)
    with pytest.raises(ValueError, match='threshold'):
        compare_predictions(frames, expected=expected, thresholds={})
    expected['n_years'] = 1
    with pytest.raises(ValueError, match='empty'):
        compare_predictions(frames, expected=expected, thresholds={'lr': 0.5, 'gru_k5': 0.5}, min_history=5)
    with pytest.raises(ValueError):
        align_test_observations({})


def test_cli_reads_files_checks_split_and_writes_report(tmp_path, expected):
    from preprocessing import file_sha256
    spec = importlib.util.spec_from_file_location('compare_models', Path(__file__).parents[1] / 'scripts/compare_models.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    split_dir = tmp_path / 'splits'
    split_dir.mkdir()
    checksums = {}
    for partition in ['train', 'validation', 'test']:
        table = expected.copy()
        if partition != 'test':
            table['company_name'] = partition + table.company_name
        table['history_group'] = table.n_years.astype(str)
        path = split_dir / f'{partition}.csv'
        table.to_csv(path, index=False)
        checksums[partition] = file_sha256(path)
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump({'split': {'directory': str(split_dir), 'sha256': checksums}}))
    runs = []
    for name, frame in predictions(expected).items():
        frame.to_csv(tmp_path / f'{name}.csv', index=False)
        runs.append({'name': name, 'csv': f'{name}.csv', 'threshold': 0.5, 'threshold_source': 'validation'})
    manifest = tmp_path / 'runs.json'
    manifest.write_text(json.dumps({'runs': runs}))
    output = tmp_path / 'out' / 'comparison.json'
    args = ['--config', str(config), '--manifest', str(manifest), '--output', str(output)]
    module.main(args)
    report = json.loads(output.read_text())
    assert report['retained_count'] == 3
    assert report['results']['lr']['average_precision'] == 1
    assert report['split_sha256'] == checksums
    with pytest.raises(ValueError, match='overwrite'):
        module.main(args[:-1] + [str(manifest)])
    csv_path = tmp_path / runs[0]['csv']
    original_csv = csv_path.read_text()
    csv_path.write_text(original_csv.replace('probability', 'target', 1))
    with pytest.raises(ValueError, match='unique'):
        module.main(args)
    csv_path.write_text(original_csv)
    runs[0]['threshold_source'] = 'test'
    manifest.write_text(json.dumps({'runs': runs}))
    with pytest.raises(ValueError, match='validation'):
        module.main(args)
