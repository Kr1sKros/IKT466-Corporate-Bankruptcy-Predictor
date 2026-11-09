"""Repeat actual RNG, loader, and small CPU training runs, plus provenance checks."""
import hashlib
import json
import random

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from reproducibility import (
    configure_reproducibility, estimator_seed_parameters, file_provenance,
    make_data_loader, write_run_metadata,
)


class RandomRows(Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, index):
        return index, random.random(), np.random.random(), torch.rand(()).item()


def test_python_numpy_torch_seeds_repeat():
    def sample(seed):
        configure_reproducibility(seed)
        return random.random(), np.random.random(), torch.rand(3)
    first, second, different = sample(42), sample(42), sample(43)
    assert first[:2] == second[:2]
    assert torch.equal(first[2], second[2])
    assert first[:2] != different[:2]
    assert torch.are_deterministic_algorithms_enabled()


@pytest.mark.parametrize('seed', [-1, 2**32, True, 1.5, None])
def test_invalid_seeds_rejected(seed):
    with pytest.raises(ValueError):
        configure_reproducibility(seed)


def test_shuffled_multiworker_loaders_repeat():
    def rows(seed):
        return torch.cat([torch.stack(batch, dim=1) for batch in make_data_loader(
            RandomRows(), seed=seed, shuffle=True, batch_size=4, num_workers=2)])
    assert torch.equal(rows(42), rows(42))
    assert not torch.equal(rows(42), rows(43))


@pytest.mark.parametrize('family', ['mlp', 'gru'])
def test_tiny_torch_training_repeats(family):
    def run():
        configure_reproducibility(7)
        if family == 'mlp':
            model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU(), torch.nn.Linear(4, 1))
            inputs = torch.arange(12, dtype=torch.float32).reshape(4, 3) / 12
            forward = lambda: model(inputs)
        else:
            model = torch.nn.GRU(3, 1, batch_first=True)
            inputs = torch.arange(24, dtype=torch.float32).reshape(4, 2, 3) / 24
            forward = lambda: model(inputs)[0][:, -1, :]
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        for _ in range(3):
            optimizer.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(forward(), torch.tensor([[0.], [1.], [0.], [1.]]))
            loss.backward()
            optimizer.step()
        return forward().detach()
    assert torch.equal(run(), run())


@pytest.mark.parametrize('family', ['logistic_regression', 'xgboost'])
def test_estimator_training_repeats(family):
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier

    features = np.random.default_rng(10).normal(size=(30, 3))
    labels = (features[:, 0] > 0).astype(int)
    constructor = LogisticRegression if family == 'logistic_regression' else XGBClassifier
    options = {} if family == 'logistic_regression' else {'n_estimators': 4, 'max_depth': 2, 'subsample': 0.8}
    def fit():
        model = constructor(**estimator_seed_parameters(family, 42), **options)
        return model.fit(features, labels).predict_proba(features)
    np.testing.assert_array_equal(fit(), fit())


@pytest.mark.parametrize('family', ['logistic_regression', 'xgboost', 'mlp', 'gru'])
def test_metadata_hashes_inputs_checkpoint_and_records_runtime(tmp_path, family):
    files = {}
    for name in ['dataset', 'train', 'validation', 'test', 'configuration', 'checkpoint']:
        files[name] = tmp_path / name
        files[name].write_bytes(name.encode())
    output = tmp_path / 'run.json'
    settings = configure_reproducibility(42)
    record = write_run_metadata(output, model=family, parameters={'threshold': 0.5},
        seed_settings=settings, dataset=files['dataset'],
        splits={name: files[name] for name in ['train', 'validation', 'test']},
        configuration=files['configuration'], checkpoint=files['checkpoint'],
        dataset_version='1', repository=tmp_path)
    assert json.loads(output.read_text()) == record
    assert record['checkpoint']['sha256'] == hashlib.sha256(b'checkpoint').hexdigest()
    assert record['dataset']['publisher_version'] == '1'
    assert record['software']['packages']['torch'] is not None
    assert record['hardware']['machine']
    assert record['source']['git_commit'] is None
    before = file_provenance(files['test'])['sha256']
    files['test'].write_bytes(b'different split')
    assert file_provenance(files['test'])['sha256'] != before
    with pytest.raises(ValueError, match='overwrite'):
        write_run_metadata(files['dataset'], model=family, parameters={}, seed_settings=settings,
            dataset=files['dataset'], configuration=files['configuration'],
            splits={name: files[name] for name in ['train', 'validation', 'test']})
    assert files['dataset'].read_bytes() == b'dataset'
