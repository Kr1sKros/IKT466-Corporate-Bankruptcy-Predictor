"""Shared seeds, deterministic data loading, and content-based run provenance."""
from __future__ import annotations

from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess

import numpy as np


def _validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**32:
        raise ValueError('Seed must be an integer in [0, 2**32)')
    return int(seed)


def configure_reproducibility(seed: int, *, deterministic: bool = True) -> dict[str, object]:
    """Call before creating models, tensors, loaders, or a CUDA context.

    Python/NumPy/PyTorch CPU and CUDA RNGs are seeded. NumPy Generator instances
    remain independent: create each with np.random.default_rng(seed). Set
    PYTHONHASHSEED before starting Python if hash iteration order matters.
    Deterministic mode raises on unsupported nondeterministic PyTorch kernels.
    """
    import torch

    seed = _validate_seed(seed)
    if not isinstance(deterministic, bool):
        raise ValueError('deterministic must be a bool')
    if deterministic:
        config = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
        if config not in {None, ':4096:8', ':16:8'}:
            raise ValueError('Unsupported CUBLAS_WORKSPACE_CONFIG for deterministic execution')
        if torch.cuda.is_initialized() and config is None:
            raise RuntimeError('Configure reproducibility before initializing CUDA')
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.set_num_threads(1)
    return {
        'seed': seed,
        'deterministic_algorithms': deterministic,
        'cudnn_benchmark': False,
        'cudnn_deterministic': deterministic,
        'torch_num_threads': torch.get_num_threads(),
        'python_hash_seed_at_startup': os.environ.get('PYTHONHASHSEED'),
        'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
    }


def estimator_seed_parameters(model: str, seed: int) -> dict[str, int]:
    """Pass explicitly to LR/XGBoost constructors; do not rely on global RNGs."""
    seed = _validate_seed(seed)
    if model == 'logistic_regression':
        return {'random_state': seed}
    if model == 'xgboost':
        return {'random_state': seed, 'n_jobs': 1}
    raise ValueError('Expected logistic_regression or xgboost; MLP/GRU use PyTorch seeds')


def seed_worker(worker_id: int) -> None:
    """Top-level callback (Windows-spawn compatible) for PyTorch DataLoader."""
    import torch

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_data_loader(dataset, *, seed: int, **kwargs):
    """Create an independently seeded loader for either the MLP or GRU.

    Give separate loaders their own generator; recreating a loader resets its
    order, while subsequent epochs on the same loader advance its RNG state.
    """
    import torch
    from torch.utils.data import DataLoader

    seed = _validate_seed(seed)
    if {'generator', 'worker_init_fn'} & kwargs.keys():
        raise ValueError('Loader generator and worker initialization are managed here')
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, generator=generator, worker_init_fn=seed_worker, **kwargs)


def file_provenance(path: str | Path) -> dict[str, object]:
    """Hash actual file bytes, not filenames or modification timestamps."""
    path = Path(path)
    digest = hashlib.sha256()
    size = 0
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
            size += len(block)
    return {'path': str(path), 'sha256': digest.hexdigest(), 'size_bytes': size}


def write_run_metadata(
    output: str | Path, *, model: str, parameters: dict, seed_settings: dict,
    dataset: str | Path, splits: dict[str, str | Path], configuration: str | Path,
    checkpoint: str | Path | None = None, repository: str | Path = '.',
    dataset_version: str | None = None,
) -> dict[str, object]:
    """Persist actual settings and hashes; call after saving the selected model.

    For pre-training metadata checkpoint may be None; write again after saving.
    Unknown publisher dataset versions stay None, never inferred from a hash.
    Include all selected hyperparameters and threshold rules in parameters.
    """
    import torch

    if model not in {'logistic_regression', 'xgboost', 'mlp', 'gru'}:
        raise ValueError('Unknown study model')
    if set(splits) != {'train', 'validation', 'test'}:
        raise ValueError('Provide the persisted train, validation, and test split files')
    _validate_seed(seed_settings.get('seed'))
    packages = {}
    for name in ['numpy', 'pandas', 'scikit-learn', 'xgboost', 'torch', 'pyyaml']:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    try:
        commit = subprocess.check_output(['git', '-C', str(repository), 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(['git', '-C', str(repository), 'status', '--porcelain'], stderr=subprocess.DEVNULL, text=True))
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    record = {
        'schema_version': 1,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'model': model,
        'parameters': parameters,
        'seed_settings': seed_settings,
        'configuration': file_provenance(configuration),
        'dataset': {**file_provenance(dataset), 'publisher_version': dataset_version},
        'splits': {name: file_provenance(path) for name, path in sorted(splits.items())},
        'checkpoint': file_provenance(checkpoint) if checkpoint is not None else None,
        'source': {'git_commit': commit, 'dirty_worktree': dirty},
        'software': {'python': platform.python_version(), 'packages': packages},
        'hardware': {
            'platform': platform.platform(), 'machine': platform.machine(),
            'cuda_available': torch.cuda.is_available(), 'cuda_version': torch.version.cuda,
            'cudnn_version': torch.backends.cudnn.version(),
            'cuda_devices': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        },
    }
    # Serialize before touching output so non-JSON values cannot leave partial metadata.
    text = json.dumps(record, indent=2, sort_keys=True, allow_nan=False)
    output = Path(output)
    inputs = [dataset, configuration, *splits.values()]
    if checkpoint is not None:
        inputs.append(checkpoint)
    if output.resolve() in {Path(path).resolve() for path in inputs}:
        raise ValueError('Metadata output cannot overwrite an input or checkpoint')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + '\n', encoding='utf-8')
    return record
