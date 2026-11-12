"""Company-level splits (#47) and fixed per-condition preprocessing (#46).

Every company appears in exactly one partition, together with all of its years.
The split is stratified by outcome and by history length (number of recorded
years, grouped 1, 2, 3, 4, 5+), so every partition has about the same share of
failed companies and of short/long histories.

The split is saved once by ``scripts/prepare_data.py`` as three CSV files
(``train.csv``, ``validation.csv``, ``test.csv``) and then only *loaded* by the
model runners: ``load_splits`` never regenerates a missing or changed split.
Each file stores the company's anchor year and target, so the input condition
(latest_year or five_year) never changes a company's anchor, outcome or
partition.

Dataset validation (schema, duplicates, labels, history) lives in
``validation.py``.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from validation import COMPANY_COLUMN, YEAR_COLUMN, anchor_targets

SPLIT_NAMES = ("train", "validation", "test")
HISTORY_GROUPS = ("1", "2", "3", "4", "5+")
SPLIT_COLUMNS = [COMPANY_COLUMN, "anchor_year", "target", "n_years", "history_group"]


def history_group(n_years: pd.Series) -> pd.Series:
    """Map recorded years per company to the groups 1, 2, 3, 4 and 5+."""
    return n_years.clip(upper=5).astype(int).astype(str).replace({"5": "5+"})


def company_table(data: pd.DataFrame) -> pd.DataFrame:
    """One row per company: anchor year, target, recorded years and group."""
    companies = anchor_targets(data)[[COMPANY_COLUMN, "anchor_year", "target"]]
    n_years = data.groupby(COMPANY_COLUMN)[YEAR_COLUMN].size()
    companies["n_years"] = companies[COMPANY_COLUMN].map(n_years).astype(int)
    companies["history_group"] = history_group(companies["n_years"])
    return companies.reset_index(drop=True)


def _check_fractions(train_fraction: float, validation_fraction: float) -> float:
    test_fraction = round(1.0 - train_fraction - validation_fraction, 9)
    if min(train_fraction, validation_fraction, test_fraction) <= 0:
        raise ValueError(
            f"Fractions must be positive and sum to 1: train={train_fraction}, "
            f"validation={validation_fraction}, test={test_fraction:.3f}"
        )
    return test_fraction


def split_companies(
    data: pd.DataFrame, *, train_fraction: float, validation_fraction: float, seed: int
) -> pd.DataFrame:
    """Assign every company to train/validation/test, stratified by
    target x history group. Returns the company table with a ``split`` column.
    """
    test_fraction = _check_fractions(train_fraction, validation_fraction)
    companies = company_table(data)
    strata = companies["target"].astype(str) + "_" + companies["history_group"]
    sizes = strata.value_counts()
    if (sizes < len(SPLIT_NAMES)).any():
        raise ValueError(
            f"Strata too small to place in all three partitions: "
            f"{sizes[sizes < len(SPLIT_NAMES)].to_dict()}"
        )

    train_idx, rest_idx = train_test_split(
        companies.index, train_size=train_fraction, stratify=strata, random_state=seed
    )
    validation_idx, test_idx = train_test_split(
        rest_idx,
        train_size=validation_fraction / (validation_fraction + test_fraction),
        stratify=strata[rest_idx],
        random_state=seed,
    )
    companies["split"] = ""
    for name, idx in zip(SPLIT_NAMES, (train_idx, validation_idx, test_idx)):
        companies.loc[idx, "split"] = name
    return companies


def _company_number(companies: pd.DataFrame) -> np.ndarray:
    return companies[COMPANY_COLUMN].str.slice(2).astype(int).to_numpy()


def split_paths(directory: str | Path) -> dict[str, Path]:
    """Paths of the three persisted split files."""
    return {name: Path(directory) / f"{name}.csv" for name in SPLIT_NAMES}


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_splits(companies: pd.DataFrame, directory: str | Path) -> dict[str, str]:
    """Write train/validation/test CSVs (sorted, LF line endings) and return
    their SHA-256 checksums, so identical splits give identical files on any OS.
    """
    paths = split_paths(directory)
    Path(directory).mkdir(parents=True, exist_ok=True)
    checksums = {}
    for name, path in paths.items():
        part = companies[companies["split"] == name]
        part = part.iloc[np.argsort(_company_number(part), kind="stable")]
        part[SPLIT_COLUMNS].to_csv(path, index=False, lineterminator="\n")
        checksums[name] = file_sha256(path)
    return checksums


class SplitChecksumError(ValueError):
    """A newly generated split differs from the approved checksums."""


def write_verified_splits(
    companies: pd.DataFrame,
    directory: str | Path,
    *,
    expected_sha256: dict[str, str] | None = None,
) -> dict[str, str]:
    """Write the split only if it matches ``expected_sha256``.

    The files are first written to a temporary folder inside ``directory`` and
    checked there. On a mismatch nothing in ``directory`` is changed and
    ``SplitChecksumError`` is raised; otherwise the files replace the old ones.
    Without expected checksums (first run) the files are written as they are.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=directory, prefix=".tmp-") as staging:
        checksums = save_splits(companies, staging)
        expected = expected_sha256 or {}
        mismatched = [n for n, sha in checksums.items() if expected.get(n) not in (None, sha)]
        if mismatched:
            raise SplitChecksumError(
                f"Generated split files {mismatched} differ from the approved "
                "checksums in configs/config.yaml (dataset, seed, fractions or "
                "library versions changed). Existing files were left unchanged."
            )
        targets = split_paths(directory)
        for name, path in split_paths(staging).items():
            os.replace(path, targets[name])
    return checksums


def load_splits(
    directory: str | Path, *, expected_sha256: dict[str, str] | None = None
) -> pd.DataFrame:
    """Load the persisted split; never regenerate it.

    Raises if a file is missing, its checksum differs from ``expected_sha256``
    (the values recorded in configs/config.yaml), or a company is in more than
    one partition.
    """
    frames = []
    for name, path in split_paths(directory).items():
        if not path.exists():
            raise FileNotFoundError(
                f"Split file {path} is missing. Run `python scripts/prepare_data.py`."
            )
        if expected_sha256 and expected_sha256.get(name) not in (None, file_sha256(path)):
            raise ValueError(
                f"{path} differs from the split recorded in the config; "
                "rerun prepare_data.py and check the dataset and seed."
            )
        frames.append(pd.read_csv(path).assign(split=name))
    companies = pd.concat(frames, ignore_index=True)
    companies["history_group"] = companies["history_group"].astype(str)
    duplicated = companies[COMPANY_COLUMN].duplicated()
    if duplicated.any():
        raise ValueError(
            f"Companies in more than one partition: "
            f"{companies.loc[duplicated, COMPANY_COLUMN].tolist()[:5]}"
        )
    return companies


def check_split_matches_data(companies: pd.DataFrame, data: pd.DataFrame) -> None:
    """Fail if the split does not cover exactly the companies, anchors and
    targets of ``data`` (e.g. after a dataset change)."""
    expected = company_table(data).set_index(COMPANY_COLUMN)
    actual = companies.set_index(COMPANY_COLUMN)
    missing = expected.index.difference(actual.index)
    extra = actual.index.difference(expected.index)
    if len(missing) or len(extra):
        raise ValueError(
            f"Split does not match the data: {len(missing)} companies missing, "
            f"{len(extra)} unknown"
        )
    columns = ["anchor_year", "target", "n_years"]
    differs = (actual.loc[expected.index, columns] != expected[columns]).any(axis=1)
    if differs.any():
        raise ValueError(
            f"Anchor/target/history differ for {int(differs.sum())} companies, "
            f"e.g. {list(differs[differs].index[:5])}"
        )


def split_summary(companies: pd.DataFrame) -> dict[str, Any]:
    """Company counts, failed share and history-group shares per partition."""
    summary: dict[str, Any] = {}
    for name in [*SPLIT_NAMES, "all"]:
        part = companies if name == "all" else companies[companies["split"] == name]
        groups = part["history_group"].value_counts().reindex(HISTORY_GROUPS, fill_value=0)
        summary[name] = {
            "n_companies": len(part),
            "company_share": round(len(part) / len(companies), 4),
            "n_failed": int(part["target"].sum()),
            "failed_share": round(float(part["target"].mean()), 4),
            "history_group_counts": {g: int(n) for g, n in groups.items()},
            "history_group_shares": {g: round(n / len(part), 4) for g, n in groups.items()},
        }
    return summary


def _check_condition(condition: str) -> None:
    if condition not in FIT_POPULATIONS:
        raise ValueError(f"Preprocessing supports {tuple(FIT_POPULATIONS)}, got {condition!r}")


def training_fit_rows(data: pd.DataFrame, companies: pd.DataFrame, *, condition: str) -> pd.DataFrame:
    """The training rows a condition's scaler is fitted on.

    latest_year: one row per training company, its final year t.
    five_year: the same records the five_year inputs use, i.e. up to the five
    most recent recorded years per training company. Older years and all
    validation/test companies are never included.
    """
    from sequences import FIVE_YEAR_WINDOW
    from validation import FEATURE_COLUMNS, check_company_ids_and_order, check_duplicates, sort_panel
    _check_condition(condition)
    check_company_ids_and_order(data)
    check_duplicates(data)
    if companies[COMPANY_COLUMN].duplicated().any() or companies['split'].isna().any():
        raise ValueError("Persisted company partitions must be unique and non-null")
    if not companies['split'].isin(SPLIT_NAMES).all():
        raise ValueError("Unknown split partition")
    check_split_matches_data(companies, data)
    train_ids = set(companies.loc[companies['split'].eq('train'), COMPANY_COLUMN])
    if not train_ids:
        raise ValueError("Training partition is empty")
    panel = sort_panel(data[data[COMPANY_COLUMN].isin(train_ids)])
    recency = panel.groupby(COMPANY_COLUMN, sort=False).cumcount(ascending=False)
    window = 1 if condition == 'latest_year' else FIVE_YEAR_WINDOW
    rows = panel[recency.to_numpy() < window].reset_index(drop=True)
    _financial_values(rows[FEATURE_COLUMNS])
    return rows


def _financial_values(features: pd.DataFrame) -> np.ndarray:
    from validation import FEATURE_COLUMNS, validate_predictor_columns
    columns = validate_predictor_columns(features.columns)
    if columns != list(FEATURE_COLUMNS):
        raise ValueError("Financial columns must be exactly X1-X18 in audited order")
    if any(not pd.api.types.is_numeric_dtype(dtype) for dtype in features.dtypes):
        raise ValueError("Financial values must be numeric")
    values = features.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Missing/nonfinite financial cells are not history padding; no imputation is configured")
    return values


MODELS = ('logistic_regression', 'xgboost', 'mlp', 'gru')
SCALED = 'signed_log1p_standard'
MODEL_STRATEGIES = {'logistic_regression': SCALED, 'mlp': SCALED,
                    'gru': SCALED, 'xgboost': 'identity'}


def signed_log1p(values: np.ndarray) -> np.ndarray:
    """sign(x) * log(1 + |x|): compresses the dollar-scale tails (#87).

    Monotone and odd, so order and sign are kept and 0 stays 0. A company 1,000x
    larger than another ends up about 7 units apart instead of 1,000x apart.
    """
    return np.sign(values) * np.log1p(np.abs(values))
# Schema 2 (before #87) used plain standardization. Its identity path for
# XGBoost is unchanged, so saved XGBoost runs bundling a schema-2 copy still
# load; any scaled model refuses a schema-2 record.
LEGACY_SCHEMA_VERSION = 2
LEGACY_MODEL_STRATEGIES = {'logistic_regression': 'standard', 'mlp': 'standard',
                           'gru': 'standard', 'xgboost': 'identity'}
FIT_POPULATIONS = {'latest_year': 'training_company_final_year',
                   'five_year': 'training_company_last_five_records'}


class FittedPreprocessor:
    """Fixed signed log1p + StandardScaler per input condition, fitted on training rows only.

    Each condition has its own mean/scale of the signed-log1p values, fitted on
    exactly the training records that condition uses. Numeric state is saved as JSON with the
    feature schemas and the fitting company/years.

    A schema-2 record (plain standardization, before #87) is accepted only for
    XGBoost's unchanged identity transform; LR/MLP/GRU refuse it.
    """

    def __init__(self, record):
        self.record = record
        self._validate_record()

    def _validate_record(self):
        from sequences import step_columns, tabular_columns
        from validation import FEATURE_COLUMNS
        r = self.record
        if (r['schema_version'], r['strategy']) == (3, SCALED):
            expected_strategies = MODEL_STRATEGIES
        elif (r['schema_version'], r['strategy']) == (LEGACY_SCHEMA_VERSION, 'standard'):
            expected_strategies = LEGACY_MODEL_STRATEGIES
        else:
            raise ValueError("Unsupported preprocessing contract")
        if r['financial_columns'] != list(FEATURE_COLUMNS):
            raise ValueError("Saved financial feature order differs")
        if r['model_strategies'] != expected_strategies:
            raise ValueError("Saved model strategies differ")
        if set(r['conditions']) != set(FIT_POPULATIONS):
            raise ValueError("Saved conditions differ")
        for condition, fit in r['conditions'].items():
            if fit['fit_population'] != FIT_POPULATIONS[condition]:
                raise ValueError("Saved fit population differs")
            if fit['schema'] != {'sequence': step_columns(condition), 'tabular': tabular_columns(condition)}:
                raise ValueError("Saved derived feature schema differs")
            mean, scale = np.asarray(fit['mean'], dtype=float), np.asarray(fit['scale'], dtype=float)
            if (mean.shape != (18,) or scale.shape != (18,) or not np.isfinite(mean).all()
                    or not np.isfinite(scale).all() or (scale <= 0).any()):
                raise ValueError("Invalid saved standardization parameters")

    @property
    def is_legacy(self) -> bool:
        """True for a schema-2 record: usable for XGBoost's identity path only."""
        return self.record['schema_version'] == LEGACY_SCHEMA_VERSION

    def transform(self, features: pd.DataFrame, *, model: str, condition: str) -> np.ndarray:
        self._validate_record()
        _check_condition(condition)
        if model not in MODELS:
            raise ValueError("Unknown model")
        if self.is_legacy and MODEL_STRATEGIES[model] != 'identity':
            raise ValueError(f"Schema-{LEGACY_SCHEMA_VERSION} preprocessor (before #87) only supports "
                             "XGBoost's identity transform; regenerate it with scripts/prepare_preprocessing.py")
        values = _financial_values(features)
        if MODEL_STRATEGIES[model] == SCALED:
            fit = self.record['conditions'][condition]
            values = (signed_log1p(values) - np.asarray(fit['mean'])) / np.asarray(fit['scale'])
        if not np.isfinite(values).all():
            raise ValueError("Transformed financial values overflowed")
        return values

    def save(self, path: str | Path) -> None:
        import json
        self._validate_record()
        text = json.dumps(self.record, indent=2, sort_keys=True, allow_nan=False)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + '\n', encoding='utf-8')

    @classmethod
    def load(cls, path: str | Path):
        import json
        return cls(json.loads(Path(path).read_text(encoding='utf-8')))


def fit_preprocessor(data: pd.DataFrame, companies: pd.DataFrame) -> FittedPreprocessor:
    """Fit one StandardScaler per condition on the signed-log1p training rows.

    latest_year is fitted on each training company's final year; five_year on
    the up to five most recent training records the five_year inputs use. Reuse
    this object for all eight experiments; XGBoost uses the identity path. No
    validation/test values contribute to the fitted statistics.
    """
    from sklearn.preprocessing import StandardScaler
    from sequences import step_columns, tabular_columns
    from validation import FEATURE_COLUMNS
    conditions = {}
    for condition, population in FIT_POPULATIONS.items():
        rows = training_fit_rows(data, companies, condition=condition)
        scaler = StandardScaler().fit(signed_log1p(_financial_values(rows[FEATURE_COLUMNS])))
        years = rows.groupby(COMPANY_COLUMN, sort=False)[YEAR_COLUMN].agg(lambda y: [int(v) for v in y])
        conditions[condition] = {
            'fit_population': population,
            'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(),
            'variance': scaler.var_.tolist(),
            'n_training_companies': int(rows[COMPANY_COLUMN].nunique()),
            'n_fitting_observations': len(rows),
            'fitting_years': years.to_dict(),
            'schema': {'sequence': step_columns(condition), 'tabular': tabular_columns(condition)},
        }
    return FittedPreprocessor({
        'schema_version': 3, 'strategy': SCALED,
        'financial_columns': list(FEATURE_COLUMNS),
        'model_strategies': dict(MODEL_STRATEGIES),
        'conditions': conditions,
        'missing_values': 'error', 'relative_offsets': 'unchanged',
        'padding': 'zero_after_transform',
    })


def transform_features(features: pd.DataFrame, preprocessor: FittedPreprocessor,
                       *, model: str, condition: str) -> np.ndarray:
    """Transform financial rows in audited order with the condition's scaler, without refitting."""
    return preprocessor.transform(features, model=model, condition=condition)


def transform_histories(histories, preprocessor: FittedPreprocessor, *, model: str):
    """Transform only real financial steps with the scaler of ``histories.condition``.

    Offsets, lengths and IDs are preserved. Flatten the returned histories for
    tabular models so the same transformation is used at every recency slot.
    Padding is always restored to zero.
    """
    from dataclasses import replace
    from sequences import FIVE_YEAR_WINDOW, step_columns, valid_step_mask
    from validation import FEATURE_COLUMNS
    _check_condition(histories.condition)
    capacity = 1 if histories.condition == 'latest_year' else FIVE_YEAR_WINDOW
    if histories.step_columns != step_columns(histories.condition):
        raise ValueError("Unexpected sequence feature order")
    if histories.features.shape != (len(histories.lengths), capacity, len(histories.step_columns)):
        raise ValueError("Unexpected sequence shape")
    lengths = np.asarray(histories.lengths)
    if lengths.dtype.kind not in 'iu' or (lengths < 1).any() or (lengths > capacity).any():
        raise ValueError("Invalid real sequence lengths")
    mask = valid_step_mask(histories)
    real = histories.features[mask].astype(float)
    if not np.isfinite(real).all():
        raise ValueError("Nonfinite values in real observations")
    real[:, :len(FEATURE_COLUMNS)] = preprocessor.transform(
        pd.DataFrame(real[:, :len(FEATURE_COLUMNS)], columns=FEATURE_COLUMNS),
        model=model, condition=histories.condition)
    values = np.zeros(histories.features.shape, dtype=np.float32)
    with np.errstate(over='ignore'):
        values[mask] = real
    if not np.isfinite(values).all():
        raise ValueError("Transformed input overflows float32")
    return replace(histories, features=values, lengths=histories.lengths.copy(),
                   targets=histories.targets.copy(), metadata=histories.metadata.copy(),
                   step_columns=list(histories.step_columns))
