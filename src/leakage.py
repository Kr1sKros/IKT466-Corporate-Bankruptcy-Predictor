"""Executable leakage checks for partitions, inputs, preprocessing and the study plan (#52).

Every ``check_*`` function raises ``LeakageError`` when a check fails and
otherwise returns a small dict of what it verified. ``describe_*`` functions do
not fail: they record known limitations (signals that are not leakage by
construction but can still be exploited by a model). ``audit`` runs them all.

See docs/leakage-audit.md for the findings and what is still open.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from preprocessing import (
    FIT_POPULATIONS,
    MODEL_STRATEGIES,
    SPLIT_NAMES,
    FittedPreprocessor,
    check_split_matches_data,
    fit_preprocessor,
    training_fit_rows,
    transform_histories,
)
from sequences import (
    CONDITIONS,
    FIVE_YEAR_WINDOW,
    OFFSET_COLUMN,
    PRESENCE_COLUMN,
    create_histories,
    flatten_histories,
    step_columns,
    valid_step_mask,
    validate_tabular_columns,
)
from validation import (
    COMPANY_COLUMN,
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    NON_FEATURE_COLUMNS,
    YEAR_COLUMN,
    verify_target_labels,
)

MODELS = tuple(MODEL_STRATEGIES)


class LeakageError(AssertionError):
    """A leakage check failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LeakageError(message)


def _partition_ids(companies: pd.DataFrame, name: str) -> list[str]:
    return companies.loc[companies["split"].eq(name), COMPANY_COLUMN].tolist()


# --- Partitions ---------------------------------------------------------------

def check_partitions(companies: pd.DataFrame, data: pd.DataFrame) -> dict[str, Any]:
    """One row per company, no overlap, every company placed once, and anchors/
    targets/history lengths exactly as derived from the data."""
    _require(not companies[COMPANY_COLUMN].duplicated().any(),
             "A company appears more than once in the split (overlap between partitions)")
    _require(companies["split"].isin(SPLIT_NAMES).all(), "Unknown partition name in split")
    counts = companies["split"].value_counts().reindex(SPLIT_NAMES, fill_value=0)
    _require((counts > 0).all(), f"Empty partition: {counts.to_dict()}")
    try:
        check_split_matches_data(companies, data)
    except ValueError as error:
        raise LeakageError(f"Split does not match the data: {error}") from error
    sets = {name: set(_partition_ids(companies, name)) for name in SPLIT_NAMES}
    for i, a in enumerate(SPLIT_NAMES):
        for b in SPLIT_NAMES[i + 1:]:
            _require(not sets[a] & sets[b], f"Companies in both {a} and {b}")
    return {
        "companies": len(companies),
        "per_partition": {k: int(v) for k, v in counts.items()},
        "one_row_per_company": True,
        "partition_overlap": 0,
        "anchors_targets_match_data": True,
    }


# --- Model inputs ---------------------------------------------------------------

def check_inputs(data: pd.DataFrame, companies: pd.DataFrame, condition: str) -> dict[str, Any]:
    """Check the inputs of every partition for one condition.

    * exactly one example per partition company, with the split's anchor/target;
    * no financial record after the anchor; the last real step is the record at t;
    * five_year uses exactly the latest min(n_years, 5) records, nothing older;
    * offsets equal t - year, are strictly decreasing and end at 0;
    * padding is zero and ``present`` marks exactly the real steps;
    * no raw ID, year or label column is a model input.
    """
    window = 1 if condition == "latest_year" else FIVE_YEAR_WINDOW
    at_anchor = data.set_index([COMPANY_COLUMN, YEAR_COLUMN])[FEATURE_COLUMNS]
    years_of = data.groupby(COMPANY_COLUMN)[YEAR_COLUMN].agg(sorted)
    expected_steps = step_columns(condition)
    leaked = [c for c in expected_steps if c in NON_FEATURE_COLUMNS or c == LABEL_COLUMN]
    _require(not leaked, f"Raw ID/year/label used as input: {leaked}")

    examples = 0
    for name in SPLIT_NAMES:
        ids = _partition_ids(companies, name)
        h = create_histories(data, condition=condition, companies=ids)
        meta = h.metadata
        examples += len(meta)
        _require(len(meta) == len(ids) and set(meta[COMPANY_COLUMN]) == set(ids),
                 f"{condition}/{name}: not exactly one example per partition company")
        split = companies.set_index(COMPANY_COLUMN).loc[meta[COMPANY_COLUMN]]
        _require((split["anchor_year"].to_numpy() == meta["anchor_year"].to_numpy()).all()
                 and (split["target"].to_numpy() == h.targets).all(),
                 f"{condition}/{name}: anchor or target differs from the persisted split")
        _require(h.step_columns == expected_steps, f"{condition}: unexpected step columns")

        lengths = h.lengths
        n_years = meta["n_years"].to_numpy()
        _require((lengths == np.minimum(n_years, window)).all(),
                 f"{condition}/{name}: lengths are not min(n_years, {window})")
        for company, used, anchor in zip(meta[COMPANY_COLUMN], meta["years"], meta["anchor_year"]):
            used = [int(y) for y in used.split(",")]
            _require(max(used) == anchor, f"{condition}: {company} uses a year after its anchor")
            _require(used == years_of[company][-len(used):],
                     f"{condition}: {company} does not use its latest {len(used)} records")

        mask = valid_step_mask(h)
        rows = np.arange(len(lengths))
        last = h.features[rows, lengths - 1, :len(FEATURE_COLUMNS)]
        anchor_values = at_anchor.loc[list(zip(meta[COMPANY_COLUMN], meta["anchor_year"]))].to_numpy(np.float32)
        _require(np.array_equal(last, anchor_values),
                 f"{condition}/{name}: last real step is not the record at the anchor")
        _require((h.features[~mask] == 0).all(), f"{condition}/{name}: padding is not zero")

        if condition == "five_year":
            offsets = h.features[:, :, step_columns(condition).index(OFFSET_COLUMN)]
            _require((offsets[rows, lengths - 1] == 0).all(), f"{name}: offset at t is not 0")
            real = np.where(mask, offsets, np.nan)
            steps = np.diff(real, axis=1)
            _require(not (steps[~np.isnan(steps)] >= 0).any(),
                     f"{name}: offsets are not strictly decreasing (oldest first)")
            for i, (used, anchor) in enumerate(zip(meta["years"], meta["anchor_year"])):
                expected = [anchor - int(y) for y in used.split(",")]
                _require(offsets[i, :lengths[i]].tolist() == expected,
                         f"{name}: offsets differ from t - year for {meta.loc[i, COMPANY_COLUMN]}")

        tabular, _ = flatten_histories(h)
        validate_tabular_columns(tabular.columns, condition)
        if condition == "five_year":
            present = tabular[[f"{PRESENCE_COLUMN}_r{r}" for r in range(window)]].to_numpy()
            _require((present.sum(axis=1) == lengths).all() and (present[:, 0] == 1).all(),
                     f"{name}: presence flags do not match the real steps")

    return {"condition": condition, "examples": examples, "window": window,
            "one_example_per_company": True, "post_anchor_records": 0,
            "last_step_is_anchor_record": True, "padding_zero": True,
            "offsets_and_presence_valid": condition == "five_year" or None,
            "raw_id_year_label_inputs": 0}


# --- Preprocessing ----------------------------------------------------------------

def check_preprocessor(preprocessor: FittedPreprocessor, data: pd.DataFrame,
                       companies: pd.DataFrame) -> dict[str, Any]:
    """The saved scaler of each condition is fitted on training companies only,
    on exactly the rows that condition uses, and transforming keeps padding 0."""
    train = set(_partition_ids(companies, "train"))
    refit = fit_preprocessor(data, companies)
    result = {}
    for condition in FIT_POPULATIONS:
        saved = preprocessor.record["conditions"][condition]
        fitted_ids = set(saved["fitting_years"])
        held_out = fitted_ids - train
        _require(not held_out, f"{condition}: scaler fitted on non-training companies {sorted(held_out)[:5]}")
        rows = training_fit_rows(data, companies, condition=condition)
        expected_years = rows.groupby(COMPANY_COLUMN)[YEAR_COLUMN].agg(lambda y: [int(v) for v in y]).to_dict()
        _require(saved["fitting_years"] == expected_years,
                 f"{condition}: saved fitting years differ from the condition's training rows")
        fresh = refit.record["conditions"][condition]
        _require(np.allclose(saved["mean"], fresh["mean"]) and np.allclose(saved["scale"], fresh["scale"]),
                 f"{condition}: saved statistics differ from a refit on the training rows")

        ids = _partition_ids(companies, "validation")
        h = create_histories(data, condition=condition, companies=ids)
        mask = valid_step_mask(h)
        for model in MODELS:
            scaled = transform_histories(h, preprocessor, model=model)
            _require((scaled.features[~mask] == 0).all(), f"{condition}/{model}: padding changed")
            if condition == "five_year":
                offset = step_columns(condition).index(OFFSET_COLUMN)
                _require(np.array_equal(scaled.features[..., offset], h.features[..., offset]),
                         f"{condition}/{model}: offsets were transformed")
        result[condition] = {"fit_population": saved["fit_population"],
                             "training_companies": len(fitted_ids),
                             "fitting_observations": saved["n_fitting_observations"],
                             "non_training_companies": 0, "matches_refit": True}
    return result


def class_weight_bounds(companies: pd.DataFrame) -> dict[str, Any]:
    """Class balance of the **training** partition only.

    Runners (#53-#56) must derive class-weight search bounds (LR balanced
    weights, XGBoost scale_pos_weight, MLP/GRU pos_weight) from this, never from
    validation/test labels.
    """
    train = companies[companies["split"].eq("train")]
    positives = int(train["target"].sum())
    negatives = len(train) - positives
    _require(positives > 0 and negatives > 0, "Training partition needs both classes")
    return {"source": "train", "positives": positives, "negatives": negatives,
            "negative_to_positive_ratio": round(negatives / positives, 6)}


# --- Study plan ---------------------------------------------------------------------

def check_study_plan(config: dict) -> dict[str, Any]:
    """The declared protocol: eight independent studies with one common budget,
    selected on validation AP, preprocessing fitted on training only.

    Saved Optuna studies do not exist yet; auditing them is left to #56.
    """
    plan, tuning = config["study_plan"], config["study_plan"]["tuning"]
    _require(sorted(plan["models"]) == sorted(MODELS) and len(set(plan["models"])) == len(MODELS),
             f"Models differ: {plan['models']}")
    _require(tuple(plan["input_conditions"]) == CONDITIONS,
             f"Conditions differ from the implemented inputs: {plan['input_conditions']}")
    n_studies = len(plan["models"]) * len(plan["input_conditions"])
    _require(plan["configuration_count"] == n_studies == 8, "Expected eight studies")
    _require(plan.get("five_year_window") == FIVE_YEAR_WINDOW, "Five-year window differs from the code")
    expected = {"framework": "optuna", "sampler": "TPESampler", "direction": "maximize",
                "objective": "validation_average_precision", "n_jobs": 1, "pruning": "none",
                "separate_study_per_model_and_condition": True, "tune_class_weights": True}
    wrong = {k: tuning.get(k) for k, v in expected.items() if tuning.get(k) != v}
    _require(not wrong, f"Tuning settings differ from the protocol: {wrong}")
    budget = tuning["attempted_trials_per_study"]
    _require(isinstance(budget, int) and budget > 0, "One positive trial budget must be declared for all studies")
    _require(isinstance(tuning["sampler_seed"], int), "Sampler seed must be fixed")
    _require(plan["shared_company_split"] is True, "All studies must share one company split")
    _require(config["preprocessing"]["fit_on_training_only"] is True, "Preprocessing must be fitted on training only")
    _require(config["evaluation"]["identical_test_companies"] is True, "Test companies must be identical across runs")
    _require(config["dataset"]["target_verified"] is False,
             "target_verified must stay false: t+1 timing is publisher-dependent")
    return {"studies": n_studies, "trials_per_study": budget, "sampler_seed": tuning["sampler_seed"],
            "objective": tuning["objective"], "common_budget": True,
            "decision_threshold": config["evaluation"]["decision_threshold"],
            "saved_studies_audited": False}


# --- Known limitations (descriptive, never fail) -----------------------------------

def describe_record_availability(data: pd.DataFrame, companies: pd.DataFrame) -> dict[str, Any]:
    """Terminal-record selection and follow-up: which year each target is anchored
    at, and how many alive companies lack follow-up after their last record."""
    labels = verify_target_labels(data)
    last_year = int(data[YEAR_COLUMN].max())
    alive = companies[companies["target"].eq(0)]
    failed = companies[companies["target"].eq(1)]
    return {
        "anchor": "last recorded year per company",
        "failed_rows_before_terminal_year": labels["failed_rows_before_terminal_year"],
        "dataset_last_year": last_year,
        "alive_anchored_before_last_year": int((alive["anchor_year"] < last_year).sum()),
        "alive_companies": len(alive),
        "failed_anchor_years": {"min": int(failed["anchor_year"].min()), "max": int(failed["anchor_year"].max())},
        "limitation": ("Alive companies anchored before the last dataset year have no observed "
                       "follow-up; t+1 timing rests on the publisher (target_verified=false)."),
    }


def describe_indirect_signals(data: pd.DataFrame, companies: pd.DataFrame) -> dict[str, Any]:
    """How well non-financial structure alone separates the classes on the
    **training** partition (ROC AUC, 0.5 = no signal). These are not inputs but a
    model can partly recover them from the financial values or the presence flags."""
    train = companies[companies["split"].eq("train")]
    h = create_histories(data, condition="five_year", companies=train[COMPANY_COLUMN])
    meta = h.metadata
    y = meta["target"].to_numpy()

    def auc(score) -> float:
        return round(float(roc_auc_score(y, score)), 3)

    return {
        "partition": "train",
        "auc_anchor_year_earlier_means_failed": auc(-meta["anchor_year"].to_numpy()),
        "auc_five_year_length_longer_means_failed": auc(meta["length"].to_numpy()),
        "auc_recorded_years": auc(meta["n_years"].to_numpy()),
        "auc_gap_in_window": auc(meta["has_gap"].astype(int).to_numpy()),
        "limitation": ("Calendar year is not an input, but failed companies end earlier than "
                       "alive ones (many alive companies end in the last year), and five_year "
                       "length is visible through the presence flags."),
    }


def audit(data: pd.DataFrame, companies: pd.DataFrame, preprocessor: FittedPreprocessor,
          config: dict) -> dict[str, Any]:
    """Run every check; raises ``LeakageError`` on the first failure."""
    return {
        "partitions": check_partitions(companies, data),
        "inputs": {c: check_inputs(data, companies, c) for c in CONDITIONS},
        "preprocessing": check_preprocessor(preprocessor, data, companies),
        "class_weight_source": class_weight_bounds(companies),
        "study_plan": check_study_plan(config),
        "record_availability": describe_record_availability(data, companies),
        "indirect_signals": describe_indirect_signals(data, companies),
    }
