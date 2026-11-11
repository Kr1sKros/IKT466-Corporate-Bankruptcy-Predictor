"""Leakage checks pass on clean inputs and fail on each injected leak (#52)."""
import copy
from pathlib import Path

import pandas as pd
import pytest
import yaml

from leakage import (
    LeakageError,
    audit,
    check_inputs,
    check_partitions,
    check_preprocessor,
    check_study_plan,
    class_weight_bounds,
)
from preprocessing import FittedPreprocessor, company_table, fit_preprocessor
from validation import FEATURE_COLUMNS

CONFIG = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "config.yaml").read_text())

# company: (years, failed, split)
COMPANIES = {
    "C_1": (range(2000, 2008), False, "train"),       # 8 records, 3 older ones excluded
    "C_2": ([2001, 2004, 2005], True, "train"),       # gap
    "C_3": ([2010], False, "train"),
    "C_4": (range(2005, 2009), True, "train"),
    "C_5": (range(2003, 2010), False, "validation"),
    "C_6": ([2011, 2012], True, "validation"),
    "C_7": ([2009, 2010, 2012], False, "test"),
    "C_8": ([2014], True, "test"),
}


@pytest.fixture
def panel():
    rows = []
    for number, (company, (years, failed, _)) in enumerate(COMPANIES.items()):
        for year in years:
            row = {"company_name": company, "status_label": "failed" if failed else "alive", "year": year}
            for j, col in enumerate(FEATURE_COLUMNS):
                row[col] = float(year - 2000 + number + j / 10)
            rows.append(row)
    data = pd.DataFrame(rows)
    splits = company_table(data)
    splits["split"] = splits["company_name"].map({c: v[2] for c, v in COMPANIES.items()})
    return data, splits


def test_clean_pipeline_passes_every_check(panel):
    data, splits = panel
    report = audit(data, splits, fit_preprocessor(data, splits), CONFIG)
    assert report["partitions"]["partition_overlap"] == 0
    assert report["inputs"]["five_year"]["examples"] == len(COMPANIES)
    assert report["preprocessing"]["five_year"]["fitting_observations"] == 5 + 3 + 1 + 4
    assert report["class_weight_source"] == {"source": "train", "positives": 2, "negatives": 2,
                                             "negative_to_positive_ratio": 1.0}
    assert report["study_plan"]["saved_studies_audited"] is False
    assert 0 <= report["indirect_signals"]["auc_anchor_year_earlier_means_failed"] <= 1


def test_company_in_two_partitions_fails(panel):
    data, splits = panel
    overlap = pd.concat([splits, splits.iloc[[0]].assign(split="test")], ignore_index=True)
    with pytest.raises(LeakageError, match="more than once"):
        check_partitions(overlap, data)


def test_changed_anchor_or_target_fails(panel):
    data, splits = panel
    moved = splits.copy()
    moved.loc[0, "anchor_year"] = 2005
    with pytest.raises(LeakageError, match="does not match"):
        check_partitions(moved, data)
    with pytest.raises(LeakageError, match="anchor or target"):
        check_inputs(data, moved, "five_year")
    flipped = splits.copy()
    flipped.loc[2, "target"] = 1
    with pytest.raises(LeakageError, match="anchor or target"):
        check_inputs(data, flipped, "latest_year")


@pytest.mark.parametrize("condition", ["latest_year", "five_year"])
def test_inputs_pass_for_both_conditions(panel, condition):
    data, splits = panel
    result = check_inputs(data, splits, condition)
    assert result["window"] == (1 if condition == "latest_year" else 5)
    assert result["post_anchor_records"] == 0


def test_scaler_fitted_on_held_out_company_fails(panel):
    data, splits = panel
    fitted = fit_preprocessor(data, splits)
    check_preprocessor(fitted, data, splits)
    record = copy.deepcopy(fitted.record)
    record["conditions"]["latest_year"]["fitting_years"]["C_8"] = [2014]
    with pytest.raises(LeakageError, match="non-training"):
        check_preprocessor(FittedPreprocessor(record), data, splits)


def test_scaler_fitted_on_other_rows_fails(panel):
    data, splits = panel
    fitted = fit_preprocessor(data, splits)
    record = copy.deepcopy(fitted.record)
    record["conditions"]["five_year"]["fitting_years"]["C_1"] = list(range(2000, 2008))
    with pytest.raises(LeakageError, match="fitting years"):
        check_preprocessor(FittedPreprocessor(record), data, splits)
    record = copy.deepcopy(fitted.record)
    record["conditions"]["five_year"]["mean"][0] += 1
    with pytest.raises(LeakageError, match="refit"):
        check_preprocessor(FittedPreprocessor(record), data, splits)
    # A scaler fitted with validation companies treated as training is caught.
    leaky = fit_preprocessor(data, splits.assign(split="train"))
    with pytest.raises(LeakageError):
        check_preprocessor(leaky, data, splits)


def test_class_weights_use_training_labels_only(panel):
    _, splits = panel
    changed = splits.copy()
    changed.loc[changed["split"].ne("train"), "target"] = 0
    assert class_weight_bounds(changed) == class_weight_bounds(splits)


@pytest.mark.parametrize("path, value, message", [
    (("study_plan", "input_conditions"), ["latest_year", "full_history"], "Conditions"),
    (("study_plan", "tuning", "objective"), "test_average_precision", "Tuning"),
    (("study_plan", "tuning", "attempted_trials_per_study"), None, "budget"),
    (("study_plan", "five_year_window"), 20, "window"),
    (("preprocessing", "fit_on_training_only"), False, "training only"),
    (("dataset", "target_verified"), True, "target_verified"),
])
def test_study_plan_rejects_protocol_violations(path, value, message):
    assert check_study_plan(CONFIG)["studies"] == 8
    config = copy.deepcopy(CONFIG)
    node = config
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    with pytest.raises(LeakageError, match=message):
        check_study_plan(config)
