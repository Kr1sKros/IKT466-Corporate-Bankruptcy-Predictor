"""Shared metric behavior, edge cases, and validation-only threshold selection."""
import json

import numpy as np
import pandas as pd
import pytest
from sklearn import metrics as reference

from evaluation import evaluate_predictions
from metrics import compute_metrics, select_validation_threshold


def test_known_confusion_and_threshold_invariance_of_ap():
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.6, 0.4, 0.9]
    result = compute_metrics(labels, scores)
    assert result['confusion_matrix'] == [[1, 1], [1, 1]]
    assert [result[k] for k in ['tn', 'fp', 'fn', 'tp']] == [1, 1, 1, 1]
    assert result['precision'] == result['recall'] == result['f1'] == 0.5
    assert result['average_precision'] == pytest.approx(5 / 6)
    assert compute_metrics(labels, scores, threshold=0.8)['average_precision'] == result['average_precision']
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('scores', [[0.1, 0.4, 0.35, 0.8], [0.5] * 4, [0, 1, 0, 1]])
def test_matches_library_for_ties_and_rankings(scores):
    labels = np.array([0, 0, 1, 1])
    result = compute_metrics(labels, scores, threshold=0.4)
    predicted = np.array(scores) >= 0.4
    for name in ['precision', 'recall', 'f1', 'accuracy']:
        assert result[name] == pytest.approx(getattr(reference, name + '_score')(labels, predicted))
    assert result['average_precision'] == pytest.approx(reference.average_precision_score(labels, scores))
    assert result['roc_auc'] == pytest.approx(reference.roc_auc_score(labels, scores))


@pytest.mark.parametrize('labels,ap,matrix', [([0, 0], None, [[1, 1], [0, 0]]), ([1, 1], 1.0, [[0, 0], [1, 1]])])
def test_single_class_has_explicit_undefined_behavior(labels, ap, matrix):
    result = compute_metrics(labels, [0.1, 0.9])
    assert result['average_precision'] == ap
    assert result['roc_auc'] is None
    assert result['confusion_matrix'] == matrix
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('labels,scores', [([], []), ([0], [0.1, 0.2]), ([[0]], [0.2]),
    ([2], [0.2]), (['failed'], [0.2]), ([np.nan], [0.2]), ([0], [np.nan]),
    ([0], [np.inf]), ([0], [-0.1]), ([1], [1.1]), ([0], ['0.2'])])
def test_invalid_inputs_rejected(labels, scores):
    with pytest.raises(ValueError):
        compute_metrics(labels, scores)


@pytest.mark.parametrize('threshold', [-1, 2, np.nan, np.inf, True, [0.5], 'invalid'])
def test_invalid_threshold_rejected(threshold):
    with pytest.raises(ValueError):
        compute_metrics([0, 1], [0.2, 0.8], threshold=threshold)


def test_threshold_selection_is_f1_optimal_and_fixed_for_test():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.6, 0.4, 0.9])
    threshold = select_validation_threshold(labels, scores)
    assert threshold == 0.4
    assert compute_metrics(labels, scores, threshold=threshold)['f1'] == max(
        reference.f1_score(labels, scores >= candidate) for candidate in np.unique(scores))
    result = compute_metrics([0, 1], [0.3, 0.35], threshold=threshold)
    assert result['threshold'] == threshold
    assert result['recall'] == 0
    with pytest.raises(ValueError):
        select_validation_threshold([0, 0], [0.1, 0.2])


def test_threshold_tie_chooses_highest():
    assert select_validation_threshold([1, 0, 0, 1], [0.1, 0.2, 0.3, 0.4]) == 0.4


def test_evaluation_validates_ids_and_labels():
    observations = pd.DataFrame({'company_name': ['a', 'b'], 'year': [2017, 2018], 'label': [0, 1]})
    assert evaluate_predictions(observations, [0, 1], [0.2, 0.8], threshold=0.5)['accuracy'] == 1
    with pytest.raises(ValueError, match='labels'):
        evaluate_predictions(observations, [1, 0], [0.2, 0.8], threshold=0.5)
    with pytest.raises(ValueError, match='unique'):
        evaluate_predictions(pd.concat([observations, observations]), [0, 1, 0, 1], [0.2]*4, threshold=0.5)
    with pytest.raises(ValueError, match='matching lengths'):
        evaluate_predictions(observations, [0], [0.2], threshold=0.5)
