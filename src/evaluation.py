"""Model-independent evaluation and observation alignment interfaces."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    import numpy as np

def evaluate_predictions(
    observations: pd.DataFrame,
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    threshold: float,
) -> dict[str, object]:
    """Evaluate vectors in observation row order, with a fixed threshold.

    Each company must occur once at its final-year anchor. If a label column is
    included, it must match y_true exactly. This checks positional alignment;
    callers must join separately sourced predictions by company/anchor first.
    """
    import numpy as np
    from metrics import compute_metrics

    required = ['company_name', 'year']
    if any(column not in observations for column in required):
        raise ValueError("Observations require company_name and year columns")
    if observations.columns.duplicated().any():
        raise ValueError("Observation column names must be unique")
    if observations[required].isna().any().any() or observations['company_name'].duplicated().any():
        raise ValueError("Observation IDs must be non-null and companies unique")
    result = compute_metrics(y_true, y_score, threshold=threshold)
    if len(observations) != result['n_observations']:
        raise ValueError("Observation IDs and labels must have matching lengths")
    if 'label' in observations and not np.array_equal(observations['label'].to_numpy(), y_true):
        raise ValueError("Observation labels do not match y_true")
    return result


def align_test_observations(predictions, *, expected=None):
    """Require identical company/anchor/target sets, sorted by company ID.

    No silent intersection. Supply persisted test IDs to detect shared omissions.
    """
    import numpy as np

    if not predictions:
        raise ValueError("At least one prediction table is required")

    def validate(frame, name):
        required = ['company_name', 'anchor_year', 'target']
        if frame.columns.duplicated().any() or any(c not in frame for c in required):
            raise ValueError(f"{name}: unique columns including {required} required")
        if frame.empty or frame[required].isna().any().any():
            raise ValueError(f"{name}: nonempty, non-null observations required")
        if not frame.company_name.map(lambda x: isinstance(x, str) and bool(x.strip())).all():
            raise ValueError(f"{name}: company IDs must be nonempty strings")
        if frame.company_name.duplicated().any():
            raise ValueError(f"{name}: company IDs must be unique")
        years = frame.anchor_year.to_numpy()
        labels = frame.target.to_numpy()
        if years.dtype.kind not in 'iuf' or not np.isfinite(years).all() or (years != np.floor(years)).any():
            raise ValueError(f"{name}: anchor years must be finite integers")
        if labels.dtype.kind not in 'biuf' or not np.isin(labels, [0, 1]).all():
            raise ValueError(f"{name}: targets must be numeric 0/1")
        return frame.sort_values('company_name').reset_index(drop=True).copy()

    reference = validate(expected if expected is not None else next(iter(predictions.values())), 'expected')
    aligned = {}
    for name in sorted(predictions):
        frame = validate(predictions[name], name)
        if frame.company_name.tolist() != reference.company_name.tolist():
            raise ValueError(f"{name}: company set differs from expected cohort")
        if not np.array_equal(frame[['anchor_year', 'target']].to_numpy(), reference[['anchor_year', 'target']].to_numpy()):
            raise ValueError(f"{name}: anchor year or target differs from expected cohort")
        aligned[name] = frame
    return aligned


def compare_predictions(predictions, *, expected, thresholds, min_history=None):
    """Evaluate full test set or one explicit common history subset.

    Inputs must cover the full test set even for a sub-analysis. Thresholds are
    fixed on validation data before calling; this function never tunes them.
    """
    import numpy as np
    from metrics import compute_metrics

    if set(thresholds) != set(predictions):
        raise ValueError("Exactly one validation-selected threshold per run is required")
    aligned = align_test_observations(predictions, expected=expected)
    reference = expected.sort_values('company_name').reset_index(drop=True)
    mask = np.ones(len(reference), dtype=bool)
    if min_history is not None:
        if isinstance(min_history, bool) or not isinstance(min_history, int) or not 1 <= min_history <= 5:
            raise ValueError("min_history must be an integer from 1 to 5")
        if 'n_years' not in reference:
            raise ValueError("Sub-analysis requires n_years from the persisted split")
        years = reference.n_years.to_numpy()
        if years.dtype.kind not in 'iuf' or not np.isfinite(years).all() or (years < 1).any() or (years != np.floor(years)).any():
            raise ValueError("n_years must contain positive integers")
        mask = years >= min_history
    if not mask.any():
        raise ValueError("The requested common cohort is empty")
    results = {}
    for name, frame in aligned.items():
        if 'probability' not in frame:
            raise ValueError(f"{name}: probability column required")
        # Validate excluded rows too, so a subset cannot conceal corrupt inputs.
        compute_metrics(frame.target.to_numpy(), frame.probability.to_numpy(), threshold=thresholds[name])
        part = frame.loc[mask]
        results[name] = compute_metrics(part.target.to_numpy(), part.probability.to_numpy(), threshold=thresholds[name])
    retained = reference.loc[mask, ['company_name', 'anchor_year', 'target']]
    return {
        'cohort': 'all_companies' if min_history is None else f'at_least_{min_history}_recorded_years',
        'test_count': len(reference),
        'test_prevalence': float(reference.target.mean()),
        'retained_count': len(retained),
        'retained_prevalence': float(retained.target.mean()),
        'excluded_count': int((~mask).sum()),
        'excluded_companies': reference.loc[~mask, 'company_name'].tolist(),
        'observations': retained.to_dict(orient='records'),
        'results': results,
    }
