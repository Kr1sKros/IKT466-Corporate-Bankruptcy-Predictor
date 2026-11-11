# Leakage audit (#52)

Checks live in `src/leakage.py` and are tested in `tests/test_leakage.py`: each
check passes on clean inputs and fails on an injected leak. Run the audit on the
real data after preparing the split and the preprocessor:

```powershell
python scripts/prepare_data.py
python scripts/prepare_preprocessing.py --csv PATH_TO_AUDITED_CSV
python scripts/audit_leakage.py
```

It writes `results/metrics/leakage_audit.json` and exits 1 on any failure.
Results below are from the audited CSV (SHA-256 `cff2c899…`) and the approved
seed-42 split, 11 November 2026.

## Verified (executable checks, all pass)

| Area | Check | Result |
|---|---|---|
| Partitions | One row per company; no company in two partitions; anchor, target and history length equal those derived from the data | 8,971 companies (6,279 / 1,346 / 1,346), 0 overlap |
| Inputs, both conditions | Exactly one example per company, with the persisted anchor and target | 8,971 examples per condition |
| | No record after the anchor; last real step is the record at *t* | 0 post-anchor records |
| | `five_year` uses exactly the latest min(n_years, 5) records, nothing older | Pass |
| | `years_before_anchor` = *t* − year, strictly decreasing, 0 at *t* | Pass |
| | Padding is 0; `present` marks exactly the real steps | Pass |
| | No company ID, year or label column among the inputs | Pass |
| Preprocessing | Each condition's scaler is fitted on training companies only, on exactly that condition's rows, and equals a refit | latest_year 6,279 rows, five_year 25,846 rows, 0 non-training companies |
| | Transforming keeps padding at 0 and offsets unchanged, for all four models | Pass |
| Class weights | `class_weight_bounds` uses training labels only (426 failed / 5,853 alive, ratio 13.74) | Changing validation/test labels does not change it |
| Study plan (config) | 4 models × 2 conditions = 8 separate studies; Optuna TPE, seed 42; objective validation AP; one budget of 20 trials for every study; `n_jobs` 1, no pruning; class weights tuned; one shared split; identical test companies; `target_verified: false` | Pass |

No financial row is duplicated across companies or partitions, so the same
filing cannot appear in two partitions under different IDs.

## Known limitations (recorded, not leakage by construction)

**Anchor timing and follow-up.** The target is the company's status at its last
recorded year *t*, meant as "files in *t*+1" only under the publisher's
convention (`target_verified: false`, #44/#45). 5,675 of 8,362 alive companies
end before 2018, so they have no observed follow-up after *t*. The `failed` label
also appears on 4,611 earlier rows of failed companies, but those rows are never
targets.

**Indirect signals.** These are not inputs, but they separate the classes
somewhat on the training partition (ROC AUC, where 0.5 = no signal):

| Signal | AUC | Why it matters |
|---|---|---|
| Earlier anchor year → failed | 0.58 | Many alive companies end in 2018, while failures spread over 1999–2018. Financial values can partly encode the calendar era. |
| Longer five-year window → failed | 0.60 | Visible through the `present` flags and the GRU length. |
| Recorded years (all) | 0.54 | Only through the five-year window length. |
| Gap inside the window | 0.51 | Negligible. |

So five_year gains may partly come from history length rather than financial
trends. Report this together with the per-history-group results (#62). The split
is by company, not by time: the same calendar years occur in every partition,
so results show generalization to unseen companies, not to future years.

## Not yet auditable (open)

- **Saved Optuna studies.** No runners exist yet (#53–#56). Once #56 saves the
  studies, they still need checking: validation-only selection, 20 attempted
  trials in each of the eight studies, and frozen settings, checkpoints and
  thresholds before test. `check_study_plan` only covers the declared plan.
- **Decision threshold.** `evaluation.decision_threshold` is still `null`. It
  must be selected on validation only and frozen before test.
- **Class weights in use.** Runners must take their weight bounds from
  `class_weight_bounds` (training labels only). Validation and test must stay
  unweighted and unresampled.
