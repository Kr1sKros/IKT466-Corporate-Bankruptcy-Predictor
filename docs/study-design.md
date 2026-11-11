# Agreed experiment protocol

This is the current study plan, revised on 9 November 2026. It supersedes the
latest-year-baselines versus GRU k=1–5 plan. The implementation status and remaining
work are listed in the README and GitHub issues; this document does not claim that
the eight training experiments have run.

## Question and comparisons

**Does using five years of financial observations improve corporate bankruptcy prediction
compared with using only the latest year, and how does this effect differ across
model families?**

| Model | `latest_year` | `five_year` |
|---|---|---|
| Logistic regression | Latest annual features | Flattened five-year history |
| XGBoost | Latest annual features | Flattened five-year history |
| MLP | Latest annual features | Flattened five-year history |
| GRU | One real timestep | Variable-length full sequence |

Train eight separate configurations, not one model reused without retraining.
Investigation 1 compares model families within each condition. Investigation 2
compares conditions within each family. Report five-year minus latest-year AP
and changes in precision/recall/F1, plus paired errors, computing cost and uncertainty.
There is no mandatory k=1–5 sweep and no claim to identify an optimal history length.
SVM is outside the agreed scope.

## Companies, target and split

- One example per company, anchored at its final recorded financial year t. All
  input observations are at or before t. Earlier rows are inputs, never additional
  positive outcomes for failed companies.
- Reuse the persisted 70/15/15, seed-42 company split and its hashes. It already
  stratifies jointly by failed/alive and recorded-history groups 1, 2, 3, 4, 5+.
  These are approximately preserved proportions, not balanced classes or equal
  group counts. Do not regenerate a different split for either input condition.
- Keep all companies with valid records, including one-year histories. Validate
  company IDs, anchors and labels across all eight prediction files; reject silent
  intersections or omissions.
- The target remains company status at t, interpreted as a t+1 filing only under
  the publisher's convention. Preserve `target_verified: false` and the audit's
  follow-up/terminal-record limitations. The scope change does not verify timing.
- Short records do not prove a company is young. Report history coverage and
  class prevalence per partition and evaluation group. Retain the existing 1/2/3/4/5+
  strata for the main analysis; descriptive subdivisions of 5+ must disclose small
  counts. Do not retune models on test subgroup outcomes.

## Five-year representation

Confirmed 10 November 2026 (#16), replacing the earlier full-history plan: use the
**five most recent recorded annual observations** ending at t, or all of them when
a company has fewer than five. Older observations are excluded. With gaps, five
records can span more than five calendar years. This is not an all-history
experiment or a k=1–5 sweep.

- Preserve each company's actual observation order and relative time offsets
  from t. Missing calendar years are skipped as observations but their gaps remain
  visible through those offsets. Never manufacture financial values for missing
  calendar years.
- GRU: real records run oldest to newest, followed by padding for batching. Pass
  real lengths to packed inference so padding cannot change the final valid state.
- Logistic regression/XGBoost/MLP: flatten a fixed-width history with slots aligned
  by recency (latest record always in the same slot), padding before short histories.
  Include presence indicators and relative time offsets. Match the GRU's available
  financial values and temporal metadata despite the different storage layout.
- X1–X18 are the financial features. Raw company IDs, labels and absolute calendar
  years are not predictors. Any derived temporal/presence fields must be explicitly
  named and validated separately from the audited raw-financial-column contract.
- Padding is not an observed zero. Fit transforms only on valid training records,
  save their fitting population/order, and transform validation/test without refits.
  Keep the representation identical within each condition across model families.
- Latest-year inputs contain just the record at t; do not append total historical
  length to this condition. A one-year GRU is a valid reference but cannot exploit
  changes across years. A company with one record has identical financial coverage
  under both conditions, although its separately trained models may predict differently.
- History length and missingness can themselves predict status; five-year gains
  need not be evidence of learning financial trends. Discuss this limitation.

Implemented in `src/sequences.py` (`create_histories`, `tabular_inputs`).

## Optuna selection and weights

- One independent study per model × input condition: eight studies, with unique
  model/condition/split identifiers and separate output directories.
- Use seeded `TPESampler`, maximise validation **Average Precision**, and use a
  common predeclared trial budget. The initial planning budget is **20 attempted
  trials per study**, subject to one documented pilot review before main searches.
  A budget change applies to every study, not selectively after viewing results.
- Use sequential study execution (`n_jobs=1`) for reproducibility and no Optuna
  pruning in the initial protocol. Neural early stopping is allowed under the same
  documented validation-AP/patience/epoch-cap rules for paired conditions. Record
  actual elapsed time; equal trial counts do not imply equal computing costs.
- Search spaces are model-specific. Within a model, use the same paired-condition
  ranges where applicable; input dimensions differ by construction. Declare any
  necessary exceptions before the main searches. Record ranges/distributions,
  sampler/training seeds and software versions.
- Optuna selects hyperparameters, including class/loss weights. LR searches
  `class_weight` (unweighted versus training-label-derived balanced weights);
  XGBoost searches `scale_pos_weight`; MLP/GRU search loss `pos_weight`. Numerical
  ranges use training-label statistics only and include the unweighted option.
  Validation/test prevalence is never changed by resampling or reweighting.
- Coefficients, tree parameters and neural network weights are fitted by each
  model's training algorithm inside a trial. Optuna does not directly search all
  learned weights. Do not require identical best hyperparameters across conditions.
- Failed trials consume attempts and must be reported; an incomplete or wholly
  failed study cannot be silently replaced by defaults. Log trial state, failure,
  objective, configuration, training/checkpoint provenance and elapsed time.
- Select classification thresholds on validation only using one declared rule.
  Freeze the selected configuration/checkpoint and threshold before final test
  evaluation. Optional seed repeats must use a predefined protocol, not select
  the best test seed. Never use test AP to choose studies, budgets or settings.

Sources: [Optuna search spaces](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/002_configurations.html),
[study optimisation and trial budgets](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html),
[reproducibility guidance](https://optuna.readthedocs.io/en/stable/faq.html#how-can-i-obtain-reproducible-optimization-results).

## Reporting and reproducibility

Save split/data/configuration hashes, preprocessing schemas, all study records,
selected hyperparameters/class weights, thresholds, prediction IDs, metrics, model
checkpoints, package/device versions, timing, and any nondeterminism. Report the
eight-cell model/condition matrix and within-family AP differences; include
history-group counts, class prevalence and single-class/undefined-metric handling.
Interpret results as performance after separate, comparable tuning, not a causal
effect of adding years while holding every other choice fixed.

Use the MSc template with Introduction sections Background and Motivation,
Objective and Research Questions, and Scope and Approach. A drop-in introduction
is in [report/introduction.tex](../report/introduction.tex); integrate it into the
actual report source and add verified dataset/method/software citations there.
Do not claim it has been compiled in the unavailable report template.

Internal completion target: **22 November 2026**. Course submission deadline:
**30 November 2026 at 23:59 Norwegian time**. Save the presentation, contribution
records and the required AI-tool disclosure with the submission; use actual model
and tool versions where available.
