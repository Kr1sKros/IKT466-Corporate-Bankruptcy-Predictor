# Fixed preprocessing from training-company anchors (#46)

Decision: StandardScaler for Logistic Regression, MLP and GRU; identity for
XGBoost. This choice is fixed and must not be included in an Optuna search.
Fit **one observation per training company: its last recorded year t**. Reuse
these same means and population standard deviations for latest_year and every
real timestep/slot in five_year. Earlier training years and validation/test
financial values never contribute to the fit. This makes a financial value use
the same units across conditions and recency positions. Constant columns use
scale=1, following StandardScaler.

No imputation, clipping or nonlinear transformation is performed. Genuine
missing/nonfinite financial cells cause an error. Padding is not a financial
observation: transform only real steps, then fill padded steps with zero.
Offsets in years before the anchor remain unchanged; flattening creates the
unchanged 0/1 presence indicators. Masks, real lengths, targets and company/
anchor IDs are preserved. Missing calendar years are not manufactured.

## Usage

Run from the repository root with project dependencies installed. First prepare
the existing verified split, then fit and inspect the fixed transform:

```powershell
python scripts/prepare_data.py --csv PATH_TO_AUDITED_CSV
python scripts/prepare_preprocessing.py --csv PATH_TO_AUDITED_CSV
```

The second command checks the audited dataset checksum and configured split
checksums. It writes `results/models/anchor_preprocessor.json` and
`results/metrics/training_anchor_distribution.json` (local generated artifacts).
The transform stores mean, scale, variance, exact financial/derived feature
order, fitting company/year IDs, and dataset/split hashes. Loading it never
refits. Treat these artifacts as tied to that dataset and split version.

```python
from preprocessing import AnchorPreprocessor, transform_histories
from sequences import create_histories, flatten_histories

preprocessor = AnchorPreprocessor.load('results/models/anchor_preprocessor.json')
# companies is the persisted split table; data is the audited panel.
ids = companies.loc[companies['split'].eq('test'), 'company_name']
h = create_histories(data, condition='five_year', companies=ids)
scaled = transform_histories(h, preprocessor, model='mlp')
tabular, metadata = flatten_histories(scaled)
# GRU instead consumes scaled.features with scaled.lengths.
# XGBoost uses the same call with model='xgboost' (financial identity transform).
```

`five_year` takes the last five recorded rows, retaining shorter histories and
excluding older rows. Gaps can make those five records span more than five
calendar years. GRU gets 19 fields per real step (X1-X18 plus relative offset);
tabular models get 100 fields (five slots of 18 financial values, offset and
presence). latest_year has 18 fields and one real step. The historical
full_history API remains available for existing callers, but this preprocessing
path explicitly rejects it. Older study_plan/inputs entries in config and
study-design documentation still describe full_history and need migration in
the corresponding input/training tasks; they do not control this script.

## Training-only distribution inspection

Measured on the audited CSV (SHA-256
`cff2c899a97ecd629415cb22f59186000e74e1c0a78cfae036c0a53025419b5e`), using the
approved seed-42 split: **6,279 final-year training observations**, 18 financial
features, no missing cells in this fitting population.

All 18 distributions are strongly right-skewed: sample skewness ranges from
approximately 11.86 (X15) to 35.35 (X5). The proportion outside each feature's
1.5-IQR fences ranges approximately from 14% to 25%. These are descriptive flags,
not proof of erroneous records and not grounds for deleting companies.
After standardization, the largest absolute value is approximately 55.51 (X8).
Standardization changes scale; it neither removes skewness nor bounds extremes.
This is a limitation to monitor during training, not evidence that an alternative
scaler has better predictive performance. No validation/test distributions or
model scores were used to select or change preprocessing.

The generated JSON includes per-feature quantiles, extrema, mean, population
standard deviation, skewness, IQR flags and maximum absolute standardized value.
These results describe anchor rows only; older five-year inputs can lie outside
the anchor distribution and are transformed without refitting or clipping.

## Verification and integration boundary

Tests verify fitting independence from older/held-out financial values, all eight
model/input combinations, five-record truncation, gap offsets, short histories,
unchanged padding/presence, constant features, exact save/load equivalence,
column-order validation and rejection of missing real financial cells.
Training runners must consume this shared path when implemented; this work does
not claim that any real model training or Optuna studies have been executed.
