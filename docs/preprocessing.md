# Fixed per-condition preprocessing from training data (#46)

Decision: signed log1p followed by StandardScaler for Logistic Regression, MLP
and GRU; identity for XGBoost. This choice is fixed and must not be included in
an Optuna search.

signed log1p is `sign(x) * log(1 + |x|)`. It keeps order and sign, maps 0 to 0,
and shrinks the dollar-scale tails: a company 1,000 times larger than another
ends up about 7 units apart instead of 1,000 times apart. XGBoost splits only on
value order, so the log step would not change it and it keeps the raw values.
Why this was added on top of plain standardization (#87): see
[Heavy-tailed inputs](#heavy-tailed-inputs-87) below.

Each input condition gets **its own scaler, fitted on exactly the training
records that condition uses**:

| Condition | Fitting population | Rows (seed-42 split) |
|---|---|---|
| `latest_year` | One row per training company: its last recorded year t | 6,279 |
| `five_year` | Up to the five most recent recorded years per training company (the same records the five_year inputs contain) | 25,846 |

One scaler per condition is shared by all its timesteps/slots, so X1 at r0 and X1
at r4 use the same mean and scale and year-to-year changes stay comparable.
The scaler's mean and scale are those of the signed-log1p training values.
Training years before the five-year window and all validation/test financial
values never contribute to a fit. Constant columns use scale=1, following
StandardScaler.

No imputation or clipping is performed; signed log1p is the only nonlinear step. Genuine
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
checksums. It writes `results/models/preprocessor.json` and
`results/metrics/preprocessing_fit_distribution.json` (local generated artifacts).
Per condition, the transform stores mean, scale, variance, the fitting
companies and years, and the exact feature schema; it also stores dataset/split
hashes. Loading it never refits. Treat these artifacts as tied to that dataset
and split version.

```python
from preprocessing import FittedPreprocessor, transform_histories
from sequences import create_histories, flatten_histories

preprocessor = FittedPreprocessor.load('results/models/preprocessor.json')
# companies is the persisted split table; data is the audited panel.
ids = companies.loc[companies['split'].eq('test'), 'company_name']
h = create_histories(data, condition='five_year', companies=ids)
scaled = transform_histories(h, preprocessor, model='mlp')  # uses the five_year scaler
tabular, metadata = flatten_histories(scaled)
# GRU instead consumes scaled.features with scaled.lengths.
# XGBoost uses the same call with model='xgboost' (financial identity transform).
```

`five_year` takes the last five recorded rows, retaining shorter histories and
excluding older rows. Gaps can make those five records span more than five
calendar years. GRU gets 19 fields per real step (X1-X18 plus relative offset);
tabular models get 100 fields (five slots of 18 financial values, offset and
presence). latest_year has 18 fields and one real step.

## Training-only distribution inspection

Measured on the audited CSV (SHA-256
`cff2c899a97ecd629415cb22f59186000e74e1c0a78cfae036c0a53025419b5e`), using the
approved seed-42 split, 18 financial features, no missing cells in either
fitting population.

| | `latest_year` | `five_year` |
|---|---|---|
| Fitting rows | 6,279 | 25,846 |
| Skewness range | 11.86 (X15) – 35.35 (X5) | 11.58 (X15) – 29.77 (X5) |
| Share outside 1.5-IQR fences | 14% – 25% | 14% – 25% |
| Largest absolute standardized value, plain z-score (before #87) | 55.51 (X8) | 60.52 (X8) |
| Skewness range after signed log1p + standardization | -0.59 – 0.82 | -0.62 – 0.76 |
| Largest absolute value after signed log1p + standardization | 4.76 (X18) | 4.85 (X18) |

All 18 distributions are strongly right-skewed in both populations. IQR flags are
descriptive, not proof of erroneous records and not grounds for deleting
companies. Plain standardization changes scale but neither removes skewness nor
bounds extremes; the signed log1p step does both (rows above). The step was
chosen with validation scores only (#87); no test data was used.

The generated JSON includes, per condition, per-feature quantiles, extrema, mean,
population standard deviation, skewness, IQR flags and maximum absolute
standardized value.

## Verification and integration boundary

Tests verify that each condition is fitted on its own training rows (and that
these equal the real training steps of its inputs), independence from older and
held-out financial values, all eight model/input combinations, five-record
windows, gap offsets, short histories, unchanged padding/presence, constant
features, exact save/load equivalence, column-order validation and rejection of
missing real financial cells. Training runners must consume this shared path
when implemented; this work does not claim that any real model training or
Optuna studies have been executed.

## Heavy-tailed inputs (#87)

The dollar-scale features are extremely right-skewed. With plain standardization
almost every company sits near 0 (median |z| 0.15) while a few giants reach
|z| 55-97, so weight-based models (LR, MLP, GRU) are pulled around by a handful
of companies. Trees (XGBoost) are unaffected. Signed log1p before standardizing
fixes this without anything extra to fit.

Validation AP on the seed-42 split (test never used). Ranges are over 5 training
seeds, or 5 sampler seeds for LR; full tables and settings are in #87.

| latest_year | Plain standard | signed log1p + standard |
|---|---|---|
| LR, 20-trial Optuna | 0.253 - 0.256 | 0.241 - 0.244 |
| MLP, fixed settings | 0.278 - 0.307 | 0.336 - 0.375 |
| MLP, 20-trial Optuna | 0.338 | 0.373 |
| GRU, fixed settings | 0.270 - 0.277 | 0.350 - 0.369 |
| GRU, 20-trial Optuna | 0.288 | 0.382 |
| XGBoost, fixed settings | 0.315 - 0.346 | identical |

five_year results are being added to #87. LR loses slightly on latest_year; one
shared policy for the three scaled models is kept and this trade-off is accepted.
