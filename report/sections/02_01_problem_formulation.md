## 2.0.1 Problem Formulation

The study treats bankruptcy classification as supervised binary learning with
one prediction per company at a fixed financial-year anchor. Let i identify a
company, let t_i be its last recorded financial year, and let x_(i,s) in R^18
contain its accounting variables in year s. Company identifiers, year identifiers
and outcome labels identify observations; they are not predictor features.

The intended target is y*_(i,t_i+1) = 1 if company i files for bankruptcy in the
year following t_i, and 0 otherwise. The publisher describes a pre-filing-year
labelling convention [1]. This intended outcome must be distinguished from the
downloaded CSV's observed label y_i: status_label = failed maps to 1 and alive
maps to 0. The committed data audit shows that this label is constant across a
company's recorded years [2]. Consequently, an earlier row belonging to a failed
company is not automatically an example of bankruptcy in the next year.

The CSV does not independently establish filing dates or complete next-year
follow-up for alive companies. The current configuration therefore records
target_definition = provided_company_status and target_verified = false [3].
Until issue #45 resolves the evidence, experiments classify the supplied company
status using data ending at t_i; they cannot establish independently verified
one-year-ahead event prediction. Selecting a company's final recorded year also
uses the retrospective panel boundary, which is not necessarily known during
live deployment. This limits interpretation as an operational warning system.

Latest-year models receive x_(i,t_i). For a company with complete consecutive
history, the k-year input is

$$
H_i^{(k)} = [x_{i,t_i-k+1},\ldots,x_{i,t_i}] \in \mathbb{R}^{k\times18},
\qquad k\in\{1,2,3,4,5\}.
$$

Changing k adds earlier input years; it does not change t_i or the intended
t_i+1 outcome. Short histories and missing calendar years require a separate
eligibility or masking policy, to be decided in issue #49.

A classifier produces a score p_i in [0,1], and a fixed threshold tau converts
it into a decision: predict failed when p_i >= tau. This probability-shaped
score is not automatically calibrated. Model selection and threshold choice
use validation data; test observations remain separate. Companies, rather than
individual historical rows, are the units of partitioning and evaluation.

The research question concerns whether additional years per company help,
not whether more training companies help. Comparisons must preserve company
and anchor identities. A GRU-versus-latest-year comparison changes architecture
and history together; the within-GRU k=1,...,5 investigation provides a more
focused study of history length. Neither comparison establishes causal effects
of financial variables or generalization to later calendar periods.

### References

1. Dataset authors, [bankruptcy_dataset: publisher documentation](https://github.com/sowide/bankruptcy_dataset). Source description; not independent verification of the downloaded labels.
2. Project [saved data audit](../../results/metrics/data_audit.json), Kaggle version 3; see provenance, measured counts and label/history checks. These are existing audit results, not new model results.
3. Project [configuration](../../configs/config.yaml), dataset target fields. Accessed 9 November 2026.
