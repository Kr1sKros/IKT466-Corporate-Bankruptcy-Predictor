## 2.0.2 Classification Models

The study compares three latest-year classifiers with a sequence classifier.
All predict failed=1 using the company/anchor and target definitions in Section
2.0.1. The table defines the experiment scope; no additional models are implied.

| Model | Input | Role |
|---|---|---|
| Logistic regression | Latest-year vector x_(i,t_i) | Linear baseline |
| XGBoost | Latest-year vector x_(i,t_i) | Nonlinear tree baseline |
| Multilayer perceptron (MLP) | Latest-year vector x_(i,t_i) | Neural baseline |
| Gated recurrent unit (GRU) | History H_i^(k), k=1,...,5 | Sequence model |

Logistic regression transforms an affine score into a binary class probability:

$$
p_i=\sigma(b+w^T x_{i,t_i}),\qquad \sigma(a)=\frac{1}{1+e^{-a}}.
$$

Its log-odds are linear in the input variables; it does not require normally
distributed inputs. Without constructed interaction terms, its decision boundary
cannot express arbitrary feature interactions. Regularization penalizes large
coefficients and controls complexity. Coefficients describe associations
conditional on the other inputs, not causal effects; correlated accounting
variables complicate their interpretation [1].

XGBoost builds an additive ensemble of decision trees. For binary logistic
classification, its margin F(x) is transformed by the sigmoid into a score.
Successive trees optimize a regularized objective using information about the
loss derivatives. Splits permit nonlinear effects and interactions without
requiring a linear relationship between inputs and log-odds. Learning rate,
tree depth and ensemble size control fitting capacity; the algorithm is not
guaranteed to outperform the other models on this dataset [2].

An MLP composes affine transformations with nonlinear activations. With hidden
representation h^(l) = phi(W^(l) h^(l-1) + b^(l)), h^(0) = x_(i,t_i), a final
scalar logit followed by a sigmoid produces p_i. Unlike logistic regression, an
MLP can represent nonlinear interactions. Training adjusts weights using
backpropagation. Its optimization is non-convex, so initialization, scaling and
regularization can affect the fitted solution. Width, depth and activation
choices belong in the eventual methods section and are not fixed here [3].

For binary training, a common per-observation objective is

$$
\ell_i=-y_i\log p_i-(1-y_i)\log(1-p_i).
$$

Numerically stable implementations may operate directly on logits. Class
weighting changes the relative contribution of failed and alive observations;
it is distinct from changing the decision threshold after fitting. The final
weighting policy is part of issue #57, not an assumption of this background.

The GRU processes ordered histories, as defined in Section 2.0.3. The MLP helps
separate a neural latest-year baseline from a linear baseline, but comparing it
with a multi-year GRU still changes both architecture and information. A GRU
with k=1 has no between-year recurrence to exploit, yet is not mathematically
identical to the MLP. Model rankings must therefore come from measured,
company-aligned experiments rather than assumptions about model complexity.

SVMs, random forests and LSTMs are not part of the agreed experiment set. Any
discussion of them elsewhere is background only and does not add training runs.

### References

1. Scikit-learn, [Logistic regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression), user guide.
2. Chen, T. and Guestrin, C. (2016). [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754). KDD. DOI: 10.1145/2939672.2939785.
3. Scikit-learn, [Neural network models (supervised)](https://scikit-learn.org/stable/modules/neural_networks_supervised.html), user guide. Cited for MLP concepts, not a commitment to a particular implementation library.

Online documentation accessed 9 November 2026.
