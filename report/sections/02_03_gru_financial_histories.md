## 2.0.3 GRU and Financial Histories

A gated recurrent unit (GRU) updates a hidden representation as it reads a
sequence. Gating controls how past information contributes to the next state
[1]. Here each timestep represents one accounting observation, not one company.
For a complete k-year history H_i^(k), inputs are supplied oldest to newest,
ending at the fixed anchor t_i. Selecting the window backwards from t_i does
not imply feeding it backwards into the GRU.

For a batch of B complete histories with d=18 features, batch-first input has
shape B x k x d. A hidden width m gives one m-dimensional state per company
at each timestep. A single-layer, unidirectional recurrence can be written
using the PyTorch convention as follows [2]:

$$
\begin{aligned}
r_j &= \sigma(W_{ir}x_j+b_{ir}+W_{hr}h_{j-1}+b_{hr}),\\
z_j &= \sigma(W_{iz}x_j+b_{iz}+W_{hz}h_{j-1}+b_{hz}),\\
n_j &= \tanh(W_{in}x_j+b_{in}+r_j\odot(W_{hn}h_{j-1}+b_{hn})),\\
h_j &= (1-z_j)\odot n_j+z_j\odot h_{j-1}.
\end{aligned}
$$

The company index is suppressed here. The reset gate r_j controls the recurrent
contribution to the candidate n_j; the update gate z_j controls retention of
the previous state. The symbol odot denotes elementwise multiplication.
Input weight matrices have shape m x d, recurrent matrices m x m, and biases
length m. This candidate equation follows PyTorch's placement of the reset
gate, which differs from the original formulation [2].

A scalar output layer maps the final valid state to a classification score,
p_i = sigma(w_o^T h_final + b_o). Binary loss supervises one company-level
output; the repeated historical status labels are not separate next-year
training targets. Backpropagation through the recurrence learns shared weights.
The equations illustrate the model family rather than fixing the implemented
number of layers, hidden width or initialization.

| k | Complete input window, in processing order | Intended outcome year |
|---|---|---|
| 1 | t_i | t_i+1 |
| 2 | t_i-1, t_i | t_i+1 |
| 3 | t_i-2, t_i-1, t_i | t_i+1 |
| 4 | t_i-3, t_i-2, t_i-1, t_i | t_i+1 |
| 5 | t_i-4, t_i-3, t_i-2, t_i-1, t_i | t_i+1 |

For example, an anchor of 2017 yields a five-year input spanning 2013-2017;
the intended outcome remains 2018 for every k. This is an illustrative window,
not a claim that a particular company's outcome timing is verified. The
observed-versus-intended target distinction in Section 2.0.1 still applies.

Longer histories might reveal deterioration or recovery hidden by a single
year, but they may also introduce noise and restrict eligible companies. This
is a hypothesis to test, not an established result. Evaluating each k on a
different population would mix the effect of history length with sample
selection. The history investigation should therefore use a common eligible
test cohort with identical company/anchor IDs and labels, and report coverage
and any changes to the training cohort.

Short histories can be excluded or represented using padding with explicit
lengths/masks. Padding alone does not prevent a recurrent model from treating
zeros as real observations; classification must use the final valid state.
An internal missing year is different from padding at a sequence boundary:
five recorded observations need not span five consecutive years. Issue #49
must choose and document the policy, including exclusions or any representation
of elapsed time. No policy is assumed to be implemented by this section.

### References

1. Cho, K. et al. (2014). [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078). EMNLP. Foundational gated recurrent model; not evidence about bankruptcy performance.
2. PyTorch, [GRU documentation](https://docs.pytorch.org/docs/2.15/generated/torch.nn.GRU.html), version 2.15, accessed 9 November 2026. Equations and dimensional conventions; the project's runtime version must be recorded separately when training.
