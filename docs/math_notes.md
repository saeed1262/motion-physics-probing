# Draft: formal results for the paper (all four now in the paper, November 17: Propositions 1 to 3, Remark 1, proofs in Appendix `app:proofs`)

Four short results. Each would be stated in the main text in two or three lines and proved in the
appendix. Numbers quoted below were computed from the result files on November 17.

Notation: $y$ is a label of the final motion, $\hat x_0$ the clean-motion estimate at the probed step
(in practice: everything it shows), $a$ a layer's activation, $u$ the prompt, $\mathcal V$ a readout
family. $R^2_B$ is the best held-out $R^2$ of the family on input $B$.

---

## 1. The $R^2$ gain is a conditional usable-information gain

**Proposition 1.** Let $\mathcal V$ be the Gaussian predictive family
$\{x \mapsto \mathcal N(g(x), \sigma^2) : g \in \mathcal G,\ \sigma > 0\}$ over a regression class
$\mathcal G$ (linear maps, or our MLP family), and let the family on $(\hat x_0, a)$ contain the
family on $\hat x_0$ (any $g(\hat x_0, a)$ may ignore $a$). Then

$$
I_{\mathcal V}(a \to y \mid \hat x_0) \;=\; \tfrac12 \log \frac{1 - R^2_{\hat x_0}}{1 - R^2_{\hat x_0, a}} \;\ge\; 0 .
$$

**Proof.** For fixed $g$, the expected log loss $\tfrac12\log(2\pi\sigma^2) + \mathbb E[(y-g)^2]/(2\sigma^2)$
is minimized at $\sigma^2 = \mathbb E[(y-g)^2]$, giving $\tfrac12\log(2\pi e\,\mathrm{MSE}(g))$. Taking
the infimum over $g$, $H_{\mathcal V}(y \mid B) = \tfrac12 \log(2\pi e\, M^*_B)$ with
$M^*_B = \inf_g \mathbb E[(y-g(B))^2] = (1-R^2_B)\,\mathrm{Var}(y)$. The conditional information
(Hewitt et al., 2021) is the difference of the two entropies; $\mathrm{Var}(y)$ cancels. Nesting gives
$M^*_{\hat x_0, a} \le M^*_{\hat x_0}$, hence $\ge 0$. $\square$

**What it buys.**
- The reported $R^2$ gain becomes a usable-information quantity in nats, with no change to the
  experiments. It is monotone in the gain for a fixed baseline: $I = -\tfrac12\log\big(1 - \Delta R^2/(1-R^2_{\hat x_0})\big)$.
- It justifies the fallback estimator: the population quantity cannot be negative, because the
  combined family contains the estimate's. A tuned combined readout can still score below the
  estimate's on finite data (selection noise). The fallback restores the nesting at the estimation
  level, choosing among the union of both candidate sets by validation error, so a negative
  measured gain can only come from validation-to-test noise.
- Values (pooled, tuned with fallback): fall rate on the jump prompts **0.063 nats** (50-step,
  $R^2$ 0.351 to 0.428) and **0.056 nats** (1000-step, 0.288 to 0.363); dynamic balance
  **0.0005** (50-step) and **-0.006** (1000-step); static balance **-0.004** (50-step).
- Caveat: for the AUC targets (flight, floats or falls) the readouts are still fitted by squared
  error, so the same formula applies to the MSE, not to the AUC we report.

## 2. What the within-prompt score can and cannot see

Let $\tilde v = v - \bar v_{u}$ denote centring within prompt (frame-weighted, empirical prompt
means), and similarly within (prompt, frame) cells $c = (u, t)$.

**Proposition 2.** For any readout of the form $\hat y = f(B) + h(c)$, where $h$ depends only on the
cell (prompt, and for the second score also frame), the within-cell score of $\hat y$ equals that of
$f(B)$. In particular, information that enters additively through the prompt (and the frame index)
cannot raise the within-prompt (within-prompt-and-frame) score.

**Proof.** Centring within cells is linear and removes any function of the cell exactly:
$\widetilde{f + h} = \tilde f + \tilde h = \tilde f$, because $h$ is constant within each cell.
The within-cell correlation is computed from $\tilde y$ and $\tilde{\hat y}$ only. $\square$

**What it buys.**
- A gain that survives within prompt and frame is information beyond anything the text and the
  frame index could add additively. That is the formal reading of our strongest control.
- It explains the 1000-step pattern: a large pooled gain (+0.076) with a small within-prompt-and-frame
  gain (+0.021) is what extra information that is mostly a function of (prompt, frame) would give,
  for example a prompt's typical timing of a jump.
- The proposition does not cover interactions (the text changing how the estimate should be read).
  That is what the text control tests: "estimate + text" against "estimate + text + layer", with
  families that can model interactions.

## 3. The most early warning can save

Per prompt $u$, let $p_u$ be the share of its samples that fail, $B$ the retry budget, and
$\tau = (k+1)/n$ the cost of an attempt aborted after step $k$ of $n$. Finish-then-filter makes
attempt $j$ ($j = 0, \dots, B-1$) with probability $p_u^j$, each at full cost. A perfect detector with
the final check kept makes the same attempts and delivers the same failure $p_u^B$, but pays $\tau$
for every failed attempt before the last.

**Proposition 3.** Against finish-then-filter at the same budget, a perfect detector at fraction
$\tau$ saves

$$
S_B \;=\; (1-\tau)\; \frac{\sum_u \sum_{j=1}^{B-1} p_u^{\,j}}{\sum_u \sum_{j=0}^{B-1} p_u^{\,j}}
\;=\; (1-\tau) \times \big(\text{share of finish-then-filter attempts that are discarded}\big).
$$

**Proof.** Attempt $j < B-1$ fails with probability $p_u$; the detector then pays $\tau n$ instead
of $n$. Summing the saved $(1-\tau)n$ over attempts $1, \dots, B-1$ and dividing by the expected
finish-then-filter cost $n\sum_{j<B} p_u^j$ gives the result; prompts are weighted equally, as in
the simulation. $\square$

**Check against our simulation** (`results/phase2/early_warning/matched_failure_compute.csv`,
detector at 20%):

| model | budget | closed form | simulated (Monte Carlo) |
|---|---|---|---|
| 50-step | 2 / 3 / 5 | 0.0672 / 0.1013 / 0.1477 | 0.0673 / 0.1018 / 0.1485 |
| 1000-step | 2 / 3 / 5 | 0.1203 / 0.1716 / 0.2264 | 0.1195 / 0.1701 / 0.2253 |

**What it buys.** The ceiling is set by the data, not the detector: it is $(1-\tau)$ times the share
of attempts that would be thrown away. Failures are rare and concentrated in a few prompts, so that
share is small at low budgets. Trained detectors in our data save less than this ceiling. To check
before writing: whether the oracle is an upper bound for every detector at matched failure (a
detector that aborts passing samples also changes the failure rate; the argument needs care).

## 4. Why chance-level linear probes on an averaged latent prove nothing

**Remark 4.** A linear readout of the time-averaged latent, $w^\top \bar z$ with
$\bar z = \tfrac1T \sum_t z_t$, depends on a video only through $\bar z$. It is therefore invariant to
any rearrangement of the frames in time and blind to any change that preserves the average, while a
physical violation such as a vanishing or teleporting object is a change over time. Chance accuracy
of such probes is consistent with the violation being fully present in the latent. Our data show the
two situations apart: the total frame-to-frame change differs by only 1.00 to 1.03 between impossible
and possible videos, while its spatial pattern, read after one fixed nonlinearity, separates them.

This one is a remark, not a theorem: the IntPhys changes are not exactly mean-preserving, so the
statement explains the failure mode rather than proving it for this dataset.

---

**Suggested use.** 1 and 2 in the method section (they define what our two main measurements
mean); 3 in the early-warning section, replacing a paragraph of simulation description; 4 as one
sentence in the video section. About half a page in the main text, about a page in the appendix.
