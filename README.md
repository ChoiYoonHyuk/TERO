# TERO: Temporal Expected Recovery Oracle

Core implementation for **An Exact Recovery Oracle for Longitudinal Evidence Retrieval**. TERO evaluates how many evidence roles remain after uncertain support is realized and an adversary removes up to two source groups.

## Quick start

Python 3.10 or later is required. Run these commands from this directory:

```bash
python -m pip install -r requirements.txt
python example.py
python -m unittest -v test_core
```

The example uses constructed probabilities, demonstrates the paper's dependence-induced ranking reversal, verifies a replacement kernel against direct evaluation, and selects three groups under a budget. It requires no dataset, model download, or GPU. NumPy runs the oracle and search; SciPy is used by the calibration module.

## Files and paper correspondence

| File | Main functions/classes | Paper component |
| --- | --- | --- |
| `tero/ownership.py` | `transition`, `_tables` | Sole/pair ownership, saturation, 992 states |
| `tero/copula.py` | `SignedPartitionMixture.joint_law` | 49-component signed-partition mixture, joint law `P(A,z)` |
| `tero/oracle.py` | `ChronologicalRecoveryOracle.evaluate`, `.analyze` | Full-history recency, forward/backward recurrence, candidate kernels |
| `tero/oracle.py` | `DependentTemporalRecovery` | Full candidate-group context and group replacements |
| `tero/search.py` | `retrieve`, `select_context` | Greedy additions, up to 32 exchange rounds, relevance completion |
| `tero/calibration.py` | `fit_marginals`, `calibrate`, `fit_dependence` | Client-weighted marginal calibration and mixture fitting |

The numerical routines are extracted from the experimental implementation. Package-local imports replace project-specific loaders. This compact package covers the method; dialogue ingestion, LLM inference, comparison-method initialization, and the full experiment driver are separate.

## Inputs and conventions

- **Role order:** baseline, goal, attempted action, recent state. Four-bit masks use these roles as bits 0 through 3.
- **Probabilities:** an `N x 4` matrix of calibrated block-role support probabilities in `[0,1]`.
- **Costs and relevance:** one positive integer token cost and one separate raw relevance score per block.
- **Sessions:** positive integers, with the baseline session numbered `1`. Keep block indices in source order for deterministic tie-breaking.
- **Groups:** one label per block. Use distinct labels for block deletion or session labels for session deletion. Each group lies within one session, and at least two groups are required.
- **Mixture:** 49 nonnegative weights summing to one, ordered as `tero.COMPONENTS`.

`retrieve` zeros baseline probabilities outside session one. Recent-role probabilities remain unmasked: temporal eligibility is computed from the entire history inside the oracle.

For each group, selected support `q` is the coordinatewise maximum over selected blocks, and full support `Q` is the maximum over all candidate blocks. Empty selected groups have `q=0` but remain in the chronological sequence. Their full-history recent event can still invalidate an earlier witness.

## Minimal retrieval call

```python
from tero import retrieve

result = retrieve(
    probabilities, costs, sessions, budget, weights,
    relevance=relevance,
    groups=sessions,                 # session deletion
    initial_sets=initial_sets,        # iterable of feasible block-index sets
    utility="two_first",             # or "mean_recovery"
)
selected = result["selected"]
core = result["core"]
```

The paper's shared initial pool includes the empty set, thresholded cores, final greedy sets from the compared objectives, and recovery-completed cores. Supply that pool through `initial_sets` for the corresponding comparison. The convenience default is the empty set; the example supplies its own constructed starts.

Greedy and exchange comparisons use 11 decimal places. Final ties prefer lower cost, fewer blocks, and earlier block indices. Completion orders blocks by descending raw relevance, then ascending cost and source order.

## Direct oracle and kernels

The oracle accepts laws shaped `(H,16,2)`: `P[h,A,z]` is the probability of selected mask `A` and full-history recent indicator `z`. Every law sums to one, selected recent support implies `z=1`, and candidate changes preserve the full-history recent marginal.

```python
import numpy as np
from tero import ChronologicalRecoveryOracle

oracle = ChronologicalRecoveryOracle(group_sessions, full_recent_probabilities)
R2, R1, R0 = oracle.evaluate(P)
analysis = oracle.analyze(P)
values = oracle.replacement_values(
    analysis, np.array([group_index]), alternative_law[None, :, :]
)
```

The objective is `R_j(E) = E[min_{|F|<=j} C_F]`: a common deletion is chosen after support realization, with recent-state eligibility fixed by the undeleted full history. Oracle values are expected role counts in `[0,4]`, ordered **`(R2,R1,R0)`**. They are distinct from empirical normalized `WR_j` measured against judge labels.

There are **992 ownership states** for four roles and **1,984 augmented states** including the session flag. A kernel query replaces one group's law. Changes to two groups require another analysis after the first replacement. Search uses removal kernels when a swap removes a group's last selected block and direct evaluation for partial-group reductions.

Exactness concerns the supplied probability model, evaluated in float64. Budgeted subset selection is heuristic. With `mean_recovery`, search `value` is the transformed tuple `((R2+R1+R0)/3,R2,R0)`; direct oracle values always retain `(R2,R1,R0)` order.

## Calibration

```python
from tero.calibration import fit_marginals, calibrate, fit_dependence

parameters, marginal_audit = fit_marginals(raw, labels, roles, client_ids)
q = calibrate(raw, roles, parameters).reshape(-1, 4)
weights, dependence_audit = fit_dependence(q, observed_masks, source_client_ids)
```

Marginal inputs are one-dimensional source-role arrays, with roles `0..3`. For the reshape above, provide each source's four roles consecutively. Dependence inputs contain one row and one four-bit observed mask per unique source. Fit dependence before applying temporal masks.

Marginal calibration uses four intercepts, a shared nonnegative slope, and slope penalty `1e-4`. Mixture fitting uses client-equalized joint log loss with the independent-component penalty `-0.01 log(w_ind)`. Use client-disjoint folds for development; fit both stages on training clients before evaluating held-out clients. The paper uses five client folds.

## Experiment context

The primary budgets are 1,200 and 2,400 tokens. Development uses 17 English MusPsy and 19 Chinese PsychEval clients. The evaluated test cohort contains 45 CPCD and 37 MusPsy clients, each with two cutoffs. CPCD's initial pool contains 100 clients and 199 histories before goal and quotation validation.

Qwen3.5-27B supplies goals, Qwen2.5-7B-Instruct supplies support/relevance scores, and Qwen3-8B/Gemma-4-E4B-it supply automated reference judgments. Oracle inputs are model probabilities; evaluation references are kept separate from selection. Empirical metrics and performance tables require the complete experiment data and comparison methods.

## Checks

`test_core.py` checks state counts, joint-law marginals, literal enumeration of support outcomes and deletions, the unselected recent-event reset, kernel/direct agreement, the ranking reversal, and search equivalence under block and session grouping. All included examples and checks use synthetic inputs.
