# Phase 11.2.1 — tabular Q-learning: what it learns, and what it does not

Produced by `python -m src.rl.qlearning --episodes 60000 --write`
(`experiments/rl/qlearning.json`, gitignored). Trained on all 993 labelled diagrams, evaluated
greedily on those 993 and on 11.2.7's 684-diagram ambiguous subset.

## Two ceilings, and neither of them is 100%

11.1.4 measured the corpus ceiling: gold play — 7.3.3's DFS replayed through this action space —
reaches full coverage on **63.06%** of the 3,993-diagram corpus, because there is no jump action
and a traversal cannot cross into a second connected component. Recomputed on the sets used here:

| set | gold full coverage | gold mean terminal reward | gold edge F1 |
| :--- | ---: | ---: | ---: |
| all labelled (993) | **41.69%** | −1.4365 | 0.3648 |
| ambiguous (684) | **19.74%** | −3.8938 | 0.2944 |

The ambiguous set is the corpus's hardest slice — 610 of its 684 pages are multi-component
hdbpmn — which is why both sets are reported.

**The second ceiling reframes 11.2.7.** The `dfs` *arm* scores 0.6247 edge F1 there, but that arm
is a whole-graph permutation: it walks into a second component because it never passes through
the action space. Gold play inside the action space scores 0.2944. That is the honest upper bound
for anything trained here; the 0.6247 is not reachable and was never on offer.

## The α × γ sweep

16 cells, 12,000 episodes each, ranked on mean terminal reward (not on edge F1 — ranking on the
transfer metric would be selecting on the test). Mean terminal reward across the grid runs from
−4.0591 (α=0.2, γ=0.9) to **−2.8672 (α=0.4, γ=1.0)**, the selected cell. Every cell reaches
0.0% full coverage at that budget except α=0.4, γ=0.95 at 1.41%. The spread across the whole grid
is 1.19 reward — the hyper-parameters move the result far less than the gap to gold (2.5) does.

## The result at 60,000 episodes (α=0.4, γ=1.0, ε 1.0 → 0.05 geometric)

| | Q-learning | gold | random | empty-policy floor | dfs arm |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **all labelled (993)** | | | | | |
| mean terminal reward | −2.5545 | −1.4365 | −4.2598 | — | — |
| full coverage | 7.35% | 41.69% | 0.00% | — | — |
| coverage vs. ceiling | **17.63%** | 100% | 0% | — | — |
| emitted share | 30.62% | 51.51% | 6.27% | — | — |
| edge F1 | 0.2729 | 0.3648 | 0.2162 | 0.2033 | 0.5849 |
| **ambiguous (684)** | | | | | |
| mean terminal reward | −4.9652 | −3.8938 | −6.1541 | — | — |
| full coverage | 0.00% | 19.74% | 0.00% | — | — |
| emitted share | 15.50% | 32.48% | 2.42% | — | — |
| edge F1 | 0.1855 | 0.2944 | 0.1532 | 0.1479 | 0.6247 |

**It learns something real and it is not enough.** On the full labelled set the agent beats a
random policy on every measure (reward −2.55 vs −4.26, edge F1 0.2729 vs 0.2162) and clears the
empty-policy floor of 0.2033 — the score an agent that emits nothing gets from the document-order
tail. It reaches 17.63% of the coverage gold reaches. On the ambiguous set it never once reaches
full coverage and its edge F1 (0.1855) sits barely above the floor (0.1479).

## It does not converge, and more budget makes it worse

| episodes | |TD| tail/head | reward tail | full coverage (993) | edge F1 (993) | reached keys |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 60,000 | 0.8224 | −2.9521 | 7.35% | 0.2729 | 1,581 |
| 250,000 | 0.8186 | −2.8887 | 1.11% | 0.2201 | 2,120 |

The TD error barely moves — 0.8224 → 0.8186 of its starting value after a 4× budget — and greedy
performance goes *down*. This is not an under-training result that a longer run fixes; the table
has found a plateau it does not leave. Only **1,581 of 128,000** bounded keys (1.24%) are ever
touched, consistent with 11.1.6's measured 1,766.

## The honest reading

The likely cause is the one 11.1.6 already recorded: the abstraction aliases 25.2% of raw states
onto a shared key, so the policy being learned is a policy over a state that cannot distinguish
the situations it needs to. That is a hypothesis this row does not prove — 11.2.5's function
approximation is the test of it, and it is stated here as the next measurement rather than as a
conclusion.

**Against a baseline, tabular Q-learning loses.** It loses to the `dfs` arm by 0.31 edge F1 on
the labelled set, and — holding the action space fixed, which is the only fair comparison — it
loses to gold play in that same action space by 0.09 edge F1 and by 34 points of coverage. That
is the result; 11.2.10 quantifies the same thing end to end on emitted code.

## Implementation notes worth keeping

- The bootstrap is masked. `terminate` is legal in every state, so an unmasked `max` would let
  the value of an illegal `follow-edge-E` leak backwards through the majority of states, where
  the mask leaves 3.29 legal actions of 9. A test pins that a follow slot with no edge behind it
  is never updated.
- `train` drives `src.rl.episode.Episode` rather than `DiagramTraversalEnv`, skipping a 28-vector
  no tabular agent reads; a test asserts the two paths produce identical trajectories.
- The permutation an arm must return is completed with **IR document order**. `gold_order` was
  tried first and rejected: it is 7.3.3's DFS, so an agent that emitted nothing inherited an edge
  F1 of 0.60 and the table scored the filler rather than the policy.
