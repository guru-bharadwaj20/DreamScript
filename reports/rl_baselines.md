# Phase 11.2.7 — baseline comparison: learned policy vs. topological sort vs. DFS/BFS

Produced by `python -m src.rl.baselines --write` (`experiments/rl/baselines.json`, gitignored).
Every number below was re-run on this commit.

## The ambiguous set had to be defined before it could be used

The plan names the evaluation set but nothing in the repo defined it. Each candidate property,
used as a binary filter, selects almost the whole corpus:

| property | diagrams (of 993) | share |
| :--- | ---: | ---: |
| at least one branch point (out-degree ≥ 2) | 930 | 93.7% |
| at least one unresolved / dropped edge | 558 | 56.2% |
| cyclic (a strongly connected component) | 631 | 63.5% |
| more than one connected component | 555 | 55.9% |
| **any of the above** | **993** | **100.0%** |

Every labelled diagram in this corpus is order-ambiguous; the minimum measured ordering freedom
is 1.0 bit. The set is therefore defined by *magnitude*: ≥ 4 bits of ordering freedom, i.e. at
least 16 emission orders consistent with the drawing. That selects **684 of 993 (68.9%)** —
610 hdbpmn, 74 fa_bresler.

## The table, on the 684-diagram ambiguous set

| arm | mean edge F1 | mean GED (norm.) | outright wins | shared best | win rate |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **dfs** | **0.6247** | — | **345** | 294 | **50.4%** |
| topological | 0.5466 | — | 44 | 226 | 6.4% |
| bfs | 0.3485 | — | 1 | 113 | 0.1% |
| reading (blind control) | 0.1743 | — | 0 | 14 | 0.0% |

294 of 684 diagrams (43%) have no outright winner; 12 have all four arms level.

**The number to beat is not 50.4%.** DFS is in the top group on **639 of 684 (93.4%)** — 345
outright plus all 294 shared. A learned arm has to beat the tie band, not the win rate.

The blind control behaves as a control should: `reading` never wins and scores 0.1743. Had it
come close, the metric would have been measuring page layout rather than flow.

Full corpus (993): dfs 0.5849, topological 0.5415, bfs 0.3986, reading 0.2021.

## Sensitivity — the 4-bit threshold is not load-bearing

| threshold | diagrams | dfs wins | topo wins | bfs | reading | dfs mean F1 | topo mean F1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ≥ 1 | 894 | 346 | 102 | 1 | 2 | 0.6064 | 0.5574 |
| ≥ 2 | 833 | 346 | 92 | 1 | 2 | 0.6130 | 0.5578 |
| ≥ 4 | 684 | 345 | 44 | 1 | 0 | 0.6247 | 0.5466 |
| ≥ 8 | 523 | 290 | 41 | 1 | 0 | 0.6240 | 0.5410 |
| ≥ 16 | 408 | 241 | 36 | 1 | 0 | 0.6129 | 0.5280 |

dfs > topological > bfs > reading at every threshold, on both wins and mean edge F1.

## The bug this row was blocked on

`compare` validated every arm against the module's own 5-node `_PROBE` instead of a diagram from
the set it was about to run, so an arm written for the caller's diagram was rejected for not
returning `_PROBE`'s five node ids — `test_a_registered_arm_joins_the_table_and_can_win` failed
for that reason and no learned policy could ever have entered the table. `check_arm` now takes
the diagram to probe on: `register_arm` keeps `_PROBE` (a generic contract check before an arm
enters the registry), `compare` passes `diagrams[0]`.

## The slot for the learned arm

The learned policy joins this table through `register_arm(name, policy)`; the module deliberately
imports no learner, so the baselines stay measurable and their regressions catchable while the
learned half of 11.2 is written. The learned-vs-baseline numbers are reported in the 11.2.1 and
11.2.10 rows, not here.
