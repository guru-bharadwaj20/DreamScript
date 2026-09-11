# Phase 11.2.5 — DQN: function approximation where the table plateaus

Produced by `python -m src.rl.dqn` in stages (`prelim`, `sweep`, six `member` runs, `merge`);
merged result `experiments/rl/dqn.json`, parts in `experiments/rl/dqn_parts/`, checkpoints
`experiments/rl/dqn/p11-dqn-features-s{0,1,2}.pt` (all gitignored). Figure
`reports/figures/p11_dqn.png`. Same reward, episode, epsilon schedule (1.0 → 0.05 geometric),
60,000-episode budget and evaluation harness (`qlearning.play`) as 11.2.1.

## 1. The hypothesis, tested before any training — and it was the wrong hypothesis

11.2.1 attributed the tabular plateau to "25.2% of raw states sharing an abstract key". That
number is 11.1.6's share of raw keys *shared by two or more diagrams*, not an aliasing rate.
Measured directly over the 10,983 states gold play visits on the 993 labelled diagrams:

| representation | distinct keys | gold states on a key whose gold actions disagree |
| :--- | ---: | ---: |
| 11.1.6 abstract key | 414 | 328 (**2.99%**) |
| `StateEncoder.features` (28) | 3,350 | 0 (0.00%) |
| features + legal mask (37) | 3,350 | 0 (0.00%) |

And the policy each representation can *express*, fitted to gold actions (no RL) and played
through the same harness:

| policy | reward | full coverage | edge F1 |
| :--- | ---: | ---: | ---: |
| gold play in the action space | −1.4365 | 41.69% | 0.3648 |
| majority gold action per abstract key | −1.5155 | 39.48% | **0.3630** |
| supervised MLP on the 28 features (train acc 1.0) | −1.4365 | 41.69% | 0.3648 |
| same two, fitted on 795, played on 198 held out: key / MLP | −1.3903 / −1.0344 | 38.38% / 45.45% | 0.3666 / 0.3801 (gold 0.3801) |

**A table over the abstract key can express 99.5% of gold's edge F1.** Representation is not
why 11.2.1 plateaus at 0.2729; learning is. A function approximator is therefore tested here as
a *learner*, not as a fix for aliasing.

## 2. Sweep (8 cells, one ensemble, 60,000 episodes each, ranked on greedy terminal reward)

| inputs | lr | γ | greedy reward (993) |
| :--- | ---: | ---: | ---: |
| features | 1e-3 | 0.95 | −1.8340 |
| features | 1e-3 | 1.0 | −1.7302 |
| features | 3e-4 | 0.95 | −2.0430 |
| **features** | **3e-4** | **1.0** | **−1.6351** |
| features+mask | 1e-3 | 0.95 | −1.9013 |
| features+mask | 1e-3 | 1.0 | −2.1343 |
| features+mask | 3e-4 | 0.95 | −1.7739 |
| features+mask | 3e-4 | 1.0 | −1.7485 |

Spread 0.50 across the grid; **every cell beats every tabular configuration on record**. Adding
the legal mask as input bought nothing (its best cell −1.7485). Edge F1 was not used to select.

## 3. Result, 3 seeds × 60,000 episodes (mean ± sd)

| | DQN | tabular (11.2.1 cell α0.4 γ1.0) | tabular_best (α0.2 γ0.99 ε-floor 0.01) | gold | random | empty floor |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **all 993** reward | **−1.534 ± 0.056** | −2.942 ± 0.717 | −2.694 ± 0.067 | −1.4365 | −4.2598 | — |
| full coverage | 28.87 ± 8.34% | 4.70 ± 2.78% | 1.58 ± 0.81% | 41.69% | 0% | — |
| emitted share | 48.06 ± 0.07% | 28.76 ± 4.06% | 25.15 ± 1.63% | 51.51% | 6.27% | — |
| edge F1 | **0.3411 ± 0.0097** | 0.2460 ± 0.0395 | 0.2797 ± 0.0128 | 0.3648 | 0.2162 | 0.2033 |
| **ambiguous 684** reward | −4.129 ± 0.092 | −5.279 ± 0.612 | −4.947 ± 0.077 | −3.8938 | −6.1541 | — |
| full coverage | 10.28 ± 2.94% | 0% | 0% | 19.74% | 0% | — |
| edge F1 | **0.2603 ± 0.0052** | 0.1718 ± 0.0261 | 0.2097 ± 0.0107 | 0.2944 | 0.1532 | 0.1479 |
| **held out 198** (trained on 795) reward | −1.562 ± 0.110 | −2.581 ± 0.184 | −2.545 ± 0.039 | −1.0344 | — | — |
| edge F1 | 0.3262 ± 0.0292 | 0.2448 ± 0.0298 | 0.3027 ± 0.0090 | 0.3801 | — | 0.2123 |

Tabular seed 0 reproduces 11.2.1 exactly (−2.5545, 0.2729); its three-seed spread (0.72 reward)
is why 11.2.1's single number was optimistic — the 5-seed mean at that cell is −2.9952 ± 0.58
(measured again in 11.2.6's `flat` arm). **The DQN closes 93% of the reward gap from tabular to
gold** (−2.94 → −1.53 against −1.44) and **85% of the edge-F1 gap from the empty floor to gold**
(0.2033 → 0.3411 of 0.3648), with a seed SD 13x smaller than the 11.2.1 table's. |TD| tail/head is
0.37 against the table's 0.82 (seed 0). The shaped table 11.2.4 recommends (quoted from that row: +0.37 over a −2.78 control, 0.3178
F1) is still below the unshaped DQN.

**What it does not do**: it does not reach gold. On the held-out 20% the reward gap to gold widens
to 0.53 and the F1 gap to 0.054, so part of the in-sample margin is fit to the 993 - and on held-out **edge F1** the DQN's lead
over `tabular_best` (0.3262 vs 0.3027) is smaller than its own seed SD (0.029), so only the
reward gap (−1.56 vs −2.55, 9x the SD) is resolved out of sample. Against the
`dfs` *arm* (0.5849 on all, 0.6247 ambiguous — a whole-graph permutation outside this action
space) it still loses, as any policy in this action space must.

## 4. "Large graphs where tabular fails" — by size, seed 0, 993 labelled

| nodes (diagrams) | tabular F1 | tabular_best F1 | DQN F1 | gold F1 | DQN full cov | gold full cov |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| ≤5 (225) | 0.4909 | 0.4124 | 0.5174 | 0.4614 | 77.8% | 99.6% |
| 6–12 (229) | 0.3569 | 0.3820 | **0.5249** | 0.5991 | 10.9% | 76.0% |
| 13–25 (462) | 0.1579 | 0.1960 | 0.2235 | 0.2442 | 0.0% | 3.5% |
| >25 (77) | 0.0759 | 0.0815 | 0.0921 | 0.1099 | 0.0% | 0.0% |

The DQN's largest gain is the 6–12 band (+0.17 F1 over the 11.2.1 table). On >25 nodes every
policy, gold included, sits near the empty floor: those pages are multi-component and the action
space cannot reach most of them, so "function approximation for large graphs" is capped by 11.1.4's
reachability ceiling rather than by the learner. On ≤5 nodes the DQN's F1 (0.5174) exceeds gold's (0.4614) while its full coverage is lower
(77.8% vs 99.6%): edge F1 of a partial order completed with document order can beat gold's DFS
adjacency on tiny pages, which is a property of the metric, not a better traversal.

## 5. Throughput (the GPU rules)

See the `dqn.py` docstring table. Eager update 9.5 ms regardless of batch → CUDA-graph capture
7.43 → 1.25 ms (3 members, batch 256); n_envs flattens at 64; one 8-member ensemble 5,689
transitions/s; three separate single-member processes 10.2k t/s vs a 3-member ensemble 6,961
(+1,077 vs +499 MiB), so the sweep ran stacked and the six seed runs as processes; peak device
allocation ≤ 700 MB per process, all under 3 GB. After capture the env loop (CPU) is 78% of the
sweep's wall time.

## 6. Bugs found on the way

- The first CUDA-graph capture dropped the first update (parameters unchanged after capture,
  TD read from warm-up weights): runs diverged from eager. Fixed by undoing the capture pass and
  replaying; capture and eager now agree bit-for-bit (tested).
- No device synchronise before the side-stream warm-up: 2 of 6 concurrent runs died with
  "illegal memory access". Fixed; all six reruns completed.
