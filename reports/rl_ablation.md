# 11.2.10 - Pipeline ablation: quality without RL

Generated from `python -m src.rl.ablation --write` (`experiments/rl/ablation.json`,
`reports/figures/p11_ablation.png`). 993 labelled diagrams (684 ambiguous), 3 seeds, tabular arm
60,000 episodes, DQN = 11.2.5's `p11-dqn-features-s{0,1,2}` checkpoints (regenerated 2026-09-12 with
`src.rl.dqn --stage member`, 11.2.5's default cell), sandbox on a seeded 150-diagram sample
(101 hdbpmn / 49 fa_bresler). Arms and the own/oracle loop-mark modes are defined in the
`src/rl/ablation.py` docstring.

## Table - 993 labelled, own loop marks (learned arms: seed mean)

| arm | semantic | terminal reward | edge F1 | coverage | loops | undefined | sandbox pass |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| topological | 0.8789 | 4.516 | 0.5415 | 1.000 | 0.314 | 0.000 | 1.000 |
| dfs | 0.8630 | 4.452 | 0.5849 | 1.000 | 0.314 | 0.000 | 1.000 |
| bfs | 0.8772 | 4.509 | 0.3986 | 1.000 | 0.314 | 0.000 | 1.000 |
| reading | 0.7997 | 4.199 | 0.2021 | 1.000 | 0.314 | 0.000 | 1.000 |
| gold_in_action_space | 0.5903 | -1.437 | 0.3648 | 0.515 | 0.314 | 0.500 | 0.473 |
| gold+dfs | 0.8662 | 4.465 | 0.5943 | 1.000 | 0.314 | 0.000 | 1.000 |
| dqn | 0.5861 | -1.771 | 0.3255 | 0.469 | 0.501 | 0.672 | 0.309 |
| dqn+dfs | 0.8890 | 4.556 | 0.5612 | 1.000 | 0.504 | 0.000 | 1.000 |
| tabular_shaped | 0.5582 | -1.994 | 0.3344 | 0.434 | 0.427 | 0.808 | 0.176 |
| tabular_shaped+dfs | 0.8783 | 4.513 | 0.5792 | 1.000 | 0.430 | 0.000 | 1.000 |

## Findings

1. **RL as the orderer loses badly, and almost all of it is the action space, not the learning.**
   DQN minus the best heuristic: semantic **-0.293 ± 0.017**, terminal reward -6.29, edge F1 -0.259,
   sandbox pass **-0.691** (0.309 vs 1.000). But gold play inside the same action space scores 0.590
   semantic / 0.473 sandbox, and DQN is within **-0.004 ± 0.017** semantic of it: a policy that cannot
   cross components leaves undefined names in 50-67% of programs no matter how well it learns.
2. **On top of DFS, RL adds a small gain that is entirely its loop marks.** `dqn+dfs - dfs` semantic
   **+0.026 ± 0.013** over seeds (seed 0 paired: +0.027, 2 s.e. 0.004, 282 wins / 615 ties / 96
   losses), terminal reward +0.104; `loops` rises 0.314 -> 0.504. With every arm given oracle loop
   marks the gain disappears (**-0.0025 ± 0.007**), so the learned *ordering* contributes nothing -
   only the `mark-as-loop` decisions do.
3. **The ordering itself costs edge F1.** `dqn+dfs - dfs` edge F1 **-0.024 ± 0.014** on the 993 and
   -0.050 ± 0.019 on the 684 ambiguous (seed 0: 2 wins / 121 losses). Even a perfect prefix is worth
   little: `gold+dfs - dfs` is +0.0094 edge F1 (41 wins, 0 losses) on the 993 and exactly 0 on the
   ambiguous set.
4. **Executable code does not need RL.** Every full-coverage arm, heuristic or hybrid, passes the
   sandbox on 150/150.

## Verdict

Removing RL and ordering with DFS/topological loses **nothing in executability** and at most
**0.026 semantic score**, all of it attributable to loop marking, while RL-first ordering costs up to
0.05 edge F1. The pipeline should order heuristically (DFS for edge fidelity) and, if RL is kept,
use it only for the loop decision.
