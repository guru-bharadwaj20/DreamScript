# Phase 11.2.3 — exploration: ε-greedy vs. softmax vs. UCB

Produced by `python -m src.rl.exploration --write` (`experiments/rl/exploration.json`,
gitignored; figure `reports/figures/p11_exploration.png`, redrawn with `--replot`). Q-learning
at 11.2.1's cell (α 0.4, γ 1.0), trained on the 993 labelled diagrams, 60,000 episodes,
**5 seeds per rule**, each rule at its own swept parameter.

## Each rule tuned on its own parameter, on reward

Sweep: 12,000 episodes, seed 0, ranked on greedy terminal reward on the 993 (never on edge F1).

| rule | swept values → greedy reward | chosen |
| :--- | :--- | :--- |
| ε-greedy (ε floor; decays 1.0 → floor) | 0.01 **−2.679** · 0.05 −2.867 · 0.1 −2.953 · 0.2 −3.830 | 0.01 |
| softmax (fixed temperature) | 0.1 −4.135 · 0.25 **−3.374** · 0.5 −3.438 · 1.0 −3.788 · 2.0 −3.973 | 0.25 |
| UCB (constant c) | 0.25 −3.080 · 0.5 −3.071 · 1.0 −3.191 · 2.0 **−2.927** · 4.0 −4.152 | 2.0 |

The ε-greedy winner sat on the edge of its grid, so the edge was checked rather than assumed:
floor 0.0 scores −3.120 and 0.005 −3.006 at the same budget, and at the full budget floor 0.0 over
5 seeds is **−2.962 ± 0.037**, worse than 0.01's −2.603. 0.01 is an interior optimum. (Floor 0.0
has the higher edge F1, 0.2825 against 0.2546 — a reminder that reward and the transfer metric
disagree, and that this study selects on reward.)

## Result (greedy evaluation, mean ± sd over 5 seeds)

| | ε-greedy (0.01) | softmax (T 0.25) | UCB (c 2.0) | gold play | random |
| :--- | ---: | ---: | ---: | ---: | ---: |
| reward, 993 labelled | **−2.6028 ± 0.1226** | −3.5728 ± 0.2026 | −2.6760 ± 0.1673 | −1.4365 | −4.2598 |
| full coverage, 993 | 2.90% | 0.00% | 1.23% | 41.69% | 0% |
| edge F1, 993 (floor 0.2033) | **0.2546** | 0.2270 | 0.2269 | 0.3648 | 0.2162 |
| greedy truncated at cap, 993 | 6.69% | **32.43%** | 7.65% | 0% | 0% |
| reward, 684 ambiguous | **−4.8456 ± 0.0761** | −5.5387 ± 0.1387 | −4.9417 ± 0.1767 | −3.8938 | −6.1541 |
| edge F1, 684 ambiguous (floor 0.1479) | **0.1944** | 0.1811 | 0.1814 | 0.2944 | 0.1532 |
| reached keys (of 128,000) | 1,512 | 1,457 | **1,678** | — | — |
| \|TD\| tail/head | 0.78 | 0.96 | 0.96 | — | — |

Greedy probe (reward on the 993, mean over seeds) at 5k / 20k / 40k / 60k episodes:
ε-greedy −4.00 / −3.52 / −3.07 / −2.60; softmax −3.85 / −3.61 / −3.65 / −3.57;
UCB **−3.11** / −2.88 / −2.69 / −2.68.

## What the measurement says

- **UCB learns fastest; ε-greedy catches it.** UCB's greedy policy is 0.89 reward ahead of
  ε-greedy's at 5,000 episodes (−3.11 ± 0.09 vs −4.00 ± 0.14) and stays ahead until ~45,000. At
  60,000 the two are a statistical tie on reward (difference 0.07, 0.8 standard errors), with
  ε-greedy ahead on edge F1 (0.2546 vs 0.2269) and coverage. UCB also visits the most of the state
  space (1,678 keys). If budget is the constraint, UCB; at this budget, ε-greedy is at least as
  good and simpler.
- **Softmax loses clearly, and its training curve hides that.** Its greedy reward is −3.57, about
  9 standard errors below ε-greedy, and its greedy policy runs into the step cap on **32% of
  diagrams**. Yet its *training* return is above ε-greedy's for the first ~35,000 episodes (left
  panel). Training return is the exploring behaviour policy's return, not the policy being
  learned — which is why the greedy probe exists, and why "reward curves" alone would have ranked
  the rules wrongly.
- **None of the three changes the 11.2.1 picture.** The best rule reaches 2.90% full coverage
  against gold's 41.69% and 0.2546 edge F1 against the 0.3648 in-action-space ceiling. Exploration
  moves greedy reward by ~1.0 across rules; the gap to gold is still 1.17.
- **One carried-forward finding: the ε floor matters more than 11.2.1 used.** Floor 0.01 (−2.60 ±
  0.12) beats 11.2.1's 0.05 at the same cell (−3.00 ± 0.58, 11.2.2's 5 seeds) and cuts seed
  spread by ~5x.

## Limits recorded, not hidden

- **The comparison is not schedule-matched.** ε-greedy anneals its exploration; softmax and UCB
  use a fixed temperature / constant, as the draft defined them. An annealed softmax was not run,
  so "softmax loses" is a result about *fixed-temperature* softmax. Their |TD| tail/head of 0.96
  (vs 0.78) is the cost of never annealing.
- UCB's counts are over 11.2.1's abstract key, which aliases 25.2% of raw states (11.1.6), so its
  bonus counts a mixture of situations. The inherited docstring predicted this would break UCB;
  **measured, it did not** — UCB ties the best rule. That prediction is removed.

## Inherited-draft corrections

- The draft claimed each rule was reported "at its own best cell" but `study` ran all three at the
  hard-coded temperature 1.0 and c 1.0 — the swept values were never used. Both measured poorly
  (−3.788 and −3.191 at 12k); the study now uses the swept winners.
- It ran softmax/UCB at α 0.2 γ 0.95 while 11.2.1 selected α 0.4 γ 1.0, trained a separate agent
  on the ambiguous set instead of evaluating the one agent on both, and reported only training
  curves. All replaced; the greedy probe was added and is checked not to perturb training (a test
  asserts a bit-identical table with and without it).
