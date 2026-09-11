# Phase 11.2.2 — SARSA against Q-learning: on-policy caution is real, and it does not pay

Produced by `python -m src.rl.sarsa --write` (`experiments/rl/sarsa.json`, gitignored; figure
`reports/figures/p11_sarsa.png`, redrawn with `--replot`). Both algorithms share
`src.rl.qlearning.train` and differ only in the bootstrap (legal max vs. the action actually
chosen next); a hand-worked single-transition test pins that difference. Trained on all 993
labelled diagrams, ε 1.0 → 0.05 geometric, 60,000 episodes, **5 seeds per configuration**;
evaluated greedily and at the ε floor ("online") on the 993 and on 11.2.7's 684 ambiguous
diagrams.

## The hypothesis, fixed before the run

The cliff in this environment is the step cap: truncation at `4n + 8` costs −2.0, and a policy
can reach it by cycling follow/backtrack without emitting or terminating. On-policy SARSA prices
its own exploration, so the prediction was: **SARSA truncates less, during training and in play,
and gains most when played with its exploration on.**

## Tuning each on its own terms, then controlling for the tune

The 16-cell α × γ sweep (12,000 episodes, seed 0, ranked on greedy terminal reward on the 993)
picks **α 0.4, γ 1.0 for Q-learning** (−2.8672, 11.2.1's cell reproduced) and **α 0.2, γ 0.99 for
SARSA** (−3.1195). Because the cells differ, both algorithms were then re-run at *both* cells.

Greedy play, mean ± sd over 5 seeds:

| configuration | algo | reward (993) | truncated (993) | full cov. (993) | edge F1 (993) | reward (684 amb.) | edge F1 (684 amb.) | train truncation, last 10% |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| α 0.4 γ 1.0 | Q | −2.9952 ± 0.5782 | 16.01% | 5.68% | 0.2357 | −5.3793 ± 0.5055 | 0.1653 | 4.89% ± 0.21 |
| α 0.4 γ 1.0 | SARSA | −3.0304 ± 0.1062 | 10.17% | 1.96% | 0.2302 | −5.2658 ± 0.1086 | 0.1698 | **1.90% ± 0.41** |
| α 0.2 γ 0.99 | Q | **−2.7773 ± 0.0985** | 9.29% | 2.76% | **0.2422** | **−5.0276 ± 0.1571** | **0.1825** | 2.44% ± 0.18 |
| α 0.2 γ 0.99 | SARSA | −2.9836 ± 0.0800 | 9.51% | 1.23% | 0.2254 | −5.1841 ± 0.0907 | 0.1575 | **1.79% ± 0.20** |
| references | gold play | −1.4365 | 0% | 41.69% | 0.3648 | −3.8938 | 0.2944 | — |
| | random | −4.2598 | 0% | 0% | 0.2162 | −6.1541 | 0.1532 | — |
| | empty-policy floor | — | — | — | 0.2033 | — | 0.1479 | — |

Online (ε 0.05) reward on the 993: at α 0.4 γ 1.0, Q −2.9006 ± 0.3455 vs SARSA −2.9799 ± 0.0659;
at α 0.2 γ 0.99, Q −2.7904 ± 0.0600 vs SARSA −2.9725 ± 0.0408.

## What the measurement says

**The cautious behaviour is real and survives the cell control.** In training, SARSA runs into the
cap less at both shared cells — 1.90% vs 4.89% and 1.79% vs 2.44% of episodes in the last 10%,
with non-overlapping seed ranges in both — and its greedy policy truncates less at α 0.4 γ 1.0
(10.17% vs 16.01%; online 3.06% vs 5.66%). SARSA is also far more *repeatable*: its greedy reward
sd is 0.08–0.11 against Q-learning's 0.58 at the same α 0.4 γ 1.0 cell. The truncation curve in
the figure shows both learners hitting the cap more as ε decays — the greedy policy is learning to
loop — and SARSA doing so later and less.

**It does not buy reward, and the textbook signature does not appear.** At a shared cell SARSA
never beats Q-learning: a tie at α 0.4 γ 1.0 (−3.03 vs −3.00, a difference 0.13 standard errors
wide) and a **loss by 0.21 at α 0.2 γ 0.99** (−2.98 vs −2.78, about 3.6 standard errors), where
the greedy truncation advantage also vanishes (9.51% vs 9.29%). Played *with* exploration on,
SARSA does not close that gap either (−2.97 vs −2.79), so the cliff-walking prediction — on-policy
wins online — is **not supported**. Q-learning also reaches full coverage more often at every cell.
On the ambiguous set the ordering is the same: best is Q-learning at α 0.2 γ 0.99, −5.03 reward
and 0.1825 edge F1, still 1.13 reward and 0.11 F1 short of gold play in the action space.

The likely reason, stated as a reading and not a proof: the cap penalty is −2.0 while an
unreachable node costs −0.5 each, so avoiding truncation by terminating early trades one penalty
for several. SARSA's caution is priced correctly by its own value function and still is not the
better trade under this reward.

## Findings that correct earlier numbers

- **11.2.1's selected cell was a single-seed choice, and it does not hold up over seeds.** Seed 0
  at α 0.4 γ 1.0 reproduces 11.2.1 exactly (−2.5545, edge F1 0.2729), but the 5-seed mean there is
  **−2.9952 ± 0.5782** and edge F1 0.2357 ± 0.0338 — seed 0 is the best of five. At α 0.2 γ 0.99,
  Q-learning is better on the mean (−2.7773) and six times more repeatable. 11.2.1's own table is
  not wrong for seed 0; it is not representative of the cell.
- **The inherited draft's figure put reward (≈ −3) and three shares (0–1) on one bar axis**, ran
  both algorithms at α 0.2 γ 0.95 — neither's tuned cell — and compared a single seed-set without
  controlling for the cell. All three are replaced.

## Rejected

- **Ranking the sweep on edge F1**: the transfer metric; selecting on it would be selecting on the
  test.
- **Reporting "each at its own best cell" as the algorithm comparison**: the cells differ, and the
  control shows the cell moves Q-learning's reward by 0.22 — as much as the algorithm does.
- **A separate SARSA loop**: every other difference between two loops would be a candidate
  explanation.
