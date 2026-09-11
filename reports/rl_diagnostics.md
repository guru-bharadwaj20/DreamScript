# Phase 11.2.8 — convergence diagnostics: the table keeps moving, and TD error cannot say why

Produced by `python -m src.rl.diagnostics --write` (`experiments/rl/diagnostics.json` and
`diagnostics_jobs.pkl`, gitignored; figures `reports/figures/p11_convergence.png` and
`p11_qvalues.png`, redrawn with `--replot`). Tabular Q-learning at 11.2.1's cell (α 0.4,
γ 1.0), ε 1.0 → 0.05 over **120,000 episodes**, 3 seeds, trained on the 993 labelled diagrams.
Two runs of the command reproduced every number below exactly.

## The numbers behind the plots

| seed | \|TD\| tail/head | \|TD\| tail | frozen-table \|TD\| floor | tail / floor | churn, final (last third) | reward slope /1k (2nd half) | reached keys | keys for ½ the updates | keys < 10 visits | final greedy reward (993) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.805 | 0.939 | 0.707 | 1.33 | 14.97% (15.39%) | +0.0109 | 1,848 | 55 | 587 | −2.8196 |
| 1 | 0.819 | 0.944 | 1.026 | 0.92 | 14.82% (14.59%) | +0.0113 | 1,877 | 56 | 617 | −2.8616 |
| 2 | 0.789 | 0.927 | 1.103 | 0.84 | 13.49% (14.56%) | +0.0104 | 1,895 | 53 | 657 | −3.0267 |

Greedy probe on the 993 (every 10,000 episodes), share of episodes truncated at the step cap:
seed 0 `.32 .45 .46 .40 .41 .43 .34 .25 .30 .37 .24 .18`, seed 1 `.31 .42 .46 .46 .44 .46 .39 .37
.27 .16 .19 .19`, seed 2 `.33 .33 .45 .46 .42 .41 .44 .44 .40 .22 .25 .23`. Greedy full coverage
never exceeds 6.75% in any probe, against gold's 41.69%.

## What the diagnostics say

- **The table has not converged, by the measure that actually tracks the policy.** Between
  snapshots 5,000 episodes apart, **13.5–15.0% of reached keys still change their greedy action**
  at the end of training. Churn falls from ~33% to a minimum of ~11% around 50,000 episodes, then
  *rises* back to ~15% as ε decays. That matches the greedy probe, which swings by up to 0.9 reward
  between consecutive probes late in training (seed 0: −3.23 → −3.65 → −4.12 → −3.22 → −2.82). A
  table that stopped moving could not do that.
- **\|TD\| error is the wrong instrument here.** It falls to ~0.80 of its opening value, the same
  plateau 11.2.1 reported, and it would be tempting to read that plateau as "not converging". But
  the **frozen-table floor** (the final table with α = 0, played at ε 0.05, so it cannot learn) is
  **0.71–1.10**. The training tail sits at 0.84–1.33× that floor. Nearly all of the remaining TD
  error is target noise: diagrams drawn at random, ε exploration, and 11.1.6's aliasing of 25.2%
  of raw states onto shared keys. It is not evidence of unfinished learning, and nothing here suggests more
  budget would remove it. The floor itself varies by seed (0.71 vs 1.10), because a frozen greedy policy that
  loops takes many more steps. That is recorded as a limit of the measure.
- **The greedy policy learns to loop before it learns to stop looping.** Mid-training, 45% of
  greedy episodes run into the step cap, and only ~18–23% still do at the end. The *training*
  truncation curve (dotted) stays below ~7%, because exploration breaks the cycles. This is the
  mechanism behind 11.2.2's rising truncation curves, and the reason training return is a poor
  proxy for policy quality in this environment.
- **Training return rises steadily** (slope +0.010–0.011 per 1,000 episodes over the second half,
  from ≈ −4.0 to ≈ −2.8). Most of that rise is ε decaying, not the policy improving. The greedy
  probe only improves in the last ~30,000 episodes, and the final greedy reward (−2.90 mean over
  seeds) is no better than 11.2.2's 60,000-episode runs at the same cell (−2.9952 ± 0.5782). Doubling
  the budget did not help. It did not make things worse either, and 11.2.1's single-seed claim
  that greedy quality *falls* with budget is **not reproduced** at 120,000.
- **Visits are concentrated but not degenerate.** About 1,850 keys are reached (1.4% of the
  128,000 bound, consistent with 11.1.6's 1,766). Half of all ~870,000 updates land on 53–56 keys,
  and 587–657 keys are updated fewer than 10 times, so a third of the table is estimated from a
  handful of samples.

## The heatmap

`p11_qvalues.png` shows the 40 most-visited keys of seed 0, labelled by their decoded factors, with
never-updated cells in grey. The top key (44,394 visits, 5% of all updates) is an unconnected
`process` node with no successors, nothing emitted and an empty stack: the entry of a
multi-component page. Only two actions were ever taken there, emit (Q −7.23) and terminate
(−11.89). Both are deeply negative, because whatever the policy does, the unreachable-node penalty
for the other components is already locked in. This is the ceiling from 11.1.4 made visible in
the table. Weighted by visits, the greedy policy is `follow-edge-0` 32.6%, `emit-node` 30.4%,
`backtrack` 12.5% and `terminate` 8.5%. Follow slots 3 and 4 are greedy on under 1% of visits.

## Inherited-draft corrections

- **Churn took the argmax over all nine actions**, including actions never taken at a key, which
  sit at exactly 0.0 and win any row of negative values. Measured on seed 0, that argmax lands on
  an untaken action for 5.0–5.8% of reached keys. It moves the churn figure by at most 0.002, so
  this is a real bug with a negligible effect on that number, and it is fixed anyway
  (`greedy_over_taken`, tested).
- **The heatmap drew never-updated cells at 0.0** on a diverging scale, the same white as a learned
  value of zero. They are now masked grey, and the colour limit is the 98th percentile of |Q|.
- **Its `converged` flag** (TD ratio < 0.25 and churn < 5%) was an arbitrary threshold that the
  noise floor shows TD error can never meet here. It is removed rather than reported as a
  permanent `False`.
- **The committed figures appear to come from a 2,000-episode run**: the only diagnostics JSON on
  disk was one, with TD tail/head 0.424, not a convergence run. Both figures are replaced.
- The draft had no greedy probe, no noise floor, one seed, and no decoded key labels. All added.
