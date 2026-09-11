# Phase 11.2.6 — curriculum: clean synthetic graphs → messy real graphs

Produced by `python -m src.rl.curriculum --write` (tabular, 5 seeds) and
`python -m src.rl.curriculum --learner dqn --write` (DQN, 3 seeds);
`experiments/rl/curriculum.json`, `experiments/rl/curriculum_dqn.json` (gitignored);
figures `reports/figures/p11_curriculum.png`, `p11_curriculum_dqn.png`. Every arm spends exactly
60,000 episodes; every arm is scored greedily on the 993 labelled diagrams with 11.2.1's `play`;
verdicts are paired over seeds and ranked on terminal reward, with a 2-standard-error band.

## The pools (measured, `--pools`)

| stage | pages | distinct structures | gold full coverage | unresolved | multi-component | max out-degree | unknown role |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| synthetic_linear | 400 | 43 | 100.00% | 0 | 0 | 2 | 16.24% |
| synthetic_structured | 1,096 | 816 | 100.00% | 0 | 0 | 3 | 15.68% |
| real_simple (1 component, ≤12 nodes) | 422 | 170 | 94.31% | 271 | 0 | 5 | 0.00% |
| real_all | 993 | 738 | 41.69% | 530 | 555 | 5 | 8.96% |

Synthetic pages come from `src.synth.graphs` (flowcharts and state machines; the clean filter
dropped 104 of 1,200 structured pages). The inherited draft's private generator was deleted:
its two synthetic stages (96 pages) held **10 distinct graphs** — the 24 nested pages were one
graph — and 31.58% of its nodes encoded as `unknown` because it used the unmapped role
`terminator`.

## Tabular (α 0.4, γ 1.0 — 11.2.1's cell), 5 seeds

| arm | reward | full coverage | edge F1 | ambiguous F1 |
| :--- | ---: | ---: | ---: | ---: |
| flat | −2.995 ± 0.578 | 5.68% | 0.2357 ± 0.0338 | 0.1653 |
| flat_restarts | −3.186 ± 0.173 | 2.38% | 0.2176 ± 0.0370 | 0.1603 |
| curriculum | −3.064 ± 0.178 | 4.41% | 0.2226 ± 0.0212 | 0.1673 |
| reversed | −3.062 ± 0.224 | 2.76% | 0.2411 ± 0.0166 | 0.1754 |
| synthetic_only | −4.046 ± 0.303 | 0.70% | 0.2193 ± 0.0239 | 0.1609 |

Paired over seeds: curriculum − flat **−0.069** (2 s.e. 0.419); curriculum − flat_restarts
+0.122 (0.228); curriculum − reversed −0.002 (0.163); synthetic_only − flat **−1.051 (0.784,
resolved)**. **No curriculum effect on the table**, in either direction, and ordering of the
stages makes no difference (curriculum ≈ reversed). Synthetic pages alone do not transfer.

## DQN (11.2.5's cell: features, lr 3e-4, γ 1.0), 3 seeds

| arm | reward | full coverage | edge F1 | ambiguous F1 |
| :--- | ---: | ---: | ---: | ---: |
| flat | −1.773 ± 0.179 | 31.02% | 0.3181 ± 0.0315 | 0.2247 |
| flat_restarts | −1.901 ± 0.123 | 24.94% | 0.3040 ± 0.0516 | 0.2141 |
| curriculum | −1.784 ± 0.052 | 29.31% | 0.3215 ± 0.0061 | 0.2310 |
| reversed | **−4.083 ± 0.183** | 0.84% | 0.2028 ± 0.0057 | 0.1454 |
| synthetic_only | −2.388 ± 0.386 | 25.08% | 0.2786 ± 0.0325 | 0.2008 |

Paired: curriculum − flat **−0.011** (2 s.e. 0.173); curriculum − flat_restarts +0.117 (0.149);
**curriculum − reversed +2.299 (0.155, resolved)**; synthetic_only − flat −0.615 (0.436,
resolved).

## Reading

1. **The curriculum does not help either learner.** Against its own control it is within noise
   for the table (−0.07 ± 0.42) and for the DQN (−0.01 ± 0.17); against the schedule-matched
   `flat_restarts` it is +0.12 for both, unresolved. The `flat_restarts` control was missing from
   the draft, which compared a four-times-restarted ε schedule with a single decay and would have
   attributed any schedule effect to the data order.
2. **Order matters for the DQN only in the destructive direction.** Ending on the easiest
   synthetic pages (`reversed`) collapses the DQN to −4.08, at the random policy's −4.26 — the
   network forgets the real pages it saw first. The table does not forget (reversed ≈
   curriculum), because a key the synthetic pages never visit keeps its value. What a curriculum
   must get right is the *last* stage, not the first.
3. **The DQN transfers from synthetic pages; the table does not.** Trained only on synthetic IR,
   the DQN reaches 25.08% full coverage and 0.2786 edge F1 on real pages it never saw — above the
   tabular agent trained on the real pages themselves (0.2357) — while the synthetic-only table
   scores 0.2193 and −4.05, barely above random. That is the first measured benefit of the
   28-feature state over the abstract key, and it is about generalisation, not aliasing.
4. **A variance note that bears on 11.2.5.** The DQN `flat` arm is 11.2.5's configuration, yet it
   scores −1.773 ± 0.179 here against 11.2.5's −1.534 ± 0.056; the only difference is that the
   three seeds ran as one ensemble, so replay mini-batches came from a different random stream.
   11.2.5's three-seed SD therefore understates run-to-run spread. Its ordering conclusions still
   hold (both sets sit far above every tabular configuration, −2.69 at best), but its point
   estimate should be read as roughly −1.5 to −1.8. The curriculum arm's much smaller spread
   (0.052 reward, 0.006 F1) is suggestive and not claimed, at n = 3.

Cost: tabular arm-seed runs 63-84 s each on CPU; DQN arms 464-520 s per 3-member ensemble, five
in parallel.
