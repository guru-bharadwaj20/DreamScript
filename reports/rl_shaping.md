# Phase 11.2.4 — reward shaping, and the exploit it has to be checked for

Produced by `python -m src.rl.shaping --write` (`experiments/rl/shaping.json`, gitignored;
figure `reports/figures/p11_shaping.png`). Q-learning at α 0.2, γ 0.99 (11.2.2's most
repeatable cell), ε 1.0 → 0.05, trained on the 993 labelled diagrams for 60,000 episodes,
**5 seeds per variant**. Every variant trains under its own shaped reward and is **evaluated
greedily under the unshaped `RewardConfig()`**, which knows nothing about the shaping.

The unshaped control reproduces 11.2.2's Q-learning run at this cell exactly (−2.7773 ± 0.0985),
so the shaping harness and 11.2.2's harness agree.

## Variants

- **Potential-based** (`γΦ(s') − Φ(s)`, Φ = 0 at the terminal, Φ ∈ [0, 4]): share of nodes
  emitted; share visited; 11.1.3's `structure_score` of the partial emission (the plan's "valid
  partial structure", taken literally).
- **Unconstrained bonuses**, +0.25 per event: per emission (bounded by n, since the mask refuses
  a second emission), per `mark-as-loop` (bounded, since a mark needs an unmarked node with a real
  back edge), per `follow-edge` *including onto visited nodes* (**unbounded**: follow, backtrack,
  and repeat up to the cap).

Verdicts use two standard errors of the difference between seed means. **Justified** means the
unshaped reward beats the control by more than that. **Exploit** means the shaped training return
rises above the control's while the unshaped reward falls below it by more than that.

## Result (993 labelled; mean over 5 seeds)

| variant | unshaped reward | Δ vs control (2 s.e.) | verdict | edge F1 | full cov. | greedy truncated | shaped train return | follows / node | revisit share of steps |
| :--- | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| none (control) | −2.7773 ± 0.0985 | — | — | 0.2422 | 2.76% | 9.29% | −3.01 | 0.49 | 5.8% |
| potential: emitted | **−2.4112 ± 0.1167** | **+0.37** (0.14) | **justified** | 0.3178 | 12.93% | 16.94% | −2.71 | 0.70 | 8.8% |
| potential: visited | −3.1152 ± 0.3212 | −0.34 (0.30) | worse, not an exploit | 0.2039 | 6.16% | 30.86% | −3.61 | 1.04 | 15.0% |
| potential: structure | −2.5490 ± 0.1437 | +0.23 (0.16) | justified | 0.2658 | **13.35%** | 15.03% | −2.69 | 0.65 | 8.1% |
| bonus: emit | −2.4216 ± 0.0985 | +0.36 (0.12) | justified | **0.3241** | 3.65% | 9.47% | −2.00 | 0.47 | 4.9% |
| **bonus: visit** | **−4.5263 ± 0.0566** | **−1.75** (0.10) | **EXPLOIT** | 0.1955 | 1.96% | **43.32%** | **−1.85** | **1.44** | **25.8%** |
| bonus: loop | −2.8755 ± 0.1823 | −0.10 (0.19) | no effect | 0.2404 | 2.38% | 13.46% | −2.86 | 0.56 | 7.3% |
| gold play | −1.4365 | | | 0.3648 | 41.69% | 0% | | 0.40 | 0% |
| random | −4.2598 | | | 0.2162 | 0% | 0% | | | |

Ambiguous set (684), unshaped reward / edge F1: control −5.0276 / 0.1825; emitted potential
−4.8835 / 0.2407; structure potential −5.0735 / 0.1848; emit bonus **−4.7873 / 0.2389**; visit bonus
−6.4367 / 0.1479 (exactly the empty-policy floor); loop bonus −5.1631 / 0.1826. Gold play:
−3.8938 / 0.2944.

## What the measurement says

**The exploit is real and it is the visit bonus.** Under its own reward it is the best learner in
the study: shaped training return −1.85, 1.16 above the control. Under the real reward it is the
worst: −4.53, **below a uniform random policy (−4.26)**. The behaviour shows what it was paid for:
1.44 follows per node (gold 0.40), 25.8% of steps onto already-visited nodes (gold 0%), episodes
twice as long (23.0 steps against 12.3), 43% of greedy episodes truncated at the cap, and 8.5% of
nodes emitted against the control's 25.8%. On the ambiguous set its edge F1 is 0.1479, exactly the
score of a policy that emits nothing.

**"Valid partial structure" shaping is justified, and the simpler emitted-share potential more
so.** The structure potential gains +0.23 unshaped reward (2 s.e. 0.16) and the most full coverage
(13.35% vs 2.76%). The emitted-share potential gains **+0.37** (2 s.e. 0.14), 12.93% coverage, and
edge F1 0.3178. That edge F1 was not used to select anything, and it closes about two thirds of the
control's gap to gold play in the action space (0.2422 → 0.3178, against gold's 0.3648). Neither
is free: both roughly double greedy cap truncation (15–17% vs 9.3%), so they trade loop penalties
for coverage and come out ahead. On the ambiguous set the emitted potential is ahead on edge F1
(0.2407 vs 0.1825) but its reward gain (+0.14) sits inside the seed noise.

**Policy invariance did not hold in practice, and the row says so.** Potential-based shaping leaves
the *optimal* policy unchanged in the raw-state MDP. What this learner produced moved in both
directions: +0.37 for "emitted" and −0.34 for "visited", with visited-potential truncation at 31%.
There are two reasons, and this study does not separate them. Invariance says nothing about what a
finite, ε-greedy, constant-α run converges to. And Φ is defined on raw states while the Q-table
indexes 11.1.6's abstract key, which aliases 25.2% of raw states, so the shaping term is not a
function of the learner's state at all. Potential-based shaping here is a design choice to be
measured, not a guarantee.

**An unconstrained bonus can be safe when the mask bounds it.** The emit bonus ties the emitted
potential on the 993 (−2.42 vs −2.41), keeps greedy truncation at the control's level (9.47%), and
is the best variant on the ambiguous set (−4.79, +0.24 against a 2 s.e. band of 0.16). The loop
bonus doubles marks per node (0.075 vs 0.036). That is the most farming the mask allows, and it
has no measurable effect on reward.

**Recommendation carried forward**: the emitted-share potential, SCALE 4. It has the largest
justified gain, a principled form, and no exploit signature. The emit bonus is an equal
performer on this corpus without that form. The visit bonus must not be used.

## The first run, kept rather than discarded

A first run at 11.2.1's cell (α 0.4, γ 1.0, 3 seeds) gave a control sd of **0.72** (11.2.2 measured
0.58 over 5 seeds there), which no effect in this study could clear. Its directions agree with the
table above: emit bonus +0.49, visit bonus −1.62 with shaped return +0.79 (flagged as an exploit
even under that noise), and potentials within the noise. The study was re-run at the stable cell
with 5 seeds, and the earlier JSON is kept as `experiments/rl/shaping_a0.4_g1.0_3seeds.json`.

## Inherited-draft corrections

- **It had the exploit backwards.** It called `loop_bonus` "the exploit" and `visit_bonus` "also
  bounded". The mask bounds marks (measured: no reward effect), and `Outcome.moved_to` is set on
  every follow including revisits. The visit bonus is the farm, now pinned by a test that collects
  it past the cap on a 3-node line.
- **Its exploit test could not detect this exploit.** It required marks-per-node above twice gold's
  *and* no reward gain, so a variant that farms follows could never be named. It is replaced by the
  shaped-return-up / unshaped-reward-down test with a standard-error band.
- The telescoping test re-implemented the sum instead of checking the loop. A test now asserts that
  the training loop itself pays exactly −Φ(s₀) per episode for every potential.
- The draft's "a variant that farms an action is farming something worth having" and the
  `test_the_loop_bonus_is_inert_on_an_acyclic_diagram` framing ("the exploit cannot fire") are
  removed. Neither describes the measured behaviour.
