# Viva preparation

Phase 17.10. Likely questions, with the answer to give, the derivation where there is one, and
the number from this project that goes with it.

## HMM — semantic roles (Phase 7.3)

**Q. What are the states and observations?**
Hidden states are node roles (`start`, `process`, `decision`, `terminal`, …). Observations are
the drawn evidence for each node, mainly the detected shape plus text features, taken in reading
order along the diagram.

**Q. Write the model.**
λ = (π, A, B) with πᵢ = P(q₁ = i), aᵢⱼ = P(qₜ₊₁ = j | qₜ = i), bⱼ(o) = P(oₜ = o | qₜ = j).
The joint probability is P(O, Q | λ) = π_{q₁} b_{q₁}(o₁) ∏ₜ a_{qₜ₋₁qₜ} b_{qₜ}(oₜ).

**Q. Viterbi.**
δ₁(i) = πᵢ bᵢ(o₁). Then δₜ(j) = maxᵢ [δₜ₋₁(i) aᵢⱼ] · bⱼ(oₜ), with backpointer ψₜ(j) = argmaxᵢ δₜ₋₁(i) aᵢⱼ.
Terminate with q*_T = argmax δ_T and backtrack. Cost O(T·N²). It is computed in log space to avoid
underflow (`src/parse/viterbi.py`).

**Q. Forward-backward and Baum-Welch.**
αₜ(i) = P(o₁..oₜ, qₜ = i). βₜ(i) = P(oₜ₊₁..o_T | qₜ = i). γₜ(i) ∝ αₜ(i)βₜ(i), and
ξₜ(i,j) ∝ αₜ(i) aᵢⱼ bⱼ(oₜ₊₁) βₜ₊₁(j). The M-step is âᵢⱼ = Σξₜ(i,j) / Σγₜ(i), and b̂ⱼ(k) is γ
summed over the steps where oₜ = k. EM never decreases the likelihood, but it only reaches a local
optimum (`p7_baumwelch.png`).

**Q. Why "per-component" Viterbi?**
A page can contain several disconnected graphs. Decoding each connected component as its own
sequence stops one component's end from conditioning the next component's start.

**Numbers.** S4 macro F1 **0.8003** against a target of 0.80, with seed std 0.0010. That is *met,
not beaten*. Ablation: replacing the HMM with the majority role per shape falls to 0.3228, so the
sequence model is worth +0.45.

## SVM kernels (Phase 6.3)

**Q. The primal and the dual.**
Primal: min ½‖w‖² + C Σξᵢ subject to yᵢ(wᵀxᵢ + b) ≥ 1 − ξᵢ, ξᵢ ≥ 0.
Dual: max Σαᵢ − ½ ΣΣ αᵢαⱼ yᵢyⱼ K(xᵢ, xⱼ) subject to 0 ≤ αᵢ ≤ C and Σαᵢyᵢ = 0. Only the support
vectors have αᵢ > 0 (`p6_support_vectors.png`).

**Q. Kernels.**
Polynomial: K(x, z) = (γ xᵀz + c₀)^d. Its feature space holds all monomials up to degree d:
c₀ trades lower-order against higher-order terms, and d sets the maximum interaction order.
RBF: K = exp(−γ‖x − z‖²). Large γ means small bumps, which gives low bias and high variance.

**Q. Kernel trick, in one line.**
The dual touches the data only through inner products, so replacing xᵀz with K(x, z) trains a
linear classifier in the feature space φ without ever computing φ. Mercer's condition (K is
positive semi-definite) is what makes this valid.

**Q. Multiclass.**
One-vs-one trains k(k−1)/2 binary SVMs and takes a vote. With 5 types that is 10 classifiers.

**Numbers.** Flowchart vs state machine is linearly separable: every cell, and the linear SVM,
score 1.0000. On circuit vs flowchart the polynomial kernel gains +0.0438. The shipped S1 model
is CLIP ViT-B/32 → PCA 128 → one-vs-one RBF SVM at **0.9871**.

## CNN parameters (Phase 9, `docs/cnn_math.md`)

**Q. Parameters of a conv layer.** (k_h·k_w·C_in + 1)·C_out with a bias, or k_h·k_w·C_in·C_out
without one. Our convolutions are followed by batch norm, so they have **no bias**: the BN shift
would make it redundant. Batch norm adds 2·C (γ, β).

**Q. Walk me through the scratch CNN.**

| Layer | Params |
| :--- | ---: |
| conv 1→16, 3×3 | 1·9·16 = **144** |
| conv 16→16 | 16·9·16 = 2,304 |
| conv 16→32 | 4,608 |
| conv 32→32 | 9,216 |
| conv 32→64 | 18,432 |
| conv 64→128 | 73,728 |
| 6 × batch norm | 32+32+64+64+128+256 = 576 |
| fc 2048→128 | 2048·128 + 128 = **262,272** |
| fc 128→7 | 128·7 + 7 = 903 |
| **total** | **372,183** |

The input is 64×64. After four 2×2 pools it is 4×4, and 128·4·4 = 2048 feeds the first FC layer.
**70% of the parameters are in that one FC layer.** That is the standard argument for global
average pooling.

**Q. Output size and receptive field.**
out = ⌊(n + 2p − k)/s⌋ + 1. The receptive field grows as r_l = r_{l−1} + (k − 1)·j_{l−1}, where the
jump j is the product of the strides. It is 64 px at the output, the whole crop.

**Q. Why is the arrowhead the weak class for YOLO?**
91% of arrowheads fit in a single P4 cell (16 px stride), so they are represented on one feature
level while other classes use all three. Resolution helps them most: arrowhead AP went
0.22 → 0.31 → 0.38 from 640 to 1280 px. Detector S2 mAP@0.5 is **0.9107**, with arrowhead AP 0.43.

## Q-learning (Phase 11)

**Q. The update.**
Q(s, a) ← Q(s, a) + α [ r + γ · max_{a′ ∈ legal(s′)} Q(s′, a′) − Q(s, a) ].
The bracketed term is the TD error. Ours uses α = 0.2 and γ = 0.95, with ε-greedy exploration decaying
geometrically from 1.0 to 0.05 (`src/rl/qlearning.py`).

**Q. Off-policy vs SARSA.**
Q-learning bootstraps from the greedy action (max), so it learns the optimal policy while acting
ε-greedily. SARSA uses the action actually taken, Q(s′, a′), and learns the value of the policy it
follows, which is safer near penalties (`p11_sarsa.png`).

**Q. Convergence.**
Tabular Q-learning converges to Q* if every (s, a) is visited infinitely often and Σα = ∞,
Σα² < ∞ (Watkins and Dayan).

**Q. Reward shaping.**
We use potential-based shaping, F = γΦ(s′) − Φ(s). It provably preserves the optimal policy
(Ng et al., 1999).

**Q. What is the MDP here?**
The state is an abstract summary of the partially traversed graph. An action chooses the next node
to emit. The reward is terminal, from the emitted program's semantic score and sandbox result.

**Q. So is RL useful here? (expect this one.)**
It was implemented and evaluated honestly, and it is not load-bearing. Over a DFS ordering it adds
**+0.026** semantic score, all from loop marks. The learned order costs 0.024–0.050 edge F1, and
executability is unchanged. The pipeline ships DFS, and the ablation table says so.

## LoRA rank (Phase 12)

**Q. What is LoRA?**
Freeze W₀ ∈ ℝ^{d×k} and learn ΔW = BA with B ∈ ℝ^{d×r}, A ∈ ℝ^{r×k}, r ≪ min(d, k). The forward
pass is h = W₀x + (α/r)·BAx. B is initialised to zero, so training starts from the base model.
Trainable parameters per matrix: r(d + k) instead of dk.

**Q. What does rank control?**
The dimension of the update's subspace, which is its capacity. α/r scales the update, so with
α = 2r (our setting) the effective step size stays comparable across ranks.

**Q. QLoRA?**
The frozen base is stored in 4-bit NormalFloat (NF4) with double quantisation. Compute is in bf16,
and the adapters are full precision. A 7B model then takes **5.60 GB** on the GPU. At r = 16 on all
seven projections the adapter has **40.37 M** trainable parameters.

**Q. How did you pick r?**
A sweep with equal token budgets (1.2 M tokens, same seed and rows) gave best validation loss:
r = 8: 0.0862, r = 16: 0.0681, r = 32: 0.0527, **r = 64: 0.0429**, all at about 22 min each, with peak
memory 14.6 → 16.7 GB. Attention-only r = 16 did worst (0.1309), so the MLP projections matter.
r = 64 was selected.

**Numbers.** Fine-tuned: functional pass@1 **70.37%** (S7) and executes **98.8%** (S6). The same
model zero-shot scores 3.09% and few-shot 39.5%. Both are measured from the IR, not from a photo.

## Questions about the whole system

- **"What's your end-to-end accuracy?"** 19.1% functional pass from a photograph, 78.4% from a
  correct graph. The error-propagation study puts 0.56 of the loss on graph structure, not OCR.
- **"What's the weakest part?"** Graph assembly: S5 median GED 13 on test against a target of 3.
- **"What happens on a diagram type it hasn't seen?"** It fails badly. 21 of 24 pages produced
  confident, wrong code. The fix is an out-of-distribution check before the router (report §8).
- **"Why a web app and not an APK?"** It has to run on iOS, and nothing in the toolchain builds
  Android binaries. Installed, a web app opens full screen. The rationale is in the Phase 16
  preamble in `contributing.md`.
- **"Is S1 real?"** It is real but inflated. Two sources are single-class, so source predicts label
  for 1,200 of 1,340 pages. Refitting the PCA inside each fold costs 0.0053.
