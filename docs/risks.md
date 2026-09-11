# Risk Register

Phase 0.3.4. What can go wrong in DreamScript, how likely it is, what it costs, and what is
actually being done about it. Every mitigation names the phase task that implements it, so a
risk cannot be "handled" by intention alone.

**Likelihood / Impact:** H = high, M = medium, L = low.
**Status:** ❌ mitigation not built yet, ✅ mitigation in place.

---

## 1. Data risks

| # | Risk | L | I | Mitigation | Owner task | Status |
| :---: | :--- | :---: | :---: | :--- | :--- | :---: |
| D1 | **Not enough labelled data.** 260 self-drawn diagrams cannot train a detector alone. | H | H | Combine 5 public datasets; generate ≥5K synthetic diagrams and "hand-ify" them; heavy augmentation | 1.1, 1.3.6, 1.3.7 | ❌ |
| D2 | **A public dataset's license forbids redistribution**, blocking the demo or the report. | M | M | License audit before any dataset enters training; raw files stay out of git; only derived metrics published | 1.1.6, 0.3.1 | ✅ (policy) |
| D3 | **Style monoculture** — a corpus drawn mostly by one person inflates every score. | H | H | ≥8 scribes required; scribe-disjoint splits; grouped CV; leave-one-scribe-out evaluation | 1.2.6, 1.3.3, 5.2.2, 14.6 | ❌ |
| D4 | **Train/test leakage** through near-duplicate images or per-image splits. | M | H | pHash deduplication; splits by person, asserted in code | 1.3.2, 1.3.3 | ❌ |
| D5 | **Annotation disagreement** on messy cases makes labels unreliable. | M | M | Written annotation guide; double-label 50 images; report Cohen's κ and fix the guide if κ < 0.75 | 2.2.3, 2.2.4 | ❌ |
| D6 | **Dataset conversion loses structure** when mapping hdBPMN/DIDI into the IR. | M | M | Round-trip test on every converter; document known losses in each data card | 2.2.2, 0.3.1 | ❌ |

## 2. Vision risks

| # | Risk | L | I | Mitigation | Owner task | Status |
| :---: | :--- | :---: | :---: | :--- | :--- | :---: |
| V1 | **Bad lighting** — shadows and glare destroy binarization, and everything downstream fails. | H | H | CLAHE + morphological background estimation; Sauvola over Otsu; ≥25% of the corpus captured adversely on purpose | 3.1.4, 3.1.5, 1.2.7 | ❌ |
| V2 | **Perspective and skew** from hand-held photos. | H | M | Page-quadrilateral detection and homography rectification; Hough deskew | 3.1.2, 3.1.3, 3.1.7 | ❌ |
| V3 | **Ruled or gridded paper** produces false lines that look like edges. | M | H | Directional morphological suppression of long straight rules before primitive extraction | 3.1.8 | ❌ |
| V4 | **Arrowheads are tiny** and get missed at normal detector resolution. | H | H | Higher input resolution plus tiling; arrowhead-specific AP tracked separately, not hidden in mAP | 9.1.5, 9.1.4 | ❌ |
| V5 | **Occlusion** — coffee stains, fingers, torn corners. | M | M | Ensemble classifier evaluated under simulated occlusion; accuracy-vs-occlusion curve reported rather than assumed | 7.1.6, 14.5 | ❌ |
| V6 | **Crossed-out elements** get parsed as real nodes, producing code for something the user deleted. | M | H | Dedicated struck-through classifier; excluded from the IR; hard negatives mined into detector training | 9.3.8, 9.1.7 | ❌ |

## 3. Parsing risks

| # | Risk | L | I | Mitigation | Owner task | Status |
| :---: | :--- | :---: | :---: | :--- | :--- | :---: |
| P1 | **Arrows that do not touch their boxes** — the defining messiness of real sketches. | H | H | Gap bridging with direction continuity; endpoint snapping to node ports; HMM posteriors repair what geometry cannot | 10.1.4, 7.3.10 | ❌ |
| P2 | **Crossing lines read as junctions**, inventing edges that do not exist. | M | H | Curvature-continuity disambiguation at intersections, with explicit test cases | 10.1.7 | ❌ |
| P3 | **Ambiguous arrow direction** flips the logic of the generated code. | M | H | Arrowhead evidence + flow-direction prior + HMM role constraints; direction accuracy tracked as its own metric | 10.1.5 | ❌ |
| P4 | **Everyone draws rectangles differently**, so template matching fails. | H | M | GMM+EM learns the shape vocabulary from data; per-scribe adaptation; soft responsibilities instead of hard labels | 7.4, 7.4.6, 7.4.7 | ❌ |
| P5 | **Illegible handwriting** yields garbage identifiers in the output code. | H | M | Lexicon-biased decoding; per-style OCR adaptation; low-confidence labels flagged for correction rather than silently used | 9.3.4, 9.3.5, 9.3.7 | ❌ |
| P6 | **Flowchart vs. state diagram confusion** — structurally similar, semantically different. | H | M | Polynomial-kernel SVM targeted at exactly this boundary; connectivity features (self-loops) separate them | 6.3.2, 4.1.7 | ❌ |

## 4. Generation risks

| # | Risk | L | I | Mitigation | Owner task | Status |
| :---: | :--- | :---: | :---: | :--- | :--- | :---: |
| G1 | **The LLM hallucinates nodes** that were never drawn, or drops ones that were. | M | H | Structural fidelity metric comparing generated AST against the IR; hallucination rate reported explicitly | 12.3.4, 12.3.8 | ❌ |
| G2 | **Generated code does not run.** | M | H | Executability and pass@1 measured in a sandbox; one automated repair attempt on failure | 12.3.2, 12.3.3, 12.3.9 | ❌ |
| G3 | **Generated code is executed** — an obvious remote-code-execution hazard in the demo. | M | H | Sandboxed subprocess with timeout and memory cap; containerized runner in the web app; never executed on the host directly | 11.2.9, 16.1.3 | ❌ |
| G4 | **QLoRA does not fit in 24 GB.** | M | H | Documented escalation path: grad checkpointing → shorter sequences → 8-bit paged optimizer → lower rank → 3B fallback | `docs/hardware.md`, 12.2.2 | ✅ (planned) |
| G5 | **Fine-tuning does not beat few-shot prompting**, undermining the whole Unit 4 claim. | M | M | Zero-shot / few-shot / LoRA compared head to head; if fine-tuning loses, that is reported as a finding, not hidden | 12.2.7 | ❌ |

## 5. Systemic and project risks

| # | Risk | L | I | Mitigation | Owner task | Status |
| :---: | :--- | :---: | :---: | :--- | :--- | :---: |
| S1 | **Error propagation** — a stage-1 mistake silently ruins the output, and the pipeline reports success. | H | H | Stage-wise metrics; error propagation study; confidence gating that asks the user instead of guessing | 14.2, 14.3, 13.5 | ❌ |
| S2 | **Unseen diagram type** photographed at demo time; the classifier confidently picks a wrong class. | M | M | Out-of-distribution behaviour tested and documented; DBSCAN outlier detection surfaces novel shapes | 14.7, 8.5 | ❌ |
| S3 | **The GPU is shared** — another process already holds ~20 GB on this machine. | H | H | Check *free* VRAM before Phase 9 and 12 runs, not total; escalation path in `docs/hardware.md` | 0.1.7 | ✅ (documented) |
| S4 | **Live demo fails** in front of an audience. | M | H | Cached-result fallback mode; preloaded example gallery; recorded demo video | 16.2.10, 16.2.9, 17.9 | ❌ |
| S5 | **Irreproducible numbers** in the report. | M | H | Determinism harness; immutable run directories capturing config, env and git commit; ≥3 seeds with spread | 0.1.6, 0.2.4, 14.10 | ✅ |
| S6 | **Scope overrun** — 18 phases, one machine. | H | M | Each task has a written Definition of Done; nothing is marked ✅ without an artifact and a passing check | contributing.md | ✅ |
| S7 | **Latency too high for the "seconds later" promise.** | M | M | Per-stage timing table and a < 10 s budget; caching; NB fast prior as a cheap first pass | 13.7, 13.6, 7.2.6 | ❌ |

## 6. Ethical and privacy risks

| # | Risk | L | I | Mitigation | Owner task | Status |
| :---: | :--- | :---: | :---: | :--- | :--- | :---: |
| E1 | **Handwriting is identifiable** and the corpus contains contributors' writing. | M | M | Raw photos stay local and out of published artifacts; contributors informed and able to withdraw | `docs/data_cards/dreamscript_chaos.md` | ✅ (policy) |
| E2 | **Personal content in sketches** (names, credentials). | L | M | Contributors instructed not to draw personal data; page cropping removes surroundings | 1.2, 3.1.2 | ✅ (policy) |
| E3 | **Silent wrong code** trusted by a user because the pipeline looked confident. | M | H | Confidences surfaced in the UI; low-confidence items flagged; the IR graph shown next to the code so the user can check the interpretation | 16.2.8, 16.2.4 | ❌ |
| E4 | **English-only labels** make the system unusable for other scripts, unstated. | H | L | Limitation stated explicitly in the data card and the report's limitations section | 17.7 | ✅ (documented) |

---

## Top five to watch

1. **P1 — broken arrows.** The single feature that separates this project from clean-wireframe
   tools. If gap bridging does not work, the demo does not work.
2. **D3 — style monoculture.** The cheapest way to produce impressive, meaningless numbers.
   Grouped CV is non-negotiable.
3. **S1 — error propagation.** Five stages at 90% each is 59% end to end. Stage-wise metrics
   must be reported next to the end-to-end number.
4. **V1 — lighting.** The demo is a photograph taken in an unknown room.
5. **G3 — executing generated code.** The one risk here that can damage the host machine
   rather than just the results.

## Review cadence

Reviewed at the end of every phase. A risk moves to ✅ only when its owner task is ✅ and the
evidence is recorded in `reports/`. New risks discovered during a phase are added here in the
same commit that discovers them.
