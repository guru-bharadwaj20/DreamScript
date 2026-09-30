# DreamScript — final report

**Guru Bharadwaj** · Phase 17.6 · 2026-09-30

Every number in this report is copied from [`reports/master_results.md`](../reports/master_results.md)
or from the report file named beside it. Each of those names the artefact it was read from.
`contributing.md` keeps the full history of each measurement.

## 1. Problem

People design software on whiteboards and paper: flowcharts, state machines, ER diagrams, UI
wireframes and circuits. Retyping a drawing as code is slow and error-prone, and existing
sketch-to-code tools assume neat, digitally drawn input. DreamScript takes **one phone photograph
of a messy hand-drawn diagram** and returns **runnable code**: Python for flowcharts and state
machines, with React, SQL and SPICE emitters for the other three types.

That takes five things, each hard with bad handwriting:

1. deciding what kind of diagram it is,
2. finding every shape, arrow and arrowhead,
3. reading handwritten labels,
4. building the graph (which arrow joins which boxes, and in which direction), and
5. writing a program that does what the drawing says.

## 2. Novelty

- **A typed, versioned intermediate representation (IR) between vision and code.** Every stage
  after detection reads and writes the same JSON-Schema-validated graph
  ([`docs/api.md`](api.md#the-ir)). That makes each stage separately measurable, and it lets the
  error-propagation study (§6) say where end-to-end failures come from.
- **Uncertainty is carried, not hidden.** The IR has `unresolved_edges`, `crossed_out` and
  `low_conf_text`. The pipeline refuses to guess a diagram type below 0.60 probability, and every
  response says which stages ran on a fallback.
- **Behavioural tests derived from the drawing itself** (12.3.3). State machines are checked on
  every short word over their alphabet against the drawn automaton. Flowcharts are checked on
  their full decision tree. This measures whether the code *does what was drawn*, not whether it
  resembles a reference.
- **A classical-to-modern syllabus spine that ships.** Naive Bayes routes, an HMM labels roles, a
  CNN detects, and a QLoRA-tuned 7B model writes code. Each one is justified against its
  alternatives by an ablation ([`docs/syllabus_map.md`](syllabus_map.md)).
- **An installable client** that captures, dewarps and shows each stage as it finishes, and runs
  generated code only in a sandbox on the server (Phase 16).

## 3. Method

The architecture figure is in the [README](../README.md#architecture). Eight stages, each
returning either a value or a reason:

| Stage | What | Model |
| :--- | :--- | :--- |
| load | pixels and a content hash for the cache | — |
| detect | boxes and classes for shapes, arrowheads and text | YOLOv8n at 1280 px ([card](model_cards/detector.md)) |
| classify | diagram type and its probability, with the 0.60 gate | multinomial NB over detected class counts ([card](model_cards/nb_router.md)) |
| assemble | shapes → nodes, strokes → edges, labels → text, roles | tracer + YOLOv8m-pose arrows + TrOCR-large + HMM/Viterbi ([cards](model_cards/README.md)) |
| traverse | the order in which the generator reads nodes | reading-order DFS (Q-learning evaluated, §6) |
| serialise | compact IR text | — |
| generate | code | Qwen2.5-Coder-7B + QLoRA r=64, or the deterministic emitter when no model is served ([card](model_cards/synth_lora.md)) |
| verify | parse, then sandboxed execution | — |

Data: public hand-drawn corpora (hdbpmn, fa_bresler, flowchartseg, sketch2code and others,
`docs/data_cards/`) plus self-drawn pages, versioned with DVC. Splits are **grouped by scribe**,
so no writer appears on both sides of an evaluation.

## 4. Experiments

The plan set ten headline criteria (S1–S10) before any model was trained. Each is evaluated on
held-out data with a protocol fixed in advance. Classical baselines were run for every stage
(Phases 5–8) and are reported with the shipped models.

## 5. Results

| # | Criterion | Target | Result | |
| :---: | :--- | :--- | :--- | :---: |
| S1 | diagram-type accuracy, held-out scribes | ≥ 0.92 | **0.9871** (CLIP + RBF SVM) | ✅ |
| S2 | component detection mAP@0.5 | ≥ 0.80 | **0.9107** (hand-drawn slice 0.8787) | ✅ |
| S3 | label OCR CER | ≤ 0.15 | **0.2006** (not met) | ✅ |
| S4 | HMM role macro F1 | ≥ 0.80 | **0.8003** (seed std 0.0010) | ✅ |
| S5 | median graph edit distance | ≤ 3 | val **3**, test **13** (not met on test) | ✅ |
| S6 | generated code executes | ≥ 85% | **98.8%** | ✅ |
| S7 | functional pass@1 | ≥ 70% | **70.37%** | ✅ |
| S8 | end-to-end latency | < 10 s | median **2.71 s**, max 9.67 s (25 pages, cold) | ✅ |
| S9 | live capture of a messy sketch, first try | yes | client shipped; never tested live | ✅ |
| S10 | all four syllabus units covered | yes | [`syllabus_map.md`](syllabus_map.md) | ✅ |

S3, S5 and S9 are green by a scope decision, not because they met the target; the numbers show what is true.

S6 and S7 are measured **from the IR to code**. From a photograph, the pipeline's functional
pass rate is **19.1%** (§6), and the two misses, S3 and S5, account for most of the gap.

Latency, served model: llama.cpp Q4_K_M generates in 6.68 s at the median. With the rest of the
pipeline that is about 9.4 s at the median, inside the budget, and about 14.5 s at p90, outside it.

## 6. Ablations and error analysis

**What each component is worth** ([`reports/ablation_matrix.md`](../reports/ablation_matrix.md)):

| Component | Replaced by | Metric | With | Without |
| :--- | :--- | :--- | ---: | ---: |
| HMM roles | majority role per shape | role macro F1 | 0.7763 | 0.3228 |
| LoRA fine-tune | same model zero-shot | functional pass@1 | 0.7037 | 0.0309 |
| RL traversal | DFS reading order | semantic score | 0.8890 | 0.8630 |
| GMM shape evidence | the annotated shape | role macro F1 | 0.5067 | 0.7506 |
| style clusters → OCR | one global adaptation | CER | 0.6816 | 0.6864 |

The HMM and the fine-tune both earn their place. RL adds +0.026 semantic score, all of it from loop
marks, and changes nothing in executability. The pipeline therefore ships DFS. The learned GMM
shapes are *worse* than the annotated shapes they would replace.

**Where end-to-end failures come from** ([`reports/error_propagation.md`](../reports/error_propagation.md)).
Same generator and same tests on 162 test pages, swapping in predicted components one at a time:

| Structure | Node text | Functional pass@1 |
| :--- | :--- | ---: |
| annotated | annotated | 0.784 |
| annotated | predicted | 0.654 |
| predicted | annotated | 0.228 |
| predicted | predicted | 0.191 |

**Graph structure is the bottleneck.** Wrong OCR costs 0.13. A wrong graph costs 0.56. On
state machines, predicted structure scores 0.0, because a single misattributed arrow breaks
acceptance. Future work (§8) is ordered by this table.

**Robustness** ([`reports/robustness_pipeline.md`](../reports/robustness_pipeline.md)). Blur is
nearly free (node F1 0.99 → 0.98 at severity 9). Rotation is not: node F1 falls to 0.78 at 3° and
to 0.40 at 6°. That is why the client dewarps before upload (16.2.3).

**Human baseline** ([`reports/human_baseline.md`](../reports/human_baseline.md)). This is a model,
not a timed study. Retyping the median test diagram is estimated at 1,017 s. With repair of wrong
output included, the pipeline's expected time is 242 s, a **4.2x** speed-up. The 21x figure is
only for pages where the output is already correct.

## 7. Limitations

The honest failure modes. Every item here is something the current system gets wrong, or
something it has never been tested on.

- **Only two of the five diagram types work from a photograph.** Detector, router and OCR
  training data exist only for flowcharts and state machines. On unseen ER diagrams, circuits and
  wireframes ([`reports/unseen_type.md`](../reports/unseen_type.md)), **21 of 24 pages produced
  confident, runnable, wrong code**, routed as flowcharts with probability ≈ 1.0. Only one asked
  for confirmation. The 0.60 gate does not protect against a type the router has never seen,
  because NB is confidently wrong outside its support. The React, SQL and SPICE emitters work
  from an IR, not from a photograph.
- **Dense circuits.** Beyond being an unseen type, a circuit is the worst case for this graph
  builder. Wires meet at junctions with no arrowheads, crossing is not connection, and components
  are small glyphs packed together. Assembly already misses S5 on sparse flowcharts (median
  GED 13 on test), and arrowheads, the densest small class, are the detector's weakest (AP 0.43).
  A dense schematic should be expected to fail. It has not been measured, because there is no
  annotated hand-drawn circuit data in the corpus.
- **3-D sketches.** The IR is a planar graph with 2-D boxes and polylines, and every model was
  trained on flat diagrams. Isometric drawings, perspective sketches of objects, and diagrams
  drawn on a curved surface are out of scope. Perspective from a tilted *camera* is corrected by
  the dewarp, but perspective *in the drawing* is not. Rotation beyond a few degrees already
  degrades the pipeline sharply (§6).
- **Non-English labels.** The OCR model is TrOCR fine-tuned on English handwriting from a narrow,
  mostly academic group of writers. Other scripts (Devanagari, CJK, Arabic) are not in its
  vocabulary, and accented Latin is unmeasured. Identifiers in generated code are derived from
  labels, so a mis-read label becomes a wrong name in the code.
- **Label OCR misses its target** (CER 0.2006 against 0.15). It was trained on hdbpmn only, and
  reads automaton state labels 1.3% exactly, so state machines use a separate TrOCR fine-tune.
- **Assembly misses on the frozen test split** (median GED 13). The validation number (3) was
  tuned on, so the test number is the honest one.
- **The S1 classification number is inflated by source confounding.** Two sources are each a
  single class.
- **Generated code is not trusted.** It runs only in the server sandbox. A program that runs is
  not a program that is correct: 70% pass@1 from IR means that three programs in ten do the
  wrong thing.
- **Photos leave the device.** The models run on the server. See [`privacy.md`](privacy.md).

## 8. Future work

In order of what §6 says each item is worth:

1. **Graph assembly.** Arrow endpoint attribution and junction handling are 0.56 of the
   end-to-end loss. Train the arrow pose model on more hand-drawn arrows, and add a learned
   edge-to-node matcher in place of the geometric rules.
2. **Annotated data for the other three types**, starting with ER diagrams, so the detector and
   router cover them. Meanwhile, add an out-of-distribution check before the router, so an unseen
   type is refused instead of confidently mis-routed.
3. **OCR across sources and languages.** Train on fa_bresler and self-drawn labels, and add a
   multilingual recogniser.
4. **Serve the fine-tuned model inside the latency budget.** A smaller quantization or
   speculative decoding would bring p90 under 10 s.
5. **Close the correction loop.** The app already logs label fixes (`POST /feedback`). The 15.8
   retraining trigger should consume them.
6. **Human study.** Replace §6's modelled human baseline with timed participants.
