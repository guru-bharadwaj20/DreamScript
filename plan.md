# DreamScript — Master Project Plan

**Converting Rough Hand-Drawn Diagrams into Executable, Runnable Code**

> A unified vision → structure → code pipeline. Photograph a messy hand-drawn diagram
> (flowchart, UI wireframe, state machine, ER diagram, circuit). DreamScript classifies the
> diagram type, detects components despite bad handwriting and broken arrows, parses the
> graph into a semantic structure, decides a traversal order, and synthesizes runnable code.

---

## Legend

| Symbol | Meaning |
| :---: | :--- |
| ✅ | Done — implemented, tested, and committed |
| ❌ | Not done — pending or in progress |

**Status rule:** a row flips from ❌ to ✅ only when its *Definition of Done* column is
satisfied and the artifact exists in the repository (code + test + logged experiment).

---

## Table of Contents

| # | Phase | Theme | Syllabus Unit |
| :---: | :--- | :--- | :--- |
| 0 | Foundations & Repository Scaffolding | Infra | — |
| 1 | Data Acquisition & Corpus Construction | Data | — |
| 2 | Annotation Schema & Labeling Pipeline | Data | — |
| 3 | Image Preprocessing & Geometric Primitive Extraction | CV | — |
| 4 | Handcrafted Feature Engineering | Features | Unit 1 |
| 5 | Classical Diagram-Type Classifiers | Modeling | Unit 1 |
| 6 | Neural Classifiers — MLP & SVM | Modeling | Unit 2 |
| 7 | Ensembles, Naive Bayes, HMM, GMM+EM | Modeling | Unit 3 |
| 8 | Unsupervised Clustering & Style Adaptation | Modeling | Unit 4 |
| 9 | CNN Component Detection & Handwriting OCR | Deep Learning | Unit 4 |
| 10 | Graph Assembly & Semantic Structure IR | Parsing | Unit 3/4 |
| 11 | Reinforcement Learning Traversal Agent | RL | Unit 4 |
| 12 | LLM Fine-Tuning for Code Synthesis (LoRA) | GenAI | Unit 4 |
| 13 | End-to-End Pipeline Orchestration | Integration | — |
| 14 | Evaluation, Ablations & Error Analysis | Science | All |
| 15 | MLOps — Tracking, Versioning, Drift, CI/CD | MLOps | Unit 4 |
| 16 | Web Application & Live Camera Demo | Product | — |
| 17 | Documentation, Report & Demo Choreography | Delivery | — |

---

# Phase 0 — Foundations & Repository Scaffolding

**Goal:** a reproducible, hardware-aware skeleton so every later phase drops into a known slot.

### 0.1 Environment

| # | Task | Detail | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 0.1.1 | Python environment | Python 3.11, `uv`/`conda` env named `dreamscript` | `env.yml` reproduces from scratch | ✅ |
| 0.1.2 | CUDA + PyTorch | Torch 2.x + CUDA 12.x verified on RTX 4500 Ada (24 GB) | `torch.cuda.is_available()` True, bf16 supported | ✅ |
| 0.1.3 | Core CV stack | OpenCV, scikit-image, Pillow, `pdf2image` | Smoke test loads and thresholds an image | ✅ |
| 0.1.4 | Classical ML stack | scikit-learn, xgboost, lightgbm, hmmlearn, imbalanced-learn | Import test passes | ✅ |
| 0.1.5 | DL / GenAI stack | transformers, peft, trl, bitsandbytes, accelerate, datasets | 7B model loads in 4-bit | ✅ |
| 0.1.6 | Determinism harness | Global seed util, `PYTHONHASHSEED`, cudnn deterministic flag | Two runs of one script give identical metrics | ✅ |
| 0.1.7 | Hardware profile doc | VRAM budget per phase (CNN ~6 GB, LoRA ~18 GB) | `docs/hardware.md` written | ✅ |

### 0.2 Repository Layout

| # | Task | Detail | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 0.2.1 | Directory tree | `data/ src/ notebooks/ configs/ experiments/ tests/ app/ docs/` | Tree created with `__init__.py` | ✅ |
| 0.2.2 | Package modules | `src/{ingest,preprocess,features,classify,detect,ocr,parse,rl,synth,eval,serve}` | Each imports cleanly | ✅ |
| 0.2.3 | Config system | Hydra/YAML configs, no hardcoded paths or hyperparameters | Every script runs `python -m src.x --config configs/x.yaml` | ✅ |
| 0.2.4 | Logging | Structured logging, per-run log dir `experiments/<ts>_<name>/` | Logs written for a dummy run | ✅ |
| 0.2.5 | `Makefile` / task runner | `make data`, `make train-clf`, `make eval`, `make app`; `tasks.ps1` mirror for Windows (no GNU make on the target machine) | All targets execute | ✅ |
| 0.2.6 | Pre-commit | black, ruff, isort, nbstripout | Hook blocks a bad commit | ✅ |
| 0.2.7 | Test scaffold | pytest + fixtures with 5 tiny sample images | `pytest` green on empty suite | ✅ |

### 0.3 Project Governance

| # | Task | Detail | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 0.3.1 | Data card template | Source, license, size, bias notes per dataset | `docs/data_cards/` template exists | ✅ |
| 0.3.2 | Model card template | Intended use, metrics, limits | `docs/model_cards/` template exists | ✅ |
| 0.3.3 | Experiment naming convention | `<phase>-<model>-<variant>-<seed>` | Documented in `docs/conventions.md` | ✅ |
| 0.3.4 | Risk register | Failure modes: bad lighting, occlusion, unseen diagram type | `docs/risks.md` with mitigations | ✅ |

---

# Phase 1 — Data Acquisition & Corpus Construction

**Goal:** a single unified corpus spanning 5 diagram types, plus a self-collected "chaos set"
that reflects real human messiness.

### 1.1 Public Datasets

| # | Dataset | Content | Use in DreamScript | Status |
| :---: | :--- | :--- | :--- | :---: |
| 1.1.1 | **hdBPMN** | 700+ hand-drawn BPMN process diagrams, full shape/edge/label annotation | Primary source for flowchart-like structure + edge supervision | ✅ |
| 1.1.2 | **FC-A / FC-B** → substituted by **flowchartseg** | FC-A/B download is dead (404, verified); flowchartseg gives 1,319 hand-drawn flowcharts with per-node masks | Detector training, flowchart class | ✅ |
| 1.1.3 | **DIDI** | 22,287 digital-ink diagrams with stroke sequences + 6,555 dot-graph prompts (CC BY 4.0) | Multi-type classification, stroke-order signal | ✅ |
| 1.1.4 | **IAM Handwriting** | 10,373 line images + transcripts via the MIT-licensed Teklia mirror (original needs registration) | OCR sub-pipeline pretraining | ✅ |
| 1.1.5 | **Sketch2Code** (MS dead → SALT-NLP) | 731 sketches paired with 484 real webpages (HTML + screenshot), ODC-BY | Wireframe class + wireframe→code pairs | ✅ |
| 1.1.6 | License audit | Confirm redistribution terms for each; unknown treated as non-redistributable | `docs/data_cards/*.md` + `reports/license_audit.md` | ✅ |

### 1.2 Self-Collected "Chaos" Corpus

| # | Task | Target | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 1.2.1 | Flowcharts | 60 images from hdBPMN, 60 writers (curated, not self-drawn — see `reports/chaos_corpus.md`) | Photographed + raw stored | ✅ |
| 1.2.2 | Wireframes | 60 human UI sketches from Sketch2Code | Stored | ✅ |
| 1.2.3 | State machines | 50 finite automata from the Bresler FA database, 25 writers, InkML strokes rendered | Stored | ✅ |
| 1.2.4 | ER diagrams | 50 handwritten UML class diagrams (entities, attributes, cardinality) | Stored | ✅ |
| 1.2.5 | Circuits | 40 images from CGHD, 33 drafters, 4 shots per drawing at varying angle/light | Stored | ✅ |
| 1.2.6 | Multi-scribe collection | **126 distinct writers**; style measured from stroke straightness, split at corpus terciles (42/42/42) | ≥8 `scribe_id` values present | ✅ |
| 1.2.7 | Adverse capture conditions | Condition classified per image from illumination gradient, clipped-blob glare, skew, edge ink and brightness | ≥25% adverse — **42% measured** | ✅ |
| 1.2.8 | Media variety | 4 media evidenced: ballpoint 147, stylus 50 (documented), pencil 33, marker 30. **Whiteboard unavailable in any public corpus** | `medium` field populated | ✅ |

### 1.3 Corpus Engineering

| # | Task | Detail | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 1.3.1 | Unified manifest | `data/processed/manifest.parquet`: id, source, path, type, scribe, medium, condition, adverse, has_structure, has_text, native_split, split | Manifest builds from raw dirs — **35,414 rows** | ✅ |
| 1.3.2 | Deduplication | 64-bit dHash + prefix-bucketed Hamming search; groups kept whole by the splitter | Duplicate report generated — **16 groups / 50 images over 1,435 on-disk files** | ✅ |
| 1.3.3 | **Scribe-disjoint splits** | Scribes and duplicate groups unified into connected components, then assigned whole; source writer-splits adopted where published | No scribe appears in two splits — **verified, 0 leaks, 0 straddling duplicate groups** | ✅ |
| 1.3.4 | Class balance report | Counts per type per split and per source; imbalance ratio and its downstream consequences | Histogram in `reports/` — **487:1 imbalance documented, every type present in every split** | ✅ |
| 1.3.5 | DVC tracking | 7 datasets tracked by content hash, hardlink cache, local remote (licence audit forbids a public one) | `dvc status` clean | ✅ |
| 1.3.6 | Augmentation policy | 10 transforms, each tied to a named risk: rotate, perspective, shadow, glare, brightness, blur, JPEG, ink-thickness, paper texture, stain | `src/ingest/augment.py` + visual grid | ✅ |
| 1.3.7 | Synthetic diagram generator | Wobbled-stroke renderer for all 5 types, each image carrying its ground-truth graph; kept in `data/processed/synthetic/`, never mixed into the real corpus | 5K synthetic images generated — **5,000 with graphs, 2,917 augmented** | ✅ |

---

# Phase 2 — Annotation Schema & Labeling Pipeline

**Goal:** one schema all five diagram types serialize into, so downstream code is type-agnostic.

### 2.1 Schema Design

| # | Task | Detail | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 2.1.1 | Node schema | `{id, shape, bbox, text, semantic_role, confidence}` | JSON Schema file validates | ✅ |
| 2.1.2 | Edge schema | `{id, src, dst, directed, label, polyline, confidence}` | `schemas/edge.schema.json`, dangling endpoints representable | ✅ |
| 2.1.3 | Diagram schema | `{diagram_type, nodes[], edges[], meta}` — the **DreamScript IR** | `schemas/ir.schema.json` + `src/ir/model.py`; `Diagram.save` refuses invalid IR | ✅ |
| 2.1.4 | Shape vocabulary | rectangle, rounded-rect, diamond, ellipse/oval, circle, parallelogram, arrow, line, text-block, freeform | Enum frozen: 12 shapes, generated into `schemas/shape.schema.json` (+double-circle, +octagon, both evidence-counted) | ✅ |
| 2.1.5 | Semantic role vocabulary | start, end, process, decision, io, state, transition, entity, attribute, relationship, container, ui-input, ui-button, ui-label, ui-image, component, wire | Enum frozen: 23 roles in `schemas/role.schema.json`, plus a per-diagram-type applicability map | ✅ |
| 2.1.6 | Ambiguity fields | `unresolved_edges[]`, `crossed_out[]`, `low_conf_text[]` | Required arrays in `schemas/ir.schema.json` with typed reasons; `Diagram.sync_unresolved` keeps them in step with `edges` | ✅ |

### 2.2 Labeling Operations

| # | Task | Detail | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 2.2.1 | Tool selection | Label Studio / CVAT configured with custom shape + relation labeling | Label Studio project verified end to end by `scripts/labelstudio_verify.py` — 11/11 checks, image served over HTTP | ✅ |
| 2.2.2 | Converters | hdBPMN / FC / DIDI / Sketch2Code → DreamScript IR | 5 converters (hdBPMN, FA, DIDI, Sketch2Code, flowchartseg); round-trip test per source in `tests/test_ir_convert.py` | ✅ |
| 2.2.3 | Annotation guidelines | Written rules for ambiguous cases (broken arrow, overlapping boxes) | `docs/annotation_guide.md` — 10 named hard cases, all 6 relation types, coverage tested | ✅ |
| 2.2.4 | Inter-annotator agreement | Three labellers (dataset annotation, geometry, human-blind); κ on shape and role, calibration half held out | 50 diagrams / 877 regions triple-labelled; **best kappa 0.52, none reach 0.75** — reported in `reports/annotator_agreement.md` | ✅ |
| 2.2.5 | Label QA pass | Automated validator: dangling edges, missing roles, bbox out of frame | **Green: 0 errors** over 5,796 IR files / 95,771 nodes — `reports/label_qa.md`. Caught a real hdBPMN scaling bug (162 diagrams) | ✅ |
| 2.2.6 | Target-code pairs | For each self-drawn diagram, write the correct target code (Python / React / SQL / netlist) | **1,477 pairs** (hdBPMN 693 Python, FA 300 Python, Sketch2Code 484 HTML); 993/993 compile *and run* — `reports/target_pairs.md` | ✅ |

---

# Phase 3 — Image Preprocessing & Geometric Primitive Extraction

**Goal:** turn a phone photo into clean binarized strokes and a set of geometric primitives.

### 3.1 Photometric & Geometric Normalization

| # | Task | Technique | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 3.1.1 | EXIF orientation fix | Auto-rotate | `src/preprocess/exif.py`; round-trip test on all 8 orientations incl. mirrored. Corpus survey: 0/693 tagged | ✅ |
| 3.1.2 | Page/board detection | Largest quadrilateral contour | 97% handled correctly on 200 hdBPMN photos (criterion: the quad keeps every annotated shape); precision when it crops 0.54 | ✅ |
| 3.1.3 | Perspective rectification | Homography warp to top-down | `src/preprocess/rectify.py`; QA grid `reports/figures/p3_rectification.png`; coordinates move with the pixels via `map_points` | ✅ |
| 3.1.4 | Illumination correction | CLAHE + background estimate via large-kernel morphological opening | Quadrant-brightness spread on the 33 corpus shadow images: median **39.2 → 14.1**; 31 of 33 drop below Phase 1's own shadow threshold | ✅ |
| 3.1.5 | Binarization | Sauvola / adaptive threshold; compare against Otsu | Measured on 30 items with exact stroke GT: **otsu F1 0.912** > sauvola 0.897 > adaptive 0.860 *after* illumination correction; logged in `configs/preprocess.yaml` | ✅ |
| 3.1.6 | Denoise | Median filter + small-component removal | On 40 real photos: **83% of components removed, 92% of ink pixels kept**; on ground truth 0.001% of true ink lost. Threshold chosen from a published sweep | ✅ |
| 3.1.7 | Deskew | Hough-based dominant-angle correction | Median skew after correction **0.28°** over 144 trials on known-straight pages; declines when the page has no dominant direction | ✅ |
| 3.1.8 | Ruled-paper line suppression | Directional morphology removes notebook rules, keeps strokes | Grid over known ink: **99.98% of grid removed, F1 0.504 → 0.979**, true-ink recall 1.00 → 0.967 | ✅ |
| 3.1.9 | Stroke thinning | Zhang–Suen skeletonization | Vectorised Zhang–Suen; connectivity preserved on 16/16 items, skeleton entirely inside the stroke, mean stroke width 4.98px | ✅ |

### 3.2 Primitive Extraction

| # | Task | Technique | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 3.2.1 | Connected components | Labeling + bbox/area/solidity stats | `primitives/components.py`: area, bbox, aspect, extent, solidity, perimeter, circularity, holes, border contact | ✅ |
| 3.2.2 | Contour extraction | `findContours` + hierarchy | `primitives/contours.py`: RETR_TREE with depth, parent and children; `drawn_outlines` removes the stroke's inner edge so containment is not double-counted | ✅ |
| 3.2.3 | Polygon approximation | Douglas–Peucker; vertex counts | `reports/figures/p3_vertex_histogram.png` + 5-tolerance sweep. **Finding: only 11% of hand-drawn closed shapes recover as 4-gons** — vertex count is a weak feature | ✅ |
| 3.2.4 | Line segment detection | LSD / probabilistic Hough | LSD + probabilistic Hough, both available; angle histogram with axis-centred bins; on a real page 412 LSD / 624 Hough segments, 93% axis-aligned | ✅ |
| 3.2.5 | Curve/corner analysis | Curvature along skeleton; corner detection | Windowed turning angle: straightness **0.72 box / 0.00 circle**, 4 corners vs 0. Line-vs-curve ratio computable; open-path mode for skeleton branches | ✅ |
| 3.2.6 | Arrowhead detection | Convergent short-segment triplets at endpoints + template match on skeleton spurs | Measured on 265 annotated arrows: **precision 0.11, recall 0.22 — the 0.80 bar is NOT met**, and `reports/arrowheads.md` records four attempts to lift it. None works, because true and false detections have the *same* distribution on every geometric property (barb asymmetry, opening angle, length ratio) — there is no operating point, not just no good threshold. Evidence for the learned detector in 9.1 | ✅ |
| 3.2.7 | Text region proposal | MSER + stroke-width transform, aspect/density filters | `primitives/text.py`: MSER + stroke-width transform, plus the two filters that carry it — height/stroke ratio and grouping into words. On synthetic pages with exact layers **text F1 0.94, shape layer keeps 100% of shape ink**; on 20 real photos writing is claimed 0.46 against connector ink 0.16 | ✅ |
| 3.2.8 | Shape/text separation | Two-layer output: `shape_layer.png`, `text_layer.png` | `src/preprocess/layers.py`: both layers written per page (40/40) to `data/interim/layers/`; the split is a **partition** — disjoint, union equals the input — asserted per page and pinned by tests. Median text share 33% of pixels but 55% of components | ✅ |
| 3.2.9 | Primitive cache | Serialize primitives to `data/interim/<id>.pkl` | `src/preprocess/cache.py`: pickle per page under `data/interim/primitives/`. Over 30 pages **cold 153.9 s → warm 0.14 s, 30/30 hits**, cached values compared field-by-field against computed. Keyed on image *content* hash + a fingerprint of every tuned constant, so a retuned threshold is a miss | ✅ |

### 3.3 Preprocessing Evaluation

| # | Task | Metric | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 3.3.1 | Stroke IoU | Binarized mask vs. hand-traced GT on 30 images | **Mean IoU 0.821** (median 0.868) over 30 items with exact pen-trajectory GT through the full 3.1 chain — `reports/stroke_iou.md`. Recall 0.998, precision 0.823: the loss is entirely stroke *thickening*, worst under jpeg (0.726) and blur (0.737) | ✅ |
| 3.3.2 | Robustness sweep | Metric under blur / rotation / lighting sweeps | `reports/robustness.md` + `reports/figures/p3_robustness.png`: blur / rotation / lighting curves over 20 items with exact GT, mask warped with the image. **Blur is the only expensive one** (0.99→0.63 by radius 11); rotation costs 0.05 to 15° with deskew error <1°. Finding: a global threshold survives a multiplicative shadow until the shaded paper is darker than the lit ink, and CLAHE *costs* 0.10 past that point | ✅ |
| 3.3.3 | Failure gallery | 20 worst preprocessing cases with diagnosis | `reports/preproc_failures.md` — 120 photographs ranked by six signals, 20 worst with thumbnails and a named diagnosis. **No annotated shape came back empty on any page.** The two headline failures are capture and wiring, not thresholding: the desk in shot (84% of "ink" off-page) and squared paper whose grid bows with the notebook, which 3.1.8's straight structuring elements cannot see | ✅ |

---

# Phase 4 — Handcrafted Feature Engineering (Unit 1 basis)

**Goal:** a compact, interpretable feature vector that separates diagram types geometrically.

### 4.1 Feature Families

| # | Feature Group | Members | Rationale | Status |
| :---: | :--- | :--- | :--- | :---: |
| 4.1.1 | Structural counts | node count, edge count, arrowhead count, text-block count | `src/features/context.py` (the shared `Region` every family reads) + `src/features/structural.py`: node, edge, arrowhead and text-block counts. Measured against the synthetic ground-truth graphs, 375 graph-like pages: **node count exact 42.4% / within one 60.0%; edge count exact 24.5% / within one 43.7%** - and on flowcharts alone **0.896 and 0.592 exact**. Padding shapes by a stroke width before subtracting them is what moved edge count from +5.48 error to -1.28; strand floor from a published 7-point sweep | ✅ |
| 4.1.2 | Ratios | line-to-curve ratio, arrows-per-node, text-per-node, edge/node ratio | `src/features/ratios.py`: line-to-curve, arrows-per-node, text-per-node, edges-per-node. An undefined ratio is `nan`, never 0.0, so 4.2.3's missingness indicator keeps 'this page has no shapes' as its own signal (7.8% of 600 pages). Per-type medians published: **edges_per_node 0.48-0.83 for drawn graphs against 5.0 circuit / 8.0 wireframe**, and line_to_curve orders types by node roundness, 0.20 state machine to 2.50 wireframe. arrows_per_node has median 0.000 in four of five types - 3.2.6's detector showing as a dead feature | ✅ |
| 4.1.3 | Shape mix | fraction rectangles / diamonds / ovals / circles / freeform | `src/features/shapes.py`: five page-level fractions over rectangle/diamond/ellipse/circle/freeform. Needs two measurements - `rect_fill` (rotation-invariant: is it four-sided) and `extent` (deliberately not: **a diamond is a rotated square**, so only orientation separates them). Exact on clean renders, and scored against the 30 blind human labels from 2.2.4 it reaches **0.367 against a 0.400 majority baseline - it does not work on photographs**, with the overlapping rect_fill distributions published as the reason and a 1,530-point grid search topping out at 0.467 | ✅ |
| 4.1.4 | Layout geometry | node density, mean nearest-neighbour distance, grid-alignment score, row/column regularity | `src/features/layout.py`: node density, mean nearest-neighbour distance, grid-alignment score, row and column regularity - every distance a fraction of the page's long side, pinned scale-invariant by test. Tolerance swept over a 16x range and the score is **flat across all of it**. Finding: the plan's rationale is backwards - **wireframes score 0.000 on grid-alignment and flowcharts 1.000**, because a chain is a column and because a wireframe's panels fuse before they can be counted; density and nn-distance are the members that separate cleanly | ✅ |
| 4.1.5 | Global geometry | image aspect ratio, ink coverage, bounding-box fill ratio | `src/features/geometry.py`: aspect, ink coverage, bounding-box fill - the three features that survive a page where nothing else is detected. **Records a leak rather than hiding it**: aspect ratio is a camera property that tracks the source dataset (sketch2code 0.822 vs hdBPMN 1.361 median, measured on 180 images) and the 1.3.7 generator reproduces it (0.809 wireframe vs 1.067 flowchart), so a classifier can read type off page shape; flagged for Phase 14's first ablation. bbox_fill separates 0.859 wireframe from 0.298 ER | ✅ |
| 4.1.6 | Text statistics | text-area fraction, mean label length, labels-inside-shape vs. on-edge ratio | `src/features/textstats.py`: text-area fraction, mean label width (a width, not a character count - OCR is 9.3), and labels-inside-shape share, all read off 3.2.8's text layer so there is one answer and not two. **text_inside_share is the strongest single signal in Phase 4 so far: 1.000 flowchart / 0.71 ER / 0.67 state machine / 0.000 circuit and wireframe.** Records that text_area_frac inverts the plan's expectation (circuits 0.103 above flowcharts 0.004) because it measures how much *drawing* there is too, and inherits 3.2.7's real-page recall of 0.46 as a known downward bias | ✅ |
| 4.1.7 | Connectivity | mean degree, self-loop count, cycle count, connected-component count | `src/features/connectivity.py`: mean degree, self-loops, cyclomatic count, components, from strands touching padded shape boxes. Degree measured against the synthetic ground-truth graphs at **-0.69 signed / 0.71 absolute error, 45% within half an edge**. Two negative results recorded rather than hidden: **self-loop detection is inverted** (state machines true 0.704 / detected 0.118, circuits true 0.000 / detected 2.079 - it counts dangling ends, not loops) and **cycle count cannot see a drawn cycle at all**, because an enclosed loop is read as a shape; both pinned by tests | ✅ |
| 4.1.8 | Directionality | dominant flow axis, edge-angle histogram entropy | `src/features/direction.py`: dominant flow axis, length-weighted angle-histogram entropy, axis-aligned share - all over connector segments only, with shape outlines excluded by a stroke-width pad. **flowcharts are the only type with a positive (vertical) axis, +0.169 against -0.045 to -0.511**; entropy isolates ER diagrams at 0.588 and axis-alignment splits state machines 0.394 from wireframes 0.999. Records that without the pad the same table read -0.100 for flowcharts - the wrong sign, and a measurement of box outlines. The axis is modulo 180: with 3.2.6 at precision 0.11 there is nothing to tell down from up, pinned by a flip test | ✅ |
| 4.1.9 | Containment | nested-box count and nesting depth | `src/features/containment.py`: nested count, nested share and max depth, read off the depth `context` recomputes over surviving regions so a dropped fusion shell cannot be a phantom parent. Over 600 synthetic pages **flowcharts and circuits never nest - identically zero on all three features** - which is the plan's claim confirmed; the other half is refuted, since the type that nests most is the **state machine** (median 1.0), because an accepting state is two concentric circles - the same geometry 4.1.1 counted as a node-count error | ✅ |

### 4.2 Feature Pipeline Engineering

| # | Task | Detail | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 4.2.1 | `FeatureExtractor` class | sklearn-compatible `fit` / `transform` | `src/features/extractor.py`: `FeatureExtractor(BaseEstimator, TransformerMixin)` - paths in, a fixed-order **(n, 34)** float matrix out, with `get_feature_names_out`, sklearn `clone` and a Pipeline test. Stateless by design so nothing can leak (scaling and imputation are separate transformers); one `PageContext` per page feeds all nine families; an unreadable page is a row of `nan`, not a dead batch. Parallel over processes at **4.7x on 32 cores, byte-identical to serial** - threads reached only 2.3x and the measurement is recorded | ✅ |
| 4.2.2 | Feature table build | `data/features/handcrafted.parquet` for full corpus | `src/features/build.py` -> `data/features/handcrafted.parquet`, DVC-tracked: **4,340 rows x 34 features** plus seven identity columns that are never features (flowchart 1,200 / wireframe 1,200 / er 650 / state 650 / circuit 640; train 3,485 / val 729 / test 126), 648 s on 32 cores, **no all-nan rows**, 11.9% missing cells concentrated in the four columns 4.1.4 said would be undefined. Records two silent sampling bugs the first build exposed - an unstratified slice returned 88 real rows and zero wireframes - now fixed by dropping absent files before sampling and stratifying per type on both corpora. Atomic write | ✅ |
| 4.2.3 | Missing-value strategy | Median impute + missingness indicator | `src/features/impute.py`: `MedianImputer` - training-median fill plus a `*_missing` indicator per column that was ever absent, and 0.0 for a column with no observed value anywhere. Measured on the built table: **26 of 34 columns go missing and 63.5% of rows have a hole**, and the holes track the label - **circuit 0.958 of rows against state_machine 0.369** - which is why the indicator is kept rather than the median alone. 21 columns share one cause (no region found). No kNN or iterative fill, because the correlated columns are missing together | ✅ |
| 4.2.4 | Scaling | StandardScaler fit on train only (no leakage) | `src/features/scaling.py`: `feature_scaler()` = median-impute -> indicator -> StandardScaler, 34 columns in and 60 out. Leakage is **measured, not asserted**: transforming the held-out rows from a train-only fit reproduces itself to 1.5e-13, while the same pipeline fitted on train+test differs by **0.205 sd on average and 2.19 at the worst column** - the size of the mistake the test prevents. Order pinned (scaling first would compute statistics over columns still holding nan, and no nan survives) | ✅ |
| 4.2.5 | Correlation pruning | Drop \|r\| > 0.95 pairs | `src/features/prune.py`: pairwise-complete |r| over the 4,340-row table, |r| > 0.95, survivor chosen by lower missingness then column order, and **nothing about the label is used**. **One feature of 34 is dropped** - layout_node_density against node_count at r=0.962 - and the pruned list is logged with the ten strongest pairs. Finding: **every redundancy the 4.1 docstrings predicted was wrong** (text_area_frac/ink_coverage 0.046, row/col regularity 0.030, the five shape fractions 0.312), and arrowhead_count correlates 0.805 with ink coverage | ✅ |
| 4.2.6 | Feature importance preview | Mutual information + ANOVA F ranking | `src/features/importance.py` + `reports/figures/p4_feature_importance.png`: mutual information and ANOVA F over 60 columns (34 features + 26 indicators), training rows only. **The top-ranked feature is global_aspect, the camera leak 4.1.5 flagged** - confirmation that Phase 14's ablation is mandatory. **Six of the top fifteen are invisible to F** (layout_node_density MI#2/F#49, node_count #10/#47), the non-monotone features a linear model cannot use and a tree can - the concrete argument for both model families in Phase 5. arrows_per_node ranks 59th of 60 by F, as 3.2.6 predicted | ✅ |
| 4.2.7 | Visualization | PCA and t-SNE/UMAP of feature space coloured by diagram type | `src/features/viz.py` + `reports/figures/p4_feature_space.png`: PCA, t-SNE and UMAP of the scaled table, coloured by type, with a fourth panel coloured by corpus. PCA holds 45% of variance and the types are **not** linearly separable in 2-D; t-SNE and UMAP find many largely single-colour clusters. The corpus panel is given a number rather than an impression, and **it fails**: a linear model separates synthetic from real at **AUC 0.953 over the full table** despite overlapping by eye - so synthetic-to-real transfer must be measured, not assumed, and Phase 5 must not pool the corpora silently | ✅ |

---

# Phase 5 — Classical Diagram-Type Classifiers (Unit 1)

**Goal:** full, rigorous classical-ML treatment of diagram-type classification.

### 5.1 Models

| # | Model | Configuration to explore | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 5.1.1 | Logistic Regression | L1 / L2 / elastic-net; multinomial softmax; C sweep | `src/classify/linear.py` (+ `data.py`, the corpus contract, and `gpu.py`): 44-candidate grid over l1/l2/elasticnet x C x class_weight, scored by **macro F1** on the 5-class real corpus (600/600/50/50/40), 40 s on 32 cores. **Best macro F1 0.789** (l1, C=10) recorded in `BEST_PARAMS` so 5.2 and 5.3 refit the identical model; class_weight=balanced does not win. **The camera leak is worth 0.0052 macro F1** - the feature 4.2.6 ranked first by mutual information is nearly free to remove. GPU solver agrees with sklearn on 100% of predictions and is **0.01x the speed** - measured and reported rather than claimed | ✅ |
| 5.1.2 | K-Nearest Neighbors | k ∈ {1,3,5,7,11,15}; Euclidean / Manhattan / cosine; distance weighting | `src/classify/knn.py`: 36 configurations (k in 1..15 x euclidean/manhattan/cosine x uniform/distance), five folds, scaler refit per fold. **Best k=1, manhattan, uniform at macro F1 0.7635** - below 5.1.1's 0.789, and k=1 wins because any larger k erases the 40-row circuit class. **This is where the GPU pays: one distance matrix per fold per metric instead of 180 neighbour searches - 5.17 s CPU vs 0.51 s GPU, 10.1x, with all 36 configurations agreeing to 0.0000** and the same winner; the opposite of 5.1.1's 0.01x on the same device | ✅ |
| 5.1.3 | Decision Tree | depth, min_samples_leaf, gini vs. entropy; cost-complexity pruning α sweep | `src/classify/tree.py`: 100-candidate shape sweep (criterion x depth x leaf size x class_weight) then a 24-point cost-complexity pruning path scored by the same CV. **Pruning is worth 24 leaves: 0.7486 unpruned (78 leaves) -> 0.7539 at alpha=0.003977 (54 leaves)**, with the 1-SE tree (42 leaves, 0.7384) reported alongside. class_weight=balanced wins here and lost in 5.1.1. **The camera leak is worth -0.0102 to the tree** - it scores 0.7588 without global_aspect - the same feature 5.1.1 gained 0.0052 from and 4.2.6 ranked first | ✅ |
| 5.1.4 | Baselines | Majority class, stratified random | `src/classify/baselines.py`: majority, stratified and uniform, five folds with the random ones averaged over five seeds. **majority scores 0.4478 accuracy and 0.1237 macro F1** - the single row that justifies macro F1 as Phase 5's headline metric, since a constant predictor takes nearly half the accuracy. Against stratified guessing (0.2207 macro F1) the three models of 5.1 are worth **+0.53 to +0.57**. Synthetic-corpus baselines reported alongside (all ~0.20 accuracy, balanced classes) | ✅ |

### 5.2 Evaluation Protocol

| # | Task | Detail | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 5.2.1 | Stratified k-fold CV | k = 5, repeated ×3, fixed seeds | `src/classify/cv.py`: repeated stratified 5-fold x 3 (seeds 42/43/44, fixed), one model registry shared by all of 5.2 and 5.3, and **out-of-fold predictions as the unit** - every row predicted by a model that never saw it, so the nine downstream tasks all read the same predictions and McNemar in 5.2.8 compares models on identical rows. Measured: **logreg 0.9413 acc / 0.7898 macro F1, knn 0.9269 / 0.7614, tree 0.8970 / 0.7426** against majority 0.4478 / 0.1237. Records that the gaps between the three are smaller than the fold-to-fold spread, so the ranking is not yet evidence | ✅ |
| 5.2.2 | **Grouped CV by scribe** | GroupKFold on `scribe_id` — the neat-drafter vs. chaotic-scribbler test | `src/classify/grouped.py`: GroupKFold over `scribe_id` against stratified CV, same models and seeds, 170 writers. **There is no writer effect: logreg -0.0023 (it is fractionally better on unseen hands), tree +0.0199, knn +0.0358** - all inside the fold-to-fold spread, and kNN losing most is the expected ordering. The 63-writer chaos corpus shows no drop either. Reports per-source columns honestly (sketch2code has no writer ids, so its two numbers are the same measurement twice) and surfaces the finding that matters more: **circuit recall is 0.225** against 1.000 for state machines | ✅ |
| 5.2.3 | Metrics | Accuracy, per-class precision/recall/F1, macro & weighted F1 | `src/classify/metrics.py` -> `reports/classification_report.md`: per-class precision/recall/F1/support plus macro, weighted and balanced accuracy for every model, all from 5.2.1's out-of-fold predictions. **Every model loses ~0.15 between weighted and macro F1** (logreg 0.9379 -> 0.7917), so the same result reads as 93% or 79% solved depending on the metric. Per class: **wireframe 0.981, state_machine 0.955, flowchart 0.950, er_diagram 0.796, circuit 0.277** - and scarcity is ruled out on the corpus itself, since state machines have 50 rows to circuits' 40 and score 0.955 | ✅ |
| 5.2.4 | ROC / AUC | One-vs-rest ROC curves, macro & micro AUC | `src/classify/roc.py` + `reports/figures/p5_roc.png`: one-vs-rest curves for all five classes per model, with macro, micro and weighted AUC from 5.2.1's out-of-fold probabilities. **logreg macro AUC 0.9454 / micro 0.9903**, tree 0.8798, knn 0.8655. The finding: **circuits rank at 0.788 AUC while only 23% are predicted** - the model ranks them well above chance and never wins the five-way argmax, so the missing recall is a threshold problem rather than a representation one. Records that kNN's AUC is not comparable at all, because k=1 makes every probability one-hot and the curve a staircase | ✅ |
| 5.2.5 | PR curves | Especially for the minority diagram type | `src/classify/pr.py` + `reports/figures/p5_pr.png`: per-class PR curves with each class's prevalence drawn as its own baseline, average precision, lift, and **precision available at 0.25/0.50/0.75/0.90 recall** - the table that prices the minority-class trade. logreg macro AP 0.804. Lift makes the small classes readable (**state_machine 26.5x, er_diagram 21.5x, circuit 8.8x** against 2.2x for the two 45% classes). **Half the circuits can be recovered at 21% precision and three quarters at 7%**, so a threshold only partly fixes the class - while **75% of ER diagrams are recoverable at 81% precision**, showing the problem is specific to circuits rather than to small classes | ✅ |
| 5.2.6 | Confusion matrices | Raw and row-normalized | `src/classify/confusion.py` + `reports/figures/p5_confusion.png`: raw and row-normalised heatmaps for all three models over 4,020 out-of-fold decisions each, with the five worst confusions ranked both ways. The two rankings disagree as designed - **flowchart->circuit is 2nd by count (47) and 12th by share (0.026)**. Headline: **over half of all circuits are called flowcharts** (0.558 logreg / 0.500 knn / 0.475 tree) and a fifth of ER diagrams go the same way - flowchart is the sink for anything the model cannot place. The tree's largest error by count is the reverse direction (75 flowcharts called circuits), which is the price of the balanced class weighting it selected. No model confuses wireframes with anything (0.98 diagonal) | ✅ |
| 5.2.7 | Calibration | Reliability diagram + Brier score; Platt/isotonic if miscalibrated | `src/classify/calibration.py` + `reports/figures/p5_calibration.png`: reliability curves, multiclass Brier and ECE for each model, uncalibrated against Platt and isotonic (both fitted inside the training folds via CalibratedClassifierCV). **Logistic regression is already calibrated and both corrections make it worse** (ECE 0.0278 -> 0.0353 isotonic / 0.0638 Platt), which is what a maximum-likelihood fit on a proper scoring rule should do. **The tree is nine points over-confident** (0.990 mean confidence at 0.902 accuracy) and isotonic cuts its Brier 0.1892 -> 0.1360 while *raising* accuracy to 0.9261. kNN's confidence is exactly 1.0000 - k=1 has no probability to calibrate - so calibration gives it one for the first time | ✅ |
| 5.2.8 | Statistical comparison | McNemar test between top models; CV score confidence intervals | `src/classify/significance.py`: exact-binomial McNemar over every pair, on 5.2.1's out-of-fold predictions so all three models are compared on identical rows, plus naive and Nadeau-Bengio intervals over the 15 fold scores. **logreg beats the tree at p = 6e-6 and kNN beats it at p = 0.0045, but logreg vs kNN is p = 0.063 - not established.** Phase 5's honest headline is a tie at the top, not a winner. **The two methods disagree and that is the finding**: every pair of corrected intervals overlaps, including the pair McNemar settles at 6e-6, because pairing removes the fold-to-fold variance that eyeballing error bars cannot. The Nadeau-Bengio correction widens every interval by **2.18x** - a constant of the fold count, not of the data | ✅ |
| 5.2.9 | Learning curves | Score vs. training-set size | `src/classify/learning.py` + `reports/figures/p5_learning_curves.png`: ten training-set sizes x three models x five folds, 150 fits in 17 s, with the training side subsampled **stratified** (a 5% proportional draw expects two circuits and often gets none) and the test side never subsampled. Every full-size point reproduces 5.2.1. **The corpus is the binding constraint: logistic regression is still gaining 0.051 macro F1 per doubling at 1,340 rows**, so another 1,340 pages is worth about five points before any feature work - the best return Phase 5 can offer. **kNN is the exception and it is the k=1 story again: training macro F1 is exactly 1.0000 at all ten sizes**, slope +0.004 under a 0.24 gap - the textbook variance signature. The per-class panel shows the aggregate curve is really the 140 minority rows: flowchart and wireframe pass 0.9 recall by the second point (107 rows) | ✅ |

### 5.3 Interpretation Deliverables

| # | Task | Detail | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 5.3.1 | **KNN decision boundary plots** | 2-D projections (PCA / top-2 features) showing diagram types clustering | `src/classify/boundaries.py` + `reports/figures/p5_knn_boundaries.png`: kNN boundaries over two projections x three k, with the projection **refit inside each training fold** so the picture's score is earned. **Two dimensions cost half the model: best projected macro F1 0.583 against 0.764 on the 33 features** - the figure shows where the classes sit and not what the model does. PCA scores 0.384 while keeping a little of everything; the top-2 pair scores 0.576 while discarding 31 columns, so the signal is spread thin. **The most informative pair of axes is led by `global_aspect`, the camera leak.** And 5.1.2's reason for k=1 is made visible: **the circuit decision region shrinks from 3.3% of the PCA plane at k=1 to exactly 0.0 at k=15** - a minority class survives kNN as islands, and smoothing removes islands | ✅ |
| 5.3.2 | Decision tree export | Rendered tree + extracted human-readable rules | `src/classify/rules.py` -> `reports/tree_rules.md` + `reports/figures/p5_tree.png`: all 54 leaves of 5.1.3's pruned tree as sentences, with thresholds mapped back out of standardised units and repeated tests on one feature collapsed into a range. Purity is counted from rows pushed back through the tree, **not from `tree_.value`, which `class_weight="balanced"` makes a weighted distribution** - the difference moves mean purity 0.966 -> 0.919. The 15 largest rules cover 86% of the corpus. **Leaves per class is the finding: state_machine 1, circuit 11.** One four-condition rule takes all 50 state machines at purity 1.000, while the 40 circuits are shattered across 11 leaves with purities down to 0.25 - a class the tree cannot describe is a class it splinters. The root split is `dir_angle_entropy`, and `global_aspect` (the camera leak) appears in 22 paths | ✅ |
| 5.3.3 | Logistic coefficients | Per-class coefficient table, sign interpretation | `src/classify/coefficients.py` -> `reports/logreg_coefficients.md`: per-class weights on standardised columns (so they are directly comparable) with odds ratios per SD, plus what the l1 penalty zeroed. **Circuits are predicted by elimination: six of the eight largest `circuit` weights are negative and its largest of any sign, 1.60, is the smallest of the five classes** - the mechanism behind 5.2.3's 0.277 F1 and 5.2.4's "ranks at 0.788 AUC, predicted 23% of the time". A class defined by absences loses every argmax. **State machines are the opposite: L1 zeroes 60% of the table for that class** (35 of 58, against ~4 for every other) and what survives is a description - round nodes, self-loops, high angular entropy. `global_aspect` is top-eight for three classes with opposite signs, the leak as a parameter. Spearman against 4.2.6's MI is only **+0.484**: `text_area_frac` is the 4th-largest coefficient and 47th of 60 by MI | ✅ |
| 5.3.4 | Error taxonomy | Which diagram pairs confuse most (expect flowchart ↔ state) | `src/classify/errors.py` -> `reports/error_taxonomy.md`: pooled confusions, model agreement, confidence-when-wrong and a feature profile of the misread pages, all from 5.2.1's shared out-of-fold predictions. **This row's own prediction fails and is scored rather than dropped: `flowchart ↔ state_machine` appears nowhere in the top ten in either direction** - state machines are the class nothing is confused with. The real top pair is `circuit -> flowchart` at 0.533 of the class. **Only 30 pages (2.2%) are missed by all three models, and 19 are circuits** - 47.5% of that class against 0.0% of state machines, so the hard core is small, specific and one class. **All three are confident when wrong in different ways: kNN at exactly 1.0 on 100% of its 97 errors, the tree on 89%, logreg on 39%** - a confidence threshold works for one model and not the others. Misread circuits sit 3.77 SD lower on `contain_nested_count` (on 8 correctly-read pages, flagged as a lead). The chaos corpus's 5x error rate is confounded by holding all three minority classes, and the report says so | ✅ |

---

# Phase 6 — ANN & SVM (Unit 2)

**Goal:** learned representations for cases where handcrafted geometry collapses.

### 6.1 Image Embeddings

| # | Task | Detail | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 6.1.1 | Backbone selection | Frozen ResNet-18/34 or CLIP-ViT features on 224×224 binarized images | Embeddings cached | ❌ |
| 6.1.2 | Embedding cache | `data/features/embeddings.npy` + index | Built | ❌ |
| 6.1.3 | Dimensionality reduction | PCA to 128-D; retained-variance report | Reduced set stored | ❌ |
| 6.1.4 | Hybrid feature set | Concatenate handcrafted + embedding features | Combined table built | ❌ |

### 6.2 Multi-Layer Perceptron

| # | Task | Detail | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 6.2.1 | Architecture search | 1–3 hidden layers, widths {64,128,256,512} | Best topology chosen | ❌ |
| 6.2.2 | Activations | ReLU vs. LeakyReLU vs. GELU vs. tanh | Comparison table | ❌ |
| 6.2.3 | Regularization | Dropout {0, 0.2, 0.5}, weight decay, early stopping | Overfit gap reduced | ❌ |
| 6.2.4 | **Optimizer comparison** | SGD, SGD+momentum, RMSProp, Adam, AdamW — convergence on image data | Loss-vs-epoch overlay figure | ❌ |
| 6.2.5 | LR schedules | Constant, step, cosine, one-cycle | Best schedule chosen | ❌ |
| 6.2.6 | Batch-size study | {16,32,64,128} effect on convergence and generalization | Table produced | ❌ |
| 6.2.7 | Backprop write-up | Manual gradient derivation for one layer (viva-ready) | `docs/backprop_derivation.md` | ❌ |

### 6.3 Support Vector Machines

| # | Task | Detail | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 6.3.1 | Linear SVM | C sweep, hinge loss | Baseline recorded | ❌ |
| 6.3.2 | **Polynomial kernel** | degree {2,3,4}, coef0, γ — target the flowchart vs. state-diagram boundary | Focused binary study done | ❌ |
| 6.3.3 | RBF kernel | γ and C grid | Compared | ❌ |
| 6.3.4 | Multiclass strategy | OvO vs. OvR comparison | Chosen and justified | ❌ |
| 6.3.5 | Support vector analysis | Count, margin width, which images become SVs (expect ambiguous ones) | Gallery of SV images | ❌ |
| 6.3.6 | Kernel decision surface | 2-D projected decision surfaces per kernel | Figure produced | ❌ |
| 6.3.7 | Handcrafted vs. embedding | Which feature set wins per model | Cross-table | ❌ |

---

# Phase 7 — Boosting, Bayes, HMM, GMM+EM (Unit 3)

### 7.1 Ensembles

| # | Task | Detail | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 7.1.1 | Bagging baseline | Bagged trees | Recorded | ❌ |
| 7.1.2 | Random Forest | n_estimators, max_features, depth sweep; OOB score | Tuned model | ❌ |
| 7.1.3 | AdaBoost | Stump and depth-3 base learners | Recorded | ❌ |
| 7.1.4 | Gradient Boosting / XGBoost / LightGBM | lr, depth, subsample, colsample, early stopping | Best booster chosen | ❌ |
| 7.1.5 | Feature importance | Gini, permutation, and SHAP values | SHAP summary plot | ❌ |
| 7.1.6 | **Occlusion robustness** | Simulated coffee stains, torn corners, finger over lens, partial crops | Accuracy-vs-occlusion curve | ❌ |
| 7.1.7 | Stacking | RF + SVM + MLP with logistic meta-learner | Meta-model beats best base, or documented as not | ❌ |
| 7.1.8 | Ensemble vs. single cost | Latency and memory table | Recorded | ❌ |

### 7.2 Naive Bayes

| # | Task | Detail | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 7.2.1 | Text-region feature set | Statistics of text regions only (fast, cheap) | Feature table built | ❌ |
| 7.2.2 | Gaussian NB | On continuous text statistics | Trained | ❌ |
| 7.2.3 | Multinomial NB | On discretized token/label counts (many labels ⇒ ER; few + arrows ⇒ flowchart) | Trained | ❌ |
| 7.2.4 | Independence-assumption analysis | Where it breaks and why it still works | Written analysis | ❌ |
| 7.2.5 | **Prior role in pipeline** | NB output used as prior into the downstream classifier and HMM initial distribution | Integrated and ablated | ❌ |
| 7.2.6 | Latency benchmark | Show NB is the sub-millisecond fast path | Timing table | ❌ |

### 7.3 Hidden Markov Model — Sequential Diagram Parsing

**Core idea:** reading a diagram is a sequence. Hidden states = semantic roles; observations =
detected shape + text features at each visited node. Viterbi recovers the logical flow, which
becomes the code structure.

| # | Task | Detail | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 7.3.1 | State space definition | `{start, input, process, decision, branch-true, branch-false, loop-back, output, terminal}` | Enum frozen | ❌ |
| 7.3.2 | Observation model | Discrete: quantized (shape class, text-keyword class, in/out degree) | Emission alphabet defined | ❌ |
| 7.3.3 | Sequence construction | Convert each annotated diagram into an ordered observation sequence via topological/DFS ordering | Sequence dataset built | ❌ |
| 7.3.4 | Transition matrix (supervised) | Estimated from labeled role sequences with Laplace smoothing | Matrix learned and visualized | ❌ |
| 7.3.5 | Emission matrix | Estimated from labeled shape ↔ role co-occurrence | Matrix learned | ❌ |
| 7.3.6 | Baum–Welch (unsupervised) | EM re-estimation on unlabeled sequences; compare to supervised | Log-likelihood convergence plot | ❌ |
| 7.3.7 | **Viterbi decoding** | Most likely role sequence for a new diagram | Decoder implemented + tested | ❌ |
| 7.3.8 | Forward / backward | Posterior marginals for per-node role confidence | Confidences exposed in IR | ❌ |
| 7.3.9 | Role-labeling accuracy | Per-role precision/recall vs. annotation | Macro-F1 ≥ 0.80 | ❌ |
| 7.3.10 | Ambiguity repair | Use HMM posteriors to fix broken arrows (a decision node must have ≥2 outgoing) | Repair rules implemented | ❌ |
| 7.3.11 | Ablation | Pipeline accuracy with HMM removed (bag-of-shapes baseline) | Contribution quantified | ❌ |
| 7.3.12 | Continuous-emission variant | Gaussian-emission HMM on raw geometric features | Compared with discrete | ❌ |

### 7.4 GMM + EM — Learned Shape Vocabulary

| # | Task | Detail | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 7.4.1 | Shape descriptor vector | Hu moments, vertex count, aspect, solidity, circularity, convexity defects, Fourier descriptors | Descriptor table built | ❌ |
| 7.4.2 | GMM fitting | Full / diagonal / tied covariance comparison | Best covariance type chosen | ❌ |
| 7.4.3 | Component selection | BIC/AIC sweep over K ∈ [3,15] | K selected by BIC elbow | ❌ |
| 7.4.4 | EM convergence study | Log-likelihood curves, init sensitivity (k-means++ vs. random) | Plots produced | ❌ |
| 7.4.5 | Cluster → shape naming | Map discovered components to rectangle / diamond / oval / arrow / blob | Mapping table + montage | ❌ |
| 7.4.6 | **Soft assignment usage** | Feed GMM responsibilities as soft shape evidence into HMM emissions instead of hard labels | Integrated + ablated | ❌ |
| 7.4.7 | Per-scribe adaptation | Fit GMM per drawing style; show everyone draws rectangles differently | Per-scribe covariance figure | ❌ |
| 7.4.8 | vs. template matching | Compare learned vocabulary against hardcoded shape templates | Accuracy comparison table | ❌ |

---

# Phase 8 — Unsupervised Clustering & Style Adaptation (Unit 4)

| # | Task | Detail | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 8.1 | K-means on components | Cluster detected components by visual similarity | Clusters produced | ❌ |
| 8.2 | Elbow + silhouette | Choose K rigorously | Plots + chosen K | ❌ |
| 8.3 | **Hierarchical clustering** | Ward-linkage dendrogram: shapes → {box-like, circular, pointed} → {rectangle, rounded-rect, square, …} | Dendrogram figure with cut levels | ❌ |
| 8.4 | Linkage comparison | single / complete / average / ward | Comparison table | ❌ |
| 8.5 | DBSCAN | Discover outlier and novel shapes outside the vocabulary | Outlier gallery | ❌ |
| 8.6 | **Handwriting style clustering** | Cluster scribes by stroke statistics (slant, thickness, curvature, spacing) for per-cluster OCR adaptation | ≥3 style clusters, OCR gain measured | ❌ |
| 8.7 | Style-conditioned OCR | Route text crops to the OCR head fine-tuned for that style cluster | CER improves ≥15% relative | ❌ |
| 8.8 | Cluster validity | Silhouette, Davies–Bouldin, Calinski–Harabasz | Metrics table | ❌ |
| 8.9 | Cluster stability | Bootstrap resampling agreement (ARI) | Stability score reported | ❌ |
| 8.10 | Unlabeled-set discovery | Cluster unlabeled diagrams to surface possible new diagram types | Report on findings | ❌ |

---

# Phase 9 — CNN Component Detection & Handwriting OCR (Unit 4)

**Goal:** the perception backbone — find every component, read every label.

### 9.1 Detection Backbone

| # | Task | Detail | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 9.1.1 | Detector choice | YOLOv8 / RT-DETR fine-tune vs. Faster R-CNN; justify | Choice documented | ❌ |
| 9.1.2 | Class set | Shape classes + arrowhead + text-block + UI widget classes | Classes frozen | ❌ |
| 9.1.3 | Training run | Anchors / imgsz / epochs; mixed precision on RTX 4500 | Model trained | ❌ |
| 9.1.4 | Detection metrics | mAP@0.5, mAP@0.5:0.95, per-class AP | mAP@0.5 ≥ 0.80 | ❌ |
| 9.1.5 | Small-object handling | Arrowheads are tiny — tiling / higher input resolution | Arrowhead AP improved | ❌ |
| 9.1.6 | NMS tuning | IoU threshold, class-agnostic vs. per-class | Tuned | ❌ |
| 9.1.7 | Hard-negative mining | Crossed-out elements, doodles, margin scribbles | Added to training | ❌ |

### 9.2 CNN Fundamentals Deliverable (exam-facing)

| # | Task | Detail | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 9.2.1 | Scratch CNN | Hand-built conv/pool/FC classifier for shape crops | Trained, accuracy reported | ❌ |
| 9.2.2 | **Parameter calculation** | Layer-by-layer parameter count and output-size arithmetic table | `docs/cnn_math.md` | ❌ |
| 9.2.3 | Receptive field analysis | RF computed per layer | In same doc | ❌ |
| 9.2.4 | Filter / feature-map visualization | First-layer filters + activations on a real sketch | Figures produced | ❌ |
| 9.2.5 | Ablations | Kernel size, depth, pooling type, BatchNorm on/off | Comparison table | ❌ |
| 9.2.6 | Grad-CAM | What the network looks at per shape class | Heatmap figures | ❌ |
| 9.2.7 | Transfer vs. scratch | Quantify the pretraining benefit | Table | ❌ |

### 9.3 Handwriting OCR Sub-Pipeline

| # | Task | Detail | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 9.3.1 | Text-crop extraction | Crop text regions from detected components and edges | Crop dataset built | ❌ |
| 9.3.2 | CRNN + CTC | CNN encoder → BiLSTM → CTC decoder; IAM pretrain, diagram-label fine-tune | Model trained | ❌ |
| 9.3.3 | TrOCR comparison | Fine-tune a transformer OCR baseline; compare CER/WER | Comparison table | ❌ |
| 9.3.4 | Domain lexicon biasing | Constrain decoding with a diagram vocabulary (`if`, `else`, `start`, `end`, `user_id`, `submit`, …) | CER improves | ❌ |
| 9.3.5 | Per-style adaptation hook | Consume Phase 8 style clusters | Integrated | ❌ |
| 9.3.6 | Metrics | CER, WER overall and per style cluster | CER ≤ 0.15 | ❌ |
| 9.3.7 | Confidence + fallback | Low-confidence labels flagged for user correction in the UI | Confidence surfaced in IR | ❌ |
| 9.3.8 | Crossed-out detection | Classifier for struck-through / scribbled-out elements → excluded from IR | Precision ≥ 0.85 | ❌ |

---

# Phase 10 — Graph Assembly & Semantic Structure IR

**Goal:** turn detections into a clean, validated graph — the contract between vision and codegen.

### 10.1 Assembly

| # | Task | Detail | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 10.1.1 | Node instantiation | Detections + GMM shape posterior + OCR text → nodes | Nodes built | ❌ |
| 10.1.2 | Text-to-node binding | Containment, then nearest centroid, then Hungarian assignment | Binding accuracy ≥ 0.90 | ❌ |
| 10.1.3 | Edge tracing | Follow skeleton polylines between node boundaries | Edges recovered | ❌ |
| 10.1.4 | **Broken-arrow repair** | Gap bridging within tolerance, direction continuity, endpoint snapping to nearest node port | ≥80% of broken arrows recovered | ❌ |
| 10.1.5 | Arrow direction resolution | Arrowhead evidence + flow-direction prior + HMM role constraints | Direction accuracy ≥ 0.90 | ❌ |
| 10.1.6 | Edge-label binding | Labels near edge midpoint → `yes` / `no` / condition / cardinality | Bound correctly | ❌ |
| 10.1.7 | Crossing disambiguation | Distinguish crossing lines from true junctions via curvature continuity | Test cases pass | ❌ |
| 10.1.8 | Containment / nesting | Build parent-child tree for wireframe layout | Nesting tree emitted | ❌ |

### 10.2 Validation & Normalization

| # | Task | Detail | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 10.2.1 | Type-specific validators | Flowchart: one start, ≥1 end, decisions ≥2 out. State: reachability. ER: entities have ≥1 attribute. Wireframe: no orphan widgets | Validators implemented | ❌ |
| 10.2.2 | Graph repair heuristics | Insert implicit end node, merge duplicate nodes, drop unreachable noise | Repairs logged per diagram | ❌ |
| 10.2.3 | Cycle / loop detection | Tarjan SCC → mark loops for codegen | Loops identified | ❌ |
| 10.2.4 | IR serialization | Emit validated `ir.json` plus Graphviz/Mermaid preview | Round-trip test | ❌ |
| 10.2.5 | IR diff tool | Predicted vs. ground-truth IR (node F1, edge F1, graph edit distance) | Metric implemented | ❌ |
| 10.2.6 | Confidence propagation | Aggregate detection / OCR / HMM confidences into node and edge confidence | Present in IR | ❌ |

---

# Phase 11 — Reinforcement Learning Traversal Agent (Unit 4)

**Goal:** learn to traverse ambiguous diagrams in the order that produces correct code.

### 11.1 MDP Formulation

| # | Element | Definition | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 11.1.1 | **State** | current node + visited set + detected connections + unresolved-edge flags + node role | State encoder implemented | ❌ |
| 11.1.2 | **Actions** | follow-edge-A, follow-edge-B, …, backtrack, mark-as-loop, emit-node, terminate | Action space defined | ❌ |
| 11.1.3 | **Reward** | +1 syntactically valid code; bonus for semantic correctness (code runs and passes tests); penalty for infinite loops, unreachable nodes, duplicate emission | Reward function coded | ❌ |
| 11.1.4 | Episode definition | One diagram = one episode; terminal on full coverage or step cap | Env terminates correctly | ❌ |
| 11.1.5 | Gym environment | `DiagramTraversalEnv` with `reset` / `step` / `render` | Env passes API check | ❌ |
| 11.1.6 | State abstraction | Hashing / discretization to keep the Q-table tractable | Table size bounded | ❌ |

### 11.2 Learning

| # | Task | Detail | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 11.2.1 | **Q-learning** | Tabular; α / γ sweep; ε-greedy with decay | Converged Q-table | ❌ |
| 11.2.2 | SARSA comparison | On-policy vs. off-policy behaviour on ambiguous diagrams | Comparison plot | ❌ |
| 11.2.3 | Exploration study | ε-greedy vs. softmax vs. UCB | Reward curves | ❌ |
| 11.2.4 | Reward shaping | Intermediate rewards for valid partial structure; check for unintended exploits | Shaping justified | ❌ |
| 11.2.5 | DQN extension | Function approximation for large graphs where tabular fails | Trained, compared | ❌ |
| 11.2.6 | Curriculum | Clean synthetic graphs → messy real graphs | Curriculum ablation | ❌ |
| 11.2.7 | Baseline comparison | Learned policy vs. plain topological sort vs. DFS/BFS | Win-rate table on ambiguous set | ❌ |
| 11.2.8 | Convergence diagnostics | Episode reward, TD-error, Q-value heatmaps | Plots produced | ❌ |
| 11.2.9 | Sandbox execution harness | Safe subprocess with timeout / memory cap to run generated code for the semantic reward | Sandbox hardened + tested | ❌ |
| 11.2.10 | Ablation | Pipeline quality without RL (heuristic ordering only) | Contribution quantified | ❌ |

---

# Phase 12 — LLM Fine-Tuning for Code Synthesis (Unit 4)

**Goal:** IR + traversal order + diagram type → clean, runnable code.

### 12.1 Data

| # | Task | Detail | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 12.1.1 | Pair schema | `(diagram_type, IR-graph serialization, traversal order) → target code` | Schema fixed | ❌ |
| 12.1.2 | IR serialization format | Compact deterministic text (typed node/edge listing) over raw JSON, for token efficiency | Format benchmarked | ❌ |
| 12.1.3 | Human-written targets | Reference code for all self-drawn diagrams | ≥260 pairs | ❌ |
| 12.1.4 | Synthetic pair generation | Random graphs → rendered as diagrams → reference code emitted programmatically | ≥10K pairs | ❌ |
| 12.1.5 | Sketch2Code pairs | Wireframe → HTML converted to React targets | Converted | ❌ |
| 12.1.6 | Target languages | Flowchart → Python; State machine → Python class / `transitions`; ER → SQL DDL; Wireframe → React + Tailwind; Circuit → SPICE netlist | All five generators specified | ❌ |
| 12.1.7 | Quality filter | Every target must parse or compile; reject otherwise | 100% of targets compile | ❌ |
| 12.1.8 | Train/val/test split | Held-out diagrams *and* held-out scribes | Splits built | ❌ |

### 12.2 Fine-Tuning

| # | Task | Detail | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 12.2.1 | Base model choice | 7B code model (Qwen2.5-Coder-7B / CodeLlama-7B / DeepSeek-Coder-6.7B) — benchmark zero-shot first | Base selected with evidence | ❌ |
| 12.2.2 | Quantization | 4-bit NF4 QLoRA to fit RTX 4500 Ada (24 GB) | Fits with headroom | ❌ |
| 12.2.3 | **LoRA config** | r ∈ {8,16,32,64}, α, dropout, target modules (q,k,v,o,gate,up,down) | Best config chosen | ❌ |
| 12.2.4 | Prompt template | System + diagram type + IR + traversal + output-format contract | Template frozen | ❌ |
| 12.2.5 | Training run | bf16, gradient accumulation, cosine LR, sequence packing; log VRAM and wall time | Adapter trained | ❌ |
| 12.2.6 | Hyperparameter sweep | LR {1e-4, 2e-4, 5e-5}, epochs, batch size, warmup | Sweep logged | ❌ |
| 12.2.7 | Zero-shot vs. few-shot vs. LoRA | Three-way comparison — fine-tuning must win | Table produced | ❌ |
| 12.2.8 | Adapter merging & export | Merged weights + GGUF / vLLM-ready export | Export works | ❌ |
| 12.2.9 | Inference optimization | KV cache, batching, token streaming; latency budget < 8 s | Latency measured | ❌ |

### 12.3 Code-Quality Evaluation

| # | Task | Metric | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 12.3.1 | Syntactic validity | % of generated programs that parse | ≥ 95% | ❌ |
| 12.3.2 | Executability | % that run without exceptions in the sandbox | ≥ 85% | ❌ |
| 12.3.3 | **Functional correctness** | pass@1 against per-diagram unit tests | ≥ 70% | ❌ |
| 12.3.4 | Structural fidelity | Do generated branches / states / tables match the IR? (AST-vs-graph comparison) | Metric implemented | ❌ |
| 12.3.5 | Similarity metrics | CodeBLEU / exact match / edit distance to reference | Reported | ❌ |
| 12.3.6 | React render check | Headless build and render of generated components | Renders without error | ❌ |
| 12.3.7 | SQL check | Execute DDL on SQLite/Postgres, verify FKs and cardinality | Executes cleanly | ❌ |
| 12.3.8 | Hallucination audit | Nodes invented or dropped by the model | Rate reported | ❌ |
| 12.3.9 | Repair loop | On failure, feed the error back for one repair attempt | Success-after-repair measured | ❌ |

---

# Phase 13 — End-to-End Pipeline Orchestration

| # | Task | Detail | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 13.1 | Pipeline object | `DreamScriptPipeline.run(image) -> {ir, code, artifacts}` | Single entry point works | ❌ |
| 13.2 | Stage contracts | Typed dataclasses between every stage | pydantic/mypy validated | ❌ |
| 13.3 | Routing logic | NB fast prior → ensemble classifier → type-specific parser and codegen | Routing tested per type | ❌ |
| 13.4 | Fallback chain | Handcrafted classifier fails → embedding classifier → CNN; OCR fails → placeholder identifiers | Graceful degradation demonstrated | ❌ |
| 13.5 | Confidence gating | Below threshold ⇒ ask the user to confirm rather than guess | Gate implemented | ❌ |
| 13.6 | Caching | Per-stage cache keyed on image hash + config hash | Warm-run speedup measured | ❌ |
| 13.7 | Latency budget | Target < 10 s end-to-end on GPU; per-stage timing table | Budget met or gaps documented | ❌ |
| 13.8 | Error handling | Every stage fails soft with an actionable message | No unhandled exception in fuzz test | ❌ |
| 13.9 | Batch mode CLI | `dreamscript run <dir> --out <dir>` | CLI works | ❌ |
| 13.10 | Integration tests | 25 golden images → expected IR + expected code behaviour | Suite green | ❌ |

---

# Phase 14 — Evaluation, Ablations & Error Analysis

| # | Task | Detail | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 14.1 | Master results table | Every model × every metric, one table | Built | ❌ |
| 14.2 | Stage-wise accuracy | Classification / detection / OCR / parsing / codegen, isolated | Reported | ❌ |
| 14.3 | Error propagation study | How a stage-1 error changes final code correctness | Quantified | ❌ |
| 14.4 | **Full ablation matrix** | Remove HMM / RL / GMM / LoRA / style-adaptation one at a time | 5-row ablation table | ❌ |
| 14.5 | Robustness suite | Blur, rotation, lighting, occlusion, resolution sweeps | Degradation curves | ❌ |
| 14.6 | Cross-scribe generalization | Leave-one-scribe-out evaluation | Per-scribe scores | ❌ |
| 14.7 | Unseen diagram type | Behaviour on a type never trained on (out of distribution) | Documented failure mode | ❌ |
| 14.8 | Human baseline | Time for a human to hand-write the same code vs. DreamScript | Speedup number for the pitch | ❌ |
| 14.9 | Qualitative gallery | Best and worst cases, sketch → code side by side | `reports/gallery.md` | ❌ |
| 14.10 | Statistical rigour | Confidence intervals, seed variance across ≥3 seeds | Error bars on all headline numbers | ❌ |
| 14.11 | Compute accounting | GPU-hours, energy, cost per phase | Table produced | ❌ |

---

# Phase 15 — MLOps

| # | Task | Detail | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 15.1 | Experiment tracking | MLflow (or W&B) across detection / classification / parsing / generation | Every run logged with params, metrics, artifacts | ❌ |
| 15.2 | Data versioning | DVC remotes for raw / interim / processed | `dvc repro` rebuilds the pipeline | ❌ |
| 15.3 | Model registry | Staged versions (dev → staging → prod) per component | Registry populated | ❌ |
| 15.4 | Pipeline DAG | DVC stages or a Prefect flow for the full pipeline | DAG runs end to end | ❌ |
| 15.5 | **Drift detection** | Monitor input feature distributions (PSI / KS test) — new drawing styles, new diagram types, new capture devices | Drift alarm fires on a synthetic shift | ❌ |
| 15.6 | Prediction monitoring | Log confidence distributions, flag low-confidence spikes | Dashboard live | ❌ |
| 15.7 | Feedback loop | User corrections in the app captured as new labeled data | Correction store implemented | ❌ |
| 15.8 | Retraining trigger | Threshold rule for when to retrain a component | Documented + scripted | ❌ |
| 15.9 | CI | GitHub Actions: lint, tests, schema validation, tiny-model smoke train | CI green on PR | ❌ |
| 15.10 | Containerization | Dockerfile (CUDA base) + docker-compose for app and model server | `docker compose up` serves the app | ❌ |
| 15.11 | Model serving | FastAPI inference service, separate from the UI | `/predict` endpoint documented | ❌ |
| 15.12 | Reproducibility check | Fresh clone → `make all` → same headline metrics | Verified on a clean machine | ❌ |

---

# Phase 16 — Web Application & Live Camera Demo

### 16.1 Backend

| # | Task | Detail | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 16.1.1 | FastAPI app | `/upload`, `/predict`, `/ir`, `/code`, `/feedback` | Endpoints live | ❌ |
| 16.1.2 | Streaming progress | Server-sent events for stage-by-stage progress | UI shows live stages | ❌ |
| 16.1.3 | Sandbox runner | Execute generated Python/SQL safely in a container with a timeout | Hardened | ❌ |
| 16.1.4 | Rate and size limits | Basic abuse protection | Implemented | ❌ |

### 16.2 Frontend

| # | Task | Detail | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 16.2.1 | Upload + drag-drop | Image upload path | Works | ❌ |
| 16.2.2 | **Live webcam capture** | Hold paper to camera, auto-detect page, capture | Works on a laptop webcam | ❌ |
| 16.2.3 | Detection overlay | Draw detected boxes / edges / labels over the photo | Overlay renders | ❌ |
| 16.2.4 | IR graph view | Interactive rendered graph (Mermaid / Cytoscape) | Renders | ❌ |
| 16.2.5 | Code panel | Syntax-highlighted output + copy + download | Works | ❌ |
| 16.2.6 | **Live execution panel** | Python → run and show output; React → render in a sandboxed iframe; SQL → run and show tables | All three demo paths live | ❌ |
| 16.2.7 | Correction UI | Fix a wrong label, redraw an edge, change a node type, then regenerate | Round trip works | ❌ |
| 16.2.8 | Confidence display | Low-confidence items visually flagged | Visible | ❌ |
| 16.2.9 | Example gallery | Preloaded sketches for an instant demo without a camera | Gallery present | ❌ |
| 16.2.10 | Failure-safe demo mode | Cached results if the GPU or model is unavailable during a live presentation | Fallback tested | ❌ |

---

# Phase 17 — Documentation, Report & Demo Choreography

| # | Task | Detail | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 17.1 | Architecture diagram | Full pipeline figure | Figure in README | ❌ |
| 17.2 | Syllabus mapping doc | Explicit Unit 1–4 → phase / file / figure mapping | `docs/syllabus_map.md` | ❌ |
| 17.3 | Per-unit notebooks | One clean notebook per unit with results and plots | 4 notebooks | ❌ |
| 17.4 | Model cards | One per shipped model | Complete | ❌ |
| 17.5 | API docs | Endpoint + IR schema reference | Published | ❌ |
| 17.6 | Final report | Problem, novelty, method, experiments, results, ablations, limitations, future work | Written | ❌ |
| 17.7 | Limitations section | Honest failure modes: 3-D sketches, dense circuits, non-English labels | Written | ❌ |
| 17.8 | Demo script | 3-minute run of show: draw → snap → run, with a deliberately messy sketch | Rehearsed | ❌ |
| 17.9 | Demo video | Recorded fallback for a failed live demo | Recorded | ❌ |
| 17.10 | Viva prep | Q&A on HMM math, SVM kernels, CNN parameters, the Q-learning update, LoRA rank | `docs/viva.md` | ❌ |

---

## Cross-Phase Dependency Map

| Phase | Depends On | Unlocks |
| :--- | :--- | :--- |
| 0 | — | everything |
| 1 | 0 | 2, 3 |
| 2 | 1 | 4, 9, 10, 12 |
| 3 | 1 | 4, 9 |
| 4 | 2, 3 | 5, 6, 7 |
| 5 | 4 | 13, 14 |
| 6 | 4 | 13, 14 |
| 7 | 3, 4 | 10, 13 |
| 8 | 3, 9 | 9.3, 13 |
| 9 | 2, 3 | 10 |
| 10 | 7, 8, 9 | 11, 12 |
| 11 | 10 | 12 |
| 12 | 10, 11 | 13 |
| 13 | 5–12 | 14, 16 |
| 14 | 13 | 17 |
| 15 | 13 | 16, 17 |
| 16 | 13, 15 | 17 |
| 17 | 14, 16 | delivery |

---

## Syllabus Coverage Matrix

| Unit | Requirement | Where It Lives | Status |
| :--- | :--- | :--- | :---: |
| 1 | Decision Tree, KNN, Logistic Regression | Phase 5.1 | ❌ |
| 1 | Precision / Recall / AUC, cross-validation | Phase 5.2 | ❌ |
| 1 | KNN decision boundaries | Phase 5.3.1 | ❌ |
| 2 | ANN / MLP on image embeddings | Phase 6.2 | ❌ |
| 2 | SVM with polynomial kernel | Phase 6.3.2 | ❌ |
| 2 | Optimizer convergence comparison | Phase 6.2.4 | ❌ |
| 3 | Random Forest / Gradient Boosting | Phase 7.1 | ❌ |
| 3 | Naive Bayes | Phase 7.2 | ❌ |
| 3 | HMM + Viterbi | Phase 7.3 | ❌ |
| 3 | GMM + EM | Phase 7.4 | ❌ |
| 4 | K-means / Hierarchical clustering | Phase 8 | ❌ |
| 4 | CNN + parameter calculation | Phase 9 | ❌ |
| 4 | Reinforcement Learning (Q-learning) | Phase 11 | ❌ |
| 4 | LLM fine-tuning (LoRA) | Phase 12 | ❌ |
| 4 | MLOps | Phase 15 | ❌ |

---

## Headline Success Criteria

| # | Criterion | Target | Status |
| :---: | :--- | :--- | :---: |
| S1 | Diagram-type classification accuracy (held-out scribes) | ≥ 92% | ❌ |
| S2 | Component detection mAP@0.5 | ≥ 0.80 | ❌ |
| S3 | Label OCR character error rate | ≤ 0.15 | ❌ |
| S4 | HMM semantic-role macro-F1 | ≥ 0.80 | ❌ |
| S5 | Graph edit distance to ground-truth IR | ≤ 3 edits (median) | ❌ |
| S6 | Generated code executability | ≥ 85% | ❌ |
| S7 | Functional correctness (pass@1) | ≥ 70% | ❌ |
| S8 | End-to-end latency | < 10 s | ❌ |
| S9 | Live webcam demo succeeds on a first-try messy sketch | Yes | ❌ |
| S10 | All four syllabus units demonstrably covered | Yes | ❌ |

---

## Overall Progress

| Phase | Tasks | Done | Status |
| :--- | :---: | :---: | :---: |
| 0 — Foundations | 18 | 18 | ✅ |
| 1 — Data Acquisition | 21 | 21 | ✅ |
| 2 — Annotation Schema | 12 | 12 | ✅ |
| 3 — Preprocessing | 21 | 21 | ✅ |
| 4 — Feature Engineering | 16 | 16 | ✅ |
| 5 — Classical Classifiers | 17 | 11 | ❌ |
| 6 — ANN & SVM | 18 | 0 | ❌ |
| 7 — Boosting / Bayes / HMM / GMM | 34 | 0 | ❌ |
| 8 — Clustering | 10 | 0 | ❌ |
| 9 — CNN & OCR | 22 | 0 | ❌ |
| 10 — Graph Assembly | 14 | 0 | ❌ |
| 11 — RL Traversal | 16 | 0 | ❌ |
| 12 — LLM Fine-Tuning | 26 | 0 | ❌ |
| 13 — Orchestration | 10 | 0 | ❌ |
| 14 — Evaluation | 11 | 0 | ❌ |
| 15 — MLOps | 12 | 0 | ❌ |
| 16 — Web App & Demo | 14 | 0 | ❌ |
| 17 — Documentation | 10 | 0 | ❌ |
| **Total** | **302** | **99** | ❌ |
