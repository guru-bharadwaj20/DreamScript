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
| 1.2.1 | Draw flowcharts | 60 sketches over 10 scenarios × 8 scribes; brief in `docs/collection/flowchart.md` | Photographed + raw stored — **tooling ✅, drawing pending** | ❌ |
| 1.2.2 | Draw wireframes | 60 sketches over 10 scenarios; brief in `docs/collection/wireframe.md` | Stored — **tooling ✅, drawing pending** | ❌ |
| 1.2.3 | Draw state machines | 50 sketches over 10 scenarios; brief in `docs/collection/state_machine.md` | Stored — **tooling ✅, drawing pending** | ❌ |
| 1.2.4 | Draw ER diagrams | 50 sketches over 10 scenarios; brief in `docs/collection/er_diagram.md` | Stored — **tooling ✅, drawing pending** | ❌ |
| 1.2.5 | Draw circuits | 40 sketches over 10 scenarios; brief in `docs/collection/circuit.md` | Stored — **tooling ✅, drawing pending** | ❌ |
| 1.2.6 | Multi-scribe collection | ≥8 people, style mix 2 neat / 4 average / 2 messy, consent recorded | ≥8 `scribe_id` values present — **registry + validation ✅, people pending** | ❌ |
| 1.2.7 | Adverse capture conditions | 8 named conditions, each mapped to the mitigation it tests; capture tool enforces the vocabulary | ≥25% of images tagged `adverse=true` — **tool + protocol ✅, photos pending** | ❌ |
| 1.2.8 | Media variety | Pencil, ballpoint, marker, whiteboard, stylus — each mapped to the failure it induces | `medium` field populated — **vocabulary enforced ✅, drawing pending** | ❌ |

### 1.3 Corpus Engineering

| # | Task | Detail | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 1.3.1 | Unified manifest | `data/processed/manifest.parquet`: id, source, path, type, scribe, medium, condition, adverse, has_structure, has_text, native_split, split | Manifest builds from raw dirs — **35,414 rows** | ✅ |
| 1.3.2 | Deduplication | 64-bit dHash + prefix-bucketed Hamming search; groups kept whole by the splitter | Duplicate report generated — **16 groups / 50 images over 1,435 on-disk files** | ✅ |
| 1.3.3 | **Scribe-disjoint splits** | Train/val/test split *by person*, not by image | No scribe appears in two splits | ❌ |
| 1.3.4 | Class balance report | Counts per diagram type per split | Histogram in `reports/` | ❌ |
| 1.3.5 | DVC / Git-LFS tracking | Data versioned, not committed raw | `dvc status` clean | ❌ |
| 1.3.6 | Augmentation policy | Rotation ±15°, perspective warp, blur, JPEG noise, brightness/contrast, ink-thickness morphology, paper-texture blend | `src/ingest/augment.py` + visual grid | ❌ |
| 1.3.7 | Synthetic diagram generator | Programmatic diagrams rendered then "hand-ified" (jitter, wobbly lines) for cheap volume | 5K synthetic images generated | ❌ |

---

# Phase 2 — Annotation Schema & Labeling Pipeline

**Goal:** one schema all five diagram types serialize into, so downstream code is type-agnostic.

### 2.1 Schema Design

| # | Task | Detail | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 2.1.1 | Node schema | `{id, shape, bbox, text, semantic_role, confidence}` | JSON Schema file validates | ❌ |
| 2.1.2 | Edge schema | `{id, src, dst, directed, label, polyline, confidence}` | Validates | ❌ |
| 2.1.3 | Diagram schema | `{diagram_type, nodes[], edges[], meta}` — the **DreamScript IR** | `schemas/ir.schema.json` | ❌ |
| 2.1.4 | Shape vocabulary | rectangle, rounded-rect, diamond, ellipse/oval, circle, parallelogram, arrow, line, text-block, freeform | Enum frozen | ❌ |
| 2.1.5 | Semantic role vocabulary | start, end, process, decision, io, state, transition, entity, attribute, relationship, container, ui-input, ui-button, ui-label, ui-image, component, wire | Enum frozen | ❌ |
| 2.1.6 | Ambiguity fields | `unresolved_edges[]`, `crossed_out[]`, `low_conf_text[]` | Present in schema | ❌ |

### 2.2 Labeling Operations

| # | Task | Detail | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 2.2.1 | Tool selection | Label Studio / CVAT configured with custom shape + relation labeling | Project loads images | ❌ |
| 2.2.2 | Converters | hdBPMN / FC / DIDI / Sketch2Code → DreamScript IR | Round-trip test passes | ❌ |
| 2.2.3 | Annotation guidelines | Written rules for ambiguous cases (broken arrow, overlapping boxes) | `docs/annotation_guide.md` | ❌ |
| 2.2.4 | Inter-annotator agreement | Double-label 50 images, report Cohen's κ on shape and role | κ ≥ 0.75 reported | ❌ |
| 2.2.5 | Label QA pass | Automated validator: dangling edges, missing roles, bbox out of frame | Validator green on corpus | ❌ |
| 2.2.6 | Target-code pairs | For each self-drawn diagram, write the correct target code (Python / React / SQL / netlist) | ≥260 pairs stored | ❌ |

---

# Phase 3 — Image Preprocessing & Geometric Primitive Extraction

**Goal:** turn a phone photo into clean binarized strokes and a set of geometric primitives.

### 3.1 Photometric & Geometric Normalization

| # | Task | Technique | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 3.1.1 | EXIF orientation fix | Auto-rotate | Unit test on rotated sample | ❌ |
| 3.1.2 | Page/board detection | Largest quadrilateral contour | Detects page in ≥90% of test photos | ❌ |
| 3.1.3 | Perspective rectification | Homography warp to top-down | Visual QA grid | ❌ |
| 3.1.4 | Illumination correction | CLAHE + background estimate via large-kernel morphological opening | Shadowed samples readable | ❌ |
| 3.1.5 | Binarization | Sauvola / adaptive threshold; compare against Otsu | Chosen method logged with F1 on stroke masks | ❌ |
| 3.1.6 | Denoise | Median filter + small-component removal | Speckle count reduced ≥80% | ❌ |
| 3.1.7 | Deskew | Hough-based dominant-angle correction | Skew < 1° after correction | ❌ |
| 3.1.8 | Ruled-paper line suppression | Directional morphology removes notebook rules, keeps strokes | Ruled samples cleaned | ❌ |
| 3.1.9 | Stroke thinning | Zhang–Suen skeletonization | Skeleton produced | ❌ |

### 3.2 Primitive Extraction

| # | Task | Technique | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 3.2.1 | Connected components | Labeling + bbox/area/solidity stats | CC table produced | ❌ |
| 3.2.2 | Contour extraction | `findContours` + hierarchy | Contours stored | ❌ |
| 3.2.3 | Polygon approximation | Douglas–Peucker; vertex counts | Vertex histogram | ❌ |
| 3.2.4 | Line segment detection | LSD / probabilistic Hough | Segment set per image | ❌ |
| 3.2.5 | Curve/corner analysis | Curvature along skeleton; corner detection | Line-vs-curve ratio computable | ❌ |
| 3.2.6 | Arrowhead detection | Convergent short-segment triplets at endpoints + template match on skeleton spurs | Precision ≥ 0.80 on labeled arrows | ❌ |
| 3.2.7 | Text region proposal | MSER + stroke-width transform, aspect/density filters | Text mask separated from shape mask | ❌ |
| 3.2.8 | Shape/text separation | Two-layer output: `shape_layer.png`, `text_layer.png` | Both emitted per image | ❌ |
| 3.2.9 | Primitive cache | Serialize primitives to `data/interim/<id>.pkl` | Cache hit path tested | ❌ |

### 3.3 Preprocessing Evaluation

| # | Task | Metric | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 3.3.1 | Stroke IoU | Binarized mask vs. hand-traced GT on 30 images | Mean IoU ≥ 0.80 | ❌ |
| 3.3.2 | Robustness sweep | Metric under blur / rotation / lighting sweeps | Degradation curves plotted | ❌ |
| 3.3.3 | Failure gallery | 20 worst preprocessing cases with diagnosis | `reports/preproc_failures.md` | ❌ |

---

# Phase 4 — Handcrafted Feature Engineering (Unit 1 basis)

**Goal:** a compact, interpretable feature vector that separates diagram types geometrically.

### 4.1 Feature Families

| # | Feature Group | Members | Rationale | Status |
| :---: | :--- | :--- | :--- | :---: |
| 4.1.1 | Structural counts | node count, edge count, arrowhead count, text-block count | Flowcharts are arrow-dense | ❌ |
| 4.1.2 | Ratios | line-to-curve ratio, arrows-per-node, text-per-node, edge/node ratio | Type-discriminative | ❌ |
| 4.1.3 | Shape mix | fraction rectangles / diamonds / ovals / circles / freeform | Diamonds ⇒ flowchart or ER | ❌ |
| 4.1.4 | Layout geometry | node density, mean nearest-neighbour distance, grid-alignment score, row/column regularity | Wireframes are grid-like | ❌ |
| 4.1.5 | Global geometry | image aspect ratio, ink coverage, bounding-box fill ratio | Cheap priors | ❌ |
| 4.1.6 | Text statistics | text-area fraction, mean label length, labels-inside-shape vs. on-edge ratio | ER diagrams are label-heavy | ❌ |
| 4.1.7 | Connectivity | mean degree, self-loop count, cycle count, connected-component count | State machines have self-loops | ❌ |
| 4.1.8 | Directionality | dominant flow axis, edge-angle histogram entropy | Flowcharts flow downward | ❌ |
| 4.1.9 | Containment | nested-box count and nesting depth | Wireframes nest; flowcharts don't | ❌ |

### 4.2 Feature Pipeline Engineering

| # | Task | Detail | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 4.2.1 | `FeatureExtractor` class | sklearn-compatible `fit` / `transform` | Unit tested | ❌ |
| 4.2.2 | Feature table build | `data/features/handcrafted.parquet` for full corpus | Table built | ❌ |
| 4.2.3 | Missing-value strategy | Median impute + missingness indicator | Documented | ❌ |
| 4.2.4 | Scaling | StandardScaler fit on train only (no leakage) | Leakage test passes | ❌ |
| 4.2.5 | Correlation pruning | Drop \|r\| > 0.95 pairs | Pruned list logged | ❌ |
| 4.2.6 | Feature importance preview | Mutual information + ANOVA F ranking | Ranked bar chart | ❌ |
| 4.2.7 | Visualization | PCA and t-SNE/UMAP of feature space coloured by diagram type | Figure in report | ❌ |

---

# Phase 5 — Classical Diagram-Type Classifiers (Unit 1)

**Goal:** full, rigorous classical-ML treatment of diagram-type classification.

### 5.1 Models

| # | Model | Configuration to explore | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 5.1.1 | Logistic Regression | L1 / L2 / elastic-net; multinomial softmax; C sweep | Best C selected by CV | ❌ |
| 5.1.2 | K-Nearest Neighbors | k ∈ {1,3,5,7,11,15}; Euclidean / Manhattan / cosine; distance weighting | k and metric selected | ❌ |
| 5.1.3 | Decision Tree | depth, min_samples_leaf, gini vs. entropy; cost-complexity pruning α sweep | Pruned tree chosen | ❌ |
| 5.1.4 | Baselines | Majority class, stratified random | Reported for context | ❌ |

### 5.2 Evaluation Protocol

| # | Task | Detail | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 5.2.1 | Stratified k-fold CV | k = 5, repeated ×3, fixed seeds | CV harness implemented | ❌ |
| 5.2.2 | **Grouped CV by scribe** | GroupKFold on `scribe_id` — the neat-drafter vs. chaotic-scribbler test | Grouped scores reported alongside plain CV | ❌ |
| 5.2.3 | Metrics | Accuracy, per-class precision/recall/F1, macro & weighted F1 | Full classification report | ❌ |
| 5.2.4 | ROC / AUC | One-vs-rest ROC curves, macro & micro AUC | Curves plotted | ❌ |
| 5.2.5 | PR curves | Especially for the minority diagram type | Curves plotted | ❌ |
| 5.2.6 | Confusion matrices | Raw and row-normalized | Heatmaps saved | ❌ |
| 5.2.7 | Calibration | Reliability diagram + Brier score; Platt/isotonic if miscalibrated | Calibration plot | ❌ |
| 5.2.8 | Statistical comparison | McNemar test between top models; CV score confidence intervals | Significance table | ❌ |
| 5.2.9 | Learning curves | Score vs. training-set size | Curves plotted | ❌ |

### 5.3 Interpretation Deliverables

| # | Task | Detail | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 5.3.1 | **KNN decision boundary plots** | 2-D projections (PCA / top-2 features) showing diagram types clustering | Figures produced | ❌ |
| 5.3.2 | Decision tree export | Rendered tree + extracted human-readable rules | `reports/tree_rules.md` | ❌ |
| 5.3.3 | Logistic coefficients | Per-class coefficient table, sign interpretation | Table in report | ❌ |
| 5.3.4 | Error taxonomy | Which diagram pairs confuse most (expect flowchart ↔ state) | Written analysis | ❌ |

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
| 1 — Data Acquisition | 21 | 0 | ❌ |
| 2 — Annotation Schema | 12 | 0 | ❌ |
| 3 — Preprocessing | 21 | 0 | ❌ |
| 4 — Feature Engineering | 16 | 0 | ❌ |
| 5 — Classical Classifiers | 17 | 0 | ❌ |
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
| **Total** | **302** | **18** | ❌ |
