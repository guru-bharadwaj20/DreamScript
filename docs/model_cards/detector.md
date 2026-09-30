# Model Card — Component detector: YOLOv8n

Phase 17.4. Every number is copied from `reports/master_results.md` or from the `contributing.md`
row named below, and those record the run that produced it.

## Identity

| Field | Value |
| :--- | :--- |
| Pipeline stage | detection |
| Owning phase | 9.1, S2 (`src/detect/train.py`, `src/detect/s2.py`) |
| Weights | `experiments/detect/final/weights/best.pt` (DVC); base `models/pretrained/yolov8n.pt` |
| Registry stage | `prod` (`s2_component_detector`) (`reports/model_registry.md`) |

## Intended use

- **What it is for:** Finds shapes, arrowheads and text regions on a diagram photograph and returns boxes with class labels.
- **Where it sits:** The first model stage. Every later stage consumes its boxes.

### Out of scope

Anything that is not a photographed or scanned hand-drawn 2-D diagram of a supported type.
That includes 3-D sketches, printed or rendered diagrams, and non-English labels.
See `docs/report.md` § Limitations.

## Architecture and training

YOLOv8n, anchor-free, fine-tuned at 1280 px for 40 epochs with mixed precision.

## Data

Detector split from 9.1.2 (hdbpmn, flowchartseg, fa_bresler exports); 308 held-out validation pages. Data cards: `docs/data_cards/`.

## Evaluation

| Metric | Value | Split | Target |
| :--- | ---: | :--- | ---: |
| mAP@0.5 | 0.9107 | val, 308 pages | ≥ 0.80 |
| mAP@0.5, hand-drawn hdbpmn slice | 0.8787 | val | — |
| arrowhead AP@0.5 | 0.4318 | val | — |

## Comparison

| Alternative | Result | Why this model was chosen |
| :--- | :--- | :--- |
| imgsz 640 → 896 → 1280 (10 ep) | 0.7972 → 0.8683 → 0.8729 | resolution is almost entirely worth it for the arrowhead class |
| RT-DETR (9.1.1) | arrowhead AP 0.4863 | yolov8n at 40 epochs reaches 0.5197 without 10.6x the parameters |

## Limitations and failure modes

- **Arrowheads are the weak class** (AP 0.43). 91% of them fit inside a single P4 cell, so they live on one feature level. Missed arrowheads become wrong edge directions in the graph (S5).
- Trained only on flowchart and state-machine data. On ER diagrams, wireframes and circuits it has never seen the shapes.

## Ethical considerations

- The training writers are a narrow, mostly academic population, and their handwriting is in English.
- A confident wrong answer becomes code that runs and does the wrong thing. The pipeline shows
  per-stage confidence and asks for confirmation below its gate. The app never runs code outside
  the sandbox.

## Maintenance

| Field | Value |
| :--- | :--- |
| Drift signals | PSI / KS on the feature table (15.5, `reports/drift.md`) |
| Retraining trigger | 15.8 (`reports/retraining_trigger.md`) |
| Owner | Guru Bharadwaj |
| Last reviewed | 2026-09-30 |
