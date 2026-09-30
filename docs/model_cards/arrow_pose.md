# Model Card — Arrow pose model: YOLOv8m-pose (tail/head keypoints)

Phase 17.4. Every number is copied from `reports/master_results.md` or from the `contributing.md`
row named below, and those record the run that produced it.

## Identity

| Field | Value |
| :--- | :--- |
| Pipeline stage | detection → assembly |
| Owning phase | 10.1.6 (`src/detect/arrows.py`) |
| Weights | `experiments/detect/arrows/pose_m/` (DVC); base `models/pretrained/yolov8m-pose.pt` |
| Registry stage | part of `s5_test_trocr_large` (assembly) (`reports/model_registry.md`) |

## Intended use

- **What it is for:** Predicts each arrow's tail and head points, so an edge can be attached to its source and target nodes with a direction.
- **Where it sits:** Inside `assemble`, alongside the tracer.

### Out of scope

Anything that is not a photographed or scanned hand-drawn 2-D diagram of a supported type.
That includes 3-D sketches, printed or rendered diagrams, and non-English labels.
See `docs/report.md` § Limitations.

## Architecture and training

YOLOv8m-pose with two keypoints per arrow.

## Data

Arrow annotations from the detector corpus. Data cards: `docs/data_cards/`.

## Evaluation

| Metric | Value | Split | Target |
| :--- | ---: | :--- | ---: |
| S5 median GED with this arrow model | 13 | test | ≤ 3 (whole assembly) |
| S5 median GED with yolov8s arrows | 14.5 | test | — |

## Comparison

| Alternative | Result | Why this model was chosen |
| :--- | :--- | :--- |
| yolov8s arrows | median GED 14.5 | the larger pose model is 1.5 edits better at the median |

## Limitations and failure modes

- Assembly as a whole misses S5 on test (median 13 against 3). The arrow model is one cause among several, so it is not the only thing to fix.

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
