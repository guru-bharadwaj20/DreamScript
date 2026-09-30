# Model Card — Pipeline router: multinomial Naive Bayes over detected class counts

Phase 17.4. Every number is copied from `reports/master_results.md` or from the `contributing.md`
row named below, and those record the run that produced it.

## Identity

| Field | Value |
| :--- | :--- |
| Pipeline stage | classification (served) |
| Owning phase | 7.2, 13.3 (`src/pipeline/routing.py`) |
| Weights | fitted from the detector's val-split output; no separate weights file |
| Registry stage | — (part of the pipeline, not registered separately) (`reports/model_registry.md`) |

## Intended use

- **What it is for:** Picks the diagram type inside the served pipeline, from the histogram of classes the detector has already found.
- **Where it sits:** Between `detect` and `assemble`. If its probability is below 0.60 the pipeline stops with `needs_confirmation` and emits no code.

### Out of scope

Anything that is not a photographed or scanned hand-drawn 2-D diagram of a supported type.
That includes 3-D sketches, printed or rendered diagrams, and non-English labels.
See `docs/report.md` § Limitations.

## Architecture and training

Multinomial NB over per-page counts of detector classes (arrowhead, rounded-rect, diamond, circle, double-circle, …).

## Data

Fitted on the detector's val split, evaluated on test. Both are disjoint from the detector's training pages. Data cards: `docs/data_cards/`.

## Evaluation

| Metric | Value | Split | Target |
| :--- | ---: | :--- | ---: |
| test accuracy | 1.0000 (162/162) | test | — |

## Comparison

| Alternative | Result | Why this model was chosen |
| :--- | :--- | :--- |
| S1 CLIP SVM | 0.9871 on 5 types | the router adds no model and no latency, and on the two types with detector data it is perfect |

## Limitations and failure modes

- **It covers two of five types.** Only flowchart and state machine have annotated detector data, so ER diagrams, wireframes and circuits cannot be routed from a photograph.
- The 1.0000 is high because the question is easy: a flowchart page averages 69.97 arrowheads and no double-circles, while a state-machine page averages 1.56 double-circles.

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
