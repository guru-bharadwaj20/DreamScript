# Model Card — S1 diagram-type classifier: CLIP ViT-B/32 + one-vs-one RBF SVM

Phase 17.4. Every number is copied from `reports/master_results.md` or from the `contributing.md`
row named below, and those record the run that produced it.

## Identity

| Field | Value |
| :--- | :--- |
| Pipeline stage | classification |
| Owning phase | 5–6, S1 (`src/classify/s1.py`) |
| Weights | fitted by `python -m src.classify.s1`; embeddings from `src/embed/` |
| Registry stage | `prod` (`s1_clip_svm`) (`reports/model_registry.md`) |

## Intended use

- **What it is for:** Assigns one of five diagram types (flowchart, state machine, ER, wireframe, circuit) to a whole page. This is the model S1 grades.
- **Where it sits:** Reads a page image and returns a type with a probability. The served pipeline routes with the NB prior (`nb_router.md`), which reuses the detector's output at no extra cost. This model is the stronger offline classifier and the one the S1 number refers to.

### Out of scope

Anything that is not a photographed or scanned hand-drawn 2-D diagram of a supported type.
That includes 3-D sketches, printed or rendered diagrams, and non-English labels.
See `docs/report.md` § Limitations.

## Architecture and training

CLIP ViT-B/32 image embedding (512-d, frozen) → PCA to 128 components → one-vs-one RBF SVM.

## Data

1,340 real pages, repeated 5-fold CV grouped by scribe (740 rows from 170 known scribes held out by writer). Data cards: `docs/data_cards/`.

## Evaluation

| Metric | Value | Split | Target |
| :--- | ---: | :--- | ---: |
| accuracy (held-out scribes) | 0.9871 | 5-fold scribe-grouped CV | ≥ 0.92 |
| accuracy std / worst repeat | 0.0009 / 0.9858 | 3 repeats | — |
| macro F1 | 0.9518 | same | — |

## Comparison

| Alternative | Result | Why this model was chosen |
| :--- | :--- | :--- |
| handcrafted logreg (5.1) | macro F1 0.7917 | 33 handcrafted features are not enough to separate wireframe from ER |

## Limitations and failure modes

- **The corpus is source-confounded.** hdbpmn is 600/600 flowchart and sketch2code is 600/600 wireframe, so source alone fixes the label for 1,200 of 1,340 pages. Part of the 0.9871 is recognising the source.
- The PCA was fitted on all 1,340 rows. Refitting it inside each fold costs 0.0053 (0.9818), which still clears the target.
- A confident wrong type sends the page to the wrong code generator. That is why the pipeline has a 0.60 confirmation gate (13.5).

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
