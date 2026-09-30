# Model Card — Label OCR: TrOCR-large, fine-tuned, with learned crop selection

Phase 17.4. Every number is copied from `reports/master_results.md` or from the `contributing.md`
row named below, and those record the run that produced it.

## Identity

| Field | Value |
| :--- | :--- |
| Pipeline stage | OCR |
| Owning phase | 9.3, S3 (`src/ocr/trocr.py`, `src/ocr/s3.py`) |
| Weights | `experiments/ocr/trocr_large/` (DVC) |
| Registry stage | `staging` (`s3_trocr_large_selected`), because it misses S3 (`reports/model_registry.md`) |

## Intended use

- **What it is for:** Reads the handwritten text inside nodes and on edges.
- **Where it sits:** Inside `assemble`, reached through `src.assemble` (node text and state labels).

### Out of scope

Anything that is not a photographed or scanned hand-drawn 2-D diagram of a supported type.
That includes 3-D sketches, printed or rendered diagrams, and non-English labels.
See `docs/report.md` § Limitations.

## Architecture and training

TrOCR-large (vision encoder + text decoder), fine-tuned on label crops, plus a learned selection of which crop to read.

## Data

Label crops built by `src/ocr/labelcrops.py`, writer-disjoint train/eval. **hdbpmn only**: `labelcrops.build` loads hdbpmn alone. Data cards: `docs/data_cards/`.

## Evaluation

| Metric | Value | Split | Target |
| :--- | ---: | :--- | ---: |
| CER | 0.2006 | frozen test | ≤ 0.15 |
| exact match | 0.6656 | frozen test | — |
| WER | 0.264 | frozen test | — |

## Comparison

| Alternative | Result | Why this model was chosen |
| :--- | :--- | :--- |
| trocr_large, 28 epochs | CER 0.2222 | longer schedule, no better |
| crnn_finetune (9.3.6) | CER 0.6864 | CRNN+CTC from 9.3; the planned model, replaced |
| trocr_zero_shot | CER 1.1702 | no fine-tuning |

## Limitations and failure modes

- **Misses S3 by about 1.3x.** One label in three is not read exactly, and a wrong label becomes a wrong identifier or string in the generated code.
- **Not trained on automaton labels.** S3 scores 1.3% exact (3 of 233) on fa_bresler state labels, so state machines use a separate TrOCR fine-tune (`src/assemble/statelabels.py`).
- English only. Labels in other languages or scripts are out of scope.

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
