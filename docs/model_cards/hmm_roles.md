# Model Card — Role decoder: HMM + per-component Viterbi

Phase 17.4. Every number is copied from `reports/master_results.md` or from the `contributing.md`
row named below, and those record the run that produced it.

## Identity

| Field | Value |
| :--- | :--- |
| Pipeline stage | parsing |
| Owning phase | 7.3, S4 (`src/parse/viterbi.py`, `src/parse/s4.py`) |
| Weights | fitted parameters from `python -m src.parse.s4`; transition and emission tables |
| Registry stage | `prod` (`s4_role_labelling`) (`reports/model_registry.md`) |

## Intended use

- **What it is for:** Assigns a semantic role (start, process, decision, terminal, input, output, …) to each node along the reading order.
- **Where it sits:** Inside `assemble`. The roles decide which code construct each node becomes.

### Out of scope

Anything that is not a photographed or scanned hand-drawn 2-D diagram of a supported type.
That includes 3-D sketches, printed or rendered diagrams, and non-English labels.
See `docs/report.md` § Limitations.

## Architecture and training

Discrete HMM. Hidden states are roles and observations are shape/text features. Baum-Welch fit, Viterbi decoding per connected component.

## Data

993 sequences, 14,056 nodes, 10 seeds. Data cards: `docs/data_cards/`.

## Evaluation

| Metric | Value | Split | Target |
| :--- | ---: | :--- | ---: |
| macro F1 | 0.8003 | held-out, 10 seeds | ≥ 0.80 |
| macro F1 std | 0.0010 | — | — |
| accuracy | 0.7873 | — | — |

## Comparison

| Alternative | Result | Why this model was chosen |
| :--- | :--- | :--- |
| 7.3.9 five-fold evaluation | macro F1 0.7763 | the S4 protocol with per-component decoding is the one that clears 0.80 |

## Limitations and failure modes

- **It meets the target but does not beat it.** The margin (0.0003) is below the seed spread (0.0010), and 6 of 10 seeds clear 0.80.
- `branch-false` is the weakest role. In 7.3.9, dropping it alone moved the macro score from 0.7763 to 0.804.

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
