# Model Card — Code synthesizer: Qwen2.5-Coder-7B-Instruct + QLoRA r=64

Phase 17.4. Every number is copied from `reports/master_results.md` or from the `contributing.md`
row named below, and those record the run that produced it.

## Identity

| Field | Value |
| :--- | :--- |
| Pipeline stage | synthesis |
| Owning phase | 12 (`src/llm/train.py`, `src/llm/score.py`) |
| Weights | adapter under `experiments/llm/` (DVC); base `Qwen/Qwen2.5-Coder-7B-Instruct`; GGUF Q4_K_M export (12.2.9) |
| Registry stage | `prod` (`lora_r64_quality`) (`reports/model_registry.md`) |

## Intended use

- **What it is for:** Turns the serialised IR (12.1.2) into code: Python for flowcharts and state machines.
- **Where it sits:** The `generate` stage. When no model is being served, 12.1.6's deterministic emitter answers instead, and the result says which one did.

### Out of scope

Anything that is not a photographed or scanned hand-drawn 2-D diagram of a supported type.
That includes 3-D sketches, printed or rendered diagrams, and non-English labels.
See `docs/report.md` § Limitations.

## Architecture and training

Qwen2.5-Coder-7B-Instruct in 4-bit NF4 with bf16 compute; LoRA r=64 on all seven projections (q, k, v, o, gate, up, down).

## Data

IR→code pairs from `src/llm/pairs.py`; 176 validation pairs, 162 test diagrams; 1.2M training tokens, LR 2e-4 cosine. Data cards: `docs/data_cards/`.

## Evaluation

| Metric | Value | Split | Target |
| :--- | ---: | :--- | ---: |
| executes in sandbox (S6) | 99.4% | 162 test | ≥ 85% |
| functional pass@1 (S7) | 70.37% | 162 test | ≥ 70% |
| pass@1 fa_bresler / hdbpmn | 66.67% / 71.93% | test | — |

## Comparison

| Alternative | Result | Why this model was chosen |
| :--- | :--- | :--- |
| zero-shot Qwen | executes 40.7% | the base model drops nodes and breaks the output contract |
| few-shot | executes 93.8% | runs, but is less often correct |
| r = 8 / 16 / 32 | val loss 0.0862 / 0.0681 / 0.0527 | r = 64 reached 0.0429 at the same cost in time |

## Limitations and failure modes

- **Measured from IR, not from a photograph.** Its input is only as good as assembly (S5), and assembly misses S5 on test.
- Served through llama.cpp Q4_K_M, p90 latency is 10.87 s, which is outside the 10 s budget. S8 was measured with the emitter answering.
- The generated code runs in the 11.2.9 sandbox from the app (`POST /run/{id}`), never on the user's host.

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
