# Model Card — <model name>

> Phase 0.3.2 template. Copy to `docs/model_cards/<slug>.md`. A model is not "shipped" —
> not wired into the pipeline, not reported in the final report — until its card is filled.

## Identity

| Field | Value |
| :--- | :--- |
| Name | |
| Pipeline stage | classification / detection / OCR / parsing / traversal / synthesis |
| Owning phase | |
| Version | |
| Trained on (date) | |
| Training run | `experiments/<timestamp>_<run_name>/` |
| Git commit | |
| Weights location | |
| Registry stage | dev / staging / prod (Phase 15.3) |

## Intended use

- **What it is for:**
- **Where it sits in the pipeline:** what it consumes, what it emits
- **Who consumes its output:** the next stage, or a person?

### Out of scope

Uses this model is *not* validated for, and must not be deployed into.

## Architecture and training

| Field | Value |
| :--- | :--- |
| Architecture | |
| Parameters | |
| Base model / pretraining | |
| Input representation | |
| Output representation | |
| Loss | |
| Optimizer, LR, schedule | |
| Batch size, epochs | |
| Precision | fp32 / bf16 / NF4 |
| Hardware, wall time, peak VRAM | |
| Seed and determinism | |
| Config | `configs/<name>.yaml` |

## Data

| Field | Value |
| :--- | :--- |
| Training data | data cards it draws on |
| Validation data | |
| Test data | |
| Split policy | scribe-disjoint? leave-one-scribe-out? |
| Augmentation | |
| Class balance | |

## Evaluation

| Metric | Value | Split | Target (contributing.md) |
| :--- | ---: | :--- | ---: |
| | | | |

Report the headline metric **and** the slice metrics that matter: per diagram type, per
scribe, and under the Phase 14.5 robustness sweeps (blur, rotation, lighting, occlusion).

| Slice | Metric | Value |
| :--- | :--- | ---: |
| neat drafters | | |
| chaotic scribblers | | |
| adverse capture (`adverse=true`) | | |

## Comparison

| Alternative | Metric | Why this model was chosen |
| :--- | ---: | :--- |
| baseline | | |
| ablation (component removed) | | |

## Limitations and failure modes

- Where it breaks, with example images from `reports/gallery.md`
- What a failure costs downstream: does the pipeline degrade or produce wrong code silently?
- Calibration: are its confidences trustworthy enough for the Phase 13.5 gate?

## Ethical considerations

- Bias inherited from the training data (writer styles, language, domain)
- Consequences of a confident wrong answer in the generated code
- Whether a human reviews the output before it is executed

## Maintenance

| Field | Value |
| :--- | :--- |
| Drift signals watched | Phase 15.5 |
| Retraining trigger | Phase 15.8 |
| Owner | |
| Last reviewed | |
