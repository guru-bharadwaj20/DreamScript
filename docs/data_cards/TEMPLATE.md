# Data Card — <dataset name>

> Phase 0.3.1 template. Copy to `docs/data_cards/<slug>.md` and fill every field before the
> dataset is used in training. "Unknown" is an acceptable answer; a blank field is not.

## Identity

| Field | Value |
| :--- | :--- |
| Name | |
| Version / release | |
| Source URL | |
| Paper / citation | |
| Retrieved on | |
| Local path | `data/raw/<slug>/` |
| DVC-tracked | yes / no |

## License and permissions

| Field | Value |
| :--- | :--- |
| License | |
| Redistribution allowed | yes / no / with attribution |
| Commercial use allowed | yes / no |
| Attribution required | |
| Restrictions carried into DreamScript | |

If redistribution is not allowed, the raw files must stay out of the repository and out of
any published artifact; only derived statistics and models may be shared.

## Contents

| Field | Value |
| :--- | :--- |
| Number of items | |
| Modality | images / strokes / text / paired code |
| Resolution or size range | |
| File formats | |
| Total size on disk | |
| Annotation types | shapes / edges / labels / bounding boxes / transcripts / none |
| Annotation format | |
| Splits provided by the source | |

## Provenance

- **Who created it, and how?** (crowdsourced, lab-collected, scraped, synthetic)
- **What were annotators asked to do?**
- **What quality control did the source apply?**

## Fit for DreamScript

| Field | Value |
| :--- | :--- |
| Diagram types covered | flowchart / wireframe / state machine / ER / circuit |
| Phases that consume it | |
| Maps to the IR how? | which fields of `schemas/ir.schema.json` it can populate |
| Conversion script | `src/ingest/converters/<name>.py` |
| Known conversion losses | what the source records that the IR drops, and vice versa |

## Bias and representativeness

- **Drawing styles represented:** how many distinct writers, and how varied?
- **Capture conditions:** scanner, phone, tablet, whiteboard?
- **Language of labels:**
- **Domain skew:** e.g. business process diagrams over-represent swimlanes
- **What this dataset will make the model bad at:**

## Known problems

- Label noise, duplicates, corrupt files, ambiguous cases
- Anything filtered out during ingest, and the filter's criterion

## Ethical and privacy notes

- Does any image contain identifiable handwriting, names, or personal content?
- Consent status for self-collected data (see `docs/risks.md`)

## Changelog

| Date | Change |
| :--- | :--- |
| | initial card |
