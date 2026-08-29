# Data Card — DreamScript Chaos Corpus

Worked example of the template, for the corpus this project collects itself (Phase 1.2).
Counts are targets until collection runs; every value marked *(target)* becomes a measured
number when `data/manifest.parquet` is built.

## Identity

| Field | Value |
| :--- | :--- |
| Name | DreamScript Chaos Corpus |
| Version / release | v0 (not yet collected) |
| Source URL | n/a — collected by this project |
| Paper / citation | n/a |
| Retrieved on | n/a |
| Local path | `data/raw/chaos/` |
| DVC-tracked | yes |

## License and permissions

| Field | Value |
| :--- | :--- |
| License | project-internal; contributors consent to research use |
| Redistribution allowed | no — photographs of identifiable handwriting stay local |
| Commercial use allowed | no |
| Attribution required | n/a |
| Restrictions carried into DreamScript | derived features, metrics and models may be published; raw photos may not |

## Contents

| Field | Value |
| :--- | :--- |
| Number of items | 260 *(target: 60 flowchart, 60 wireframe, 50 state, 50 ER, 40 circuit)* |
| Modality | photographs of paper and whiteboard sketches |
| Resolution or size range | phone and webcam captures, 1–12 MP |
| File formats | JPEG, PNG |
| Total size on disk | ~1.5 GB *(target)* |
| Annotation types | shapes, edges, labels, semantic roles, plus target code per diagram |
| Annotation format | DreamScript IR (`schemas/ir.schema.json`) |
| Splits provided by the source | none — splits are scribe-disjoint, built in Phase 1.3.3 |

## Provenance

- **Who created it, and how?** Drawn by ≥8 people (Phase 1.2.6) across pencil, ballpoint,
  marker, whiteboard and tablet stylus, then photographed under deliberately varied
  conditions: shadows, oblique angles, ruled paper, whiteboard glare, coffee stains and
  crossed-out elements (Phase 1.2.7).
- **What were annotators asked to do?** Draw a named diagram naturally, without neatening;
  the point is that the images are messy.
- **What quality control did the source apply?** Per-image validator (Phase 2.2.5) plus
  double-labelling of 50 images for inter-annotator agreement (Phase 2.2.4).

## Fit for DreamScript

| Field | Value |
| :--- | :--- |
| Diagram types covered | all five |
| Phases that consume it | 3–14; it is the only source with paired target code for every type |
| Maps to the IR how? | annotated directly in IR form, so no lossy conversion |
| Conversion script | n/a — native |
| Known conversion losses | none by construction |

## Bias and representativeness

- **Drawing styles represented:** ≥8 writers *(target)*, deliberately spanning neat drafters
  and chaotic scribblers — the split axis for the Phase 5.2.2 grouped cross-validation.
- **Capture conditions:** phone and webcam; no flatbed scans, which the public datasets cover.
- **Language of labels:** English only. The model will not read other scripts.
- **Domain skew:** textbook-style examples (login forms, traffic lights, blog schemas), not
  production diagrams; expect degradation on dense real-world diagrams.
- **What this dataset will make the model bad at:** very large diagrams (>20 nodes),
  non-English labels, colour-coded semantics, and 3-D or isometric sketches.

## Known problems

- Small: 260 images cannot carry a detector on its own, so it is used with the public sets
  and the synthetic generator (Phase 1.3.7).
- Self-drawn diagrams risk an author-style monoculture; the ≥8-scribe requirement exists
  specifically to bound that, and leave-one-scribe-out evaluation (Phase 14.6) measures it.

## Ethical and privacy notes

- Handwriting is biometric-adjacent and identifiable. Contributors are told the images stay
  local, are used only for this project, and can be withdrawn on request.
- No personal content is to be drawn: no names, addresses, or real credentials in labels.
- Faces, hands and surroundings are cropped out during the page-detection step (Phase 3.1.2).

## Changelog

| Date | Change |
| :--- | :--- |
| 2026-08-29 | initial card written alongside the Phase 0.3 template |
