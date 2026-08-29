# Data Card — DIDI (Digital Ink Diagram data)

Verified by `python -m src.ingest.datasets.didi` on 2026-08-29. Counts measured locally.

## Identity

| Field | Value |
| :--- | :--- |
| Name | DIDI — Digital Ink Diagram data |
| Version / release | `diagrams_20200131` |
| Source URL | https://github.com/google-research/google-research/tree/master/didi_dataset |
| Data host | `https://storage.googleapis.com/digital_ink_diagram_data/` |
| Paper / citation | Gervais, Deselaers, Aksan, Hilliges — *The DIDI dataset: Digital Ink Diagram data* (arXiv:2002.09303) |
| Retrieved on | 2026-08-29 |
| Local path | `data/raw/didi/` |
| DVC-tracked | yes |

## License and permissions

| Field | Value |
| :--- | :--- |
| License | **CC BY 4.0**, licensed by Google LLC |
| Redistribution allowed | yes, with attribution |
| Commercial use allowed | yes |
| Attribution required | yes — cite the arXiv paper |
| Restrictions carried into DreamScript | none beyond attribution |

## Contents (measured)

| Field | Value |
| :--- | :--- |
| Number of items | **22,287 diagrams** |
| Modality | **digital ink** — stroke sequences with per-point coordinates, not raster images |
| Mean strokes per diagram | 36.7 (over a 5,000-record sample) |
| Mean points per diagram | 2,455 |
| File formats | NDJSON (one diagram per line); prompts as `.dot`, `.xdot`, `.png` |
| Total size on disk | 1,399,638,205 bytes (~1.3 GB) |
| Annotation types | structural: each record's `label_id` names the Graphviz prompt the writer reproduced |
| Prompt files | **6,555 `.dot` + 6,555 `.png`** |
| Splits provided by the source | yes — `split` field per record (in-sample: 3,747 train / 652 valid / 601 test per 5,000) |

Record shape:

```json
{"key": "000076d733f34f13",
 "label_id": "40c2d065...",           // -> prompts/dot/<label_id>.dot
 "split": "train",
 "writing_guide": {"width": 2400, "height": 1432},
 "drawing": [[[x...], [y...], [t...]], ...]}   // one entry per stroke
```

## Provenance

- **Who created it, and how?** A prompted collection effort: writers were shown a rendered
  Graphviz diagram and asked to redraw it on a touch device, so the ink is captured natively
  rather than photographed.
- **Quality control by the source:** every drawing is tied to a known prompt graph, so the
  intended structure is unambiguous.

## Fit for DreamScript — why this dataset is uniquely valuable

**DIDI is the only source in the corpus that records drawing *order*.** Every other dataset
gives a finished image. Phase 7.3's premise — that diagrams are produced and read as a
sequence — can be *measured* here rather than assumed:

| What it enables | Phase |
| :--- | :--- |
| Empirical stroke-order prior (are nodes really drawn before edges?) | 7.3.3 |
| Ground-truth node/edge structure from the `.dot` prompt, free of annotation cost | 7.3.4, 5 |
| Rendering ink to raster at any resolution/thickness — cheap augmentation | 1.3.6 |
| A large, clean class for diagram-type classification | 5 |

| Field | Value |
| :--- | :--- |
| Diagram types covered | graph-structured diagrams (map onto flowchart and state-machine classes) |
| Maps to the IR how? | `.dot` prompt → nodes and edges directly; ink → rendered image |
| Conversion script | `src/ingest/converters/didi.py` (Phase 2.2.2) |
| Known conversion losses | The prompt gives the *intended* graph, not what the writer actually drew — a sloppy or incomplete drawing still carries a perfect label |

## Bias and representativeness

- **Drawing styles:** many writers, but drawn on **touchscreens with a stylus or finger**, not
  on paper. Stroke dynamics differ from pen-on-paper, and there is no paper texture, shadow,
  or camera distortion.
- **Capture conditions:** none — this is synthetic-clean ink. It cannot exercise Phase 3's
  preprocessing at all.
- **Language of labels:** the prompts are graph structures; text content is minimal.
- **Domain skew:** Graphviz-style node-and-edge graphs; no UI wireframes, ER diagrams or
  circuits.
- **What this dataset will make the model bad at:** nothing directly, but it will *flatter*
  any preprocessing metric it is included in. It is deliberately excluded from the Phase 3
  robustness evaluation for that reason.

## Known problems

- **Label leakage risk:** the prompt graph is the label, and it describes the target, not the
  drawing. A writer who omitted an edge still carries the full-edge label. Anything using DIDI
  for edge supervision must treat its labels as *intent*, not observation — hdBPMN is the
  source for observed edges.
- 1.2 GB single NDJSON file: it is streamed line by line, never loaded whole.
- Prompt count (6,555) is smaller than the diagram count (22,287): prompts are reused across
  writers, which is what makes per-prompt style comparison possible.

## Ethical and privacy notes

Digital ink from anonymous contributors; no personal content, no identifiable metadata.

## Changelog

| Date | Change |
| :--- | :--- |
| 2026-08-29 | initial card; 22,287 diagrams, 6,555 prompts, stroke data verified locally |
