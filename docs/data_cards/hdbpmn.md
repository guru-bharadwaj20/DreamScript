# Data Card — hdBPMN

Verified by `python -m src.ingest.datasets.hdbpmn --check` on 2026-08-29. All counts below
are measured from the local copy, not quoted from the paper.

## Identity

| Field | Value |
| :--- | :--- |
| Name | hdBPMN (hand-drawn BPMN diagrams) |
| Version / release | GitHub `main`, shallow clone |
| Source URL | https://github.com/dwslab/hdBPMN |
| Paper / citation | Schäfer, van der Aa, Leopold, Stuckenschmidt — *Sketch2BPMN / hdBPMN* (Data & Knowledge Engineering) |
| Retrieved on | 2026-08-29 |
| Local path | `data/raw/hdbpmn/` |
| DVC-tracked | yes |

## License and permissions

| Field | Value |
| :--- | :--- |
| License | **CC-BY-4.0** (confirmed via the GitHub license API) |
| Redistribution allowed | yes, with attribution |
| Commercial use allowed | yes |
| Attribution required | yes — cite the paper and the repository |
| Restrictions carried into DreamScript | none beyond attribution; the citation goes in the final report and the model cards of anything trained on it |

## Contents

| Field | Value |
| :--- | :--- |
| Number of items | **704 images** |
| Modality | photographs and scans of hand-drawn BPMN diagrams |
| Resolution or size range | mixed; 573 MB of image data (~815 KB/image average) |
| File formats | `.jpg` / `.png` images; `.bpmn` (BPMN 2.0 XML) annotations; `.xml` word annotations |
| Total size on disk | 1.1 GB including git history |
| Annotation types | **704 BPMN structure files** (elements + sequence flows) and **704 word-level transcripts** |
| Annotation format | BPMN 2.0 XML with `BPMNDiagram`/`BPMNShape`/`BPMNEdge` geometry |
| Splits provided by the source | `data/writer_split.csv` — **writer-disjoint**: 65 train / 21 val / 21 test writers |

Layout:

```
data/raw/hdbpmn/data/
├── images/ex00..exNN/<exercise>_<writer>.jpg   704 photographs
├── annotations/ex00../<exercise>_<writer>.bpmn 704 structure files
├── words/ex00../<exercise>_<writer>.xml        704 word transcripts
├── writer_split.csv                            107 writers, official split
└── exercises.pdf                               the modelling tasks writers were given
```

## Provenance

- **Who created it, and how?** 107 writers each drew a subset of a fixed set of BPMN
  modelling exercises (`exercises.pdf`) on paper, then photographed or scanned them.
- **What were annotators asked to do?** Model the described process; the resulting diagrams
  were then annotated into valid BPMN 2.0 XML with per-element geometry.
- **Quality control by the source:** annotations are schema-valid BPMN, and the writer split
  is published so results are comparable across papers.

## Fit for DreamScript

| Field | Value |
| :--- | :--- |
| Diagram types covered | flowchart (BPMN is a superset — tasks ≈ process nodes, gateways ≈ decisions, sequence flows ≈ directed edges) |
| Phases that consume it | 1.1.1 acquisition, 9.1 detector training, 10 graph assembly, 7.3 HMM role sequences |
| Maps to the IR how? | `BPMNShape` → nodes (bbox + type), `BPMNEdge` → edges with waypoints, word XML → node/edge labels |
| Conversion script | `src/ingest/converters/hdbpmn.py` (Phase 2.2.2) |
| Known conversion losses | BPMN has richer semantics than our IR: pools, lanes, event subtypes (message/timer/error) and boundary events collapse to `container` / `start` / `end` / `process`; that collapse is lossy and is recorded per file during conversion |

**This is the single most valuable public source for the project** because it is the only one
with annotated *connections*, not just annotated components. Phase 10's broken-arrow repair
has ground truth to train and evaluate against only because of this dataset.

## Bias and representativeness

- **Drawing styles:** 107 writers — the widest style variety available to the project, and
  the reason the official writer split is adopted rather than a random one.
- **Capture conditions:** mostly clean, well-lit scans and photographs. Notably *less* adverse
  than the Phase 1.2 chaos corpus, so it will overstate preprocessing robustness on its own.
- **Language of labels:** English and German.
- **Domain skew:** business process modelling only — swimlanes, gateways and events dominate.
  There are no UI wireframes, state machines, ER diagrams or circuits here.
- **What this dataset will make the model bad at:** the other four diagram types, and messy
  capture conditions. Both gaps are filled deliberately elsewhere (Phase 1.1.5, 1.2.7).

## Known problems

- BPMN vocabulary is larger than the DreamScript IR, so conversion is lossy in one direction
  (see above). The conversion script logs every collapsed element type.
- Class imbalance within the set: sequence flows and tasks vastly outnumber gateways, which
  are exactly the elements Phase 7.3 needs most.
- One `.gitignore` sits inside `data/annotations/`; the inventory ignores it.

## Ethical and privacy notes

- Images contain identifiable handwriting but no personal content; the source published them
  under CC-BY-4.0 for exactly this use.
- No faces or surroundings are present in the frames.

## Changelog

| Date | Change |
| :--- | :--- |
| 2026-08-29 | initial card; 704 images / 704 annotations / 704 word files / 107 writers verified locally |
