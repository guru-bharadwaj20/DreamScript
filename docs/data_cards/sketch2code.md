# Data Card — Sketch2Code (wireframe → code)

Covers Phase 1.1.5. Like the flowchart card, this documents both the dataset the plan named
and the one actually acquired.

---

## Part 1 — Microsoft Sketch2Code (named in contributing.md, NOT acquired)

| Field | Value |
| :--- | :--- |
| Source | https://github.com/microsoft/ailab/tree/master/Sketch2Code |
| **Availability** | **UNAVAILABLE** |
| Verified on | 2026-08-29 |

Probed by `src.ingest.datasets.sketch2code.probe_microsoft()`:

| URL | Result |
| :--- | :--- |
| GitHub repository | 200 — the code is still there |
| Hosted demo (`sketch2code.azurewebsites.net`) | **URLError — does not resolve** |

The repository publishes the application code, not the training corpus; the corpus lived
behind the Azure service, which is gone. Recorded as unavailable.

---

## Part 2 — SALT-NLP/Sketch2Code (ACQUIRED)

### Identity

| Field | Value |
| :--- | :--- |
| Name | Sketch2Code (SALT-NLP) |
| Source URL | https://huggingface.co/datasets/SALT-NLP/Sketch2Code |
| Project page | https://salt-nlp.github.io/Sketch2Code-Project-Page/ |
| Related | built on the Design2Code webpage corpus |
| Retrieved on | 2026-08-29 |
| Local path | `data/raw/sketch2code/` |
| DVC-tracked | yes |

### License and permissions

| Field | Value |
| :--- | :--- |
| License | **ODC-BY** (Open Data Commons Attribution) |
| Redistribution allowed | yes, with attribution |
| Commercial use allowed | yes, with attribution |
| Attribution required | yes |
| Restrictions carried into DreamScript | attribution in the report and in the model card of anything trained on it |

### Contents (measured)

| Field | Value |
| :--- | :--- |
| Number of items | **731 hand-drawn sketches** over **484 webpages** |
| Sketches per page | 1.51 average (1–3 annotators drew each page) |
| Paired artifacts | 484 source `.html` files **and** 484 rendered screenshots |
| Orphan sketches | **0** — every sketch resolves to its webpage |
| Total size on disk | 125,583,952 bytes (~120 MB) |
| Splits provided by the source | none (it is published as a benchmark) |

Layout: `sketches/{page_id}_{sketch_id}.png`, `webpages/{page_id}.html`,
`webpages/{page_id}.png`.

### Why the substitute is better than the original for this project

The Microsoft set provided hand-drawn wireframes labelled with **widget bounding boxes** — a
detection dataset. SALT-NLP's provides hand-drawn wireframes paired with **real, working
HTML** — a generation dataset.

Phase 12 needs `(sketch → code)` supervision. The Microsoft set would have required inventing
the target code for every image; here the target already exists and is real-world HTML rather
than a toy template. This is the single largest source of genuine wireframe→code pairs the
project has (Phase 12.1.5).

| Field | Value |
| :--- | :--- |
| Diagram types covered | wireframe |
| Phases that consume it | 1.1.5, **12.1.5** (React target conversion), 5 (wireframe class) |
| Maps to the IR how? | sketch → nested container/widget tree; HTML gives the ground-truth nesting |
| Conversion script | `src/ingest/converters/sketch2code.py` (Phase 2.2.2) |
| Known conversion losses | HTML is far richer than the IR's wireframe vocabulary; styling, scripts and semantics beyond layout are dropped |

### Bias and representativeness

- **Drawing styles:** a small number of annotators (1–3 per page); far fewer writers than
  hdBPMN's 107. Style variety here is **weak**, and it is not a valid source for cross-writer
  generalization claims.
- **Capture conditions:** clean digital sketches, not photographs.
- **Domain skew:** real-world webpage layouts — headers, navs, card grids. Not app UIs, not
  forms-heavy admin screens.
- **Placeholder images:** all images inside the HTML are replaced by the authors with a single
  placeholder (`rick.jpg`), so a model trained here learns "image goes here", never real image
  content. That is the desired behaviour for wireframe→code, but it must be stated.
- **What this dataset will make the model bad at:** dense application UIs, mobile layouts, and
  anything requiring real image assets.

### Known problems

- No splits: this project builds its own, and because writer identity is not published the
  split cannot be scribe-disjoint. Like flowchartseg, it is excluded from the grouped-CV
  protocol (Phase 5.2.2).
- 484 unique layouts is small for code generation; it is combined with synthetic wireframe →
  React pairs from Phase 12.1.4.
- The published `.zip` duplicates the flat directories; the downloader ignores it to avoid
  storing the corpus twice.

### Ethical and privacy notes

Sketches are of public webpages; no personal content. The underlying webpages are third-party
sites captured by the Design2Code authors — generated code derived from them is used for
research evaluation only.

## Changelog

| Date | Change |
| :--- | :--- |
| 2026-08-29 | Microsoft Sketch2Code probed and recorded unavailable; SALT-NLP Sketch2Code acquired — 731 sketches / 484 HTML+screenshot pairs verified |
