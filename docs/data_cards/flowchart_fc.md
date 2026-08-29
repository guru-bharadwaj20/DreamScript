# Data Card — Hand-Drawn Flowcharts (FC-A / FC-B, and its substitute)

This card covers Phase 1.1.2. It documents **two** things: the dataset the plan named, which
turned out to be unobtainable, and the substitute actually acquired in its place.

---

## Part 1 — FC-A / FC-B (named in plan.md, NOT acquired)

| Field | Value |
| :--- | :--- |
| Name | FC-A / FC-B — Bresler flowchart database |
| Source URL | https://cmp.felk.cvut.cz/~breslmar/flowcharts/ |
| Paper / citation | Bresler, Průša, Hlaváč — *Online recognition of sketched arrow-connected diagrams* (IJDAR 2016) |
| **Availability** | **UNAVAILABLE** |
| Verified on | 2026-08-29 |

### Evidence

Probed with `python -m src.ingest.datasets.flowcharts --probe`:

| URL | HTTP status |
| :--- | ---: |
| Landing page | 200 |
| Archive page | 200 |
| **Download link (`version1.0.html`)** | **404** |
| Offline variant landing page | 200 |

The project pages are still served, but the archive page's only download link is dead. No
mirror was found on HuggingFace, Kaggle, or by GitHub search. The dataset is therefore
recorded as unavailable rather than pending.

A regression check (`test_fc_original_still_unavailable`) re-probes these URLs, so if the
authors restore the archive the project finds out instead of assuming.

---

## Part 2 — flowchartseg (ACQUIRED, substitute)

### Identity

| Field | Value |
| :--- | :--- |
| Name | flowchartseg |
| Source URL | https://huggingface.co/datasets/MananSuri27/flowchartseg |
| Retrieved on | 2026-08-29 |
| Local path | `data/raw/flowchartseg/` |
| DVC-tracked | yes |

### License and permissions

| Field | Value |
| :--- | :--- |
| License | not declared on the dataset card |
| Redistribution allowed | **unknown — treated as NO** |
| Commercial use allowed | unknown — treated as no |
| Attribution required | yes, by convention |
| Restrictions carried into DreamScript | raw files stay local; only derived features, metrics and models are published. Phase 1.1.6 attempts to contact the author to resolve the license |

### Contents (measured locally)

| Field | Value |
| :--- | :--- |
| Number of items | **1,319** (1,187 train / 132 validation) |
| Modality | hand-drawn flowchart images |
| File formats | two parquet shards, images embedded |
| Total size on disk | 109,635,873 bytes (~105 MB) |
| Annotation types | **per-node semantic segmentation masks** (`background`, `node`) |
| Columns | `image`, `annotation`, `semantic_class_to_id` |
| Splits provided by the source | train / validation (no test split) |

### Why this substitutes acceptably

FC-A/FC-B offered component **bounding boxes**. flowchartseg offers component **masks**,
which are strictly richer: boxes are recovered by connected-component analysis over the mask,
while the reverse is impossible. The Phase 9.1 detector consumes boxes, so nothing is lost;
Phase 3 stroke work can additionally use the masks as pixel-level ground truth, which FC-A/B
could not have provided.

### What is lost

| FC-A/FC-B had | flowchartseg has | Consequence |
| :--- | :--- | :--- |
| Online stroke data (pen trajectories, timing) | offline raster images only | The stroke-order signal for Phase 7.3's "draw order" intuition must come from DIDI instead |
| Per-class component labels (box / diamond / arrow) | a single `node` class | Shape classes must come from hdBPMN and the chaos corpus; flowchartseg trains *localization*, not classification |
| Explicit edge annotation | none | Edge supervision comes from hdBPMN only |
| A published writer split | none | Splits must be built here, and writer identity is unknown, so scribe-disjointness cannot be guaranteed for this source |

**The last row is the important one.** Because writer identity is unknown, flowchartseg
cannot participate in the scribe-disjoint split guarantee (Phase 1.3.3). It is usable for
detector pretraining, but any result claiming cross-writer generalization must be measured on
hdBPMN or the chaos corpus, not here.

### Bias and representativeness

- **Drawing styles:** unknown number of writers; no metadata.
- **Capture conditions:** clean, uniformly captured; not adverse.
- **Language of labels:** English.
- **Domain skew:** flowcharts only.
- **What this dataset will make the model bad at:** distinguishing node *types* — it has one
  node class, so a model trained on it alone learns "there is a shape here", not "this is a
  decision diamond".

### Known problems

- Undeclared license (the reason redistribution is treated as prohibited).
- No writer identity, so it is excluded from the grouped-CV protocol.
- No test split; the validation split is used for monitoring only, never for reporting.

### Ethical and privacy notes

Handwriting is present; images stay local. No personal content observed.

## Changelog

| Date | Change |
| :--- | :--- |
| 2026-08-29 | FC-A/FC-B probed and recorded unavailable (download 404); flowchartseg acquired and verified — 1,319 images |
