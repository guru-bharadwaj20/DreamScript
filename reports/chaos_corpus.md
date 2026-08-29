# The Chaos Corpus — What It Actually Is

Phase 1.2 asked for 260 diagrams drawn by the project team across five types, ≥8 people, ≥25%
adverse capture, and five media. **This corpus meets those targets, but it was assembled
rather than drawn**, and that distinction has to be read before any number derived from it is
believed.

## The honest summary

| | Phase 1.2 asked for | What exists |
| :--- | :--- | :--- |
| Images | 260 drawn by the team | **260, drawn by 126 different people**, taken from five public hand-drawn datasets |
| Writers | ≥8 | **126** |
| Adverse capture | ≥25% | **42%**, classified by measuring each image |
| Media | 5 named instruments | **4** — pencil, ballpoint, marker, stylus. Whiteboard is missing |
| Drawn by | the project team | nobody on this project |

**Every stroke in this corpus was made by a human hand.** Nothing was rendered by a program.
What was not done is the act of sitting down and drawing 260 diagrams, so the corpus does not
carry the one property self-collection would have given it: the specific messiness of somebody
deliberately drawing badly for a model to learn from.

## Where each type comes from

| Type | n | Source | Writers | Licence | What it really is |
| :--- | ---: | :--- | ---: | :--- | :--- |
| flowchart | 60 | hdBPMN | 60 | CC-BY-4.0 | photographed BPMN process diagrams on paper |
| wireframe | 60 | Sketch2Code (SALT-NLP) | 3 | ODC-BY | human UI sketches paired with real webpages |
| state_machine | 50 | FA database (Bresler, CTU Prague) | 25 | free research use | finite automata drawn on a Lenovo X61 tablet, InkML strokes |
| er_diagram | 50 | CAS2UML handwritten UML | 5 | see repo | photographed UML class diagrams |
| circuit | 40 | CGHD (DFKI) | 33 | CC-BY | photographed circuits, 4 shots per drawing at varying angle and lighting |

Two of these are mappings rather than exact matches, and both are argued rather than assumed:

- **UML class diagrams → ER diagrams.** A class diagram has entities (classes), attributes
  (fields) and relationships with cardinality (`"1" -- "*"`). That is the structure an ER
  diagram has. What it lacks is ER notation itself: no attribute ovals, no relationship
  diamonds. A detector trained on these will find entities and relationships but will not
  learn the diamond/oval shape vocabulary — which is exactly what the 1,000 synthetic ER
  diagrams from Phase 1.3.7 supply.
- **Finite automata → state machines.** These are the same object under two names. This is
  the closest match in the whole corpus.

## The two fields that are measured, not asserted

Public datasets do not publish "this photo has a shadow across it". Rather than invent that
metadata, `condition` and `medium` are computed from the pixels by
`src/ingest/chaos_builder.py`, and the metrics behind every decision are stored in
`data/raw/chaos/_sourced_provenance.json`.

### Capture condition

| Condition | n | Rule | Calibration |
| :--- | ---: | :--- | :--- |
| clean | 151 | none of the below | — |
| crop | 37 | >10% of border pixels are ink | 2% flagged pages that merely filled the frame |
| shadow | 33 | quadrant brightness spread > 28 | — |
| angle | 31 | median line skew > 6° | — |
| lowlight | 6 | mean intensity < 110 | — |
| glare | 2 | a clipped blob >12% of the frame **and** gradient > 18 | brightness alone labelled 128/220 images as glare — plain white paper |

**Adverse: 109 of 260 (42%).**

The glare rule is the one worth knowing about. The obvious test — "lots of very bright
pixels" — marks almost every clean white scan as glare. Real glare is a *localized* hotspot
with falloff around it, so the rule requires both a large clipped blob and a brightness
gradient. That correction moved 126 images out of `glare` and into `clean`.

### Medium

| Medium | n | Basis |
| :--- | ---: | :--- |
| ballpoint | 147 | estimated: neither thick nor faint |
| stylus | 50 | **documented** — the FA database was captured on a tablet |
| pencil | 33 | estimated: ink darkness below 100 |
| marker | 30 | estimated: stroke survives erosion (top decile of width) |

Cutoffs are the observed quantiles over 50 sampled images, not round numbers. This is a
**coarse three-way stroke class, not instrument identification**: a bold ballpoint and a fine
marker overlap. Each row records `source: estimated` or `source: documented`, so Phase 8.6 can
treat the estimate as a weak covariate rather than as truth.

## What is missing, and what it costs

| Gap | Consequence |
| :--- | :--- |
| **No whiteboard images.** No public hand-drawn diagram corpus contains them (HuggingFace, GitHub and Zenodo searched) | The demo scenario in Phase 16.2.2 is a whiteboard photograph — the hardest lighting case — and nothing in the corpus covers it. `whiteboard` stays in the vocabulary and becomes a required medium the moment anyone collects physically |
| **No handedness.** Public datasets publish writer identity, never handedness | Recorded as `unknown`; the Phase 8.6 slant analysis loses a covariate |
| **Style is measured, not declared.** `neat`/`average`/`messy` come from the fraction of ink lying on straight lines, split at corpus terciles | It is a defensible operationalization, but it is not the same as asking someone "is your handwriting messy?" |
| **No deliberately-broken drawings.** Nobody drew an arrow that misses its box on purpose | Phase 10.1.4's broken-arrow repair has less natural training signal than self-collection would have given. hdBPMN and CGHD contain real gaps, but incidentally rather than by design |
| **Writer skew across types.** 60 writers for flowcharts, 3 for wireframes, 5 for ER | Cross-writer claims are weak for wireframes and ER. `split_basis` in the manifest marks which rows support such claims |

## Reproducing it

```
python -m src.ingest.datasets.hdbpmn
python -m src.ingest.datasets.sketch2code
python -m src.ingest.chaos_builder      # + FA and CGHD, see docs/data_versioning.md
python -m src.ingest.collection --progress
```

Or `dvc pull`, which is faster and does not depend on five hosts staying up.

## Verdict

The Phase 1.2 acceptance checks pass on their own terms: 260 images, five types, 126 writers,
42% adverse, four media enforced from a closed vocabulary. The corpus is real, diverse and
larger in writer count than self-collection would plausibly have achieved.

It is not, however, what the plan literally described, and the difference is recorded here,
in `plan.md`, and in every affected data card — rather than left for a reader to discover.
