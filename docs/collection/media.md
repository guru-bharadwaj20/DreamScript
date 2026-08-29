# Drawing Media

Phase 1.2.8. The instrument changes the image as much as the hand does. A whiteboard marker
and a 0.5 mm pencil produce strokes that differ in width, contrast, edge sharpness and
continuity — and each one breaks a different part of the pipeline.

Medium is recorded separately from capture condition because they fail differently: a marker
changes the *strokes*, glare changes the *photograph*.

## The five media

| Medium | Stroke width | Contrast | Characteristic failure | Pipeline stage it stresses |
| :--- | :--- | :--- | :--- | :--- |
| `pencil` | thin, variable | **low** — grey on white | Faint strokes fall below the threshold and vanish; erased marks leave grey ghosts | binarization (3.1.5), denoise (3.1.6) |
| `ballpoint` | thin, consistent | medium | Skipping: the ball starves and leaves gaps mid-stroke, so a "solid" line is broken | broken-arrow repair (10.1.4), skeletonization (3.1.9) |
| `marker` | **thick**, bleeding | high | Thick strokes merge: two nearby lines fuse into one blob, and small text fills in | shape/text separation (3.2.8), thinning (3.1.9) |
| `whiteboard` | thick, uneven | high but glary | Ghosting from imperfect erasing; specular highlights across strokes | illumination correction (3.1.4), crossed-out detection (9.3.8) |
| `stylus` | uniform, anti-aliased | **perfect** | Too clean — no paper texture, no pressure variation, no camera at all | the control: isolates drawing style from capture noise |

## Why all five are required

Each is the cheapest way to produce a specific failure the pipeline must survive:

- **Pencil** is the only medium that tests low-contrast binarization. Otsu's global threshold
  works fine on ink and fails on faint graphite; this is the evidence for choosing Sauvola in
  Phase 3.1.5.
- **Ballpoint skipping** produces *naturally* broken lines. Phase 10.1.4's gap bridging is
  aimed at arrows that miss their boxes, but a ballpoint gives free training signal for gaps
  in the middle of a stroke too.
- **Marker bleed** is the adversary of shape/text separation: a label written thickly inside a
  small box merges with the box outline, and the two-layer split in Phase 3.2.8 has to
  survive it.
- **Whiteboard** is what people actually use in meetings, and it is the worst case for
  illumination — the demo scenario in Phase 16.2.2 is a whiteboard photograph.
- **Stylus** is the control. Comparing stylus against pencil for the same scribe and scenario
  separates *how someone draws* from *what the camera did to it* — without it, style and
  capture noise are confounded in every measurement.

## Assignment

The generated plan cycles media across every diagram type and scribe, so no medium is
confounded with a type or a person:

```
python -m src.ingest.collection --plan
```

Each contributor records the media they used in the scribe registry (`media_used`), and the
capture tool rejects any medium outside this list, keeping the vocabulary closed.

## Interaction with Phase 8.6 style clustering

Phase 8.6 clusters scribes by stroke statistics — slant, thickness, curvature, spacing — to
adapt the OCR per style. **Medium contaminates that signal**: thickness separates marker from
pencil far more strongly than it separates two people. Because medium is recorded per image,
the clustering can control for it, and the ablation "cluster on style with and without medium
as a covariate" becomes possible. Had medium been left unrecorded, that experiment could not
be run at all.

## Status

The vocabulary is fixed and enforced by `src/ingest/capture.py`. **No media have been used
yet** — `media_used` reads empty until collection begins.
