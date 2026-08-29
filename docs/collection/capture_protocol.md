# Capture Protocol — Adverse Conditions

Phase 1.2.7. Every public dataset in this corpus was captured cleanly: flatbed scans, even
light, straight-on framing. A model trained only on those will look excellent in evaluation
and fail the moment someone holds a napkin up to a webcam.

**At least 25% of the chaos corpus is captured badly on purpose.** This document says exactly
how badly, and how to file it.

## The conditions

`clean` is the control group. Everything else is adverse and counts toward the 25%.

| Condition | How to produce it | What it attacks | Mitigation under test |
| :--- | :--- | :--- | :--- |
| `clean` | Flat page, even light, camera perpendicular, whole diagram in frame | nothing — the control | — |
| `shadow` | Let your hand or head cast a shadow across part of the drawing | binarization: a global threshold splits the page into two exposures | CLAHE + morphological background estimate (3.1.4), Sauvola (3.1.5) |
| `angle` | Photograph 20–40° off perpendicular so the page is a trapezoid | geometry: boxes stop being rectangles, parallel lines converge | page-quad detection + homography (3.1.2, 3.1.3) |
| `ruled` | Draw on ruled or squared notebook paper | false structure: printed rules look exactly like diagram edges | directional morphological suppression (3.1.8) |
| `glare` | Whiteboard or glossy paper reflecting a lamp or window | saturation: strokes vanish into a white blob | illumination correction (3.1.4) |
| `stain` | Coffee ring, smudge, thumbprint over part of the diagram | occlusion: components partly destroyed | ensemble robustness (7.1.6) |
| `crossedout` | Draw an element, then strike it through or scribble it out | semantics: the model must *ignore* what a person deleted | crossed-out classifier (9.3.8) |
| `lowlight` | Dim room, no flash, let the sensor go noisy | signal-to-noise: speckle looks like ink | denoise + min component area (3.1.6) |
| `crop` | Frame so part of the diagram runs off an edge | completeness: a node or edge is simply missing | graph repair heuristics (10.2.2) |

Each condition maps to a specific mitigation. If a condition has no mitigation, collecting it
only produces noise — that is why the list is fixed rather than "take some bad photos".

## Rules

1. **The drawing must still be legible to you.** Adverse means degraded, not destroyed. If you
   cannot read it, neither can a person, and it is not a fair test.
2. **One condition per image.** Combined conditions cannot be attributed — if a model fails on
   `shadow+angle+glare` you learn nothing about which one broke it.
3. **Spread conditions across scribes and types.** All the glare coming from one person's
   whiteboard confounds condition with handwriting.
4. **`clean` and adverse pairs are valuable.** Where practical, photograph the *same* drawing
   both ways. That gives a controlled measurement of exactly what the condition costs — the
   cleanest possible input to the Phase 14.5 robustness curves.

## Taking the photograph

```
python -m src.ingest.capture --webcam \
    --scribe scribe03 --type flowchart --scenario atm_withdraw \
    --medium ballpoint --condition shadow
```

or file a photo you already took:

```
python -m src.ingest.capture --file DCIM/IMG_4831.jpg \
    --scribe scribe03 --type flowchart --scenario atm_withdraw \
    --medium ballpoint --condition shadow
```

The tool refuses anything that would corrupt the corpus:

| Rejection | Reason |
| :--- | :--- |
| unknown scenario for that diagram type | a typo would create a phantom scenario |
| unknown medium or condition | keeps the metadata vocabulary closed |
| out of focus (Laplacian variance < 40) | a motion smear has no recoverable strokes |
| too dark (mean < 25) or blown out (mean > 250) | lens cap, or a white frame |
| no strokes found (ink fraction < 0.001) | nothing was drawn, or nothing survived |

**The quality gate is deliberately permissive.** Shadows, glare and odd angles all pass — they
are the point. It rejects only frames carrying no signal at all. Use `--force` if you are
certain a rejected frame is worth keeping, and say why in the scribe registry notes.

## Target

| Requirement | Value |
| :--- | :--- |
| Minimum adverse fraction | **25%** of all collected sketches (Phase 1.2.7) |
| Assignment | every 4th sketch in the generated plan, cycling through the 8 adverse conditions |

Check with:

```
python -m src.ingest.collection --progress
```

## Status

The tool and the protocol are in place and verified against a synthetic image (metadata
rejection and successful filing both confirmed). **No real photographs have been captured
yet**; the adverse fraction reads 0% until collection begins.
