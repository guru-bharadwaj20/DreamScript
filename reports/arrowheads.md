# Phase 3.2.6 — arrowhead detection, and why the rule does not reach its bar

contributing.md asks for precision ≥ 0.80. The geometric detector reaches 0.11. This is the
record of four attempts to lift it, each measured against hdBPMN's annotated arrow
positions on the same 20 pages, so the comparison is like for like.

| variant | detections | precision | recall |
| :--- | ---: | ---: | ---: |
| `as_built` | 770 | 0.106 | 0.215 |
| `on_the_shape_layer` | 767 | 0.107 | 0.215 |
| `symmetric_barbs_only` | 279 | 0.097 | 0.091 |
| `terminating_barbs_only` | 13 | 0.231 | 0.011 |
| `double_match_radius` | 770 | 0.261 | 0.385 |

**Every variant lands within a few points of the baseline.** Removing the handwriting
first does not help, which rules out text being the source of the false positives.
Demanding symmetric barbs does not help, nor does demanding that the barbs terminate
rather than merge into the shape they point at. Doubling the match radius — the most
generous reading of the annotation, in case the drawn head simply sits away from the
annotated waypoint — does not help either.

The reason they all fail the same way is one measurement. Comparing the two
populations of detections directly, the true and the false ones are **not separated by
any geometric property this detector can see**:

| property | true positives (p25 / p50 / p75) | false positives |
| :--- | :--- | :--- |
| barb asymmetry, degrees | 8.7 / 18.1 / 28.6 | 5.3 / 15.1 / 29.0 |
| half-opening angle, degrees | 32.4 / 39.6 / 48.0 | 33.4 / 41.7 / 49.7 |
| shaft reach, px | 70 / 70 / 70 | 59 / 70 / 70 |
| barb length ratio | 1.0 / 1.0 / 1.0 | 1.0 / 1.0 / 1.0 |

The distributions are the same. There is no threshold to tune and no operating point
on a precision-recall curve that reaches 0.80, because the score that would order the
curve does not exist. A page of handwritten BPMN contains hundreds of places where
three strokes meet at a plausible angle, and by shape alone an arrowhead is not one of
them.

## What this is evidence for

The plan puts a learned arrow detector at 9.1. This is the third independent
measurement in the project pointing the same way — Phase 2.2.4 got κ 0.30 for shape
from geometry, Phase 3.2.3 recovered 11% of hand-drawn boxes as quadrilaterals, and
this reaches precision 0.11. Hand-drawn marks resist hand-written rules.

The detector is kept and cached by 3.2.9 as a **feature**, not as an answer: “there is
an arrowhead-like junction near this stroke end” is worth something to a classifier
that also sees fifty other things. It must not be used to decide whether an edge is
directed.
