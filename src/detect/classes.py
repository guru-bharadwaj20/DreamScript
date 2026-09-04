"""Phase 9.1.2 - the detection class set, frozen against what actually carries a box.

    python -m src.detect.classes            # the survey that froze the list
    python -m src.detect.classes --census   # per-source, per-class instance counts

The plan asks for "shape classes + arrowhead + text-block + UI widget classes". Three of those
four groups survive contact with the corpus and one does not, so this module exists to write
down *which* and *why* rather than to hold a tuple.

## The rule the list is frozen by

A detection class needs three things, and a class missing any of them cannot be trained or
scored:

1. **A box.** Not a label, a box - in the coordinate frame of an image that exists on disk.
2. **Enough instances** to appear in all three splits.
3. **A definition a human could check.** A class whose boxes come from a derivation has to say
   so, because its score then measures the derivation as much as the detector.

## What that admits, and what it refuses

`src/ir/vocab.py` freezes twelve shapes. Five of the twelve are unreachable here:

    line / arrow        an edge is a polyline, not a region. `arrow` survives only as its
                        **head** - see the derivation below - and `line` not at all.
    ellipse            zero rows anywhere in the corpus with a box (7.4.1 found the same).
    octagon            2,413 DIDI prompts and no image: DIDI publishes stroke data whose IR
                        coordinates are a graphviz layout *fitted* to the ink extent, so the
                        boxes describe where the prompt said to draw, not where ink is.
    text-block          175 boxed instances, all of them hdbpmn `textAnnotation` elements on
                        115 pages - the marginal notes a BPMN author staples to the side of a
                        diagram, and **every one of them has an empty `text` string**. The
                        label text 9.3 actually needs to read is annotated as a *string on a
                        node*, never as its own region, in every source. So the class that
                        exists is not the class the plan wanted, and training on it would
                        teach the detector to find margin notes.

**The UI widget classes are the one group the plan names that cannot be built at all.**
sketch2code is the corpus's only wireframe source with a role vocabulary (`ui-input`,
`ui-button`, `ui-label`, `ui-image` - 37,372 nodes over 484 pages, more annotated nodes than
every other source combined) and its IR carries `"geometry": "absent"`: the structure comes
from the real HTML and nothing was ever rendered, so **not one of those 37,372 nodes has a
box**. This is recorded here rather than discovered in 9.1.3 with a training run that quietly
learns five classes instead of nine. Phase 16's wireframe route needs either a rendering pass
over the sketch2code HTML or a box annotation campaign, and 9.1.4's mAP says nothing about
wireframes until one of those exists.

## The three sources that qualify

    hdbpmn        693 photographs of hand-drawn BPMN, boxes from the reference model.
                  The only *real photographs* in the set, and therefore the only rows that
                  measure the deployment question.
    fa_bresler    300 state machines, rendered from real pen trajectories by
                  `chaos_builder.render_inkml` into the same 1024px canvas the IR
                  coordinates already live in. Real ink, synthetic paper.
    flowchartseg  1,319 **computer-rendered** pages. Not hand-drawn, and included with that
                  stated: it is the only source of `parallelogram` and it triples the
                  `rectangle` count, and 9.1.1 measures what the domain gap costs rather than
                  assuming it is free.

## `arrowhead`, and why its boxes are derived

There is no arrowhead annotation in any source. The class is built from hdbpmn's sequence-flow
waypoints: the last waypoint of a BPMN edge is its arrival point on the target's boundary,
which is exactly where the writer drew the head. `arrowhead_box` places a square there,
extending back along the final segment.

Two consequences are stated up front so 9.1.4 is read correctly. The box **size** is a
convention, not a measurement - it is scaled from the page's median node height, because
nothing in the annotation records how large anyone drew their heads - so arrowhead AP is
partly a measurement of that convention, and IoU-based metrics will punish it more than the
shape classes. And the derivation is only trustworthy where the polyline has a *direction*:
hdbpmn's waypoints run source-to-target by construction, while **fa_bresler's polylines are raw
ink traces whose direction is whichever way the writer happened to move the pen**, so fa
contributes states and no arrowheads. That asymmetry is deliberate and it is why `arrowhead`
appears on 693 of the 2,312 pages.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.config import ROOT

IR = ROOT / "data" / "processed" / "ir"

#: Frozen at 9.1.2. Index is the YOLO class id and must never be reordered - a trained
#: checkpoint stores integers, not names, so a reorder silently relabels every prediction.
CLASSES: tuple[str, ...] = (
    "rectangle",
    "rounded-rect",
    "diamond",
    "circle",
    "double-circle",
    "parallelogram",
    "freeform",
    "arrowhead",
)

CLASS_INDEX: dict[str, int] = {name: i for i, name in enumerate(CLASSES)}

#: The shape vocabulary entries deliberately left out, with the reason. Read by the tests, so
#: dropping a class silently is not possible.
EXCLUDED: dict[str, str] = {
    "line": "an edge is a polyline, not a region",
    "arrow": "kept only as its head; the shaft is an edge",
    "ellipse": "zero boxed instances anywhere in the corpus",
    "octagon": "DIDI only, and DIDI has no images - its boxes are a fitted layout",
    "text-block": (
        "175 boxed instances, all hdbpmn textAnnotation margin notes with empty text; "
        "the label text 9.3 needs is a string on a node, never its own region"
    ),
    "ui-input": "sketch2code geometry is absent - 37,372 nodes, no boxes",
    "ui-button": "sketch2code geometry is absent - 37,372 nodes, no boxes",
    "ui-label": "sketch2code geometry is absent - 37,372 nodes, no boxes",
    "ui-image": "sketch2code geometry is absent - 37,372 nodes, no boxes",
}

#: source -> why it is in, so the mix is a decision rather than whatever happened to load.
SOURCES: dict[str, str] = {
    "hdbpmn": "real photographs of hand-drawn pages; boxes from the reference BPMN model",
    "fa_bresler": "real pen trajectories rendered onto synthetic paper at 1024px",
    "flowchartseg": "computer-rendered; the only source of parallelogram, and it triples rectangle",
}

#: Sources with an IR but no usable detection supervision.
REJECTED_SOURCES: dict[str, str] = {
    "didi": "no image; coordinates are a graphviz layout fitted to the ink extent",
    "sketch2code": 'geometry is "absent" - the HTML was never rendered',
}

#: The arrowhead square's side, as a multiple of the page's median node height, then clipped
#: to a band of the page diagonal so a page of giant or tiny nodes cannot produce an absurd box.
ARROWHEAD_SCALE = 0.30
ARROWHEAD_MIN_DIAG = 0.008
ARROWHEAD_MAX_DIAG = 0.030


def arrowhead_box(
    polyline: list[list[float]], side: float
) -> tuple[float, float, float, float] | None:
    """A square at the polyline's arrival end, extending back along the final segment.

    The tip is the last waypoint; the head's body lies behind it, so the square is centred
    half a side back along the incoming direction rather than on the tip itself. Returns None
    when the polyline has no usable final segment - a degenerate two-identical-points edge,
    which fa_bresler's traces contain in quantity.
    """
    points = [p for p in polyline or [] if p is not None and len(p) >= 2]
    if len(points) < 2:
        return None
    tip = np.asarray(points[-1], dtype=float)
    for previous in reversed(points[:-1]):
        direction = tip - np.asarray(previous, dtype=float)
        length = float(np.hypot(*direction))
        if length > 1e-6:
            unit = direction / length
            centre = tip - unit * (side / 2.0)
            return (float(centre[0] - side / 2), float(centre[1] - side / 2), side, side)
    return None


def arrowhead_side(diagram: dict[str, Any]) -> float:
    """The head size convention: a fraction of the median node height, clipped to the diagonal."""
    width, height = diagram["meta"].get("image_size") or (0, 0)
    diagonal = float(np.hypot(width, height))
    heights = [n["bbox"][3] for n in diagram["nodes"] if n.get("bbox")]
    base = float(np.median(heights)) * ARROWHEAD_SCALE if heights else diagonal * 0.015
    return float(np.clip(base, ARROWHEAD_MIN_DIAG * diagonal, ARROWHEAD_MAX_DIAG * diagonal))


def clip_box(box, width: float, height: float):
    """Trim a box to the page and refuse the ones that leave nothing behind.

    A derived arrowhead at the very edge of a page, and an annotated node whose reference
    geometry runs slightly off the photograph, both land here. Dropping them is right: a box
    with no pixels inside it is a label the detector can only be punished for.
    """
    x, y, w, h = box
    x0, y0 = max(0.0, x), max(0.0, y)
    x1, y1 = min(float(width), x + w), min(float(height), y + h)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return (x0, y0, x1 - x0, y1 - y0)


def boxes_for(diagram: dict[str, Any], arrowheads: bool = True) -> list[dict[str, Any]]:
    """Every detection target on one page, as `{cls, bbox, basis}` in image pixels.

    `basis` is `annotated` for a node whose box came from the source and `derived-waypoint`
    for an arrowhead, so 9.1.4 can report the two populations separately instead of averaging
    a measurement with a convention.
    """
    size = diagram["meta"].get("image_size")
    if not size:
        return []
    width, height = size
    out: list[dict[str, Any]] = []

    for node in diagram["nodes"]:
        if not node.get("bbox"):
            continue
        name = node["shape"]
        if name not in CLASS_INDEX:
            continue
        box = clip_box(node["bbox"], width, height)
        if box is None:
            continue
        out.append({"cls": name, "bbox": list(box), "basis": "annotated", "id": node["id"]})

    if arrowheads and diagram["meta"].get("source") == "hdbpmn":
        side = arrowhead_side(diagram)
        for edge in diagram["edges"]:
            if not edge.get("directed", True):
                continue
            derived = arrowhead_box(edge.get("polyline"), side)
            if derived is None:
                continue
            box = clip_box(derived, width, height)
            if box is None:
                continue
            out.append(
                {
                    "cls": "arrowhead",
                    "bbox": list(box),
                    "basis": "derived-waypoint",
                    "id": edge["id"],
                }
            )
    return out


def pages(source: str) -> list[Path]:
    return sorted((IR / source).glob("*.ir.json"))


def census(sources=tuple(SOURCES)) -> dict:
    """Per-source, per-class instance counts - the table the class list was frozen from."""
    import collections

    counts: dict[str, collections.Counter] = {}
    pages_per_class: dict[str, collections.Counter] = {}
    dropped = collections.Counter()
    for source in sources:
        counter: collections.Counter = collections.Counter()
        page_counter: collections.Counter = collections.Counter()
        for path in pages(source):
            diagram = json.loads(path.read_text(encoding="utf-8"))
            found = boxes_for(diagram)
            for box in found:
                counter[box["cls"]] += 1
            for name in {b["cls"] for b in found}:
                page_counter[name] += 1
            for node in diagram["nodes"]:
                if node.get("bbox") and node["shape"] not in CLASS_INDEX:
                    dropped[node["shape"]] += 1
        counts[source] = counter
        pages_per_class[source] = page_counter
    totals: dict[str, int] = {name: 0 for name in CLASSES}
    for counter in counts.values():
        for name, n in counter.items():
            totals[name] += n
    return {
        "classes": list(CLASSES),
        "per_source": {s: dict(c) for s, c in counts.items()},
        "pages_per_class": {s: dict(c) for s, c in pages_per_class.items()},
        "totals": totals,
        "boxed_shapes_dropped": dict(dropped),
        "excluded": EXCLUDED,
        "rejected_sources": REJECTED_SOURCES,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--census", action="store_true", help="count instances per source and class")
    args = ap.parse_args(argv)

    if args.census:
        print(json.dumps(census(), indent=2))
        return 0

    print(f"classes ({len(CLASSES)}): " + ", ".join(f"{i}={n}" for i, n in enumerate(CLASSES)))
    print("\nexcluded from the shape vocabulary:")
    for name, reason in EXCLUDED.items():
        print(f"  {name:<14} {reason}")
    print("\nsources:")
    for name, reason in SOURCES.items():
        print(f"  {name:<14} {reason}")
    for name, reason in REJECTED_SOURCES.items():
        print(f"  {name:<14} REJECTED - {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
