"""Phase 10.1.6 - binding a floating text region to the edge it annotates, and typing it.

    python -m src.assemble.edgelabels

`bind()` attaches each detected text region to the edge whose polyline it sits closest to, or
refuses it - because it belongs to a node, or because it is too far from any polyline to be an
edge's annotation at all. `classify()` then types a bound label as `yes` / `no` / `condition` /
`cardinality` / `symbol` / `other`. Both are scored on the 470 held-out pages, with **ground-truth
polylines** so the numbers describe this stage alone rather than 10.1.3's tracing error, and
against ground-truth text (there is no annotated label bbox in this corpus, so a region is
identified as a label only by OCR-reading it and string-matching the read to a ground-truth label
- see `match_read`, and its `ambiguous_share` caveat below).

## Binding: triples the midpoint baseline, and is still a hard problem

    rule        precision   recall    f1
    polyline       0.344     0.290   0.3145
    midpoint       0.226     0.073   0.1108

**The polyline rule beats the plan's literal wording - nearest edge by midpoint only - by 0.2037
F1, nearly tripling it** (150 vs. 38 true positives out of the same 518 ground-truth regions).
Midpoint recall is crippled because most bound labels do not sit near the midpoint (see below);
scoring against the true closest point on the polyline recovers most of the misses.

Both numbers are still modest in absolute terms, and the reason is visible per source:

    source          f1 (polyline)   tp   fp   fn
    hdbpmn              0.6296      17    4   16
    fa_bresler          0.2956     133  282  352
    flowchartseg        0.0000       0    0    0

**fa_bresler is the hard corpus and hdbpmn is the easy one, which is the opposite of what the
label vocabulary alone would suggest.** fa_bresler's transition alphabet is single letters and
digits (`a`, `b`, `0`, `1`, `c`...), and single characters are exactly where `match_read`'s CER
gate cannot help: `MIN_NEAR_LEN` refuses a near-match below 2 characters because at length 1 every
string is a neighbour of every other, so a label only counts as ground truth on an exact OCR read.
That, plus **86% of the corpus's ground-truth regions share their string with at least one other
edge on the same page** (`ambiguous_share = 0.8629`, driven almost entirely by fa_bresler's small
repeated alphabet), means the correctness criterion - bound edge's label equals the matched string
- cannot tell two same-labelled transitions apart on a large share of the corpus, and is scored as
correct either way. **flowchartseg contributes zero rows because none of its edges carry a
`label` string at all** - it is not a corpus this stage can be measured on, not a corpus it fails.

**Typing accuracy is 0.9867 on the 150 correctly bound labels** (149/150) - once a label is bound
to the right edge, classifying its string is close to solved.

**Node theft is zero by construction, not by measurement.** `steal_rate = 0.0` because `bind()`
refuses any region whose area is `>= NODE_GUARD` (0.5) inside a ground-truth node box before it is
ever scored against an edge - the guard cannot fail on this corpus by definition, and the number
records that the guard fires (119,622 of 149,547 regions were node-owned) rather than that the
edge-binding rule is somehow immune to stealing text a looser guard would have let through.

**Contested labels never occurred in the corpus scoring** (`contested = 0` under both rules): no
correctly bound region ever had a runner-up edge within `CONTEST_RATIO` (1.25x) of the winner's
distance. The competition case - two parallel flows out of one gateway, a label genuinely between
them - is real on this corpus's geometry but did not coincide with a *correctly matched* ground
truth region in this run, so it is exercised in `tests/test_assemble_edgelabels.py` on a
constructed page instead of claimed here.

**Unknown regions are the honest blind spot.** 29,407 of 149,547 regions (an OCR read that
matched no ground-truth label) are excluded from precision/recall entirely, because a region the
recogniser turned to noise may be a label this rule bound correctly or a stray mark it invented -
counting it either way would be inventing a number. `unknown_bound = 15,255`: just over half of
them were bound to *some* edge anyway, and that is the upper bound on false positives this
construction is structurally unable to see.

## Arc position: labels do not sit at the midpoint

    n = 150 (correctly bound labels only)     mean 0.3958   median 0.3603
    within 0.1 of the midpoint:  16.67%       in the outer thirds: 50.67%
    decile histogram (0.0 -> 1.0, 10 bins):  8 13 29 51 19 6 8 4 3 9

**The plan's "near edge midpoint" phrasing does not hold.** Only one bound label in six sits
within 0.1 of the true midpoint; half sit in the outer thirds of their edge. The histogram peaks
at deciles 3-4 (arc 0.3-0.5), so the typical label sits noticeably *before* the midpoint rather
than symmetrically around it - consistent with labels drawn near a gateway's outgoing branch
rather than centred on a long, possibly L-shaped, flow. `midpoint_distance_ratio_median = 2.227`:
for a typical correctly bound label, the literal midpoint point is **more than twice as far away**
as the true nearest point on the polyline - which is exactly why the midpoint rule's recall
(0.073) is a fifth of the polyline rule's (0.290).

## Vocabulary: three of the plan's four categories are populated, and a fifth exists

    class          count   share      classes appear in
    condition        824   0.5034     hdbpmn only
    symbol            788   0.4814     fa_bresler only
    yes                11   0.0067     hdbpmn only
    no                 10   0.0061     hdbpmn only
    cardinality         0   0.0        nowhere
    other               4   0.0024     hdbpmn only

Over 1,637 labelled edges and 450 distinct label strings. **`cardinality` is completely
unpopulated: zero of 1,637 labels match the ER notation regex, on any of the three sources this
stage sees.** None of hdbpmn, fa_bresler or flowchartseg is an ER diagram, so this is not a
detector failure - the plan's fourth category has no corpus to be tested on here, and that is
worth saying plainly rather than padding the number. **`symbol` is the category the plan did not
name and the corpus made unavoidable**: fa_bresler's automaton alphabet (`a` 173, `b` 172, `0`
118, `1` 102, `c` 76, plus combinations like `0,1` and `a,b,c`) is not `yes`/`no`/condition/
cardinality by any reading, and without a fifth class every one of fa_bresler's 788 labels would
have been forced into `other`. **`condition` is the biggest class and the loosest one**: of its
824 members, only 274 (33.25%) contain an explicit marker word (`if`, `above`, `rejected`, `>` ...)
- the rest are strings with a letter in them and no boolean/cardinality/symbol shape (`offer`,
`expertise`, `cheque`), so `condition` is closer to "uncategorised phrase" than "boolean
condition" as the plan's word implies. `yes`/`no` together are 1.28% of the corpus - real but rare.

## What this does not measure

Ground-truth polylines isolate binding from 10.1.3's tracing error, which is a stated simplification
in the other direction: the real end-to-end number is `polyline_f1 x tracing_accuracy`, not this
module's 0.3145. The OCR-match construction cannot identify a label that the recogniser mangled
past `MATCH_CER`, so `unknown_regions` is a floor on the true miss rate, not the whole of it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.assemble.corpus import RUNS, Page, centre, contains_fraction, pages, truth
from src.ocr.metrics import edit_distance, normalise
from src.utils.config import ROOT
from src.utils.parallel import pmap

PRIMITIVES = ROOT / "data" / "interim" / "primitives_detect"

#: A text region with this share of its area inside a ground-truth node box belongs to that node.
#: 10.1.2 owns those; this module must refuse them, and every one it binds is a stolen label.
NODE_GUARD = 0.5

#: The furthest a label may sit from its edge's polyline, as a share of the page diagonal.
#: Beyond this the region is left unbound rather than attached to the least-distant edge on the
#: page, which is what an unbounded nearest-edge rule does with a page title.
MAX_DIST = 0.04

#: Two edges are in competition when the runner-up is within this factor of the winner's
#: distance. Parallel flows out of one gateway are the case; the flag exists to score them.
CONTEST_RATIO = 1.25

#: Ground truth is built by matching an OCR read to a ground-truth label string. Multi-character
#: labels may match within this CER; single-character labels must match exactly, because at one
#: character every string is within one edit of every other.
MATCH_CER = 0.34
MIN_NEAR_LEN = 2

#: Padding around a text region before it is cut for the recogniser, in pixels.
CROP_PAD = 2

#: Labels that are a boolean branch. `y` and `n` are deliberately absent: fa_bresler's alphabet
#: is single letters, and a rule that read `n` as "no" would mistype hundreds of automaton
#: symbols to buy a handful of branches.
YES = {"yes", "true", "ja", "oui", "si", "y e s"}
NO = {"no", "false", "nein", "non", "n o"}

#: An ER cardinality, in the forms an ER diagram writes them.
CARDINALITY = re.compile(
    r"^(\*|[01nm]\s*\.\.\s*[*nm0-9]|[01nm]\s*:\s*[*nm0-9]|\(?[01]\s*,\s*[nm]\)?)$"
)

#: An automaton transition symbol: single alphanumerics or epsilon, joined by , + | /.
SYMBOL = re.compile(r"^[0-9a-zελ](\s*[,+|/]\s*[0-9a-zελ])*$")

#: Markers that make a phrase an actual condition rather than the name of a message or document.
CONDITION_MARKERS = (
    "not ",
    "no ",
    "if ",
    "above",
    "below",
    "greater",
    "less",
    "than",
    "invalid",
    "valid",
    "complete",
    "incomplete",
    "empty",
    "exceed",
    "too ",
    ">",
    "<",
    "=",
    "available",
    "accepted",
    "rejected",
    "approved",
    "denied",
    "ok",
    "success",
    "fail",
    "insured",
    "possible",
    "enough",
    "match",
)

CLASSES = ("yes", "no", "condition", "cardinality", "symbol", "other")


# ------------------------------------------------------------------------------------------
# typing
# ------------------------------------------------------------------------------------------


def classify(label: str) -> str:
    """One of `CLASSES` for a label string. Pure, so the tests can pin every rule."""
    text = normalise(label)
    if not text:
        return "other"
    if text in YES:
        return "yes"
    if text in NO:
        return "no"
    if CARDINALITY.match(text):
        return "cardinality"
    if SYMBOL.match(text):
        return "symbol"
    if any(ch.isalpha() for ch in text):
        return "condition"
    return "other"


def has_condition_marker(label: str) -> bool:
    """Does a phrase express a condition, or is it merely the name of a message or document?"""
    text = normalise(label)
    return any(marker in text for marker in CONDITION_MARKERS)


# ------------------------------------------------------------------------------------------
# geometry
# ------------------------------------------------------------------------------------------


def _segment(point, a, b) -> tuple[float, float]:
    """(distance from `point` to segment `ab`, how far along `ab` the foot fell, in [0, 1])."""
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    length2 = dx * dx + dy * dy
    if length2 <= 0:
        return (float(np.hypot(point[0] - ax, point[1] - ay)), 0.0)
    t = ((point[0] - ax) * dx + (point[1] - ay) * dy) / length2
    t = min(1.0, max(0.0, t))
    return (float(np.hypot(point[0] - (ax + t * dx), point[1] - (ay + t * dy))), float(t))


def polyline_distance(point, polyline) -> tuple[float, float]:
    """(distance to the polyline, arc position of the closest point, normalised to [0, 1]).

    The arc position is the quantity that tests the plan's "near edge midpoint" phrasing: a rule
    written against the midpoint is only right if these cluster at 0.5.
    """
    points = [(float(x), float(y)) for x, y in polyline]
    if len(points) < 2:
        if not points:
            return (float("inf"), 0.5)
        return (float(np.hypot(point[0] - points[0][0], point[1] - points[0][1])), 0.5)
    lengths = [
        float(np.hypot(b[0] - a[0], b[1] - a[1])) for a, b in zip(points, points[1:], strict=False)
    ]
    total = sum(lengths) or 1.0
    best, best_arc, travelled = float("inf"), 0.0, 0.0
    for (a, b), length in zip(zip(points, points[1:], strict=False), lengths, strict=False):
        d, t = _segment(point, a, b)
        if d < best:
            best, best_arc = d, (travelled + t * length) / total
        travelled += length
    return (best, float(best_arc))


def midpoint(polyline) -> tuple[float, float]:
    """The point half way along the polyline by arc length - the plan's literal rule."""
    points = [(float(x), float(y)) for x, y in polyline]
    if len(points) == 1:
        return points[0]
    lengths = [
        float(np.hypot(b[0] - a[0], b[1] - a[1])) for a, b in zip(points, points[1:], strict=False)
    ]
    total = sum(lengths)
    if total <= 0:
        return points[0]
    travelled = 0.0
    for (a, b), length in zip(zip(points, points[1:], strict=False), lengths, strict=False):
        if travelled + length >= total / 2:
            t = (total / 2 - travelled) / length
            return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
        travelled += length
    return points[-1]


# ------------------------------------------------------------------------------------------
# binding
# ------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Binding:
    """One text region's verdict: the edge it annotates, or nothing and why."""

    region: int
    bbox: list[float]
    edge: str | None
    distance: float
    arc: float
    contested: bool
    rejected: str | None = None
    label: str = ""
    kind: str = "other"


def _polyline_of(edge):
    if isinstance(edge, dict):
        return edge.get("polyline")
    return getattr(edge, "polyline", None)


def _id_of(edge, index: int) -> str:
    value = edge.get("id") if isinstance(edge, dict) else getattr(edge, "id", None)
    return str(value) if value else f"e{index}"


def bind(
    edges,
    text_regions,
    *,
    node_boxes=(),
    diagonal: float = 1000.0,
    rule: str = "polyline",
    texts=(),
) -> list[Binding]:
    """Attach each text region to the edge it annotates, or to nothing.

    `edges` carry a `polyline` in the same pixels as `text_regions`, which are `[x, y, w, h]`.
    `node_boxes` are the shapes whose text belongs to a node rather than to an edge; a region
    mostly inside one is refused here, because 10.1.2 owns it and a label stolen from a node is
    a silent corruption of the graph rather than a merely missing annotation.

    `rule="midpoint"` scores the plan's literal wording - nearest edge *midpoint* - as a baseline.
    """
    geometry = [(_id_of(e, i), _polyline_of(e)) for i, e in enumerate(edges)]
    geometry = [(i, p) for i, p in geometry if p]
    limit = MAX_DIST * diagonal
    out: list[Binding] = []
    for index, box in enumerate(text_regions):
        bbox = [float(v) for v in box]
        point = centre(bbox)
        text = texts[index] if index < len(texts) else ""
        owned = max((contains_fraction(bbox, n) for n in node_boxes), default=0.0)
        if owned >= NODE_GUARD:
            out.append(
                Binding(index, bbox, None, float("inf"), 0.5, False, "node", text, classify(text))
            )
            continue
        scored = []
        for edge_id, polyline in geometry:
            if rule == "midpoint":
                mx, my = midpoint(polyline)
                d, arc = float(np.hypot(point[0] - mx, point[1] - my)), 0.5
            else:
                d, arc = polyline_distance(point, polyline)
            scored.append((d, edge_id, arc))
        scored.sort(key=lambda row: row[0])
        if not scored or scored[0][0] > limit:
            distance = float(scored[0][0]) if scored else float("inf")
            out.append(
                Binding(index, bbox, None, distance, 0.5, False, "far", text, classify(text))
            )
            continue
        best = scored[0]
        contested = len(scored) > 1 and scored[1][0] <= CONTEST_RATIO * max(best[0], 1e-6)
        out.append(
            Binding(
                index,
                bbox,
                best[1],
                float(best[0]),
                float(best[2]),
                contested,
                None,
                text,
                classify(text),
            )
        )
    return out


# ------------------------------------------------------------------------------------------
# the corpus: regions, reads, and the ground truth built out of the two
# ------------------------------------------------------------------------------------------


def page_geometry(page: Page) -> dict:
    """Everything about one page that does not need the recogniser: regions, edges, node boxes."""
    from src.preprocess.cache import load_or_compute

    primitives, _ = load_or_compute(page.name, page.image, PRIMITIVES)
    diagram = truth(page)
    edges = [
        {"id": e.id, "label": e.label, "polyline": e.polyline} for e in diagram.edges if e.polyline
    ]
    return {
        "page": page.name,
        "source": page.source,
        "diagonal": page.diagonal,
        "regions": [[float(v) for v in b] for b in primitives.text_boxes],
        "edges": edges,
        "nodes": [n.bbox for n in diagram.nodes if n.bbox],
        "labels": sorted({normalise(e["label"]) for e in edges if normalise(e["label"])}),
    }


def read_regions(rows: list[dict], model_name: str = "finetune", batch: int = 64) -> None:
    """Fill each row's `reads` with the recogniser's string for every text region, in place.

    There is no annotation saying which pixels are a label, so the only way to know that a
    region *is* one is to read it. The reads are 9.3's fine-tuned CRNN at CER 0.6864, which is
    the honest ceiling on how much ground truth this construction can recover.
    """
    import cv2

    from src.ocr.crnn import fit, greedy_decode, load
    from src.ocr.lexicon import posteriors

    model, chars, device = load(model_name)
    for row in rows:
        row["reads"] = []
        if not row["regions"]:
            continue
        image = cv2.imread(str(row["image"]), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        crops = []
        height, width = image.shape
        for x, y, w, h in row["regions"]:
            x1 = max(0, int(x) - CROP_PAD)
            y1 = max(0, int(y) - CROP_PAD)
            x2 = min(width, int(x + w) + CROP_PAD)
            y2 = min(height, int(y + h) + CROP_PAD)
            patch = image[y1:y2, x1:x2]
            crops.append(fit(patch) if patch.size else fit(np.full((8, 8), 255, np.uint8)))
        for start in range(0, len(crops), batch):
            chunk = crops[start : start + batch]
            row["reads"].extend(
                greedy_decode(c, chars) for c in posteriors(model, chunk, device, batch=batch)
            )


def match_read(read: str, labels: list[str]) -> tuple[str | None, str]:
    """(the ground-truth label this read is, how it matched) - or `(None, "none")`.

    The label's own bbox is *not annotated anywhere in this corpus*, so a text region can only be
    identified as a label by agreeing with one. Exact agreement after normalisation is the strict
    form; a near match within `MATCH_CER` recovers labels the recogniser dented rather than
    destroyed, and is refused for one-character labels where every string is a neighbour.
    """
    text = normalise(read)
    if not text or not labels:
        return (None, "none")
    if text in labels:
        return (text, "exact")
    best, best_rate = None, MATCH_CER
    for label in labels:
        if len(label) < MIN_NEAR_LEN:
            continue
        rate = edit_distance(label, text) / max(1, len(label))
        if rate < best_rate:
            best, best_rate = label, rate
    return (best, "near") if best else (None, "none")


# ------------------------------------------------------------------------------------------
# scoring
# ------------------------------------------------------------------------------------------


def score(rows: list[dict], rule: str = "polyline") -> dict:
    """Bind every page under `rule` and count precision, recall and typing against the truth.

    A binding is correct when the bound edge carries the label string the region was matched to.
    Where a page repeats a string on two edges - fa_bresler writes `a` on several transitions -
    that criterion cannot tell them apart, so the share of ground-truth regions whose string is
    ambiguous on its page is reported beside the score as the size of the concession.

    Only two kinds of region are scored: the ones the string match identified as a label, and the
    ones that sit inside a ground-truth node box and therefore must be refused. **Everything else
    is `unknown` and excluded**, because a region the recogniser turned to noise may equally be a
    label this rule found correctly or a stray mark it invented, and counting it either way would
    be inventing a number. `unknown_bound` reports how many of them were bound anyway, which is
    the honest upper bound on the false positives this construction cannot see.
    """
    tp = fp = fn = 0
    unknown = unknown_bound = 0
    stolen = node_regions = 0
    contested = contested_right = 0
    ambiguous = 0
    typed = typed_right = 0
    arcs: list[float] = []
    mid_ratio: list[float] = []
    per_source: dict[str, Counter] = {}
    matched_kind: Counter = Counter()

    for row in rows:
        labels = row["labels"]
        edges = row["edges"]
        by_id = {e["id"]: normalise(e["label"]) for e in edges}
        counts = Counter(by_id.values())
        reads = row.get("reads", [])
        truths: dict[int, str] = {}
        for index, read in enumerate(reads):
            label, how = match_read(read, labels)
            if label:
                truths[index] = label
                matched_kind[how] += 1

        node_boxes = row["nodes"]
        owned = {
            index
            for index, box in enumerate(row["regions"])
            if max((contains_fraction(box, n) for n in node_boxes), default=0.0) >= NODE_GUARD
        }
        # A region that reads as a label and sits inside a node is the ambiguous case the
        # construction cannot resolve; it is counted as node-owned, which is the strict reading.
        truths = {k: v for k, v in truths.items() if k not in owned}
        node_regions += len(owned)

        bindings = bind(
            edges,
            row["regions"],
            node_boxes=node_boxes,
            diagonal=row["diagonal"],
            rule=rule,
            texts=reads,
        )
        bucket = per_source.setdefault(row["source"], Counter())
        for binding in bindings:
            gold = truths.get(binding.region)
            if binding.region in owned:
                if binding.edge is not None:
                    stolen += 1
                    fp += 1
                    bucket["fp"] += 1
                continue
            if gold is None:
                unknown += 1
                unknown_bound += int(binding.edge is not None)
                continue
            if counts.get(gold, 0) > 1:
                ambiguous += 1
            if binding.edge is None:
                fn += 1
                bucket["fn"] += 1
                continue
            correct = by_id.get(binding.edge, "") == gold
            if binding.contested:
                contested += 1
                contested_right += int(correct)
            if correct:
                tp += 1
                bucket["tp"] += 1
                typed += 1
                typed_right += int(classify(binding.label) == classify(gold))
                if rule == "polyline":
                    point = centre(binding.bbox)
                    polyline = next(e["polyline"] for e in edges if e["id"] == binding.edge)
                    _, arc = polyline_distance(point, polyline)
                    arcs.append(arc)
                    mx, my = midpoint(polyline)
                    to_mid = float(np.hypot(point[0] - mx, point[1] - my))
                    mid_ratio.append(to_mid / max(binding.distance, 1e-6))
            else:
                fp += 1
                fn += 1
                bucket["fp"] += 1
                bucket["fn"] += 1

    def prf(t: int, f: int, n: int) -> dict:
        precision = t / (t + f) if t + f else 0.0
        recall = t / (t + n) if t + n else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {
            "tp": t,
            "fp": f,
            "fn": n,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    result = prf(tp, fp, fn)
    result.update(
        {
            "rule": rule,
            "gt_regions": tp + fn,
            "match_kind": dict(matched_kind),
            "unknown_regions": unknown,
            "unknown_bound": unknown_bound,
            "node_owned_regions": node_regions,
            "labels_stolen_from_nodes": stolen,
            "steal_rate": round(stolen / node_regions, 4) if node_regions else 0.0,
            "ambiguous_share": round(ambiguous / max(1, tp + fn), 4),
            "contested": contested,
            "contested_accuracy": round(contested_right / contested, 4) if contested else None,
            "typing_accuracy": round(typed_right / typed, 4) if typed else None,
            "typed": typed,
            "by_source": {
                source: prf(c["tp"], c["fp"], c["fn"]) for source, c in sorted(per_source.items())
            },
        }
    )
    if arcs:
        array = np.array(arcs)
        hist, _ = np.histogram(array, bins=10, range=(0.0, 1.0))
        result["arc"] = {
            "n": len(arcs),
            "mean": round(float(array.mean()), 4),
            "median": round(float(np.median(array)), 4),
            "within_0.1_of_mid": round(float((np.abs(array - 0.5) <= 0.1).mean()), 4),
            "in_outer_thirds": round(float(((array < 1 / 3) | (array > 2 / 3)).mean()), 4),
            "deciles": hist.tolist(),
            "midpoint_distance_ratio_median": round(float(np.median(mid_ratio)), 3),
        }
    return result


def vocabulary(rows: list[dict] | None = None) -> dict:
    """The corpus's actual edge-label vocabulary and its class distribution.

    The plan asserts four categories. This is the test of whether they are the right four, and
    it is computed over every labelled ground-truth edge rather than over the ones this module
    manages to bind, so a class cannot look unpopulated merely because it is hard to find.
    """
    counts: Counter = Counter()
    classes: Counter = Counter()
    per_source: dict[str, Counter] = {}
    marked = phrases = 0
    for page in pages():
        diagram = truth(page)
        for edge in diagram.edges:
            text = normalise(edge.label)
            if not text:
                continue
            counts[text] += 1
            kind = classify(text)
            classes[kind] += 1
            per_source.setdefault(page.source, Counter())[kind] += 1
            if kind == "condition":
                phrases += 1
                marked += int(has_condition_marker(text))
    total = sum(counts.values())
    return {
        "labelled_edges": total,
        "distinct_labels": len(counts),
        "top": counts.most_common(25),
        "classes": {k: classes.get(k, 0) for k in CLASSES},
        "class_share": {k: round(classes.get(k, 0) / max(1, total), 4) for k in CLASSES},
        "by_source": {s: dict(c) for s, c in sorted(per_source.items())},
        "condition_phrases": phrases,
        "condition_with_marker": marked,
        "condition_marker_share": round(marked / max(1, phrases), 4),
    }


def run(limit: int | None = None) -> dict:
    held = pages()[:limit] if limit else pages()
    rows = pmap(page_geometry, held, n_jobs=3, desc="edge-label geometry")
    for row, page in zip(rows, held, strict=False):
        row["image"] = str(page.image)
    read_regions(rows)
    result = {
        "pages": len(rows),
        "regions": sum(len(r["regions"]) for r in rows),
        "edges_with_polyline": sum(len(r["edges"]) for r in rows),
        "vocabulary": vocabulary(),
        "polyline": score(rows, "polyline"),
        "midpoint_baseline": score(rows, "midpoint"),
        "constants": {
            "node_guard": NODE_GUARD,
            "max_dist": MAX_DIST,
            "contest_ratio": CONTEST_RATIO,
            "match_cer": MATCH_CER,
        },
        "polylines": "ground-truth (this stage is isolated from 10.1.3 tracing error)",
    }
    best = result["polyline"]["f1"]
    result["beats_midpoint_by"] = round(best - result["midpoint_baseline"]["f1"], 4)
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=RUNS / "edgelabels.json")
    args = ap.parse_args(argv)

    result = run(args.limit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    printable = dict(result)
    printable["vocabulary"] = {k: v for k, v in result["vocabulary"].items() if k != "top"}
    print(json.dumps(printable, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
