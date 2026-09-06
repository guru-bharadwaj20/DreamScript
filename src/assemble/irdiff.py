"""Phase 10.2.5 - the IR diff metric: node F1, edge F1 and GED, none of them free of the
correspondence problem.

    python -m src.assemble.irdiff
    python -m src.assemble.irdiff predicted.ir.json truth.ir.json --match combined

A predicted node carries a different id from the true node it represents, so "node F1" is not
one number - it is a family of numbers, one per rule for deciding which predicted node stands
for which true node. This module implements four such rules and measures how much the choice
moves the answer, plus an edge F1 and a graph-edit-distance built on top of whichever rule was
chosen, and reports honestly that the GED is an **approximation** to an NP-hard exact problem,
not the exact answer.

## The four matching rules

`match_nodes` solves the assignment with `scipy.optimize.linear_sum_assignment` - the
similarity-maximising pairing, not a greedy nearest-first one, because greedy can strand a
correct pair on a dense page where the first pick consumes the partner the second one needed.
Four similarity matrices feed it:

    geometric   IoU only, gated at 0.5 (9.1's mAP@50 convention).
    text        1 - normalised edit distance, gated at 0.5. Undefined for text-free nodes -
                see below.
    combined    0.5 geometry + 0.5 text, gated at the blended threshold. The naive mix, kept
                deliberately because its failure is the measurement.
    gated       geometry decides admissibility; text only breaks ties among boxes that already
                pass the IoU gate. Cannot promote a pair geometry rejected.

## What moved, measured on 120 held-out ground-truth graphs (945 nodes, 910 edges) with two
perturbations per graph (`matcher_spread`)

    matcher      node F1   edge F1
    geometric     0.9005    0.8089
    text          0.7015    0.8047
    combined      0.8773    0.8946
    gated         0.9005    0.8089

**Matching rule moves node F1 by 0.199** (geometric/gated's 0.9005 down to text's 0.7015) on
identical damage. `gated` never once diverged from `geometric` on this corpus
(`gated_differs_from_geometric: 0`): the label tie-break had no overlapping-box case to settle.
The worst single graph swung 1.0 between rules - a page where text matching failed completely
while geometry still worked, or the reverse.

**Identity is not free for every rule.** `diff(x, x)` is exact - 1.0 node F1, zero-cost GED - for
`geometric` and `gated` on all 120 real ground-truth graphs. It is **not** exact for `text` and
`combined`: their self-diff node F1 average 0.795 and 0.9193 respectively, because
`empty_text_share` is **0.5048** on this corpus - just over half of all nodes carry no OCR'd
text - and an empty label matched against itself scores `EMPTY_TEXT_SIMILARITY = 0.0` by design
(matching two blanks is evidence of missing text, not of correspondence; scoring it 1.0 would let
text matching pair every unlabelled box with every other one). That is not a bug in the identity
check, it is the text rule's real behaviour, which is exactly why a node F1 must always be quoted
with the matching rule that produced it.

## The edge cascade (`cascade`, 288 cases, `jitter_boxes` damage only)

Deleting a node deletes its incident edges along with it, so a predicted graph built by node
deletion never contains an edge with an unmatchable endpoint - the first version of this study
measured 0.0 lost edges per missed node, and the zero was an artefact of the damage model, not a
property of the metric. `jitter_boxes` moves a box past the IoU gate (by `JITTER = 0.35` of its
own width/height) while leaving every edge still attached to it: the real failure mode of a
detector that puts a box in slightly the wrong place, not a redrawn diagram.

Under that damage, one missed node costs **2.5757 edges**, on both sides of the ledger: the same
figure holds for edges the prediction throws away (`edges_lost_per_missed_node`, unprojectable
because an endpoint never matched) and true edges no predictor could ever have scored given the
same misses (`true_edges_unreachable_per_missed_node`) - identical because every jittered node in
this sample is a shared endpoint counted the same way on both sides. **59.25%** of all true edges
in the damaged cases become unreachable this way. Node F1 averages 0.5442 against edge F1's
0.3587 on the same damaged graphs - **edge F1 trails node F1 by 0.1855** - which is the honest
reason edge F1 is always the harder number to move: it inherits every node miss and adds its own.

## GED: approximation vs. brute-force exact (`exact_ged`, graphs of <= 7 nodes only)

`ged_bound` is the cost of *one particular* edit path - the one implied by the similarity-optimal
node assignment - not the cheapest possible path, so it is an **upper bound on exact GED, never a
lower one**: `ged_bound >= GED` always, with equality only when the appearance-optimal assignment
happens to also be the cost-optimal one. `exact_ged` brute-forces every injective node mapping
(7! = 5,040, milliseconds; 8! is not) and is the only way to know how loose that bound is. Exact
GED itself is NP-hard in general, which is why this check is confined to small graphs at all.

On 825 small-graph perturbation pairs: **never_below_exact holds (never a violation of the
bound)**, the approximation lands on the exact value in **99.39%** of pairs, and the mean gap is
**0.0412** edit operations (mean relative gap 3.76% of a mean exact cost of 2.7309). The gap is
not spread evenly: `delete_nodes`, `delete_edges` and `rename_labels` are exact every time
(0.0 mean gap each, because the optimal node assignment is also unambiguous there); `rewire_edges`
opens a small gap (0.0242); `jitter_boxes` opens the largest (0.1818, max single-pair gap 14.0),
because a jittered box can make the IoU-optimal assignment disagree with the cost-optimal one.

## Monotonicity (`monotonicity`, k = 0..4, five perturbation kinds)

Every curve for every (matcher, kind, metric) triple in this run is monotone in the intended
direction - node F1 and edge F1 non-increasing in k, `ged_normalised` non-decreasing - with one
instructive flat line rather than a violation: `text`'s node F1 is flat under `jitter_boxes` (it
does not look at boxes) and `geometric`/`gated` are flat under `rename_labels` (they do not look
at text), which is each rule correctly being blind to damage outside its own evidence, not a
monotonicity failure.

## End to end: trivial predictor (every detection a node, no edges) vs. truth, 470 held-out pages

`geometric`/`gated` node F1 is **0.6615**; `text` and `combined` both collapse to **0.0**, because
the trivial predictor's boxes carry no OCR text at all, so no predicted node clears the
text-similarity gate against anything, and `combined`'s blended score can never reach threshold
either. Edge F1 is **0.2809** for every matcher (the trivial predictor emits no edges, so this
number is carried entirely by the 132 of 470 pages with zero true edges scoring a vacuous 1.0
under the empty/empty F1 convention: `edge_f1_on_pages_with_edges` - the real number, restricted
to pages that have edges to miss - is **0.0** for every matcher, as it must be for a predictor
that never emits one).
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from dataclasses import dataclass, field
from itertools import combinations, permutations
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.ir.model import Diagram, Node
from src.ocr.metrics import edit_distance, normalise
from src.utils.config import ROOT
from src.utils.parallel import pmap

RUNS = ROOT / "experiments" / "assemble"
OUT = RUNS / "irdiff.json"

#: A predicted box must overlap a true box by at least this much to be allowed to represent it.
#: 0.5 is the detection convention (9.1's mAP@50) and is kept so that a node F1 here and a box
#: recall there are talking about the same notion of "found it".
IOU_THRESHOLD = 0.5

#: A predicted label must be at least this similar - 1 - CER against the true label - to be
#: allowed to represent it. 0.5 is deliberately loose: 9.3.6's best recogniser sits at CER 0.68,
#: so a strict text gate would match nothing at all on real predictions.
TEXT_THRESHOLD = 0.5

#: Weight on geometry in the combined score; the remainder goes to text.
GEOMETRY_WEIGHT = 0.5

#: Two nodes that both carry no text are not evidence of a correspondence, they are evidence of
#: an absence. Scoring them 1.0 would let text matching pair every unlabelled box with every
#: other one, so the empty/empty case scores zero and text-only matching is left visibly weak.
EMPTY_TEXT_SIMILARITY = 0.0

#: Unit edit costs. Every operation costs 1 so that the approximation and the brute-force exact
#: GED below are computed against the same cost model and their gap is the approximation's.
COST = 1.0

#: Graphs at or below this many nodes get an exact GED by brute force over every injective node
#: mapping. 7! = 5,040 mappings is a few milliseconds; 8! is not, and 12! is a career.
EXACT_MAX_NODES = 7

#: Weight on the label when geometry has already decided a pair is admissible. Small on purpose:
#: `gated` is geometry's answer with text used only to break ties between overlapping boxes, so
#: it must never be able to promote a pair that IoU rejected.
TIE_BREAK = 0.1

#: The four rules a predicted node can be matched to a true one by. `combined`'s naive mix is
#: kept in the comparison even though it collapses on textless predictions, because that
#: collapse is the measurement this module exists to make visible.
MATCHERS = ("geometric", "text", "combined", "gated")

SEED = 42


# ------------------------------------------------------------------------------------------
# the correspondence problem
# ------------------------------------------------------------------------------------------


def text_similarity(a: str, b: str) -> float:
    """1 - normalised edit distance between two labels, on 9.3.6's normalisation."""
    x, y = normalise(a), normalise(b)
    if not x and not y:
        return EMPTY_TEXT_SIMILARITY
    if not x or not y:
        return 0.0
    return max(0.0, 1.0 - edit_distance(x, y) / max(len(x), len(y)))


def box_similarity(a: list[float] | None, b: list[float] | None) -> float:
    """IoU, with a missing box scoring zero rather than raising."""
    if a is None or b is None:
        return 0.0
    from src.assemble.corpus import iou

    return iou(list(a), list(b))


def similarity_matrix(predicted: list[Node], truth_nodes: list[Node], match: str) -> np.ndarray:
    """`S[i, j]` in [0, 1]: how much predicted node `i` looks like true node `j`."""
    if match not in MATCHERS:
        raise ValueError(f"unknown matcher {match!r}, expected one of {MATCHERS}")
    matrix = np.zeros((len(predicted), len(truth_nodes)), dtype=float)
    for i, p in enumerate(predicted):
        for j, t in enumerate(truth_nodes):
            if match == "geometric":
                matrix[i, j] = box_similarity(p.bbox, t.bbox)
            elif match == "text":
                matrix[i, j] = text_similarity(p.text, t.text)
            elif match == "gated":
                overlap = box_similarity(p.bbox, t.bbox)
                matrix[i, j] = (
                    overlap + TIE_BREAK * text_similarity(p.text, t.text)
                    if overlap >= IOU_THRESHOLD
                    else 0.0
                )
            else:
                matrix[i, j] = GEOMETRY_WEIGHT * box_similarity(p.bbox, t.bbox) + (
                    1.0 - GEOMETRY_WEIGHT
                ) * text_similarity(p.text, t.text)
    return matrix


def threshold_for(match: str) -> float:
    if match in ("geometric", "gated"):
        return IOU_THRESHOLD
    if match == "text":
        return TEXT_THRESHOLD
    return GEOMETRY_WEIGHT * IOU_THRESHOLD + (1.0 - GEOMETRY_WEIGHT) * TEXT_THRESHOLD


def match_nodes(
    predicted: list[Node], truth_nodes: list[Node], match: str = "geometric"
) -> list[tuple[int, int]]:
    """Hungarian-optimal node correspondence, as `(predicted index, true index)` pairs.

    Greedy matching - take the best pair, remove it, repeat - is the obvious implementation and
    it is wrong in a way that shows up on exactly this data: two overlapping boxes on a dense
    BPMN page can be assigned so that the globally best pairing is unreachable, because the
    first greedy pick consumed the partner the second one needed. `linear_sum_assignment`
    maximises the total similarity instead, which is a different answer often enough to matter.
    """
    matrix = similarity_matrix(predicted, truth_nodes, match)
    if matrix.size == 0:
        return []
    rows, columns = linear_sum_assignment(-matrix)
    limit = threshold_for(match)
    return [(int(i), int(j)) for i, j in zip(rows, columns, strict=True) if matrix[i, j] >= limit]


# ------------------------------------------------------------------------------------------
# the diff
# ------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Diff:
    """One predicted graph scored against one true graph under one matching rule."""

    match: str
    node_f1: float
    node_precision: float
    node_recall: float
    edge_f1: float
    edge_precision: float
    edge_recall: float
    #: The assignment-based approximation. An **upper bound** on exact GED - see `ged_bound`.
    ged: float
    #: `ged` divided by the largest possible edit cost, so graphs of different sizes compare.
    ged_normalised: float
    matched: list[tuple[str, str]] = field(default_factory=list)
    edits: dict[str, list] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "match": self.match,
            "node_f1": round(self.node_f1, 4),
            "node_precision": round(self.node_precision, 4),
            "node_recall": round(self.node_recall, 4),
            "edge_f1": round(self.edge_f1, 4),
            "edge_precision": round(self.edge_precision, 4),
            "edge_recall": round(self.edge_recall, 4),
            "ged": round(self.ged, 4),
            "ged_normalised": round(self.ged_normalised, 4),
            "matched": [list(pair) for pair in self.matched],
            "edits": dict(self.edits),
            "counts": dict(self.counts),
        }


def _f1(true_positive: int, predicted: int, actual: int) -> tuple[float, float, float]:
    precision = true_positive / predicted if predicted else (1.0 if not actual else 0.0)
    recall = true_positive / actual if actual else (1.0 if not predicted else 0.0)
    total = precision + recall
    return (2 * precision * recall / total if total else 0.0, precision, recall)


def _edge_keys(diagram: Diagram) -> set[tuple[str, str]]:
    return {
        (e.src, e.dst) if e.directed else tuple(sorted((e.src, e.dst)))  # type: ignore[misc]
        for e in diagram.edges
        if e.src is not None and e.dst is not None
    }


def diff(predicted: Diagram, truth: Diagram, *, match: str = "geometric") -> Diff:
    """Score `predicted` against `truth`. The public entry point of Phase 10's metric.

    Node F1 is undefined until the correspondence is fixed, so the rule is a keyword argument
    with no default that can be forgotten: `match` is always in the returned object and always
    in the JSON, because a node F1 quoted without it is not a number anyone can reproduce.
    """
    pairs = match_nodes(predicted.nodes, truth.nodes, match)
    node_f1, node_precision, node_recall = _f1(len(pairs), len(predicted.nodes), len(truth.nodes))

    forward = {predicted.nodes[i].id: truth.nodes[j].id for i, j in pairs}
    matched = sorted(forward.items())

    # Edges are scored on the *mapped* endpoints, which is what makes a node miss cascade: an
    # edge whose source or target was never matched cannot be projected into the true graph's
    # id space at all, so it is lost before any comparison happens.
    predicted_edges: set[tuple[str, str]] = set()
    unprojectable = dangling = 0
    for e in predicted.edges:
        if e.src is None or e.dst is None:
            # 2.1.6's open ends. Not scored on either side - `_edge_keys` drops them from the
            # truth too - because charging for them here would make `diff(x, x)` non-zero on
            # any diagram that records an unresolved edge, and 87 of the first 120 held-out
            # pages carry exactly one.
            dangling += 1
            continue
        if e.src not in forward or e.dst not in forward:
            unprojectable += 1
            continue
        key = (forward[e.src], forward[e.dst])
        predicted_edges.add(key if e.directed else tuple(sorted(key)))  # type: ignore[arg-type]
    true_edges = _edge_keys(truth)

    # An edge lost to an unmatched endpoint still counts against precision: it is an edge the
    # prediction asserts and the truth cannot confirm. Dropping it from the denominator instead
    # would let a predictor raise its edge precision by losing nodes, which is the wrong way up.
    hits = predicted_edges & true_edges
    edge_f1, edge_precision, edge_recall = _f1(
        len(hits), len(predicted_edges) + unprojectable, len(true_edges)
    )

    # The edit list. Deletions are true things the prediction missed; insertions are predicted
    # things that are not there. Substitutions are matched nodes whose label or shape is wrong.
    matched_predicted = {i for i, _ in pairs}
    matched_true = {j for _, j in pairs}
    unmatched_true_ids = {n.id for k, n in enumerate(truth.nodes) if k not in matched_true}
    substitutions = [
        {
            "predicted": predicted.nodes[i].id,
            "truth": truth.nodes[j].id,
            "text": [predicted.nodes[i].text, truth.nodes[j].text],
            "shape": [predicted.nodes[i].shape, truth.nodes[j].shape],
        }
        for i, j in pairs
        if normalise(predicted.nodes[i].text) != normalise(truth.nodes[j].text)
        or predicted.nodes[i].shape != truth.nodes[j].shape
    ]
    edits = {
        "node_insert": [n.id for k, n in enumerate(predicted.nodes) if k not in matched_predicted],
        "node_delete": [n.id for k, n in enumerate(truth.nodes) if k not in matched_true],
        "node_substitute": substitutions,
        "edge_insert": [list(k) for k in sorted(predicted_edges - true_edges)],
        "edge_delete": [list(k) for k in sorted(true_edges - predicted_edges)],
    }

    cost = COST * (
        len(edits["node_insert"])
        + len(edits["node_delete"])
        + len(substitutions)
        + len(edits["edge_insert"])
        + len(edits["edge_delete"])
        + unprojectable
    )
    worst = COST * (
        len(predicted.nodes) + len(truth.nodes) + len(predicted.edges) + len(true_edges)
    )
    return Diff(
        match=match,
        node_f1=node_f1,
        node_precision=node_precision,
        node_recall=node_recall,
        edge_f1=edge_f1,
        edge_precision=edge_precision,
        edge_recall=edge_recall,
        ged=cost,
        ged_normalised=cost / worst if worst else 0.0,
        matched=matched,
        edits=edits,
        counts={
            "predicted_nodes": len(predicted.nodes),
            "true_nodes": len(truth.nodes),
            "predicted_edges": len(predicted.edges),
            "true_edges": len(true_edges),
            "matched_nodes": len(pairs),
            # Predicted edges thrown away because an endpoint never matched. This is the whole
            # cascade in one integer.
            "edges_lost_to_unmatched_nodes": unprojectable,
            # The recall side of the same cascade: true edges touching a true node that was
            # never matched. No predictor can score these, whatever its edges look like.
            "true_edges_at_unmatched_nodes": len(
                [k for k in true_edges if k[0] in unmatched_true_ids or k[1] in unmatched_true_ids]
            ),
            "dangling_edges": dangling,
        },
    )


# ------------------------------------------------------------------------------------------
# graph edit distance: the approximation, and the exact answer it is checked against
# ------------------------------------------------------------------------------------------


def ged_bound(predicted: Diagram, truth: Diagram, *, match: str = "geometric") -> float:
    """The assignment-based GED. **This is an upper bound on exact GED, not exact GED.**

    Exact GED is NP-hard. What this computes is the cost of one *particular* valid edit path -
    the one implied by the similarity-optimal node assignment - and any valid edit path costs at
    least as much as the cheapest one. So `ged_bound >= GED`, always, with equality only when
    the similarity-optimal assignment happens to also be the cost-optimal one.

    It is an upper bound and not the usual assignment *lower* bound because the assignment here
    is chosen by appearance (IoU, label) and not by edit cost, and the edge costs are added
    afterwards rather than folded into the assignment matrix. That is deliberate: the whole
    point of this module is that the node correspondence is a stated, inspectable choice, and a
    bound derived from a different correspondence than the one `node_f1` used would be quoting
    two different matchings in one report.
    """
    return diff(predicted, truth, match=match).ged


def _path_cost(
    predicted: Diagram, truth: Diagram, mapping: dict[str, str], unmapped_predicted: int
) -> float:
    """Cost of the edit path implied by one injective node mapping, under the same unit costs."""
    truth_by_id = {n.id: n for n in truth.nodes}
    predicted_by_id = {n.id: n for n in predicted.nodes}
    substitutions = sum(
        1
        for p, t in mapping.items()
        if normalise(predicted_by_id[p].text) != normalise(truth_by_id[t].text)
        or predicted_by_id[p].shape != truth_by_id[t].shape
    )
    projected: set[tuple[str, str]] = set()
    lost = 0
    for e in predicted.edges:
        if e.src is None or e.dst is None:
            continue
        if e.src in mapping and e.dst in mapping:
            key = (mapping[e.src], mapping[e.dst])
            projected.add(key if e.directed else tuple(sorted(key)))  # type: ignore[arg-type]
        else:
            lost += 1
    true_edges = _edge_keys(truth)
    return COST * (
        unmapped_predicted
        + (len(truth.nodes) - len(mapping))
        + substitutions
        + len(projected - true_edges)
        + len(true_edges - projected)
        + lost
    )


def exact_ged(predicted: Diagram, truth: Diagram, max_nodes: int = EXACT_MAX_NODES) -> float:
    """Brute-force GED over every injective node mapping. Only for tiny graphs.

    This is the evidence that `ged_bound` is usable. It enumerates every way of mapping some
    subset of predicted nodes onto distinct true nodes - including the empty mapping, which is
    "delete everything and insert everything" and is the cost ceiling - and returns the cheapest
    edit path under the same unit costs `diff` uses.
    """
    if max(len(predicted.nodes), len(truth.nodes)) > max_nodes:
        raise ValueError(f"exact GED refuses graphs above {max_nodes} nodes")
    predicted_ids = [n.id for n in predicted.nodes]
    truth_ids = [n.id for n in truth.nodes]
    best = float("inf")
    for size in range(min(len(predicted_ids), len(truth_ids)) + 1):
        for chosen in combinations(range(len(predicted_ids)), size):
            for targets in permutations(range(len(truth_ids)), size):
                mapping = {
                    predicted_ids[a]: truth_ids[b] for a, b in zip(chosen, targets, strict=True)
                }
                best = min(best, _path_cost(predicted, truth, mapping, len(predicted_ids) - size))
    return float(best)


# ------------------------------------------------------------------------------------------
# controlled perturbations - the metric validated on truth we control
# ------------------------------------------------------------------------------------------


def _renamed(diagram: Diagram) -> Diagram:
    """A copy with every node and edge id replaced, so no matcher can cheat off the ids.

    Every perturbation goes through this. A predicted node never carries the true node's id -
    that is the entire reason this module exists - so a validation corpus in which it does would
    validate nothing.
    """
    copied = copy.deepcopy(diagram)
    renames = {n.id: f"p{i}" for i, n in enumerate(copied.nodes)}
    for i, node in enumerate(copied.nodes):
        node.id = f"p{i}"
    for i, edge in enumerate(copied.edges):
        edge.id = f"pe{i}"
        edge.src = renames.get(edge.src) if edge.src is not None else None
        edge.dst = renames.get(edge.dst) if edge.dst is not None else None
    copied.unresolved_edges, copied.crossed_out, copied.low_conf_text = [], [], []
    return copied


#: How far a jittered box is moved, as a share of its own width and height. 0.35 is chosen to
#: straddle the 0.5 IoU gate: a smaller shift never breaks a geometric match and a larger one
#: always does, and neither of those measures anything.
JITTER = 0.35


def perturb(diagram: Diagram, kind: str, k: int, seed: int = SEED) -> Diagram:
    """One of the five controlled damages, applied `k` times. Ids are always renamed."""
    out = _renamed(diagram)
    rng = random.Random(seed)
    if k <= 0:
        return out
    if kind == "delete_nodes":
        victims = set(rng.sample([n.id for n in out.nodes], min(k, len(out.nodes))))
        out.nodes = [n for n in out.nodes if n.id not in victims]
        out.edges = [e for e in out.edges if e.src not in victims and e.dst not in victims]
    elif kind == "delete_edges":
        keep = list(out.edges)
        for _ in range(min(k, len(keep))):
            keep.pop(rng.randrange(len(keep)))
        out.edges = keep
    elif kind == "rewire_edges":
        ids = [n.id for n in out.nodes]
        for edge in rng.sample(out.edges, min(k, len(out.edges))) if len(ids) > 1 else []:
            others = [i for i in ids if i != edge.dst]
            if others:
                edge.dst = rng.choice(others)
    elif kind == "rename_labels":
        for node in rng.sample(out.nodes, min(k, len(out.nodes))):
            node.text = "".join(rng.choice("zxqvw") for _ in range(max(3, len(node.text))))
    elif kind == "jitter_boxes":
        for node in rng.sample(out.nodes, min(k, len(out.nodes))):
            if node.bbox is None:
                continue
            x, y, w, h = node.bbox
            node.bbox = [x + JITTER * w, y + JITTER * h, w, h]
    else:
        raise ValueError(f"unknown perturbation {kind!r}")
    return out


PERTURBATIONS = (
    "delete_nodes",
    "delete_edges",
    "rewire_edges",
    "rename_labels",
    "jitter_boxes",
)


# ------------------------------------------------------------------------------------------
# the trivial predictor - a real end-to-end number rather than a synthetic one
# ------------------------------------------------------------------------------------------


def detections_as_diagram(page, boxes: list[dict]) -> Diagram:
    """Every detection is a node, nothing is an edge. The floor Phase 10 has to beat.

    This is not a strawman for its own sake: it is what "assembly" reduces to if the binding,
    tracing and repair tasks all fail, and a metric that cannot separate it from a real IR is
    not measuring anything.
    """
    return Diagram(
        id=f"{page.name}:detections",
        diagram_type="flowchart",
        nodes=[
            Node(id=f"d{i}", shape=b["cls"], bbox=list(b["bbox"]), text="", confidence=b["score"])
            for i, b in enumerate(boxes)
        ],
        edges=[],
    )


# ------------------------------------------------------------------------------------------
# the run
# ------------------------------------------------------------------------------------------


def _identity_check(diagram: Diagram) -> list[dict]:
    return [diff(diagram, diagram, match=m).to_dict() for m in MATCHERS]


def _sweep_one(argument: tuple[Diagram, str, int, int]) -> dict:
    diagram, kind, k, seed = argument
    damaged = perturb(diagram, kind, k, seed)
    row = {"kind": kind, "k": k}
    for matcher in MATCHERS:
        d = diff(damaged, diagram, match=matcher)
        row[matcher] = {
            "node_f1": d.node_f1,
            "edge_f1": d.edge_f1,
            "ged": d.ged,
            "ged_normalised": d.ged_normalised,
            "edges_lost": d.counts["edges_lost_to_unmatched_nodes"],
            "true_edges": d.counts["true_edges"],
        }
    return row


def monotonicity(diagrams: list[Diagram], ks=(0, 1, 2, 3, 4), seed: int = SEED) -> dict:
    """Does the metric move the right way as k rises? The measurement this task reports."""
    jobs = [
        (diagram, kind, k, seed + i)
        for kind in PERTURBATIONS
        for k in ks
        for i, diagram in enumerate(diagrams)
    ]
    rows = pmap(_sweep_one, jobs, n_jobs=4)

    table: dict[str, dict[int, dict[str, list[float]]]] = {}
    for row in rows:
        bucket = table.setdefault(row["kind"], {}).setdefault(
            row["k"], {m: {"node_f1": [], "edge_f1": [], "ged_normalised": []} for m in MATCHERS}
        )
        for matcher in MATCHERS:
            for field_ in ("node_f1", "edge_f1", "ged_normalised"):
                bucket[matcher][field_].append(row[matcher][field_])

    summary: dict[str, Any] = {}
    for kind, per_k in table.items():
        curves = {
            matcher: {
                field_: [round(float(np.mean(per_k[k][matcher][field_])), 4) for k in sorted(per_k)]
                for field_ in ("node_f1", "edge_f1", "ged_normalised")
            }
            for matcher in MATCHERS
        }
        summary[kind] = {
            "k": sorted(per_k),
            "curves": curves,
            "monotone": {
                matcher: {
                    "node_f1": _non_increasing(curves[matcher]["node_f1"]),
                    "edge_f1": _non_increasing(curves[matcher]["edge_f1"]),
                    "ged_normalised": _non_decreasing(curves[matcher]["ged_normalised"]),
                }
                for matcher in MATCHERS
            },
        }
    return summary


def _non_increasing(values: list[float], tolerance: float = 1e-9) -> bool:
    return all(b <= a + tolerance for a, b in zip(values, values[1:], strict=False))


def _non_decreasing(values: list[float], tolerance: float = 1e-9) -> bool:
    return all(b >= a - tolerance for a, b in zip(values, values[1:], strict=False))


def _exact_gap_one(argument: tuple[Diagram, str, int, int]) -> dict | None:
    diagram, kind, k, seed = argument
    if len(diagram.nodes) > EXACT_MAX_NODES:
        return None
    damaged = perturb(diagram, kind, k, seed)
    if max(len(damaged.nodes), len(diagram.nodes)) > EXACT_MAX_NODES:
        return None
    approximate = ged_bound(damaged, diagram, match="combined")
    exact = exact_ged(damaged, diagram)
    return {
        "kind": kind,
        "k": k,
        "nodes": len(diagram.nodes),
        "approximate": approximate,
        "exact": exact,
        "gap": approximate - exact,
    }


def exact_gap(diagrams: list[Diagram], ks=(1, 2, 3), seed: int = SEED) -> dict:
    """Approximate GED against brute-force exact GED on graphs of at most 7 nodes."""
    jobs = [
        (diagram, kind, k, seed + i)
        for kind in PERTURBATIONS
        for k in ks
        for i, diagram in enumerate(diagrams)
        if len(diagram.nodes) <= EXACT_MAX_NODES
    ]
    rows = [r for r in pmap(_exact_gap_one, jobs, n_jobs=4) if r is not None]
    if not rows:
        return {"pairs": 0}
    gaps = np.array([r["gap"] for r in rows], dtype=float)
    exacts = np.array([r["exact"] for r in rows], dtype=float)
    return {
        "pairs": len(rows),
        "max_nodes": EXACT_MAX_NODES,
        "never_below_exact": bool((gaps >= -1e-9).all()),
        "exact_share": round(float((gaps <= 1e-9).mean()), 4),
        "mean_gap": round(float(gaps.mean()), 4),
        "max_gap": round(float(gaps.max()), 4),
        "mean_exact": round(float(exacts.mean()), 4),
        "mean_relative_gap": round(float((gaps / np.maximum(exacts, 1.0)).mean()), 4),
        "by_kind": {
            kind: round(
                float(np.mean([r["gap"] for r in rows if r["kind"] == kind])),
                4,
            )
            for kind in PERTURBATIONS
            if any(r["kind"] == kind for r in rows)
        },
    }


def _trivial_one(page) -> dict:
    from src.assemble.corpus import detections, truth

    true = truth(page)
    predicted = detections_as_diagram(page, detections(page))
    row: dict[str, Any] = {"page": page.name, "source": page.source}
    for matcher in MATCHERS:
        d = diff(predicted, true, match=matcher)
        row[matcher] = {
            "node_f1": d.node_f1,
            "edge_f1": d.edge_f1,
            "ged_normalised": d.ged_normalised,
            "matched": d.counts["matched_nodes"],
            "true_nodes": d.counts["true_nodes"],
            "true_edges": d.counts["true_edges"],
        }
    return row


def trivial_predictor(limit: int | None = None) -> dict:
    from src.assemble.corpus import pages

    held = list(pages())[:limit]
    rows = pmap(_trivial_one, held, n_jobs=4)
    return {
        "pages": len(rows),
        "by_matcher": {
            matcher: {
                field_: round(float(np.mean([r[matcher][field_] for r in rows])), 4)
                for field_ in ("node_f1", "edge_f1", "ged_normalised")
            }
            for matcher in MATCHERS
        },
        # A page with no true edges and a predictor with no edges is a perfect edge F1 by the
        # empty/empty convention, and 9.1's held-out set is full of them. The mean edge F1 above
        # is therefore almost entirely this artefact; the restricted mean is the real number.
        "pages_with_no_true_edges": int(sum(1 for r in rows if not r["geometric"]["true_edges"])),
        "edge_f1_on_pages_with_edges": {
            matcher: round(
                float(
                    np.mean(
                        [r[matcher]["edge_f1"] for r in rows if r[matcher]["true_edges"]] or [0.0]
                    )
                ),
                4,
            )
            for matcher in MATCHERS
        },
        "true_edges_total": int(sum(r["geometric"]["true_edges"] for r in rows)),
    }


def _cascade_one(argument: tuple[Diagram, str, int, int]) -> dict:
    diagram, kind, k, seed = argument
    damaged = perturb(diagram, kind, k, seed)
    d = diff(damaged, diagram, match="geometric")
    return {
        "kind": kind,
        "k": k,
        "node_f1": d.node_f1,
        "edge_f1": d.edge_f1,
        "edge_recall": d.edge_recall,
        "nodes_missed": d.counts["true_nodes"] - d.counts["matched_nodes"],
        "edges_lost": d.counts["edges_lost_to_unmatched_nodes"],
        "true_edges_at_unmatched": d.counts["true_edges_at_unmatched_nodes"],
        "true_edges": d.counts["true_edges"],
    }


def cascade(diagrams: list[Diagram], ks=(1, 2, 3), seed: int = SEED) -> dict:
    """How many edges one missed node costs. The honest reason edge F1 trails node F1.

    The damage has to be `jitter_boxes` and not `delete_nodes`. Deleting a node deletes its
    edges too, so the predicted graph never contains an edge with an unmatchable endpoint and
    the cascade is invisible - the first version of this study measured 0.0 lost edges per
    missed node and the zero was an artefact of the damage model, not a property of the metric.
    Jitter moves a box past the IoU gate while leaving every edge attached to it, which is the
    real failure: a detector that puts the box in slightly the wrong place.
    """
    jobs = [
        (diagram, "jitter_boxes", k, seed + i)
        for k in ks
        for i, diagram in enumerate(diagrams)
        if diagram.edges
    ]
    rows = [
        r for r in pmap(_cascade_one, jobs, n_jobs=4) if r["nodes_missed"] > 0 and r["true_edges"]
    ]
    if not rows:
        return {"cases": 0}
    missed = np.array([r["nodes_missed"] for r in rows], dtype=float)
    lost = np.array([r["edges_lost"] for r in rows], dtype=float)
    unreachable = np.array([r["true_edges_at_unmatched"] for r in rows], dtype=float)
    true_total = float(sum(r["true_edges"] for r in rows))
    return {
        "cases": len(rows),
        "damage": "jitter_boxes",
        "nodes_missed": int(missed.sum()),
        # Precision side: predicted edges thrown away because an endpoint never matched.
        "edges_lost_per_missed_node": round(float(lost.sum() / missed.sum()), 4),
        # Recall side: true edges no predictor could have scored, given those misses.
        "true_edges_unreachable_per_missed_node": round(float(unreachable.sum() / missed.sum()), 4),
        "share_of_true_edges_unreachable": round(float(unreachable.sum() / true_total), 4),
        "mean_node_f1": round(float(np.mean([r["node_f1"] for r in rows])), 4),
        "mean_edge_f1": round(float(np.mean([r["edge_f1"] for r in rows])), 4),
        "edge_f1_below_node_f1_by": round(
            float(np.mean([r["node_f1"] - r["edge_f1"] for r in rows])), 4
        ),
    }


def sample_diagrams(limit: int = 120) -> list[Diagram]:
    """Real ground-truth graphs from the held-out pages, in the corpus's fixed order."""
    from src.assemble.corpus import pages, truth

    return [truth(page) for page in list(pages())[:limit]]


def matcher_spread(diagrams: list[Diagram], seed: int = SEED) -> dict:
    """How far apart do the three matching rules land on the same damaged corpus?

    The headline of this task. A node F1 quoted without its matching rule is a number that can
    be moved by whichever rule flatters the result.
    """
    jobs = [
        (diagram, kind, k, seed + i)
        for kind in PERTURBATIONS
        for k in (1, 2)
        for i, diagram in enumerate(diagrams)
    ]
    rows = pmap(_sweep_one, jobs, n_jobs=4)
    means = {
        matcher: round(float(np.mean([r[matcher]["node_f1"] for r in rows])), 4)
        for matcher in MATCHERS
    }
    edge_means = {
        matcher: round(float(np.mean([r[matcher]["edge_f1"] for r in rows])), 4)
        for matcher in MATCHERS
    }
    per_graph = np.array(
        [
            max(r[m]["node_f1"] for m in MATCHERS) - min(r[m]["node_f1"] for m in MATCHERS)
            for r in rows
        ]
    )
    return {
        "cases": len(rows),
        "node_f1": means,
        "edge_f1": edge_means,
        "spread": round(max(means.values()) - min(means.values()), 4),
        "worst_single_graph_spread": round(float(per_graph.max()), 4),
        "mean_single_graph_spread": round(float(per_graph.mean()), 4),
        "best": max(means, key=lambda m: means[m]),
        "worst": min(means, key=lambda m: means[m]),
        # Does the label tie-break ever change geometry's assignment on this corpus?
        "gated_differs_from_geometric": int(
            sum(1 for r in rows if r["gated"]["node_f1"] != r["geometric"]["node_f1"])
        ),
    }


def run(limit: int = 120, exact_limit: int = 60) -> dict:
    diagrams = sample_diagrams(limit)
    empty_text = sum(1 for d in diagrams for n in d.nodes if not normalise(n.text))
    total_nodes = sum(len(d.nodes) for d in diagrams)
    result: dict[str, Any] = {
        "diagrams": len(diagrams),
        "nodes": total_nodes,
        "edges": sum(len(d.edges) for d in diagrams),
        "empty_text_share": round(empty_text / max(1, total_nodes), 4),
        "thresholds": {
            "iou": IOU_THRESHOLD,
            "text": TEXT_THRESHOLD,
            "geometry_weight": GEOMETRY_WEIGHT,
        },
        "identity": [row for diagram in diagrams[:20] for row in _identity_check(diagram)][:3],
        # Per matcher, because one of them fails it. `diff(x, x)` is the cheapest possible
        # sanity check and a rule that cannot match a diagram to *itself* has no business
        # scoring a prediction.
        "identity_holds": {
            matcher: all(
                diff(diagram, diagram, match=matcher).node_f1 == 1.0
                and diff(diagram, diagram, match=matcher).ged == 0.0
                for diagram in diagrams
            )
            for matcher in MATCHERS
        },
        "identity_node_f1": {
            matcher: round(float(np.mean([diff(d, d, match=matcher).node_f1 for d in diagrams])), 4)
            for matcher in MATCHERS
        },
        "matcher_spread": matcher_spread(diagrams),
        "monotonicity": monotonicity(diagrams),
        "cascade": cascade(diagrams),
        "exact_ged": exact_gap(diagrams[:exact_limit]),
        "trivial_predictor": trivial_predictor(),
    }
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="*", type=Path, help="predicted.ir.json truth.ir.json")
    ap.add_argument("--match", default="geometric", choices=MATCHERS)
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)

    if args.files:
        if len(args.files) != 2:
            print("give exactly two .ir.json files: predicted, then truth", file=sys.stderr)
            return 2
        predicted, truth_ = (Diagram.load(p) for p in args.files)
        print(json.dumps(diff(predicted, truth_, match=args.match).to_dict(), indent=2))
        return 0

    result = run(args.limit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "identity"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
