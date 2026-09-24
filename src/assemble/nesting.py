"""Phase 10.1.8 - the containment tree, and the corpus that owns the structure but not the pixels.

    python -m src.assemble.nesting

The plan's line is *"build parent-child tree for wireframe layout"*, and the corpus it points at
is sketch2code: 484 IR files whose containment is already annotated as edges with
`attrs.kind == "contains"` - **43,649 relations over 44,133 nodes**. Everything the plan asks for
is sitting there in the structure. It is also, geometrically, empty.

## The trap, verified before anything else was written

**Every one of sketch2code's 44,133 nodes has `bbox: null`, and all 484 files carry
`meta.geometry == "absent"`.** Not an oversight in a few files: sketch2code's IR is converted from
the real HTML behind each wireframe and nothing was ever rendered, so the tree is exact and the
boxes do not exist. 9.1.2 hit the same wall from the other side and refused to train UI widget
classes on it.

So the geometric half of this task **cannot be scored on the corpus that owns the structure**, and
a single accuracy quoted over both halves would be a number about nothing. They are split, and
each is measured where it is honest:

    the tree builder        relations -> tree. Scored on sketch2code's 484 pages, where the
                            containment relations are ground truth and geometry is irrelevant.
    the geometric finder    boxes -> relations. Scored on the 242 held-out hdbpmn pages, whose
                            pools, lanes and subprocesses are real nested rectangles carrying
                            real boxes - both drawn and detected.

## Half one: the tree builder, on sketch2code

`from_contains` reads the edges and `tree` turns proposed parents into a `Nesting`. A set of
containment relations is not a tree by construction - it can have several roots, a cycle, a node
claimed by two parents, or a parent that is not on the page - so each is repaired by a stated rule
and every repair is recorded in `Nesting.dropped` rather than applied silently.

**All 484 pages yield a well-formed tree, all 484 have exactly one real root before any synthetic
one is attached, and not one repair fired: no cycles, no multi-parent nodes, no dangling parents
anywhere in 43,649 relations.** So the repair rules are, on this corpus, dead code - which is
stated here and covered by unit tests instead, because a rule whose only evidence is that it never
ran is a rule nobody has checked.

    nodes            44,133 over 484 pages      internal nodes    22,276
    mean depth        5.85 (node-weighted)      max depth             20
    depth p99            16                     branching median     1.0
    branching p99        10

**The depth is the number to carry forward.** A wireframe nests **20 levels deep** with a median
branching factor of **1.0** - long single-child chains of layout `div`s, not the wide shallow tree
a flowchart suggests. Anything downstream that recurses over this structure recurses 20 deep on
real input, and anything that assumes a container has several children is wrong half the time.

## Half two: the geometric parent finder, on hdbpmn

`parents_from_boxes` proposes, for each box, the **smallest** box that encloses at least `COVER`
of its area and is at least `MIN_SHRINK` larger. Smallest, because a task inside a lane inside a
pool is enclosed by both and only the lane is its parent; the `MIN_SHRINK` floor because two
nearly identical boxes each enclose the other and the innermost rule alone would pick one at
random.

The reference is read off the vector-exact ground-truth geometry: a node's parent is the innermost
ground-truth box tagged `participant`, `lane` or `subProcess` covering 99% of it. **3,141 of 4,419
boxed nodes on 242 pages have a real parent, under 659 containers.** What the finder is *not*
given is the tag - the detector has no pool or lane class, pools and lanes are both plain
`rectangle` - so it must decide from geometry alone which rectangle is a container.

    boxes                       parent accuracy   exact parent recovered
    ground truth                     0.9941               0.9940
    detected (IoU >= 0.5)            0.9226               0.8950
    ---- controls, on ground-truth boxes ----
    outermost instead of innermost   0.6382
    nobody has a parent              0.2892

**The ceiling is 0.9941 and the real number is 0.9226: the detector costs 0.0715 of parent
accuracy**, far less than this task expected to pay. The reason is that containment is forgiving
of exactly the errors a detector makes - a pool box wrong by twenty pixels still encloses the same
tasks - and 9.1's detector finds **95.9% of the 659 containers and 97.1% of all nodes** at IoU
0.5. Nodes whose own box was never detected are excluded from the 0.9226 and counted separately in
`node_match_rate`, so that figure is the accuracy over the nodes assembly actually gets to place
and detection recall is not charged twice.

**Both controls matter, and the second one is why the ceiling is not vacuous.** 28.92% of nodes
genuinely have no parent, so a finder that proposed nothing at all would score 0.2892; and
swapping the innermost rule for the outermost - the naive "the biggest box that contains it"
- collapses to **0.6382, a loss of 0.3559**. Nesting on these pages is real and two levels deep
(task in lane in pool), and the entire difficulty of the geometric half is picking the right level.

**The tolerance barely matters, which was not the expected finding.** Sweeping `COVER` from 0.70
to 0.99 moves detected accuracy over a range of **0.0063** - 0.9168 at 0.99, 0.9231 at 0.80 - and
the nominal peak beats the value kept here by 0.0005, which is **two nodes**. That is not a basis
for choosing a constant, so 0.90 is kept as the middle of a flat region. The truth column of the
sweep rises monotonically toward 0.99 and **must not be read as evidence**: the reference is
itself defined at cover 0.99, so agreement there is partly definitional. **`COVER` exists to
forgive hand-drawn overshoot and hdbpmn's reference boxes are vector-exact, so this corpus cannot
exercise it** - the detected column, where boxes really are sloppy, is flat, and a corpus of
*drawn* container annotations is what would actually set this number.

## What this does not measure

The geometric finder is scored on BPMN pools and lanes because that is the only place nested boxes
and ground truth coexist. Wireframe containment is a different distribution - **20 levels deep
rather than 2**, tighter, with siblings that abut instead of sitting apart - and **nothing here
says 0.9226 transfers to it.** Half one shows the builder handles depth 20; half two shows the
geometry works at depth 2; the corpus that would join those claims is a rendered sketch2code, and
it does not exist.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.assemble.corpus import RUNS, contains_fraction, iou, pages, truth
from src.ir.model import Diagram
from src.utils.parallel import pmap

#: Share of a child's area that must fall inside a candidate parent. Loose enough to forgive a
#: hand-drawn box overshooting its container; the sweep in the write-up shows 0.7-0.95 are within
#: 0.013 of each other, so this is a peak and not a cliff.
COVER = 0.90

#: A parent must be at least this much bigger than its child. Two nearly identical boxes each
#: enclose the other, and "innermost wins" alone would pick one of them arbitrarily.
MIN_SHRINK = 0.90

#: Two children are on the same row if they overlap vertically by this share of the shorter one.
#: Reading order is top-to-bottom *then* left-to-right, which needs a notion of "same row".
ROW_OVERLAP = 0.5

#: Detection-to-truth matching for the "real number" run. 0.5 is the mAP convention.
MATCH_IOU = 0.5

#: The synthetic parent attached above a page's real roots, so every page is a single tree.
PAGE_ROOT = "__page__"

#: BPMN's container elements. The only tags whose ground-truth boxes may legally be a parent.
CONTAINER_TAGS = ("participant", "lane", "subProcess")


@dataclass(frozen=True)
class NestNode:
    """One node offered to the tree builder: an id, optional geometry, a proposed parent."""

    id: str
    bbox: list[float] | None = None
    parent: str | None = None


@dataclass(frozen=True)
class Nesting:
    """A parent-child tree over one page."""

    parent: dict[str, str | None]
    roots: list[str]
    depth: dict[str, int]
    children: dict[str, list[str]]
    #: Relations the builder had to drop to make this a tree, each with the reason.
    dropped: list[dict[str, str]] = field(default_factory=list)

    @property
    def max_depth(self) -> int:
        return max(self.depth.values(), default=0)

    @property
    def well_formed(self) -> bool:
        """One parent each, no cycles, every node reachable from a root."""
        return len(self.depth) == len(self.parent) and all(v >= 0 for v in self.depth.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "roots": list(self.roots),
            "parent": dict(self.parent),
            "children": {k: list(v) for k, v in self.children.items()},
            "depth": dict(self.depth),
            "dropped": list(self.dropped),
        }


# ------------------------------------------------------------------------------------------
# the tree builder
# ------------------------------------------------------------------------------------------


def _reading_order(ids: Sequence[str], boxes: dict[str, list[float] | None]) -> list[str]:
    """Top-to-bottom, then left-to-right, over the ids that have geometry.

    Ids without a box keep their input order and are appended after the placed ones, because
    sketch2code has no geometry at all and a sort that silently reorders those would make the
    tree's shape depend on a value that does not exist.
    """
    placed = [i for i in ids if boxes.get(i)]
    loose = [i for i in ids if not boxes.get(i)]
    rows: list[list[str]] = []
    for node_id in sorted(placed, key=lambda i: (boxes[i][1], boxes[i][0])):
        y, h = boxes[node_id][1], boxes[node_id][3]
        for row in rows:
            ry, rh = boxes[row[0]][1], boxes[row[0]][3]
            overlap = min(y + h, ry + rh) - max(y, ry)
            if overlap > ROW_OVERLAP * min(h, rh):
                row.append(node_id)
                break
        else:
            rows.append([node_id])
    ordered: list[str] = []
    for row in rows:
        ordered.extend(sorted(row, key=lambda i: boxes[i][0]))
    return ordered + loose


def tree(nodes: Iterable[NestNode], *, page_root: bool = True) -> Nesting:
    """Turn proposed parents into a well-formed tree.

    Three things can be wrong with a set of containment relations and each is repaired by a
    stated rule rather than assumed away: a parent that is not a node on this page is dropped;
    a cycle is broken at the edge that closes it, walking nodes in input order; several roots
    are joined under a synthetic `PAGE_ROOT` so that "one root per page" holds by construction.
    Each repair is recorded in `dropped`.
    """
    nodes = list(nodes)
    boxes = {n.id: n.bbox for n in nodes}
    order = [n.id for n in nodes]
    known = set(order)
    dropped: list[dict[str, str]] = []

    parent: dict[str, str | None] = {}
    for node in nodes:
        proposed = node.parent
        if proposed == node.id:
            dropped.append({"node": node.id, "parent": proposed, "reason": "self-parent"})
            proposed = None
        elif proposed is not None and proposed not in known:
            dropped.append({"node": node.id, "parent": proposed, "reason": "unknown-parent"})
            proposed = None
        parent[node.id] = proposed

    # Break cycles: walk up from each node; if we return to a node already on this walk, the
    # last link is the one that closed the loop and it is the one dropped.
    for node_id in order:
        seen = {node_id}
        current = node_id
        while (up := parent[current]) is not None:
            if up in seen:
                dropped.append({"node": current, "parent": up, "reason": "cycle"})
                parent[current] = None
                break
            seen.add(up)
            current = up

    children: dict[str, list[str]] = {i: [] for i in order}
    roots = [i for i in order if parent[i] is None]
    for node_id in order:
        if parent[node_id] is not None:
            children[parent[node_id]].append(node_id)

    if page_root and len(roots) != 1:
        children[PAGE_ROOT] = list(roots)
        for node_id in roots:
            parent[node_id] = PAGE_ROOT
        parent[PAGE_ROOT] = None
        boxes[PAGE_ROOT] = None
        roots = [PAGE_ROOT]

    children = {k: _reading_order(v, boxes) for k, v in children.items()}

    depth: dict[str, int] = {}
    stack = [(r, 0) for r in reversed(roots)]
    while stack:
        node_id, level = stack.pop()
        depth[node_id] = level
        for child in reversed(children.get(node_id, [])):
            stack.append((child, level + 1))
    return Nesting(parent=parent, roots=roots, depth=depth, children=children, dropped=dropped)


def from_contains(diagram: Diagram, *, page_root: bool = True) -> Nesting:
    """Build the tree from IR edges whose `attrs.kind == "contains"` (`src` contains `dst`).

    A node claimed by two `contains` edges keeps the first in file order; the rest are dropped
    with reason `multi-parent`. With no geometry there is nothing to prefer one claim by, which
    is precisely what a rendered corpus would fix.
    """
    proposed: dict[str, str] = {}
    dropped: list[dict[str, str]] = []
    for edge in diagram.edges:
        if (edge.attrs or {}).get("kind") != "contains" or edge.src is None or edge.dst is None:
            continue
        if edge.dst in proposed:
            dropped.append({"node": edge.dst, "parent": edge.src, "reason": "multi-parent"})
            continue
        proposed[edge.dst] = edge.src
    nodes = [NestNode(n.id, n.bbox, proposed.get(n.id)) for n in diagram.nodes]
    built = tree(nodes, page_root=page_root)
    return Nesting(
        parent=built.parent,
        roots=built.roots,
        depth=built.depth,
        children=built.children,
        dropped=dropped + built.dropped,
    )


# ------------------------------------------------------------------------------------------
# the geometric parent finder
# ------------------------------------------------------------------------------------------


def parents_from_boxes(
    boxes: Sequence[tuple[str, list[float]]],
    *,
    cover: float = COVER,
    shrink: float = MIN_SHRINK,
    innermost: bool = True,
) -> dict[str, str | None]:
    """The innermost box that encloses each box, or None.

    Innermost, because a task inside a lane inside a pool is enclosed by both and only the lane
    is its parent. Ties in area are broken by id so two runs agree.
    """
    areas = {i: max(b[2], 0.0) * max(b[3], 0.0) for i, b in boxes}
    out: dict[str, str | None] = {}
    for inner_id, inner in boxes:
        best: str | None = None
        for outer_id, outer in boxes:
            if outer_id == inner_id or areas[inner_id] > shrink * areas[outer_id]:
                continue
            if contains_fraction(inner, outer) < cover:
                continue
            key = (areas[outer_id], outer_id)
            if best is None or (key < (areas[best], best)) == innermost:
                best = outer_id
        out[inner_id] = best
    return out


def from_boxes(
    boxes: Sequence[tuple[str, list[float]]],
    *,
    cover: float = COVER,
    shrink: float = MIN_SHRINK,
    page_root: bool = True,
) -> Nesting:
    """Geometry in, tree out: `parents_from_boxes` then `tree`."""
    proposed = parents_from_boxes(boxes, cover=cover, shrink=shrink)
    return tree(
        [NestNode(i, list(b), proposed.get(i)) for i, b in boxes],
        page_root=page_root,
    )


def reference_parents(diagram: Diagram, *, cover: float = 0.99) -> dict[str, str | None]:
    """Ground-truth nesting for a BPMN page, read off the vector-exact reference boxes.

    Only `participant`, `lane` and `subProcess` may be a parent - that is BPMN's own semantics,
    and it is what makes this a reference rather than a second run of the algorithm under test:
    the finder is never told which rectangles are containers.
    """
    boxed = [n for n in diagram.nodes if n.bbox]
    containers = [n for n in boxed if (n.attrs or {}).get("bpmn_tag") in CONTAINER_TAGS]
    out: dict[str, str | None] = {}
    for node in boxed:
        best = None
        for candidate in containers:
            if candidate.id == node.id or contains_fraction(node.bbox, candidate.bbox) < cover:
                continue
            area = candidate.bbox[2] * candidate.bbox[3]
            if best is None or (area, candidate.id) < (best[0], best[1]):
                best = (area, candidate.id)
        out[node.id] = best[1] if best else None
    return out


def match(detected: Sequence[dict], reference: Sequence[Any], threshold: float = MATCH_IOU) -> dict:
    """Greedy highest-IoU detection -> ground-truth-node mapping. Returns `{det_index: node_id}`."""
    pairs = []
    for d_index, det in enumerate(detected):
        for node in reference:
            score = iou(det["bbox"], node.bbox)
            if score >= threshold:
                pairs.append((score, d_index, node.id))
    pairs.sort(key=lambda p: (-p[0], p[1], p[2]))
    used_d: set[int] = set()
    used_n: set[str] = set()
    out: dict[int, str] = {}
    for _, d_index, node_id in pairs:
        if d_index in used_d or node_id in used_n:
            continue
        used_d.add(d_index)
        used_n.add(node_id)
        out[d_index] = node_id
    return out


# ------------------------------------------------------------------------------------------
# the run
# ------------------------------------------------------------------------------------------


def _structure_page(path: Any) -> dict:
    diagram = Diagram.load(path)
    nesting = from_contains(diagram, page_root=False)
    internal = [len(v) for k, v in nesting.children.items() if v]
    return {
        "id": diagram.id,
        "nodes": len(diagram.nodes),
        "relations": sum(1 for e in diagram.edges if (e.attrs or {}).get("kind") == "contains"),
        "roots": len(nesting.roots),
        "max_depth": nesting.max_depth,
        "mean_depth": statistics.fmean(nesting.depth.values()) if nesting.depth else 0.0,
        "branching": internal,
        "well_formed": nesting.well_formed,
        "dropped": nesting.dropped,
        "no_geometry": all(n.bbox is None for n in diagram.nodes),
    }


def structure(source: str = "sketch2code", n_jobs: int = 4) -> dict:
    """Half one: the tree builder over every page whose containment is ground truth."""
    from src.assemble.corpus import IR

    files = sorted((IR / source).glob("*.ir.json"))
    rows = pmap(_structure_page, files, n_jobs=n_jobs, desc=f"nesting/{source}")
    branching = [b for r in rows for b in r["branching"]]
    depths = [r["max_depth"] for r in rows]
    dropped: dict[str, int] = {}
    for row in rows:
        for entry in row["dropped"]:
            dropped[entry["reason"]] = dropped.get(entry["reason"], 0) + 1
    return {
        "source": source,
        "pages": len(rows),
        "nodes": sum(r["nodes"] for r in rows),
        "contains_relations": sum(r["relations"] for r in rows),
        # The trap, asserted from the files rather than from the write-up.
        "pages_without_geometry": sum(1 for r in rows if r["no_geometry"]),
        "well_formed_pages": sum(1 for r in rows if r["well_formed"]),
        "pages_with_one_root": sum(1 for r in rows if r["roots"] == 1),
        "pages_needing_repair": sum(1 for r in rows if r["dropped"]),
        "repairs": dropped,
        "repaired_pages": sorted({r["id"] for r in rows if r["dropped"]})[:10],
        "max_depth": max(depths, default=0),
        "mean_depth": (
            round(statistics.fmean([d for r in rows for d in [r["mean_depth"]] * r["nodes"]]), 4)
            if rows
            else 0.0
        ),
        "depth_p99": round(_quantile(depths, 0.99), 2),
        "branching_median": round(statistics.median(branching), 3) if branching else 0.0,
        "branching_p99": round(_quantile(branching, 0.99), 2),
        "internal_nodes": len(branching),
    }


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return float(ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))])


def _geometry_page(name: str) -> dict:
    from src.assemble.corpus import detections

    page = next(p for p in pages() if p.name == name)
    diagram = truth(page)
    boxed = [n for n in diagram.nodes if n.bbox]
    reference = reference_parents(diagram)
    detected = [d for d in detections(page) if d["cls"] != "arrowhead"]
    mapping = match(detected, boxed)
    truth_boxes = [(n.id, n.bbox) for n in boxed]
    det_boxes = [(f"d{i}", d["bbox"]) for i, d in enumerate(detected)]
    to_node = {f"d{i}": node_id for i, node_id in mapping.items()}
    containers = {n.id for n in boxed if (n.attrs or {}).get("bpmn_tag") in CONTAINER_TAGS}
    row: dict[str, Any] = {
        "page": name,
        "nodes": len(boxed),
        "with_parent": sum(1 for p in reference.values() if p),
        "containers": len(containers),
        "containers_matched": sum(1 for node_id in to_node.values() if node_id in containers),
        "nodes_matched": len(to_node),
        "sweep": {},
    }
    outermost = parents_from_boxes(truth_boxes, innermost=False)
    row["outermost_correct"] = sum(1 for i, p in reference.items() if outermost.get(i) == p)
    row["no_parent_baseline"] = sum(1 for p in reference.values() if p is None)
    for cover in COVER_SWEEP:
        predicted_truth = parents_from_boxes(truth_boxes, cover=cover)
        predicted_det_raw = parents_from_boxes(det_boxes, cover=cover)
        predicted_det = {
            to_node[k]: to_node.get(v) if v else None
            for k, v in predicted_det_raw.items()
            if k in to_node
        }
        row["sweep"][str(cover)] = {
            "outer_correct": 0,
            "truth_correct": sum(1 for i, p in reference.items() if predicted_truth.get(i) == p),
            "truth_parent_correct": sum(
                1 for i, p in reference.items() if p and predicted_truth.get(i) == p
            ),
            "det_n": len(predicted_det),
            "det_correct": sum(1 for i, p in predicted_det.items() if reference.get(i) == p),
            "det_parent_n": sum(1 for i in predicted_det if reference.get(i)),
            "det_parent_correct": sum(
                1 for i, p in predicted_det.items() if reference.get(i) and reference[i] == p
            ),
        }
    return row


#: The tolerance sweep. Wide on purpose: the claim that this constant barely matters is only
#: worth making if the range that was tried is stated.
COVER_SWEEP = (0.70, 0.80, 0.90, 0.95, 0.99)


def geometry(n_jobs: int = 4) -> dict:
    """Half two: boxes -> containment, on ground-truth boxes and on detected boxes."""
    names = [p.name for p in pages() if p.source == "hdbpmn"]
    rows = pmap(_geometry_page, names, n_jobs=n_jobs, prefer="threads", desc="nesting/hdbpmn")
    nodes = sum(r["nodes"] for r in rows)
    with_parent = sum(r["with_parent"] for r in rows)
    containers = sum(r["containers"] for r in rows)
    sweep = {}
    for cover in COVER_SWEEP:
        key = str(cover)
        cells = [r["sweep"][key] for r in rows]
        det_n = sum(c["det_n"] for c in cells)
        det_parent_n = sum(c["det_parent_n"] for c in cells)
        sweep[key] = {
            "truth_accuracy": round(sum(c["truth_correct"] for c in cells) / max(1, nodes), 4),
            "truth_parent_recall": round(
                sum(c["truth_parent_correct"] for c in cells) / max(1, with_parent), 4
            ),
            "detected_accuracy": round(sum(c["det_correct"] for c in cells) / max(1, det_n), 4),
            "detected_parent_recall": round(
                sum(c["det_parent_correct"] for c in cells) / max(1, det_parent_n), 4
            ),
        }
    chosen = sweep[str(COVER)]
    return {
        "source": "hdbpmn",
        "pages": len(rows),
        "nodes": nodes,
        "nodes_with_parent": with_parent,
        "containers": containers,
        "container_match_rate": round(
            sum(r["containers_matched"] for r in rows) / max(1, containers), 4
        ),
        "node_match_rate": round(sum(r["nodes_matched"] for r in rows) / max(1, nodes), 4),
        "cover": COVER,
        # Two controls. A page where nothing contains anything scores the first of these, and
        # any parent finder that does not beat it has found nothing.
        "no_parent_baseline": round(sum(r["no_parent_baseline"] for r in rows) / max(1, nodes), 4),
        "outermost_ablation": round(sum(r["outermost_correct"] for r in rows) / max(1, nodes), 4),
        "truth_accuracy": chosen["truth_accuracy"],
        "detected_accuracy": chosen["detected_accuracy"],
        # The number the plan should quote: what the detector costs this task.
        "detector_cost": round(chosen["truth_accuracy"] - chosen["detected_accuracy"], 4),
        "sweep": sweep,
    }


def run(n_jobs: int = 4) -> dict:
    return {
        "structure": structure(n_jobs=n_jobs),
        "geometry": geometry(n_jobs=n_jobs),
        "constants": {
            "COVER": COVER,
            "MIN_SHRINK": MIN_SHRINK,
            "ROW_OVERLAP": ROW_OVERLAP,
            "MATCH_IOU": MATCH_IOU,
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", type=Path, default=RUNS / "nesting.json")
    args = ap.parse_args(argv)

    result = run(n_jobs=args.jobs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
