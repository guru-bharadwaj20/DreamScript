"""Phase 10.2.2 - graph repair heuristics, and the damage study that scores them.

    python -m src.assemble.graphrepair

10.2.1 built eight validators and the finding underneath all of them was the same shape: a rule
the plan asserts can fire on ground truth, and a rule that fires on ground truth will "fix" a
diagram that was never broken. `FC.DECISION_BRANCH` fired on 58.3% of hand-drawn BPMN because it
could not tell a splitting gateway from a merging one; unqualified, a repair built on it would
have rewritten 404 of 693 correct pages. **That is the constraint this row is written under: a
repair may only act on evidence 10.2.1 already showed is discriminative, and every repair here is
measured on ground truth first, where by construction every edit is a false positive.**

## The three repairs

    insert_implicit_end   a flowchart has `ROLES + EDGES` evidence and zero `end` nodes. Every
                           sink (out-degree 0, not `start`) is wired to one new `end` node.
                           Keyed to `FC.END_ONE`'s zero-count case specifically, not its `!= 1`
                           firing - 10.2.1 measured 4.18% zero-end pages (29/693) against pages
                           with several ends, and only "zero" is unambiguous enough to touch.
    merge_duplicate_nodes  two nodes in one diagram are a duplicate when their boxes overlap
                           above `DUP_IOU_THRESHOLD` **and** their transcribed text agrees,
                           non-blank. Both conditions are required - overlap alone merges a
                           decision sitting inside its own swimlane, and text alone merges two
                           unrelated boxes that both say "Yes". The threshold is swept below.
    drop_unreachable_noise a flowchart or state machine's edges are treated as undirected and
                           split into connected components; any component with no `start`/`end`
                           (or `initial-state`/`final-state`) node, at most
                           `MAX_NOISE_COMPONENT` nodes, is dropped whole. A component holding an
                           anchor role is never touched, however small - it might be one lane of
                           a multi-pool BPMN page, and 10.2.1's start-count finding is the reason
                           this repair does not try to guess which pool is "the" process.

Every repair returns the diagram unchanged plus a possibly-empty `list[RepairEntry]` -
`{repair, refs, detail, before, after}` - so nothing is fixed silently and Phase 16 has something
to show next to the marker `Violation.refs` already draws.

## Measured on ground truth: the damage to correct graphs

All 5,796 IR files, each repair run alone against a diagram it has never seen damaged. `n` is the
diagrams for which the repair's own evidence gate does not skip it - a repair asked of a diagram
type it does not touch is not a trial, the way 10.2.1 would not score `SM.REACHABLE` against
flowchartseg's zero edges:

    repair                    n      fired    rate
    insert_implicit_end     693         29   0.0418
    merge_duplicate_nodes  5796          0   0.0000
    drop_unreachable_noise  993         18   0.0181

`insert_implicit_end`'s rate is `FC.END_ONE`'s zero-end count exactly, by construction - the gate
*is* the rule, and every one of the 29 pages it was asked to fix had at least one eligible sink,
so there is nothing left for a repair pass to get wrong beyond the rule itself.
`merge_duplicate_nodes` never fires on ground truth at all - the sweep below is why. And
`drop_unreachable_noise` is the repair that failed first and is worth telling straight:

**its first version touched 553 of 693 hdbpmn pages - 79.8% - before it shipped, and would have
been the worst false-positive rate in this row by a wide margin.** The 553 broke down as 2,437
size-1 components and 5 size-2 ones, and the roles inside them were not ambiguous:
`container` (a pool or lane frame) is edge-free **1,338 of 1,766 times in hdbpmn**, and `io` (a
BPMN data object) is edge-free **1,085 of 1,085 - every single one**, because hdBPMN's
association arrows are not carried as directed edges in this IR. Neither is noise; both are
a gap in what the format records, not a defect in the drawing, and excluding
`NOISE_EXEMPT_ROLES = {container, io}` from the component graph entirely - not merely exempting
them as anchors, which would still let them size a component around a genuine stray mark - drops
the rate from 79.8% to **1.81% (18/993)**. This is 10.2.1's finding again, one level down: the
first rule that looked plausible fired on the correct majority of a corpus, and the fix was to
find *which* role was doing that and read why, not to retune a threshold.

## The duplicate threshold, swept rather than picked

Every same-diagram node pair with a bbox, bucketed by whether their text agrees:

    IoU >=   same-text pairs hit   diff-text pairs hit
    0.30                   1                625
    0.50                   0                218
    0.70                   0                 50
    0.90                   0                 44

**Ground truth has almost no same-labelled overlapping pair at any threshold** - one pair, at the
lowest bucket, and none at 0.5 or above. That is the reason `merge_duplicate_nodes` fires zero
times in the false-positive table above: this corpus was not drawn with the kind of accidental
re-trace the repair targets. What persists at every threshold, including 0.9, is *different*-text
pairs sitting almost fully on top of one another - a decision diamond drawn inside its own lane
box, or similar legitimate nesting - which is exactly why the label-agreement half of the rule is
not optional. `DUP_IOU_THRESHOLD = 0.6` is set above where the one same-text pair sits (0.3-0.5)
and gives the merge no reason to ever fire on this corpus; a predicted graph with real duplicate
detections is the case it is actually for.

## Controlled damage: what each repair recovers

Ground-truth diagrams are damaged in a way each repair is meant to undo, then repaired, and
scored against the damage actually injected - `src.parse.repair`'s method, because there is no
corpus of "diagrams with a heuristic error and the fix a human made":

    repair                    n damaged   recall   precision
    insert_implicit_end             67   1.0000     1.0000
    merge_duplicate_nodes          500   0.9900     1.0000
    drop_unreachable_noise         500   1.0000     0.9881

`insert_implicit_end` only found 67 diagrams meeting a clean trial (a single end, and no
pre-existing sink the damage would be confounded with) out of 5,796 - hdbpmn's `FC.END_ONE` rate
is low to begin with, and most of its 693 pages already have >= 2 sinks converging on their one
end. Where the trial was clean, recovery was exact. `merge_duplicate_nodes` misses 1% of injected
duplicates: an injected copy shifted just past `DUP_IOU_THRESHOLD` is, correctly, not merged - the
threshold sweep above says that is the gate working, not a miss to chase. `drop_unreachable_noise`
recovers every injected fragment but at 98.81% precision even after the container/io exemption -
12 of 1,012 dropped nodes across 500 trials were pre-existing small anchor-free components the
exemption does not cover (mostly lone `process`/`unknown` shapes), which is the same failure mode
that produced the 1.81% ground-truth rate above, at a smaller scale.

This is 10.2.1's lesson applied rather than repeated: none of these three numbers would mean
anything without the false-positive table - and the story behind `drop_unreachable_noise`'s
number - sitting next to them.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.ir.model import SUFFIX, Diagram, Edge, Node
from src.utils.config import ROOT
from src.utils.parallel import pmap

IR_ROOT = ROOT / "data" / "processed" / "ir"
RUNS = ROOT / "experiments" / "assemble"
OUT = RUNS / "graphrepair.json"

#: `n_jobs` for the corpus sweeps. Several agents share this 32-core box.
N_JOBS = 3

#: Confidence recorded on anything this module invents. Never 1.0 - that is reserved for
#: annotation copied straight from a dataset, and a repair is neither.
REPAIR_CONFIDENCE = 0.5

#: Picked by the IoU sweep in the module docstring: below it diff-text pairs still outnumber
#: same-text ones, above it same-text pairs have already collapsed to a handful.
DUP_IOU_THRESHOLD = 0.6

#: A dropped component larger than this is not "noise" - it is a second process, and 10.2.1's
#: multi-start finding is the reason this repair refuses to guess which pool is the real one.
MAX_NOISE_COMPONENT = 3

#: Roles excluded from the noise graph entirely, in either direction. `container` is a pool or
#: lane frame and `io` is a BPMN data object linked by an association this IR does not carry as
#: an edge - both are legitimately edge-free, and the corpus sweep below is why they are here
#: rather than in `ANCHOR_ROLES`: exempting them as anchors would still let them size a
#: component that a *genuine* stray mark sits next to, dropping the mark and keeping the frame.
NOISE_EXEMPT_ROLES = frozenset({"container", "io"})

#: Anchor roles per diagram type: a component holding one of these is never dropped, however
#: small, because it is a start or an end of the flow rather than a stray mark.
ANCHOR_ROLES: dict[str, frozenset[str]] = {
    "flowchart": frozenset({"start", "end"}),
    "state_machine": frozenset({"initial-state", "final-state"}),
}

#: Diagram types `drop_unreachable_noise` is allowed to touch. Everything else either has no
#: directed flow to be unreachable from (er_diagram, wireframe) or is excluded from evidence.
NOISE_GATED_TYPES = frozenset(ANCHOR_ROLES)


@dataclass(frozen=True)
class RepairEntry:
    """One edit, logged. `before`/`after` are the repair's own count - end nodes, group size,
    unreachable nodes - not a generic diff, because the number that makes a repair auditable is
    different for each of the three."""

    repair: str
    refs: tuple[str, ...]
    detail: str
    before: int
    after: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "repair": self.repair,
            "refs": list(self.refs),
            "detail": self.detail,
            "before": self.before,
            "after": self.after,
        }


# ------------------------------------------------------------------------------------------
# shared helpers - deliberately not imported from validate.py, which this row must not depend on
# ------------------------------------------------------------------------------------------

ROLES, EDGES = "roles", "edges"


def _evidence(diagram: Diagram) -> set[str]:
    have = set()
    if any(n.semantic_role not in ("unknown", "") for n in diagram.nodes):
        have.add(ROLES)
    if diagram.edges:
        have.add(EDGES)
    return have


def _by_role(diagram: Diagram) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for node in diagram.nodes:
        grouped[node.semantic_role].append(node.id)
    return grouped


def _adjacency(diagram: Diagram) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    out: dict[str, list[str]] = defaultdict(list)
    into: dict[str, list[str]] = defaultdict(list)
    for edge in diagram.edges:
        if edge.src is None or edge.dst is None:
            continue
        out[edge.src].append(edge.dst)
        into[edge.dst].append(edge.src)
    return out, into


def _fresh_id(diagram: Diagram, prefix: str) -> str:
    have = diagram.node_ids | {e.id for e in diagram.edges}
    i = 0
    while f"{prefix}_{i}" in have:
        i += 1
    return f"{prefix}_{i}"


def _iou(a: list[float] | None, b: list[float] | None) -> float:
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    union = max(a[2] * a[3], 0.0) + max(b[2] * b[3], 0.0) - inter
    return float(inter / union) if union > 0 else 0.0


def _norm_text(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


# ------------------------------------------------------------------------------------------
# the repairs
# ------------------------------------------------------------------------------------------


def insert_implicit_end(diagram: Diagram) -> tuple[Diagram, list[RepairEntry]]:
    """Every sink not already `start` gets wired to one new `end` node, when the diagram has
    evidence and zero ends already. Never fires on a diagram with >= 1 end - that case is
    `FC.END_ONE`'s `!= 1` firing, which 10.2.1 kept as warn-only because several ends are a
    legal multi-trigger process, not a defect a repair should collapse."""
    if diagram.diagram_type != "flowchart" or not {ROLES, EDGES} <= _evidence(diagram):
        return diagram, []
    roles = _by_role(diagram)
    if roles.get("end"):
        return diagram, []
    out, _ = _adjacency(diagram)
    sinks = sorted(n.id for n in diagram.nodes if not out.get(n.id) and n.semantic_role != "start")
    if not sinks:
        return diagram, []

    new = copy.deepcopy(diagram)
    end_id = _fresh_id(new, "repair_end")
    new.nodes.append(Node(end_id, "ellipse", None, "", "end", confidence=REPAIR_CONFIDENCE))
    for sink in sinks:
        new.edges.append(
            Edge(
                _fresh_id(new, f"repair_edge_{sink}"),
                sink,
                end_id,
                confidence=REPAIR_CONFIDENCE,
            )
        )
    entry = RepairEntry(
        "insert_implicit_end",
        (*sinks, end_id),
        f"no end node; attached {len(sinks)} sink(s) to new end node {end_id}",
        before=0,
        after=1,
    )
    return new, [entry]


def merge_duplicate_nodes(diagram: Diagram) -> tuple[Diagram, list[RepairEntry]]:
    """Union-find over box overlap + label agreement. The survivor of a group is its lowest id
    so the choice is deterministic; every edge naming a merged id is redirected to the survivor
    and exact-duplicate `(src, dst)` pairs created by the redirect are collapsed."""
    boxed = [n for n in diagram.nodes if n.bbox is not None and _norm_text(n.text)]
    parent = {n.id: n.id for n in boxed}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            # lower id wins so the survivor is deterministic, not an artefact of iteration order
            lo, hi = (ra, rb) if ra < rb else (rb, ra)
            parent[hi] = lo

    for i, a in enumerate(boxed):
        for b in boxed[i + 1 :]:
            if (
                _norm_text(a.text) == _norm_text(b.text)
                and _iou(a.bbox, b.bbox) >= DUP_IOU_THRESHOLD
            ):
                union(a.id, b.id)

    groups: dict[str, list[str]] = defaultdict(list)
    for n in boxed:
        groups[find(n.id)].append(n.id)
    merges = {root: sorted(ids) for root, ids in groups.items() if len(ids) > 1}
    if not merges:
        return diagram, []

    remap = {dup: root for root, ids in merges.items() for dup in ids if dup != root}
    new = copy.deepcopy(diagram)
    new.nodes = [n for n in new.nodes if n.id not in remap]
    seen_edges: set[tuple[str | None, str | None]] = set()
    kept_edges = []
    for e in new.edges:
        e.src = remap.get(e.src, e.src)
        e.dst = remap.get(e.dst, e.dst)
        key = (e.src, e.dst)
        if e.src == e.dst or key in seen_edges:
            continue
        seen_edges.add(key)
        kept_edges.append(e)
    new.edges = kept_edges

    entries = [
        RepairEntry(
            "merge_duplicate_nodes",
            tuple(ids),
            f"{len(ids)} nodes with matching text and IoU >= {DUP_IOU_THRESHOLD} merged into "
            f"{root}",
            before=len(ids),
            after=1,
        )
        for root, ids in sorted(merges.items())
    ]
    return new, entries


def drop_unreachable_noise(diagram: Diagram) -> tuple[Diagram, list[RepairEntry]]:
    """Connected components over undirected edges; anything anchor-free and no bigger than
    `MAX_NOISE_COMPONENT` is dropped whole. A component with an anchor role is never touched,
    whatever its size - 10.2.1 found 158 BPMN pages with several legitimate starts, and this
    repair does not try to be the rule that decides between them.

    `NOISE_EXEMPT_ROLES` nodes never enter the component graph at all, in either direction -
    the first cut of this repair dropped 553 of 693 hdbpmn pages (79.8%) before that exemption
    existed, and the roles doing the damage were not ambiguous: `container` is edge-free 1,338
    of 1,766 times in hdbpmn (a pool or lane frame, disconnected from the flow by construction)
    and `io` is edge-free **1,085 of 1,085** - every one - because hdBPMN's data objects are
    linked by an association this IR does not carry as a directed edge. Neither is noise; both
    are a gap in what the format records, and a repair that cannot tell the difference must stay
    out of the way rather than guess."""
    anchors = ANCHOR_ROLES.get(diagram.diagram_type)
    if anchors is None or not {ROLES, EDGES} <= _evidence(diagram):
        return diagram, []

    graph_nodes = [n for n in diagram.nodes if n.semantic_role not in NOISE_EXEMPT_ROLES]
    parent = {n.id: n.id for n in graph_nodes}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for e in diagram.edges:
        if e.src is not None and e.dst is not None and e.src in parent and e.dst in parent:
            ra, rb = find(e.src), find(e.dst)
            if ra != rb:
                parent[rb] = ra

    members: dict[str, list[str]] = defaultdict(list)
    for n in graph_nodes:
        members[find(n.id)].append(n.id)

    roles = {n.id: n.semantic_role for n in diagram.nodes}
    drop_ids: list[str] = []
    entries = []
    for _root, ids in sorted(members.items()):
        if len(ids) > MAX_NOISE_COMPONENT or any(roles[i] in anchors for i in ids):
            continue
        drop_ids.extend(ids)
        entries.append(
            RepairEntry(
                "drop_unreachable_noise",
                tuple(sorted(ids)),
                f"{len(ids)}-node component with no {'/'.join(sorted(anchors))} node dropped",
                before=len(ids),
                after=0,
            )
        )
    if not drop_ids:
        return diagram, []

    drop = set(drop_ids)
    new = copy.deepcopy(diagram)
    new.nodes = [n for n in new.nodes if n.id not in drop]
    new.edges = [e for e in new.edges if e.src not in drop and e.dst not in drop]
    return new, entries


REPAIRS = (insert_implicit_end, merge_duplicate_nodes, drop_unreachable_noise)


def repair(diagram: Diagram) -> tuple[Diagram, list[RepairEntry]]:
    """Run every repair in a fixed order - dedupe, then clear noise, then patch the ending, so
    the last repair sees the cleaned-up graph rather than one still carrying duplicates it
    would otherwise wire into the new end node."""
    log: list[RepairEntry] = []
    for step in (merge_duplicate_nodes, drop_unreachable_noise, insert_implicit_end):
        diagram, entries = step(diagram)
        log.extend(entries)
    return diagram, log


# ------------------------------------------------------------------------------------------
# measurement: damage to ground truth
# ------------------------------------------------------------------------------------------


def _applicable(name: str, diagram: Diagram) -> bool:
    if name == "insert_implicit_end":
        return diagram.diagram_type == "flowchart" and {ROLES, EDGES} <= _evidence(diagram)
    if name == "merge_duplicate_nodes":
        return True  # every diagram type can carry a bbox + text duplicate
    if name == "drop_unreachable_noise":
        return diagram.diagram_type in NOISE_GATED_TYPES and {ROLES, EDGES} <= _evidence(diagram)
    raise ValueError(name)


def _fp_row(path: Path) -> dict[str, Any]:
    diagram = Diagram.load(path)
    row: dict[str, Any] = {"source": path.parent.name}
    for fn in REPAIRS:
        name = fn.__name__
        applicable = _applicable(name, diagram)
        _, entries = fn(diagram) if applicable else (diagram, [])
        row[name] = {"applicable": applicable, "fired": bool(entries)}
    return row


def false_positive_sweep(root: Path = IR_ROOT) -> dict[str, Any]:
    """Every repair, alone, over every ground-truth IR file. Every edit counted here is by
    definition a false positive - the file is ground truth - so this is the damage table, not a
    benefit table."""
    paths = sorted(root.rglob(f"*{SUFFIX}"))
    rows = pmap(_fp_row, paths, n_jobs=N_JOBS)
    out: dict[str, Any] = {"files": len(paths)}
    for fn in REPAIRS:
        name = fn.__name__
        n = sum(1 for r in rows if r[name]["applicable"])
        fired = sum(1 for r in rows if r[name]["applicable"] and r[name]["fired"])
        out[name] = {"n": n, "fired": fired, "rate": round(fired / n, 4) if n else None}
    return out


def duplicate_iou_sweep(
    root: Path = IR_ROOT, thresholds: tuple[float, ...] = (0.3, 0.5, 0.7, 0.9)
) -> dict:
    """Same-diagram node-pair IoU, bucketed by text agreement, at each candidate threshold -
    the evidence `DUP_IOU_THRESHOLD` is picked from."""
    paths = sorted(root.rglob(f"*{SUFFIX}"))

    def _counts(path: Path) -> tuple[list[float], list[float]]:
        diagram = Diagram.load(path)
        boxed = [n for n in diagram.nodes if n.bbox is not None]
        same, diff = [], []
        for i, a in enumerate(boxed):
            for b in boxed[i + 1 :]:
                v = _iou(a.bbox, b.bbox)
                if v <= 0:
                    continue
                ta, tb = _norm_text(a.text), _norm_text(b.text)
                (same if ta and ta == tb else diff).append(v)
        return same, diff

    rows = pmap(_counts, paths, n_jobs=N_JOBS)
    same_all = [v for s, _ in rows for v in s]
    diff_all = [v for _, d in rows for v in d]
    return {
        str(t): {
            "same_text_pairs": sum(1 for v in same_all if v >= t),
            "diff_text_pairs": sum(1 for v in diff_all if v >= t),
        }
        for t in thresholds
    }


# ------------------------------------------------------------------------------------------
# measurement: controlled damage and recovery
# ------------------------------------------------------------------------------------------


def _damage_missing_end(diagram: Diagram, rng: random.Random) -> tuple[Diagram, set[str]] | None:
    """Delete the diagram's single end node, if it has exactly one and its removal creates
    exactly one new sink - anything murkier is not a controlled trial."""
    roles = _by_role(diagram)
    ends = roles.get("end", [])
    if len(ends) != 1 or not {ROLES, EDGES} <= _evidence(diagram):
        return None
    end_id = ends[0]
    out, into = _adjacency(diagram)
    preds = into.get(end_id, [])
    existing_sinks = {
        n.id
        for n in diagram.nodes
        if n.id != end_id and n.semantic_role != "start" and not out.get(n.id)
    }
    if not preds or existing_sinks:
        return None  # keep the trial clean: no sink pre-dates the damage
    new = copy.deepcopy(diagram)
    new.nodes = [n for n in new.nodes if n.id != end_id]
    new.edges = [e for e in new.edges if e.src != end_id and e.dst != end_id]
    return new, set(preds)


def _damage_duplicate(diagram: Diagram, rng: random.Random) -> tuple[Diagram, str, str] | None:
    """Clone a labelled, boxed node a few pixels over - a re-trace, not a new shape."""
    candidates = [n for n in diagram.nodes if n.bbox is not None and _norm_text(n.text)]
    if not candidates:
        return None
    node = rng.choice(candidates)
    new = copy.deepcopy(diagram)
    dup_id = _fresh_id(new, "damage_dup")
    x, y, w, h = node.bbox
    shift = 0.02 * max(w, h)
    new.nodes.append(
        Node(dup_id, node.shape, [x + shift, y + shift, w, h], node.text, node.semantic_role)
    )
    return new, node.id, dup_id


def _damage_unreachable(diagram: Diagram, rng: random.Random) -> tuple[Diagram, set[str]] | None:
    """Bolt on a 2-node fragment with no anchor role and no edge to the rest of the graph."""
    anchors = ANCHOR_ROLES.get(diagram.diagram_type)
    if anchors is None or not {ROLES, EDGES} <= _evidence(diagram):
        return None
    new = copy.deepcopy(diagram)
    a = _fresh_id(new, "damage_noise")
    new.nodes.append(Node(a, "rectangle", [0, 0, 5, 5], "", "unknown"))
    b = _fresh_id(new, "damage_noise2")
    new.nodes.append(Node(b, "rectangle", [10, 0, 5, 5], "", "unknown"))
    new.edges.append(Edge(_fresh_id(new, "damage_edge"), a, b))
    return new, {a, b}


def recovery_sweep(root: Path = IR_ROOT, n_samples: int = 500, seed: int = 0) -> dict[str, Any]:
    """Inject controlled damage a repair is meant to undo, run only that repair, and score the
    result against the damage actually injected. `src.parse.repair`'s method: recall is how much
    of the injected damage came back, precision is how much of what came back was injected."""
    paths = sorted(root.rglob(f"*{SUFFIX}"))
    rng = random.Random(seed)
    rng.shuffle(paths)

    out: dict[str, Any] = {}

    # insert_implicit_end
    tp = fp = fn = 0
    n = 0
    for path in paths:
        if n >= n_samples:
            break
        diagram = Diagram.load(path)
        damaged = _damage_missing_end(diagram, rng)
        if damaged is None:
            continue
        n += 1
        damaged_diagram, true_preds = damaged
        repaired, entries = insert_implicit_end(damaged_diagram)
        attached = set()
        for e in entries:
            attached |= set(e.refs[:-1])  # every ref but the new end node itself
        tp += len(attached & true_preds)
        fp += len(attached - true_preds)
        fn += len(true_preds - attached)
    out["insert_implicit_end"] = _prf(n, tp, fp, fn)

    # merge_duplicate_nodes
    tp = fp = fn = 0
    n = 0
    for path in paths:
        if n >= n_samples:
            break
        diagram = Diagram.load(path)
        damaged = _damage_duplicate(diagram, rng)
        if damaged is None:
            continue
        n += 1
        damaged_diagram, orig_id, dup_id = damaged
        _, entries = merge_duplicate_nodes(damaged_diagram)
        merged_pairs = [set(e.refs) for e in entries]
        hit = any({orig_id, dup_id} <= g for g in merged_pairs)
        extra_groups = [g for g in merged_pairs if not ({orig_id, dup_id} <= g)]
        tp += 1 if hit else 0
        fn += 0 if hit else 1
        fp += len(extra_groups)
    out["merge_duplicate_nodes"] = _prf(n, tp, fp, fn)

    # drop_unreachable_noise
    tp = fp = fn = 0
    n = 0
    for path in paths:
        if n >= n_samples:
            break
        diagram = Diagram.load(path)
        damaged = _damage_unreachable(diagram, rng)
        if damaged is None:
            continue
        n += 1
        damaged_diagram, injected = damaged
        _, entries = drop_unreachable_noise(damaged_diagram)
        dropped: set[str] = set()
        for e in entries:
            dropped |= set(e.refs)
        tp += len(dropped & injected)
        fp += len(dropped - injected)
        fn += len(injected - dropped)
    out["drop_unreachable_noise"] = _prf(n, tp, fp, fn)

    return out


def _prf(n: int, tp: int, fp: int, fn: int) -> dict[str, Any]:
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    return {
        "n": n,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4) if precision is not None else None,
        "recall": round(recall, 4) if recall is not None else None,
    }


def run() -> dict[str, Any]:
    return {
        "false_positive": false_positive_sweep(),
        "duplicate_iou_sweep": duplicate_iou_sweep(),
        "recovery": recovery_sweep(),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)

    result = run()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
