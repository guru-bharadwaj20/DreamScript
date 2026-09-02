"""Phase 7.3.10 - repairing broken structure from the posteriors, and the damage study that scores it.

    python -m src.parse.repair --damage 0.1

The plan's line: *use HMM posteriors to fix broken arrows (a decision node must have >= 2
outgoing)*. That is a rule about the graph, and the reason it belongs to 7.3 rather than to Phase
10's validators is the qualifier - **the posteriors decide which violations to believe**.

A validator sees a diamond with one outgoing edge and knows something is wrong. It does not know
whether the fix is "add the missing edge" or "this is not a decision". 7.3.8's marginals answer
exactly that: if the node's posterior says `decision` at 0.98, the missing edge is missing; if it
says `decision` at 0.41 with `process` at 0.39, the shape was a misread and the graph is fine.

## The three rules

    missing branch     a node decoded `decision` with fewer than 2 outgoing edges. Repaired by
                       attaching the geometrically nearest node that comes later in reading order
                       and is not already reached, within `MAX_SPAN` of the page diagonal.
    stray terminal     a node decoded `terminal` with outgoing edges. Repaired by dropping the
                       outgoing edge with the lowest confidence - a terminal that continues is
                       usually a mis-traced line crossing the node.
    orphan             a node with no edges at all whose posterior is not `input`/`output` (the
                       two states 7.3.4 found to be genuinely isolated). Repaired by attaching the
                       geometrically nearest node above it, within the same span limit.

Each rule fires only when the deciding node's posterior confidence exceeds `THRESHOLD`, and
7.3.8's calibration is what makes that threshold meaningful: at 0.9 the role is right 97.5% of the
time, so a rule gated there is acting on evidence rather than on a decoder's shrug.

## How it is scored, which is the harder half

There is no corpus of "diagrams with broken arrows and the repairs a human would make". So the
evaluation **injects the damage**: delete a known fraction of the edges at random, run the repair,
and ask how many of the deleted edges came back. That gives three numbers rather than one -

    recovered        deleted edges the repair restored exactly (same source, same target)
    wrong            edges the repair added that were never in the diagram
    missed           deleted edges never restored

- and the second is the one that decides whether this is worth running. A repair that recovers
80% of broken arrows while inventing an equal number of false ones has made the graph worse, and
a study that reported only the recovery rate would not show it.

## What it measured

Damage injected into all 993 labelled diagrams, repaired from the posteriors, scored against the
edges that were deleted:

    damage   gate    deleted   recovered   recovery   false repairs   precision
     5%      0.9        796         7        0.009         688          0.010
    10%      0.9      1,609        16        0.010         970          0.016
    20%      0.9      3,254        51        0.016       1,165          0.042

    gate sweep at 10% damage
     0.5              1,609        85        0.053       1,393          0.058
     0.7              1,609        70        0.044       1,279          0.052
     0.9              1,609        16        0.010         970          0.016

**This does not work, and the failure is worth more than a working version of it would have
been.** Best case across every configuration: 5.3% of broken arrows recovered against a plan
target of 80%, at a precision of 5.8% - **for every edge correctly restored, seventeen wrong ones
are invented.** Running this repair on a real page would degrade the graph badly.

## The gate is anti-correlated with the damage, which is the mechanism

At the 0.9 threshold the `missing_branch` rule - the plan's own example - **fires zero times**.
Not rarely: never, in 993 diagrams. The reason is a feedback loop between the two halves of 7.3:
deleting a decision's outgoing edge changes its degree class from `branching` to `linear`, which
changes its observation symbol from `diamond|empty|branching` (0.73 emission mass on `decision`)
to `diamond|empty|linear` (0.22). Its posterior confidence falls below the gate, and the rule that
exists to repair damaged decisions is switched off by the damage.

Lowering the gate to 0.5 lets it fire 454 times and *does* triple the recovery rate - to 5.3%. The
gate is behaving exactly as 7.3.8's calibration says it should; the problem is that a
well-calibrated confidence in a damaged graph is *correctly* low.

## Why the recovery rate is so low even when the rules fire

A deleted edge leaves nothing behind. The repair is choosing among nodes by position alone, and on
a 20-node BPMN page the nearest later node is the right one perhaps one time in fifteen. The
`orphan` rule fires most (986 times at 10% damage) and is the least accurate, because an isolated
node in this corpus is usually a data object that was *always* isolated rather than a step that
lost its arrows.

## What this actually establishes for Phase 10

**A repair driven by posteriors and node positions is not viable; the missing information is the
stroke.** A real broken arrow is not a deleted edge - it is a polyline with a gap in it, and the
gap has two endpoints, a direction and a length. 10.1.4's gap bridging works on exactly that
evidence and can ask "does this stub point at that node", which is a question this task cannot
form because its damage model destroys the stub.

So the study is a lower bound by construction, and that is stated rather than used as an excuse:
deleting an edge is strictly harder than repairing a broken one. The finding that survives the
caveat is the feedback loop - **whatever repairs a decision in 10.1.4 must not be gated on that
decision's own role confidence**, because the damage suppresses it. The rules are implemented,
exercised and left in place with their measured numbers attached; Phase 10 should re-derive them
on stroke evidence rather than inherit these.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys

import numpy as np

from src.parse.roles import STATES

#: A rule fires only above this posterior confidence. 7.3.8 measured 0.9 as the point where the
#: decoded role is right 97.5% of the time.
THRESHOLD = 0.9

#: A repair may not connect two nodes further apart than this share of the page diagonal. Without
#: it "the nearest unreached node" is unbounded and will happily join opposite corners.
MAX_SPAN = 0.25

SEED = 42


def _centre(node: dict) -> tuple[float, float]:
    bbox = node.get("bbox") or [0, 0, 0, 0]
    return (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)


def _diagonal(nodes: dict) -> float:
    """The page diagonal, from the nodes themselves - the IR carries no canvas size."""
    if not nodes:
        return 1.0
    points = np.array([_centre(node) for node in nodes.values()])
    span = points.max(axis=0) - points.min(axis=0)
    return float(np.hypot(*span)) or 1.0


def _nearest(node_id: str, candidates, nodes: dict, limit: float) -> str | None:
    """The closest candidate within `limit`, or None. Distance, not reading order: a missing
    arrow is short, and the node that follows in reading order can be anywhere on the page."""
    origin = np.array(_centre(nodes[node_id]))
    best, best_distance = None, limit
    for other in candidates:
        distance = float(np.linalg.norm(np.array(_centre(nodes[other])) - origin))
        if distance < best_distance:
            best, best_distance = other, distance
    return best


def roles_and_confidence(
    diagram: dict, sequence: dict, model: dict, alphabet: list[str]
) -> tuple[dict, dict]:
    """Decoded role and its posterior confidence, per node id."""
    from src.parse.posteriors import forward_backward
    from src.parse.viterbi import decode, logs

    index = {symbol: i for i, symbol in enumerate(alphabet)}
    symbols = [index.get(s, 0) for s in sequence["observations"]]
    parameters = logs(model)
    path = decode(symbols, *parameters)
    gamma = forward_backward(symbols, *parameters)
    roles = {node_id: STATES[path[i]] for i, node_id in enumerate(sequence["node_ids"])}
    confidence = {
        node_id: float(gamma[i, path[i]]) for i, node_id in enumerate(sequence["node_ids"])
    }
    return roles, confidence


def repair(diagram: dict, roles: dict, confidence: dict, threshold: float = THRESHOLD) -> dict:
    """Apply the three rules. Returns the repairs made; the diagram is modified in place."""
    nodes = {node["id"]: node for node in diagram["nodes"]}
    outgoing: dict[str, list] = {node_id: [] for node_id in nodes}
    incoming: dict[str, list] = {node_id: [] for node_id in nodes}
    for edge in diagram["edges"]:
        if edge.get("src") in outgoing:
            outgoing[edge["src"]].append(edge)
        if edge.get("dst") in incoming:
            incoming[edge["dst"]].append(edge)

    order = sorted(nodes, key=lambda i: (_centre(nodes[i])[1], _centre(nodes[i])[0]))
    position = {node_id: i for i, node_id in enumerate(order)}
    limit = MAX_SPAN * _diagonal(nodes)
    made = []

    def add_edge(src: str, dst: str, rule: str) -> None:
        edge = {
            "id": f"r{len(diagram['edges'])}",
            "src": src,
            "dst": dst,
            "directed": True,
            "label": "",
            "confidence": round(min(confidence.get(src, 0.0), confidence.get(dst, 1.0)), 4),
            "attrs": {"repaired_by": rule},
        }
        diagram["edges"].append(edge)
        outgoing[src].append(edge)
        incoming[dst].append(edge)
        made.append({"rule": rule, "src": src, "dst": dst})

    # Rule 1 - a decision must have two ways out.
    for node_id, role in roles.items():
        if role != "decision" or confidence.get(node_id, 0) < threshold:
            continue
        if len(outgoing.get(node_id, [])) >= 2:
            continue
        reached = {e["dst"] for e in outgoing.get(node_id, [])}
        candidates = [
            other
            for other in order[position[node_id] + 1 :]
            if other not in reached and other != node_id
        ]
        target = _nearest(node_id, candidates, nodes, limit)
        if target is not None:
            add_edge(node_id, target, "missing_branch")

    # Rule 2 - a terminal does not continue.
    for node_id, role in roles.items():
        if role != "terminal" or confidence.get(node_id, 0) < threshold:
            continue
        extra = outgoing.get(node_id, [])
        if not extra:
            continue
        worst = min(extra, key=lambda e: e.get("confidence", 1.0))
        diagram["edges"].remove(worst)
        outgoing[node_id].remove(worst)
        made.append({"rule": "stray_terminal", "src": node_id, "dst": worst.get("dst")})

    # Rule 3 - an orphan that is not a data object belongs to the flow.
    for node_id, role in roles.items():
        if role in ("input", "output") or confidence.get(node_id, 0) < threshold:
            continue
        if outgoing.get(node_id) or incoming.get(node_id):
            continue
        above = [other for other in order[: position[node_id]] if other != node_id]
        source = _nearest(node_id, above, nodes, limit)
        if source is not None:
            add_edge(source, node_id, "orphan")

    return {"repairs": made, "count": len(made)}


def damage(diagram: dict, fraction: float, seed: int = SEED) -> tuple[dict, list]:
    """Delete a fraction of the edges. Returns (damaged copy, the deleted edges)."""
    copied = copy.deepcopy(diagram)
    rng = random.Random(seed)
    keep, removed = [], []
    for edge in copied["edges"]:
        (removed if rng.random() < fraction else keep).append(edge)
    copied["edges"] = keep
    return copied, removed


def evaluate(
    fraction: float = 0.1, limit: int | None = None, threshold: float = THRESHOLD, seed: int = SEED
) -> dict:
    from src.parse.sequences import build, labelled_diagrams, sequence_of
    from src.parse.viterbi import build_model

    data = build(limit=limit)
    alphabet = sorted({s for seq in data["sequences"] for s in seq["observations"]})
    model = build_model(data["sequences"], alphabet, 1.0)

    recovered = wrong = missed = 0
    fired = {"missing_branch": 0, "stray_terminal": 0, "orphan": 0}
    diagrams = labelled_diagrams(limit=limit)
    touched = 0
    for i, diagram in enumerate(diagrams):
        broken, removed = damage(diagram, fraction, seed + i)
        if not removed:
            continue
        touched += 1
        sequence = sequence_of(broken, labelled=False)
        if sequence is None:
            missed += len(removed)
            continue
        roles, confidence = roles_and_confidence(broken, sequence, model, alphabet)
        result = repair(broken, roles, confidence, threshold)
        added = {(r["src"], r["dst"]) for r in result["repairs"] if r["rule"] != "stray_terminal"}
        gone = {(e.get("src"), e.get("dst")) for e in removed}
        recovered += len(added & gone)
        wrong += len(added - gone)
        missed += len(gone - added)
        for entry in result["repairs"]:
            fired[entry["rule"]] = fired.get(entry["rule"], 0) + 1

    deleted = recovered + missed
    return {
        "damage_fraction": fraction,
        "threshold": threshold,
        "diagrams_damaged": touched,
        "edges_deleted": deleted,
        "recovered": recovered,
        "recovery_rate": round(recovered / deleted, 4) if deleted else 0.0,
        "false_repairs": wrong,
        "precision": round(recovered / max(1, recovered + wrong), 4),
        "missed": missed,
        "rules_fired": fired,
        "net_edges_changed": recovered + wrong,
    }


def run(fractions=(0.05, 0.1, 0.2), limit: int | None = None, thresholds=(0.5, 0.7, 0.9)) -> dict:
    rows = [evaluate(fraction, limit) for fraction in fractions]
    # The gate is swept at one damage level, because 7.3.8's calibration is what the threshold is
    # supposed to be buying and a single operating point cannot show whether it delivers.
    sweep = [evaluate(0.1, limit, threshold) for threshold in thresholds]
    return {
        "target": "the plan asks for >= 80% of broken arrows recovered (10.1.4)",
        "rows": rows,
        "threshold_sweep": sweep,
        "best_recovery": max(row["recovery_rate"] for row in rows + sweep),
        "best_precision": max(row["precision"] for row in rows + sweep),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--damage", type=float, nargs="*", default=[0.05, 0.1, 0.2])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)
    try:
        print(json.dumps(run(tuple(args.damage), args.limit), indent=2))
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
