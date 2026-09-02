"""Phase 7.3.1 - the hidden state space, frozen, and the four states nothing in the IR carries.

    python -m src.parse.roles              # the space, the mapping, the judgement calls
    python -m src.parse.roles --coverage   # ... and the annotation/derivation split, once
                                           #     7.3.3 exists to supply the traversal

7.3 reads a diagram as a **sequence**: the hidden states are semantic roles, the observations are
what was drawn at each visited node, and Viterbi recovers the logical flow that Phase 12 turns
into code. This task freezes the state alphabet, which is the decision every later task in 7.3 is
built on and the one that cannot be changed afterwards without invalidating four learned matrices.

The plan names nine states:

    start, input, process, decision, branch-true, branch-false, loop-back, output, terminal

## The problem this task has to solve, stated plainly

2.1.5's `ROLES` vocabulary is a **node** vocabulary - what a box means on its own - and it has 17
entries. The plan's nine are not a subset of it. Five of the nine map onto roles directly and
**four do not exist as node annotations at all**:

    branch-true    a node is not "the true branch"; the *edge* reaching it is
    branch-false   the same
    loop-back      a node is not a loop; the edge that returns to it is
    input/output   `io` in 2.1.5 is one role covering both directions

So four of the nine states are properties of **how a node is reached**, not of the node. That is
not a flaw in the plan's state space - it is what makes the sequence model worth building, since
a per-node classifier cannot express any of them - but it does mean the supervision for those
states has to be *derived*, and the derivation has to be written down honestly rather than
buried. `derive_states` below is that derivation, and `coverage()` reports how much of each state
comes from a real annotation against how much comes from a rule.

## The mapping, and the two judgement calls in it

    IR role            HMM state      note
    start              start
    initial-state      start          a state machine's entry is a start
    end                terminal
    final-state        terminal
    process            process
    state              process        a state is what the machine is doing: a process
    event              process        judgement call, see below
    container          process        BPMN pools and lanes; they carry no flow of their own
    join               process        judgement call, see below
    decision           decision
    fork               decision       one input, several outputs, chosen or taken in parallel
    io (in-degree 0)   input          a source `io` is where data enters
    io (out-degree 0)  output         a sink `io` is where it leaves
    io (otherwise)     input          see below

**`event` -> process** is the first judgement call. BPMN's intermediate events (12% of hdbpmn's
labelled nodes) are timers and message receipts - things that happen and then flow continues -
and the plan's nine states have no `event`. `process` is the state whose transition structure
they share; calling them `input` would be defensible for message events and wrong for timers.

**`join` -> process** is the second. A join has several inputs and one output, which is the
mirror of `fork`, and the plan's space has no merge state. It is mapped to `process` rather than
`decision` because a decision *chooses* an outgoing edge and a join does not choose anything -
grouping it with `decision` would corrupt the one transition row 7.3.10's repair rule depends on
("a decision node must have >= 2 outgoing").

Both calls are recorded in `JUDGEMENT_CALLS` so 7.3.9's per-role scores can be read with them in
view, and 7.3.11's ablation can be told which states are annotation and which are inference.

## What the derived states cost, measured rather than asserted

`coverage()` over the 993 sequential labelled diagrams (693 hdbpmn flowcharts, 300 fa_bresler
state machines, 14,056 nodes):

    state          rows    annotation   degree   edge label   traversal order
    process       4,916      4,742         -          -            174
    branch-true   1,513          -         -         67          1,446
    branch-false  1,384          -         -         38          1,346
    decision      1,494      1,494         -          -              -
    loop-back     1,368          -         -          -          1,368
    terminal      1,160      1,160         -          -              -
    start         1,136      1,136         -          -              -
    input           782          -       338          -            444
    output          303          -       303          -              -

**5,524 of 14,056 assignments - 39.3% - are rules rather than annotations**, which is four times
what a glance at the state list suggests, because the three derived states are not rare: branches
and loop-backs are 30% of all nodes between them.

Two numbers inside that are the ones to carry forward. **Only 105 of 2,897 branch assignments
(3.6%) come from an edge label**; the rest come from the ordering fallback, because hdbpmn labels
its edges with message names ("claim", "offer") and fa_bresler labels them with alphabet symbols,
and neither dataset writes "yes" on a page. And **every `loop-back` and every `output` is a
derivation** - 1,368 and 303 rows with no human ever having said so.

That is the honest ceiling on what 7.3.9 can claim: for four states it scores against a human
label, and for the other five, wholly or mostly, against this file's own definitions. A model
that reproduces `branch-true` perfectly has reproduced `derive_states`, not a reading of the
page. What makes the exercise worth running anyway is that `derive_states` uses information the
HMM never sees - the arrival edge, the sibling ordering, the DFS stack - so reproducing it from
`(shape, keyword, degree)` alone is still a real inference, and 7.3.9 measures how far it gets.
"""

from __future__ import annotations

import argparse
import json
import sys

# ---------------------------------------------------------------------------------------
# The frozen state space
# ---------------------------------------------------------------------------------------

#: Frozen at Phase 7.3.1, in the plan's order. Every matrix in 7.3.4 through 7.3.12 is indexed
#: by `STATES.index(...)`, so inserting a state anywhere but the end invalidates them all.
STATES: tuple[str, ...] = (
    "start",
    "input",
    "process",
    "decision",
    "branch-true",
    "branch-false",
    "loop-back",
    "output",
    "terminal",
)

#: Index of each state, for building matrices without a linear scan per node.
STATE_INDEX: dict[str, int] = {name: i for i, name in enumerate(STATES)}

#: 2.1.5 role -> 7.3.1 state, for the roles where the node annotation decides it alone.
#: `io` is deliberately absent: its direction is a property of the graph, not the node.
ROLE_TO_STATE: dict[str, str] = {
    "start": "start",
    "initial-state": "start",
    "end": "terminal",
    "final-state": "terminal",
    "process": "process",
    "state": "process",
    "event": "process",
    "container": "process",
    "join": "process",
    "transition": "process",
    "decision": "decision",
    "fork": "decision",
}

#: The two mappings that are arguments rather than facts, kept where a reader will find them.
JUDGEMENT_CALLS: dict[str, str] = {
    "event": "BPMN intermediate events flow like a process; the state space has no event state",
    "join": "a join does not choose an edge, so grouping it with decision would corrupt 7.3.10",
}

#: States that no annotation carries and `derive_states` has to infer.
DERIVED_STATES: tuple[str, ...] = ("branch-true", "branch-false", "loop-back", "input", "output")

#: Edge labels that name a branch outright, lowercased and stripped. Anything else falls back to
#: the ordering rule, and `coverage()` reports how often that happens.
TRUE_LABELS: frozenset[str] = frozenset(
    {"yes", "y", "true", "t", "1", "ok", "accept", "accepted", "valid", "pass", "passed"}
)
FALSE_LABELS: frozenset[str] = frozenset(
    {"no", "n", "false", "f", "0", "reject", "rejected", "invalid", "fail", "failed", "error"}
)

#: Roles a wireframe carries. They are listed to be excluded: a wireframe is a containment tree,
#: not a traversal, and 7.3.3 drops those diagrams rather than inventing an order for them.
NON_SEQUENTIAL_ROLES: frozenset[str] = frozenset(
    {"ui-label", "ui-button", "ui-input", "ui-image", "widget", "screen"}
)


def state_of_role(role: str) -> str | None:
    """The state a node annotation determines on its own, or None if the graph has to decide."""
    return ROLE_TO_STATE.get(role)


def branch_of_label(label: str) -> str | None:
    """`branch-true` / `branch-false` from an edge label, or None when the label does not say."""
    key = (label or "").strip().lower()
    if key in TRUE_LABELS:
        return "branch-true"
    if key in FALSE_LABELS:
        return "branch-false"
    return None


def is_sequential(diagram) -> bool:
    """Whether this diagram is a traversal at all.

    A wireframe is not: its structure is containment, and 7.3.3's DFS over it would produce an
    order with no semantic content. Diagrams with no edges are excluded for the same reason -
    flowchartseg's 1,319 pages carry node polygons and no connectors, so a sequence over them
    would be a sort by position pretending to be a flow.
    """
    roles = {node.get("semantic_role", "unknown") for node in diagram.get("nodes", [])}
    if roles & NON_SEQUENTIAL_ROLES:
        return False
    return bool(diagram.get("edges"))


def degrees(diagram) -> tuple[dict[str, int], dict[str, int]]:
    """In-degree and out-degree per node id.

    A **dangling** edge - one whose other end 2.1.6 could not resolve - still counts for the end
    it does have. 2,021 of the labelled corpus's 16,249 edges (12.4%) are dangling, and ignoring
    them would tell 7.3.2's degree class that a node with three unresolved outgoing arrows is a
    sink. The drawing shows the arrow leaving; only the tracer is unsure where it goes.

    7.3.3's traversal takes the opposite view for the opposite reason - it cannot walk an edge
    with no target - so the two functions disagree deliberately and the difference is what
    `component_breaks` ends up recording.
    """
    incoming: dict[str, int] = {node["id"]: 0 for node in diagram.get("nodes", [])}
    outgoing: dict[str, int] = dict.fromkeys(incoming, 0)
    for edge in diagram.get("edges", []):
        if edge.get("src") in outgoing:
            outgoing[edge["src"]] += 1
        if edge.get("dst") in incoming:
            incoming[edge["dst"]] += 1
    return incoming, outgoing


def derive_states(diagram, order: list[str], back_edges: set[tuple[str, str]]) -> dict[str, str]:
    """State per node id, given a traversal order and the back edges that order induced.

    `order` and `back_edges` come from 7.3.3, which is where a traversal is defined; this function
    does no graph search of its own so that the two tasks cannot disagree about what a back edge
    is. The precedence is deliberate and is the whole content of the derivation:

        1. loop-back   the node is the target of a back edge. This wins over everything, because
                       "the flow returns here" is the fact a sequence model exists to represent,
                       and it is true whatever the node is otherwise.
        2. branch-*    the node is reached from a *decision* by an edge that names a branch, or,
                       failing a label, by the first / a later outgoing edge of that decision.
        3. role        the node annotation, through ROLE_TO_STATE.
        4. io          direction from the degrees.
        5. process     the fallback for an unannotated node in a sequential diagram.
    """
    incoming, outgoing = degrees(diagram)
    roles = {node["id"]: node.get("semantic_role", "unknown") for node in diagram.get("nodes", [])}
    position = {node_id: i for i, node_id in enumerate(order)}

    # Which edge reached each node first in the traversal, and from where.
    arrival: dict[str, dict] = {}
    for edge in sorted(
        diagram.get("edges", []),
        key=lambda e: position.get(e.get("src"), len(position)),
    ):
        src, dst = edge.get("src"), edge.get("dst")
        if src is None or dst is None or dst in arrival:
            continue
        arrival[dst] = edge

    # Sibling order among a decision's outgoing edges, used when no label names the branch.
    sibling_rank: dict[str, int] = {}
    for node_id in roles:
        if ROLE_TO_STATE.get(roles[node_id]) != "decision":
            continue
        outs = [e for e in diagram.get("edges", []) if e.get("src") == node_id]
        outs.sort(key=lambda e: position.get(e.get("dst"), len(position)))
        for rank, edge in enumerate(outs):
            sibling_rank[edge.get("id", f"{node_id}->{edge.get('dst')}")] = rank

    states: dict[str, str] = {}
    for node_id, role in roles.items():
        if any(dst == node_id for _, dst in back_edges):
            states[node_id] = "loop-back"
            continue

        edge = arrival.get(node_id)
        if edge is not None and ROLE_TO_STATE.get(roles.get(edge.get("src"), "")) == "decision":
            named = branch_of_label(edge.get("label", ""))
            if named is not None:
                states[node_id] = named
                continue
            rank = sibling_rank.get(edge.get("id", ""), 0)
            states[node_id] = "branch-true" if rank == 0 else "branch-false"
            continue

        mapped = state_of_role(role)
        if mapped is not None:
            states[node_id] = mapped
            continue

        if role == "io":
            states[node_id] = "output" if outgoing[node_id] == 0 and incoming[node_id] else "input"
            continue

        states[node_id] = "process"
    return states


def provenance(diagram, order: list[str], back_edges: set[tuple[str, str]]) -> dict[str, str]:
    """Where each node's state came from: `annotation`, `degree`, `label` or `order`."""
    incoming, outgoing = degrees(diagram)
    roles = {node["id"]: node.get("semantic_role", "unknown") for node in diagram.get("nodes", [])}
    assigned = derive_states(diagram, order, back_edges)
    position = {node_id: i for i, node_id in enumerate(order)}

    arrival: dict[str, dict] = {}
    for edge in sorted(
        diagram.get("edges", []), key=lambda e: position.get(e.get("src"), len(position))
    ):
        if edge.get("src") and edge.get("dst") and edge["dst"] not in arrival:
            arrival[edge["dst"]] = edge

    out: dict[str, str] = {}
    for node_id, state in assigned.items():
        if state == "loop-back":
            out[node_id] = "order"
        elif state in ("branch-true", "branch-false"):
            edge = arrival.get(node_id, {})
            out[node_id] = "label" if branch_of_label(edge.get("label", "")) else "order"
        elif state in ("input", "output") and roles[node_id] == "io":
            out[node_id] = (
                "degree" if (outgoing[node_id] == 0 or incoming[node_id] == 0) else "rule"
            )
        elif state_of_role(roles[node_id]) is not None:
            out[node_id] = "annotation"
        else:
            out[node_id] = "rule"
    return out


def coverage(limit: int | None = None) -> dict:
    """How much of each state is a human annotation and how much is this file's own rule."""
    from src.parse.sequences import labelled_diagrams, traversal

    counts: dict[str, dict[str, int]] = {name: {} for name in STATES}
    diagrams = labelled_diagrams(limit=limit)
    for diagram in diagrams:
        order, back_edges = traversal(diagram)
        assigned = derive_states(diagram, order, back_edges)
        source = provenance(diagram, order, back_edges)
        for node_id, state in assigned.items():
            bucket = counts[state]
            bucket[source[node_id]] = bucket.get(source[node_id], 0) + 1

    total = sum(sum(bucket.values()) for bucket in counts.values())
    derived = sum(
        sum(count for key, count in bucket.items() if key != "annotation")
        for bucket in counts.values()
    )
    return {
        "diagrams": len(diagrams),
        "assignments": total,
        "derived_assignments": derived,
        "derived_share": round(derived / total, 4) if total else 0.0,
        "by_state": {
            name: {
                "rows": sum(bucket.values()),
                "sources": dict(sorted(bucket.items(), key=lambda kv: -kv[1])),
            }
            for name, bucket in counts.items()
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--coverage", action="store_true", help="scan the corpus (needs the IR)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    result = {
        "states": list(STATES),
        "n_states": len(STATES),
        "role_to_state": ROLE_TO_STATE,
        "derived_states": list(DERIVED_STATES),
        "judgement_calls": JUDGEMENT_CALLS,
    }
    if args.coverage:
        try:
            result["coverage"] = coverage(args.limit)
        except FileNotFoundError as error:
            print(error, file=sys.stderr)
            return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
