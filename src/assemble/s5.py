"""Success criterion S5 - graph edit distance from the assembled IR to the ground-truth IR.

    python -m src.assemble.s5                    # val, the split every choice is made on
    python -m src.assemble.s5 --split test       # the frozen split, run once
    python -m src.assemble.s5 --limit 40         # a quick pass while iterating

The plan's bar is **median GED <= 3 edits**. This module is what actually composes Phase 10's
stages into one predicted `Diagram` per page and diffs it against truth, and it reports the
edit *mass* by kind rather than the aggregate alone, because a single median says nothing about
which stage to fix.

## The metric, and why the number this row used to carry was not it

The previous write-up of S5 quoted a **median of 22**, from an "edge-only lower bound" computed
as `|traced| + |truth| - 2 * matched`. That expression is not the GED this project validated in
10.2.5, and it is harsher than one in three separate ways:

    1  it counts traced polylines with **multiplicity**, so five fragments lying between the same
       two boxes cost five edits. `irdiff.diff` compares **sets** of endpoint pairs, so they cost
       one - which is right, because inserting the same edge five times is one wrong edge in the
       graph, not five.
    2  it counts polylines with a **dangling end**. `irdiff.diff` scores neither side's open ends
       (2.1.6 records them in the ground truth too), so a stub that attached to nothing is not an
       asserted edge and is not charged as one.
    3  it counts **no node edits at all**, which sounds conservative and is, but it also means the
       number was never comparable to the bar, which is a GED over the whole graph.

Run through `irdiff.diff` - the approximation validated against brute-force exact GED on 99.39%
of small-graph pairs - the same pipeline on the same pages measures a **median of 10 on val**,
not 22. The bar is still missed by a wide margin; the point is that the figure the row was
reasoning from was 2.2x the real one and pointed at the wrong stage.

## Where the edits actually are (308 val pages, `match="geometric"`)

    kind                edits    share   what it is
    node_substitute      1959    36.9%   a matched node whose text or shape is wrong
      - text only        1947    36.7%     **the predicted diagram carries no text at all**
      - shape              12     0.2%     the detector's class is essentially right
    edge_delete          1825    34.4%   a true edge no polyline was traced for
    edge_insert          1263    23.8%   an endpoint pair asserted that is not in the truth
    node_insert           113     2.1%   a box that is not a node
    node_delete            86     1.6%   a node that was never detected
    unprojectable          52     1.0%   a traced edge whose endpoint never matched

**Node instantiation is not the problem and has not been for some time**: inserts, deletes and
shape substitutions together are **3.9%** of all edit mass, which is 10.1.1's 0.9837 F1 showing
up where it should. Everything else is text (36.9%) and edges (59.2%).

## The oracle ladder - what each stage is worth if it were perfect

Medians, so the rows are on the bar's own scale rather than on a total's:

    condition                       all val   hdbpmn   fa_bresler   flowchartseg
    as assembled                       10.0     37.0         10.0            0.0
    + perfect text                      5.0     23.5          5.0            0.0
    + perfect edges (text as is)        5.0     14.0          5.0            0.0
    + both perfect                      0.0      1.0          0.0            0.0

**hdbpmn is the whole criterion and the pooled median hides it.** flowchartseg's 132 IR files
carry zero edges and no node text, so they diff to 0.0 for free and carry 93.9% of the pooled
pass rate on their own - the pooled "41.6% of pages at <= 3" is very nearly a statement about how
many flowchartseg pages are in the corpus. On pages that actually have edges the median is
**28.0** and the pass rate is **2.3%**.

## The bar

Perfect text and perfect edges together still leave hdbpmn at a median of 1.0, so the bar is not
unreachable in principle - it is unreachable given what the two upstream stages can deliver.
10.1.3's own perfect-ink ceiling is recall 0.7239 / precision 0.6386, and S3's node-label exact
match is 0.654 on ground-truth crops; neither is anywhere near the "perfect" rows above, and the
floor those two ceilings imply is computed in `ceiling_floor` rather than argued.

## A negative result, measured rather than assumed

10.1.4's gap bridging is built, works at 0.9196 recovery on synthetic gaps, and is **not worth
wiring into this pipeline**: pairing the traced fragments that have exactly one attached end,
with an *oracle* that is told which pairing is correct, recovers **33 of ~1,300 missed hdbpmn
edges (2.5%)**. The tracer's misses are not broken arrows with two stubs; they are edges for
which no usable polyline was traced at all. See `bridge_ceiling`.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

from src.assemble import irdiff, tracing
from src.assemble.corpus import Page, pages, truth
from src.ir.model import Diagram
from src.utils.config import ROOT

#: The report S5 is read from.
REPORT = ROOT / "reports" / "s5_graph_ged.json"

#: The plan's bar: median graph edit distance, in edits.
TARGET = 3.0

#: Every choice in this module is made on `val`; `test` is run once, after freezing.
TUNING_SPLIT = "val"

#: The node-correspondence rule. `geometric` is the only one whose `diff(x, x)` is exact on all
#: 120 real ground-truth graphs (10.2.5), which a GED must have.
MATCH = "geometric"

#: The edit kinds, in the order they are reported.
KINDS = (
    "node_insert",
    "node_delete",
    "sub_text",
    "sub_shape",
    "edge_insert",
    "edge_delete",
    "unprojectable",
)


# ------------------------------------------------------------------------------------------
# assembling one page
# ------------------------------------------------------------------------------------------


#: Sources whose pages are state machines, so 10.1.2's state-label path applies.
STATE_MACHINE_SOURCES = ("fa_bresler",)

#: Sources whose node labels are read by `statelabels`, each by the recogniser trained on it.
#: **flowchartseg is deliberately absent**: its IR records no node text at all, so a label written
#: there cannot match anything and can only manufacture substitutions - 218 of them when
#: `nodetext` was last allowed to do it. That is a property of the annotation, not of the page.
TEXT_SOURCES = ("fa_bresler", "hdbpmn")

#: Where 10.1.5's fitted log-odds weights live, per ablation and per source.
DIRECTION_WEIGHTS = ROOT / "experiments" / "assemble" / "direction.json"


#: A self-loop whose longest supporting polyline is shorter than this fraction of the page
#: diagonal is a fragment, not a drawn loop. Fitted on val, which is s5's tuning split; the
#: claim it supports is only worth what the test run says.
MIN_LOOP_LENGTH = 0.072


def _arc_length(points) -> float:
    if not points or len(points) < 2:
        return 0.0
    return sum(
        ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5
        for a, b in zip(points, points[1:], strict=False)
    )


def _prune_short_loops(page: Page, diagram: Diagram) -> None:
    """Drop self-loops with no polyline long enough to be one (in place).

    The tracer attaches both ends of a stub to the nearest box, and when that box is the same one
    twice the result is a self-loop that was never drawn. On fa_bresler val these are **80 of the
    147 false predicted edges**, and they separate cleanly from real loops by length: the longest
    polyline supporting a true loop has a median of 0.166 of the page diagonal against 0.035 for a
    false one. Cutting at `MIN_LOOP_LENGTH` removes all 80 and costs 4 true loops.

    Grouped by endpoint pair rather than per polyline, because `irdiff` scores pairs as a set: one
    long trace is enough to establish the loop however many short ones accompany it.
    """
    diagonal = getattr(page, "diagonal", 0.0) or 1.0
    longest: dict[tuple[str, str], float] = {}
    for edge in diagram.edges:
        if edge.src is None or edge.dst is None or edge.src != edge.dst:
            continue
        key = (edge.src, edge.dst)
        length = _arc_length(getattr(edge, "polyline", None) or []) / diagonal
        longest[key] = max(longest.get(key, 0.0), length)

    doomed = {key for key, length in longest.items() if length < MIN_LOOP_LENGTH}
    if not doomed:
        return
    diagram.edges = [
        e
        for e in diagram.edges
        if not (e.src is not None and e.src == e.dst and (e.src, e.dst) in doomed)
    ]


def _orient(page: Page, diagram: Diagram) -> None:
    """Give every traced edge a direction, using 10.1.5's fitted model (in place).

    The tracer returns `src`/`dst` in walk order and `directed=False`, while the truth records
    directed edges. `irdiff._edge_keys` keys a directed edge as an ordered pair and an undirected
    one as a sorted pair, so an unoriented prediction can never match a truth edge that does not
    happen to already be in sorted order - 52.6% of hdbpmn's and 23.5% of fa_bresler's. 10.1.5
    was built, but never run or wired; this is that connection.
    """
    from src.assemble import direction

    try:
        fitted = json.loads(DIRECTION_WEIGHTS.read_text(encoding="utf-8"))["weights"]["all"]
    except (OSError, ValueError, KeyError):
        return
    # flowchartseg carries no edges of its own to fit on, so it borrows hdbpmn's - the other
    # flowchart corpus - rather than falling back to an unweighted coin flip.
    weights = fitted.get(page.source) or fitted.get("hdbpmn") or {}
    nodes = {n.id: n for n in diagram.nodes}
    evidence = direction.evidence_for(page, weights)
    for edge in diagram.edges:
        if edge.src is None or edge.dst is None or edge.src == edge.dst:
            continue
        if edge.src not in nodes or edge.dst not in nodes:
            continue
        try:
            decision = direction.resolve(edge, nodes, evidence)
        except Exception:  # noqa: BLE001 - a malformed polyline leaves the edge undirected
            continue
        if decision is None:
            continue
        edge.src, edge.dst, edge.directed = decision.src, decision.dst, True


def assemble(
    page: Page,
    *,
    mask: Any = None,
    text: bool = False,
    state_text: bool = False,
    direct: bool = False,
    prune_loops: bool = False,
) -> Diagram:
    """Phase 10's stages composed into the one predicted `Diagram` S5 is scored on.

    Nodes are 10.1.1's boxes as the tracer sees them (arrowheads dropped, `MIN_SCORE` applied),
    edges are 10.1.3's traced polylines, and containers stay in the node list because they are
    real nodes in the IR even though the tracer refuses to attach to them.

    `text` reads the labels through `src.assemble.nodetext` and is **off by default, because it
    was measured and does not pay**: 9.0 -> 9.0 median on val at the break-even containment
    policy, and 9.0 -> 10.0 once the looser ownership stages are allowed. The edits it removes by
    reading a label exactly are cancelled by the ones it creates on nodes whose truth is blank and
    by labels attached to the wrong node. That table is in `nodetext`, and it is the reason the
    36.7% text edit mass is not recoverable by composing the parts that already exist.
    """
    boxes = tracing.node_boxes(page)
    diagram = tracing.to_diagram(page, boxes, tracing.trace(page, boxes, mask=mask))

    if state_text and page.source in TEXT_SOURCES:
        # 10.1.2 for state machines: the label is inside the circle, so the node box is the crop
        # and no text detector is involved. See `statelabels` for why `nodetext` reads nothing
        # here and why S3's recogniser cannot read what it does find.
        from src.assemble import statelabels

        labels = statelabels.cached(page)
        if labels is None:
            labels = statelabels.read_page(page, boxes)
        for node in diagram.nodes:
            if labels.get(node.id):
                node.text = labels[node.id]

    if prune_loops:
        _prune_short_loops(page, diagram)

    if direct:
        _orient(page, diagram)

    if text:
        from src.assemble import nodetext
        from src.assemble.nodetext import cached

        labels = cached(page, boxes)
        if labels is None:
            labels = nodetext.assign(nodetext.read_page(page, boxes), boxes)
        for node in diagram.nodes:
            if labels.get(node.id):
                node.text = labels[node.id]
    return diagram


def decompose(predicted: Diagram, actual: Diagram, *, match: str = MATCH) -> dict[str, Any]:
    """One page's GED split into the edit kinds that make it up.

    `node_substitute` is split into text and shape because they are two different upstream
    stages and pooling them hides that one of them is 36.7% of all error mass and the other is
    0.2%.
    """
    result = irdiff.diff(predicted, actual, match=match)
    edits = result.edits
    substitutions = edits["node_substitute"]
    shape_wrong = sum(1 for s in substitutions if s["shape"][0] != s["shape"][1])
    return {
        "ged": result.ged,
        "node_insert": len(edits["node_insert"]),
        "node_delete": len(edits["node_delete"]),
        "sub_text": len(substitutions) - shape_wrong,
        "sub_shape": shape_wrong,
        "edge_insert": len(edits["edge_insert"]),
        "edge_delete": len(edits["edge_delete"]),
        "unprojectable": result.counts["edges_lost_to_unmatched_nodes"],
        "true_nodes": result.counts["true_nodes"],
        "true_edges": result.counts["true_edges"],
        "predicted_nodes": result.counts["predicted_nodes"],
        "predicted_edges": result.counts["predicted_edges"],
        "node_f1": round(result.node_f1, 4),
        "edge_f1": round(result.edge_f1, 4),
    }


def score_page(page: Page) -> dict[str, Any]:
    """`decompose` for one held-out page, with the page's identity attached."""
    row = decompose(
        assemble(page, state_text=True, direct=True, prune_loops=True), truth(page)
    )
    return {"page": page.name, "source": page.source, "split": page.split, **row}


# ------------------------------------------------------------------------------------------
# aggregation, and the oracle ladder
# ------------------------------------------------------------------------------------------


def _text_free(row: dict) -> float:
    return row["ged"] - row["sub_text"]


def _edge_free(row: dict) -> float:
    return row["ged"] - row["edge_insert"] - row["edge_delete"] - row["unprojectable"]


def _both_free(row: dict) -> float:
    return _edge_free(row) - row["sub_text"]


def summarise(rows: list[dict], target: float = TARGET) -> dict[str, Any]:
    """Median GED, the pass rate against the bar, the error mass by kind, and the ladder."""
    if not rows:
        return {"pages": 0}
    total = {k: sum(r[k] for r in rows) for k in KINDS}
    mass = sum(total.values())
    median = st.median(r["ged"] for r in rows)
    values = sorted(r["ged"] for r in rows)
    quartiles = st.quantiles(values, n=4) if len(values) > 1 else [median, median, median]
    return {
        "pages": len(rows),
        "median_ged": median,
        "mean_ged": round(st.mean(values), 3),
        "p25_ged": quartiles[0],
        "p75_ged": quartiles[2],
        "pass_share": round(sum(1 for r in rows if r["ged"] <= target) / len(rows), 4),
        "passes": median <= target,
        "edit_mass": total,
        "edit_share": {k: round(v / mass, 4) if mass else 0.0 for k, v in total.items()},
        "ladder": {
            "as_assembled": median,
            "perfect_text": st.median(_text_free(r) for r in rows),
            "perfect_edges": st.median(_edge_free(r) for r in rows),
            "both_perfect": st.median(_both_free(r) for r in rows),
        },
    }


def ceiling_floor(rows: list[dict]) -> dict[str, Any]:
    """The median GED the two upstream ceilings imply, rather than a perfect-stage oracle.

    10.1.3's perfect-ink control traces at recall 0.7239 / precision 0.6386, and S3 reads a node
    label exactly right 0.654 of the time on ground-truth crops. Applying both to each page's own
    true counts gives the best median this assembly chain could reach if every remaining stage
    were run at its own measured ceiling and nothing else changed.
    """
    recall, precision, exact = 0.7239, 0.6386, 0.654
    floors = []
    for row in rows:
        true_edges = row["true_edges"]
        hits = recall * true_edges
        asserted = hits / precision if precision else 0.0
        edge_cost = (true_edges - hits) + max(0.0, asserted - hits)
        text_cost = (1.0 - exact) * row["sub_text"]
        node_cost = row["node_insert"] + row["node_delete"] + row["sub_shape"]
        floors.append(edge_cost + text_cost + node_cost)
    return {
        "assumptions": {"edge_recall": recall, "edge_precision": precision, "text_exact": exact},
        "median_floor": round(st.median(floors), 2) if floors else None,
        "share_at_or_below_target": (
            round(sum(1 for f in floors if f <= TARGET) / len(floors), 4) if floors else None
        ),
    }


# ------------------------------------------------------------------------------------------
# the bridging ceiling: 10.1.4 against real fragments rather than synthetic gaps
# ------------------------------------------------------------------------------------------


def bridge_ceiling(page: Page) -> dict[str, Any]:
    """How many missed edges an *oracle* pairing of one-ended traced fragments could recover.

    An oracle, deliberately: it is told which pairing is right. Whatever 10.1.4's geometry would
    actually achieve is bounded above by this, so a small number here closes the question without
    building the adapter.
    """
    diagram = truth(page)
    boxes = tracing.node_boxes(page)
    edges = tracing.trace(page, boxes)
    identity = tracing.identify(boxes, diagram)
    wanted = {
        frozenset((e.src, e.dst))
        for e in diagram.edges
        if e.src is not None and e.dst is not None and e.src != e.dst
    }

    have: set[frozenset] = set()
    anchors: list[str] = []
    for edge in edges:
        left, right = identity.get(edge.src or ""), identity.get(edge.dst or "")
        if edge.src is not None and edge.dst is not None:
            if left and right and left != right:
                have.add(frozenset((left, right)))
        elif edge.src is not None or edge.dst is not None:
            anchor = left if edge.src is not None else right
            if anchor:
                anchors.append(anchor)

    recovered: set[frozenset] = set()
    used: set[int] = set()
    for i, j in combinations(range(len(anchors)), 2):
        if i in used or j in used or anchors[i] == anchors[j]:
            continue
        key = frozenset((anchors[i], anchors[j]))
        if key in wanted and key not in have and key not in recovered:
            recovered.add(key)
            used |= {i, j}
    return {
        "page": page.name,
        "source": page.source,
        "true_pairs": len(wanted),
        "traced_pairs": len(have),
        "missed": len(wanted - have),
        "one_ended_fragments": len(anchors),
        "oracle_recovered": len(recovered),
    }


# ------------------------------------------------------------------------------------------
# the run
# ------------------------------------------------------------------------------------------


def run(split: str = TUNING_SPLIT, limit: int | None = None, n_jobs: int = 6) -> dict[str, Any]:
    """Score every page of one split, by source, with both ceilings."""
    from src.utils.parallel import pmap

    held = [p for p in pages() if p.split == split]
    if limit:
        held = held[:limit]

    # Read the labels here, once, before the workers start: see `nodetext.build_cache`.
    from src.assemble.nodetext import build_cache

    read = build_cache(held, tracing.node_boxes)
    if read:
        print(f"[s5] read labels on {read} pages", flush=True)

    # The state-machine recogniser is a second model, and the same argument applies: one copy in
    # the parent, looked up by the workers.
    from src.assemble.statelabels import build_cache as build_state_cache

    state_pages = [p for p in held if p.source in TEXT_SOURCES]
    if state_pages:
        read = build_state_cache(state_pages, tracing.node_boxes)
        if read:
            print(f"[s5] read state labels on {read} pages", flush=True)

    rows = pmap(score_page, held, n_jobs=n_jobs, prefer="threads", desc=f"s5-{split}")
    bridges = pmap(bridge_ceiling, held, n_jobs=n_jobs, prefer="threads", desc="s5-bridge")

    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_source[row["source"]].append(row)

    with_edges = [r for r in rows if r["true_edges"] > 0]
    overall = summarise(rows)
    return {
        "criterion": "S5",
        "target_median_ged": TARGET,
        "split": split,
        "match": MATCH,
        "metric": (
            "src.assemble.irdiff.diff - the assignment-based GED validated against brute-force "
            "exact GED on 99.39% of small-graph pairs (10.2.5). Endpoint pairs are compared as "
            "sets and open ends are scored on neither side, which is what makes this smaller "
            "than the multiplicity-counting edge-only bound this row used to quote."
        ),
        "overall": overall,
        "pages_with_edges": summarise(with_edges),
        "by_source": {s: summarise(rs) for s, rs in sorted(by_source.items())},
        "ceiling_floor": ceiling_floor(with_edges),
        "bridging": {
            "missed": sum(b["missed"] for b in bridges),
            "one_ended_fragments": sum(b["one_ended_fragments"] for b in bridges),
            "oracle_recovered": sum(b["oracle_recovered"] for b in bridges),
            "note": (
                "10.1.4's gap bridging against the fragments 10.1.3 really produces, with an "
                "oracle pairing. The recovery is a rounding error, so the module is left "
                "unwired and the reason is measured rather than assumed."
            ),
        },
        "passes": overall["passes"],
        "pages": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default=TUNING_SPLIT, choices=("val", "test"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    result = run(args.split, args.limit, args.jobs)
    default = REPORT if args.split == "val" else REPORT.with_name("s5_graph_ged_test.json")
    out = args.out or default
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "pages"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
