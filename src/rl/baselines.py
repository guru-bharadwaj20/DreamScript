"""Phase 11.2.7 - the non-learned arms of the baseline comparison, the definition of the
"ambiguous set", and the harness that turns the two into a win-rate table.

    python -m src.rl.baselines                 # the full table over the 993 labelled diagrams
    python -m src.rl.baselines --distribution  # only the ambiguity distribution

11.2.7 asks for "learned policy vs. plain topological sort vs. DFS/BFS - win-rate table on
ambiguous set". Three things in that sentence did not exist yet: the baselines, the *ambiguous
set* (nothing in the repo defined it), and any way to score an emission order at all. This module
is those three. The learned arm is deliberately absent - it is registered from outside through
`register_arm`, see "The slot for the learned policy".

## Defining the ambiguous set, and the first inconvenient result

The plan offers candidate properties - multiple valid topological orders, gateway branch points,
cycles, disconnected components, unresolved edges - and every one of them, used as a *binary*
filter, selects almost the whole corpus:

    property                                      diagrams (of 993)
    at least one branch point (out-degree >= 2)     930   93.7%
    at least one unresolved / dropped edge          558   56.2%
    cyclic (a strongly connected component)         631   63.5%
    more than one connected component               555   55.9%
    **any of the above**                            993  100.0%

**Every labelled diagram in this corpus is order-ambiguous.** There is not a single page whose
edges pin down one emission order: the minimum measured ordering freedom is 1.0 bit. So "the
ambiguous set" cannot be a yes/no property - defined that way it *is* the corpus, the table over
it is the table over everything, and the phrase carries no information.

It is therefore defined as a **magnitude**. `ambiguity()` measures
`log2(number of emission orders the flow permits)` as a lower bound, in two additive parts:
Kahn ready-set choices on the SCC-condensation, plus log2(k!) for every cycle of size k, because
inside a cycle the edges imply no order at all. A diagram is in the ambiguous set when that
exceeds `AMBIGUITY_BITS = 4.0` - **at least 16 orders are consistent with the drawing**.

    threshold   diagrams   share    hdbpmn   fa_bresler   median nodes
      >= 0.5       993     100.0%     693       300           14
      >= 1         894      90.0%     693       201           15
      >= 2         833      83.9%     684       149           16
      >= 4  <--    684      68.9%     610        74           18
      >= 8         523      52.7%     521         2           19
      >= 16        408      41.1%     408         0           20
      >= 32        143      14.4%     143         0           24

    median 10.17 bits, quartiles 3.00 / 10.17 / 24.42, max 101.18

Why 4 and not the median: past 8 bits the set is **pure hdbpmn** - fa_bresler's state machines
have a median of 5 nodes and cannot carry that much freedom - and an evaluation set that is one
source is a source confound of the kind the S1 audit already had to chase out of this repo. 4.0
bits is the largest threshold that still keeps both sources (610 / 74) while dropping the 309
pages where a policy has fewer than 16 orders to choose between and 11.2's question is close to
trivial. It is a judgement call, and `sensitivity()` exists so that it does not have to be
trusted: **the ranking of the arms is identical at every threshold from 1 to 16 bits** (below),
so nothing in the conclusion rests on this number.

Two smaller measurement notes. `share_cyclic` here is **0.6354**, not 7.3.3's 0.6862, and the gap
is exactly 50 diagrams - all fa_bresler, all of them pages whose *only* cycle is a **self-loop**.
7.3.3's DFS calls a self-loop a back edge; `adjacency` drops self-loops because they impose no
ordering constraint and would stop Kahn's in-degree for that node ever reaching zero. Both
numbers are right about different questions. Likewise `share_with_a_branch_point` is 0.9366
rather than the 0.987 a raw out-degree count gives, because parallel edges between the same pair
are collapsed to one constraint.

## The win-rate table

Arms are scored by `edge_f1` of the chain their order asserts against the diagram's own edges
(see `score` for why that metric, and what was rejected). A **win is outright** - strictly the
best score on that diagram; where the best score is shared it is counted in `shared_best`
instead, so the win rates never sum past 1.

### Ambiguous set, `total_bits >= 4`, **684 diagrams**

    arm            mean edge F1   mean GED(norm)   outright wins   win rate   shared best
    dfs               0.6247          0.1852            345          50.4%        294
    topological       0.5466          0.2236             44           6.4%        226
    bfs               0.3485          0.3231              1           0.1%        113
    reading           0.1743          0.4043              0           0.0%         14
    all four tied on 12 diagrams

    head-to-head (row strictly beat column)
                  topological    dfs    bfs   reading
    topological         -         44    525     657
    dfs                414         -    568     670
    bfs                 92         1      -     485
    reading              2         0    113       -

### Whole corpus, 993 diagrams - the same shape, weaker margins

    arm            mean edge F1   outright wins   win rate   shared best
    dfs               0.5849           346          34.8%        482
    topological       0.5415           102          10.3%        426
    bfs               0.3986            19           1.9%        328
    reading           0.2021             2           0.2%         82
    all four tied on 64 diagrams

### What the table says, including the parts that are not flattering

**DFS wins, and its margin over plain topological sort is smaller than the win rate suggests.**
0.6247 against 0.5466 mean edge F1 is a real gap, but the two arms produce *identical* orders
often enough that topological sort shares the best score on 226 of the 684 ambiguous diagrams -
a third of the set - and outright beats DFS on 44 of them. On a single-path page there is nothing
for them to disagree about. The 50.4% win rate is earned mostly on multi-component hdbpmn pages,
where DFS finishes a pool before starting the next while Kahn interleaves them in reading order
and breaks the flow at every switch.

**BFS is not a serious contender: 1 outright win in 684.** 7.3.3 rejected it on the argument that
it keeps siblings adjacent and destroys the path a reader follows; this is that argument as a
number. It is still far above the blind control, so the failure is not that BFS ignores the graph
- it is that level order is the wrong linearisation of one.

**The blind control behaves as a control should.** `reading` never once wins on the ambiguous set
and scores 0.1743 against DFS's 0.6247. Had it come close, the metric would have been measuring
page layout rather than flow and the whole table would have been void. It is reported for that
reason, not as a candidate.

**Ties are 43% of the result and are reported as ties.** 294 of the 684 ambiguous diagrams have
no outright winner - 12 of them have all four arms level. A table that split shared wins among
the winners would have made DFS look like a 93% policy, because **DFS is in the top group on
639 of 684 (93.4%)**: 345 outright plus every one of the 294 shared. That is the number to beat,
and it is not 50.4%. A learned arm has to beat the *tie band*, not the win rate, for 11.2 to have
been worth doing.

### Sensitivity - the threshold is not load-bearing

    threshold  diagrams   dfs wins   topological   bfs   reading   dfs mean F1   topo mean F1
      >= 1        894        346         102        1       2         0.6064        0.5574
      >= 2        833        346          92        1       2         0.6130        0.5578
      >= 4        684        345          44        1       0         0.6247        0.5466
      >= 8        523        290          41        1       0         0.6240        0.5410
      >= 16       408        241          36        1       0         0.6129        0.5280

dfs > topological > bfs > reading at every threshold, on both wins and mean edge F1. What *does*
move is topological sort's outright wins - 102 at 1 bit down to 44 at 4 bits - because most of
them were on nearly-linear pages that the harder thresholds remove. That is the one place where
the definition of the ambiguous set changes an arm's number, and it changes it against the arm
the plan named first.

## What was rejected

    a binary ambiguous set        it selects 993 of 993. This is the finding that forced the
                                  bit-magnitude definition, not a preference.
    "has a gateway" as the filter  93.7% of the corpus. Not a filter.
    thresholding at the median
    (10.17 bits)                  gives a set that is 99.6% hdbpmn. Rejected as a source
                                  confound, not because the number was inconvenient.
    bits normalised per node      tried; still correlates 0.66 with node count and still strands
                                  fa_bresler (0 of 300 above 1.2 bits/node), so it buys no source
                                  balance in exchange for losing the direct reading "this many
                                  orders are consistent with the drawing".
    Kendall tau vs. a reference   no ground-truth order exists in this corpus; the only candidate
                                  reference is 7.3.3's DFS, which hands the DFS arm 1.0 by
                                  definition. Circular.
    node F1                       constant 1.0 across arms by construction - all arms permute the
                                  same node set.
    refusing to order a cyclic
    graph                         would delete 63.5% of the corpus from the table. The cycle is
                                  broken by a stated rule instead - see `topological_order`.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from src.assemble.irdiff import diff
from src.ir.model import Diagram, Edge, Node
from src.parse.sequences import components, labelled_diagrams, traversal
from src.utils.config import ROOT

RUNS = ROOT / "experiments" / "rl"
OUT = RUNS / "baselines.json"

#: The matching rule handed to `irdiff.diff`. The reconstructed graph and the truth graph share
#: a node set *by construction* (an emission order is a permutation of the diagram's own nodes),
#: so any identity-exact rule agrees here; `geometric` is chosen because 10.2.5 measured it as
#: one of the two rules for which `diff(x, x)` is exact on all 120 held-out graphs, and because
#: all 14,056 nodes in this corpus carry a bbox (0 without) so it is never undefined. `text` and
#: `combined` are *not* identity-exact - 10.2.5 put their self-diff node F1 at 0.795 and 0.9193 -
#: and would charge every arm equally for a defect of the matcher.
MATCH = "geometric"

#: A diagram counts as ambiguous when at least this many bits of ordering freedom remain after
#: the flow has constrained everything it can - i.e. when at least 2**4 = 16 distinct emission
#: orders are consistent with the diagram's edges. See "Defining the ambiguous set".
AMBIGUITY_BITS = 4.0

#: Thresholds the win-rate table is re-computed at, to show the ranking does not depend on the
#: one above.
SENSITIVITY_BITS = (1.0, 2.0, 4.0, 8.0, 16.0)


# ------------------------------------------------------------------------------------------
# the graph, defensively
# ------------------------------------------------------------------------------------------


def _reading_key(node: dict) -> tuple[float, float]:
    """Top-to-bottom then left-to-right; a node with no bbox sorts last, deterministically.

    The same rule as `src.parse.sequences._reading_key`, restated rather than imported because
    it is a private name there and this module must not reach into another phase's privates.
    """
    bbox = node.get("bbox")
    if not bbox:
        return (float("inf"), float("inf"))
    x, y, _, _ = bbox
    return (round(y, 3), round(x, 3))


def adjacency(diagram: dict) -> tuple[list[str], dict[str, list[str]], dict[str, int], int]:
    """(node ids in reading order, successors, in-degree, edges dropped as unusable).

    Three kinds of edge are dropped and counted rather than crashed on:

        src or dst is None      2.1.6's unresolved ends. 530 of the 993 labelled diagrams
                                (53.4%) carry at least one.
        endpoint is not a node
        of this diagram         a converter artefact; same treatment.
        src == dst              a self-loop implies no ordering constraint and would stop Kahn's
                                in-degree for that node ever reaching zero.

    A node reachable only through a dropped edge therefore looks like a root to every arm, which
    is the honest reading: nothing in the IR says where it comes from. Every arm sees the same
    dropped set, so no arm is advantaged by the choice.
    """
    nodes = sorted(diagram.get("nodes", []), key=_reading_key)
    ids = [n["id"] for n in nodes]
    known = set(ids)
    successors: dict[str, list[str]] = {i: [] for i in ids}
    indegree: dict[str, int] = {i: 0 for i in ids}
    dropped = 0
    seen: set[tuple[str, str]] = set()
    for edge in diagram.get("edges", []):
        src, dst = edge.get("src"), edge.get("dst")
        if src not in known or dst not in known or src == dst:
            dropped += 1
            continue
        if (src, dst) in seen:  # a parallel edge is one constraint, not two
            continue
        seen.add((src, dst))
        successors[src].append(dst)
        indegree[dst] += 1
    rank = {node_id: i for i, node_id in enumerate(ids)}
    for node_id in successors:
        successors[node_id].sort(key=lambda t: rank[t])
    return ids, successors, indegree, dropped


# ------------------------------------------------------------------------------------------
# the three arms the plan names, plus one deliberately blind control
# ------------------------------------------------------------------------------------------


def topological_order(diagram: dict) -> list[str]:
    """Kahn's algorithm, reading-order tie-break, with a stated rule for breaking cycles.

    **A topological order is undefined on a cyclic graph and 681 of the 993 labelled diagrams
    (68.6%) are cyclic** - all 300 state machines and 381 of the 693 BPMN pages. Refusing to
    answer would delete two pages in three from the comparison, so the cycle is *broken*:

        when the ready queue empties with nodes still unemitted, the unemitted node with the
        smallest (remaining in-degree, reading key) is force-admitted, its outstanding incoming
        edges are discarded, and Kahn resumes.

    Smallest remaining in-degree first because that node is the one closest to being legitimate;
    reading key to settle the remaining tie deterministically. The discarded edges are exactly
    the loop-back arcs, so on a single simple cycle this reduces to "enter the loop at the node a
    reader reaches first", which is what a code generator wants. It is a *choice*, not a theorem:
    another break gives another order, and that arbitrariness is a large part of what makes these
    diagrams ambiguous in the first place.

    Disconnected components: every in-degree-0 node is in the initial queue, so components are
    interleaved in reading order rather than emitted one after another. That is a real difference
    from the DFS arm, which finishes a component before starting the next, and the win-rate table
    below is largely a measurement of it.
    """
    ids, successors, indegree, _ = adjacency(diagram)
    remaining = dict(indegree)
    rank = {node_id: i for i, node_id in enumerate(ids)}
    ready = sorted((n for n in ids if remaining[n] == 0), key=lambda n: rank[n])
    emitted: list[str] = []
    done: set[str] = set()
    while len(emitted) < len(ids):
        if not ready:
            stuck = min((n for n in ids if n not in done), key=lambda n: (remaining[n], rank[n]))
            remaining[stuck] = 0
            ready = [stuck]
        node_id = ready.pop(0)
        if node_id in done:
            continue
        done.add(node_id)
        emitted.append(node_id)
        for target in successors[node_id]:
            if target in done:
                continue
            remaining[target] -= 1
            if remaining[target] <= 0:
                ready.append(target)
        ready = sorted(set(ready), key=lambda n: rank[n])
    return emitted


def dfs_order(diagram: dict) -> list[str]:
    """DFS preorder. **Delegates to `src.parse.sequences.traversal`** rather than reimplementing.

    7.3.3 already settled this traversal - roots are the in-degree-0 nodes in reading order, then
    any unvisited node in reading order, children visited in reading order - and its back edges
    are what 7.3.1's `loop-back` state is defined from. A second DFS here would let the RL
    baseline and the HMM sequence disagree about what a depth-first walk of the same page is, so
    this is a thin wrapper and `traversal` stays the single definition.

    Cycles need no special case: a target already on the stack is recorded as a back edge and not
    re-emitted. Disconnected components are concatenated, each finished before the next begins.
    Unresolved edges are skipped by `traversal`'s own `src in children and dst in nodes` guard,
    the same rule `adjacency` applies here.
    """
    order, _ = traversal(diagram)
    seen = set(order)
    tail = [
        n["id"] for n in sorted(diagram.get("nodes", []), key=_reading_key) if n["id"] not in seen
    ]
    # `traversal` reaches every node on this corpus; the tail is belt-and-braces so an arm can
    # never return a non-permutation and silently change the denominators of the table.
    return list(order) + tail


def bfs_order(diagram: dict) -> list[str]:
    """Breadth-first from every in-degree-0 node in reading order, then from any unvisited node.

    Cycles: the visited set makes a cycle terminate, exactly as in DFS. Disconnected components:
    each root's whole component is drained before the next root is taken, so components stay
    contiguous. Unresolved edges are dropped by `adjacency`, which promotes their orphaned
    targets to roots.

    Included because 7.3.3 rejected BFS for the HMM with an argument - it keeps siblings adjacent
    and destroys the path a reader follows - and 11.2.7 is where that argument gets a number.
    """
    ids, successors, indegree, _ = adjacency(diagram)
    rank = {node_id: i for i, node_id in enumerate(ids)}
    roots = [n for n in ids if indegree[n] == 0] + [n for n in ids if indegree[n]]
    order: list[str] = []
    visited: set[str] = set()
    queue: collections.deque[str] = collections.deque()
    for root in roots:
        if root in visited:
            continue
        visited.add(root)
        queue.append(root)
        while queue:
            node_id = queue.popleft()
            order.append(node_id)
            for target in sorted(successors[node_id], key=lambda t: rank[t]):
                if target not in visited:
                    visited.add(target)
                    queue.append(target)
    return order


def reading_order(diagram: dict) -> list[str]:
    """Sort by bbox, top-to-bottom then left-to-right. Ignores the edges entirely.

    Not in the plan's list. It is here as the control that makes the other three interpretable:
    an arm that can see the flow graph should beat one that cannot, and if it does not, the
    metric or the arm is wrong. It cannot fail on a cycle, a disconnection or an unresolved edge
    because it never looks at an edge.
    """
    return [n["id"] for n in sorted(diagram.get("nodes", []), key=_reading_key)]


#: name -> callable(diagram: dict) -> list[str]. The arms of the table.
ARMS: dict[str, Callable[[dict], list[str]]] = {
    "topological": topological_order,
    "dfs": dfs_order,
    "bfs": bfs_order,
    "reading": reading_order,
}

_PROBE: dict[str, Any] = {
    "id": "probe",
    "diagram_type": "flowchart",
    "nodes": [
        {"id": "a", "shape": "terminator", "bbox": [0.0, 0.0, 10.0, 10.0], "text": "start"},
        {"id": "b", "shape": "decision", "bbox": [0.0, 20.0, 10.0, 10.0], "text": "q"},
        {"id": "c", "shape": "process", "bbox": [20.0, 40.0, 10.0, 10.0], "text": "x"},
        {"id": "d", "shape": "process", "bbox": [0.0, 40.0, 10.0, 10.0], "text": "y"},
        {"id": "e", "shape": "terminator", "bbox": [0.0, 60.0, 10.0, 10.0], "text": "end"},
    ],
    "edges": [
        {"id": "e1", "src": "a", "dst": "b", "directed": True},
        {"id": "e2", "src": "b", "dst": "c", "directed": True},
        {"id": "e3", "src": "b", "dst": "d", "directed": True},
        {"id": "e4", "src": "c", "dst": "e", "directed": True},
        {"id": "e5", "src": "d", "dst": "e", "directed": True},
    ],
    "meta": {"source": "probe"},
}


def check_arm(name: str, policy: Callable[[dict], list[str]], diagram: dict | None = None) -> None:
    """Raise unless `policy` returns a permutation of the node ids on a probe diagram.

    `diagram` is the diagram to probe *on*. Passing None falls back to the built-in 5-node
    `_PROBE`, which is right for `register_arm` - a generic contract check before an arm enters
    the registry - and **wrong for `compare`**, which must probe on a diagram from the set it is
    about to run: an arm written for a 3-node diagram legitimately returns 3 ids and was being
    rejected for not returning `_PROBE`'s 5. That is why `compare` passes `diagrams[0]`.
    """
    probe = diagram if diagram is not None else _PROBE
    expected = [n["id"] for n in probe.get("nodes", [])]
    got = policy(probe)
    if sorted(got) != sorted(expected):
        raise ValueError(
            f"arm {name!r} did not return a permutation of the node ids: got {len(got)} ids "
            f"({len(set(got))} distinct), expected {len(expected)}"
        )


def register_arm(name: str, policy: Callable[[dict], list[str]]) -> None:
    """**The slot for the learned policy.** Register one more arm and it joins the same table.

    The contract, checked by `check_arm`:

        policy(diagram: dict) -> list[str]

    a **permutation of `[n["id"] for n in diagram["nodes"]]`** - every node exactly once, no node
    twice, nothing invented. Nothing else is assumed: an arm may look at bboxes, text, roles,
    confidences, or a Q-table. 11.2.1's agent wraps its greedy rollout in a function of that
    shape and calls this:

        from src.rl.baselines import register_arm, run
        register_arm("q_learning", lambda d: agent.rollout(d))
        table = run()["table_ambiguous"]

    Deliberately a registry and not an import: this module must not import `src.rl.env` or any
    learner, so that the baselines can be measured, and regressions in them caught, while the
    learned half of 11.2 is still being written.
    """
    check_arm(name, policy)
    ARMS[name] = policy


# ------------------------------------------------------------------------------------------
# defining the ambiguous set
# ------------------------------------------------------------------------------------------


def _sccs(ids: list[str], successors: dict[str, list[str]]) -> list[list[str]]:
    """Tarjan, iterative. hdbpmn has 200-node pages and Python's recursion limit is 1000."""
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    out: list[list[str]] = []
    counter = 0
    for root in ids:
        if root in index:
            continue
        work: list[tuple[str, int]] = [(root, 0)]
        while work:
            node_id, position = work[-1]
            if position == 0:
                index[node_id] = low[node_id] = counter
                counter += 1
                stack.append(node_id)
                on_stack.add(node_id)
            recursed = False
            for i in range(position, len(successors[node_id])):
                target = successors[node_id][i]
                if target not in index:
                    work[-1] = (node_id, i + 1)
                    work.append((target, 0))
                    recursed = True
                    break
                if target in on_stack:
                    low[node_id] = min(low[node_id], index[target])
            if recursed:
                continue
            work.pop()
            if low[node_id] == index[node_id]:
                component: list[str] = []
                while True:
                    popped = stack.pop()
                    on_stack.discard(popped)
                    component.append(popped)
                    if popped == node_id:
                        break
                out.append(component)
            if work:
                low[work[-1][0]] = min(low[work[-1][0]], low[node_id])
    return out


@dataclass(frozen=True)
class Ambiguity:
    """How much ordering freedom one diagram leaves, and where the freedom comes from."""

    id: str
    source: str
    nodes: int
    #: log2 of the number of emission orders the flow permits, split into its two causes.
    choice_bits: float  # Kahn ready-set choices on the acyclic condensation
    cycle_bits: float  # log2(k!) for each strongly connected component of size k > 1
    total_bits: float
    cyclic: bool
    components: int
    branch_points: int  # out-degree >= 2
    merge_points: int  # in-degree >= 2
    dropped_edges: int  # unresolved / out-of-graph / self-loop

    @property
    def ambiguous(self) -> bool:
        return self.total_bits >= AMBIGUITY_BITS

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "nodes": self.nodes,
            "choice_bits": round(self.choice_bits, 4),
            "cycle_bits": round(self.cycle_bits, 4),
            "total_bits": round(self.total_bits, 4),
            "cyclic": self.cyclic,
            "components": self.components,
            "branch_points": self.branch_points,
            "merge_points": self.merge_points,
            "dropped_edges": self.dropped_edges,
            "ambiguous": self.ambiguous,
        }


def ambiguity(diagram: dict) -> Ambiguity:
    """Measure the ordering freedom of one diagram in **bits**.

    The quantity is `log2(number of emission orders consistent with the diagram's edges)`, the
    only definition of "ambiguous" that is about the thing 11.2 actually chooses - an order -
    rather than about a feature that merely correlates with it. Counting linear extensions
    exactly is #P-complete, so it is bounded below in two additive pieces:

        choice_bits  sum of log2(|ready|) over the steps of one Kahn run on the **condensation**
                     (SCCs collapsed, so the run always completes). Each such step is a place
                     where two nodes are simultaneously legal and the flow does not say which
                     comes first.
        cycle_bits   sum of log2(k!) over strongly connected components of size k > 1. Inside a
                     cycle *no* order is implied by the edges at all, so all k! are consistent.

    It is a lower bound - one Kahn run sees one branch of the choice tree - and it is reported as
    one, but it never claims freedom that is not there.
    """
    ids, successors, indegree, dropped = adjacency(diagram)
    scc = _sccs(ids, successors)
    of = {node_id: i for i, component in enumerate(scc) for node_id in component}
    condensed: dict[int, set[int]] = {i: set() for i in range(len(scc))}
    for node_id in ids:
        for target in successors[node_id]:
            if of[node_id] != of[target]:
                condensed[of[node_id]].add(of[target])
    remaining = {i: 0 for i in range(len(scc))}
    for _source, targets in condensed.items():
        for j in targets:
            remaining[j] += 1
    ready = [i for i in remaining if remaining[i] == 0]
    choice_bits = 0.0
    while ready:
        if len(ready) > 1:
            choice_bits += math.log2(len(ready))
        current = ready.pop(0)
        for target in sorted(condensed[current]):
            remaining[target] -= 1
            if remaining[target] == 0:
                ready.append(target)
    cycle_bits = sum(math.lgamma(len(c) + 1) / math.log(2) for c in scc if len(c) > 1)
    order, _ = traversal(diagram)
    piece = components(diagram, order) if order else []
    return Ambiguity(
        id=diagram.get("id", "?"),
        source=diagram.get("meta", {}).get("source", "unknown"),
        nodes=len(ids),
        choice_bits=choice_bits,
        cycle_bits=cycle_bits,
        total_bits=choice_bits + cycle_bits,
        cyclic=any(len(c) > 1 for c in scc),
        components=len(set(piece)),
        branch_points=sum(1 for n in ids if len(successors[n]) >= 2),
        merge_points=sum(1 for n in ids if indegree[n] >= 2),
        dropped_edges=dropped,
    )


def ambiguous_set(diagrams: list[dict], bits: float = AMBIGUITY_BITS) -> list[dict]:
    """The diagrams whose ordering freedom is at least `bits`. The 11.2.7 evaluation set."""
    return [d for d in diagrams if ambiguity(d).total_bits >= bits]


def ambiguity_distribution(diagrams: list[dict]) -> dict[str, Any]:
    """The measured distribution that justifies the threshold."""
    scores = [ambiguity(d) for d in diagrams]
    total = [s.total_bits for s in scores]
    n = len(scores)
    per_threshold = []
    for bits in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0):
        kept = [s for s in scores if s.total_bits >= bits]
        per_threshold.append(
            {
                "bits": bits,
                "diagrams": len(kept),
                "share": round(len(kept) / n, 4) if n else 0.0,
                "by_source": dict(collections.Counter(s.source for s in kept)),
                "median_nodes": statistics.median([s.nodes for s in kept]) if kept else 0,
            }
        )

    def share(predicate: Callable[[Ambiguity], bool]) -> float:
        return round(sum(1 for s in scores if predicate(s)) / n, 4) if n else 0.0

    return {
        "diagrams": n,
        "median_bits": round(statistics.median(total), 4) if total else 0.0,
        "quartile_bits": [round(q, 4) for q in statistics.quantiles(total, n=4)] if n > 3 else [],
        "min_bits": round(min(total), 4) if total else 0.0,
        "max_bits": round(max(total), 4) if total else 0.0,
        # the single-property candidates 11.2.7 suggests, each measured on the whole corpus
        "share_cyclic": share(lambda s: s.cyclic),
        "share_multi_component": share(lambda s: s.components > 1),
        "share_with_a_branch_point": share(lambda s: s.branch_points > 0),
        "share_with_a_dropped_edge": share(lambda s: s.dropped_edges > 0),
        "share_with_any_freedom": share(lambda s: s.total_bits > 0),
        "by_threshold": per_threshold,
    }


# ------------------------------------------------------------------------------------------
# scoring one emission order
# ------------------------------------------------------------------------------------------


def as_chain(diagram: dict, order: list[str]) -> Diagram:
    """The control flow a code generator would imply from `order`: node i -> node i+1.

    This is the metric's whole premise. An emission order is not itself right or wrong; what is
    right or wrong is the program it produces, and a generator walking an order emits statement
    i+1 immediately after statement i - it *asserts* an edge between them. So an order is scored
    by the graph it asserts, against the graph the diagram actually has.
    """
    nodes = {n["id"]: n for n in diagram.get("nodes", [])}
    return Diagram(
        id=f"{diagram.get('id', '?')}::chain",
        diagram_type=diagram.get("diagram_type", "flowchart"),
        nodes=[
            Node(
                id=node_id,
                shape=nodes[node_id].get("shape", "unknown"),
                bbox=nodes[node_id].get("bbox"),
                text=nodes[node_id].get("text", ""),
            )
            for node_id in order
        ],
        edges=[
            Edge(id=f"c{i}", src=order[i], dst=order[i + 1], directed=True)
            for i in range(len(order) - 1)
        ],
    )


def as_truth(diagram: dict) -> Diagram:
    """The diagram's own graph, as an `irdiff`-shaped `Diagram`."""
    return Diagram(
        id=diagram.get("id", "?"),
        diagram_type=diagram.get("diagram_type", "flowchart"),
        nodes=[
            Node(
                id=n["id"],
                shape=n.get("shape", "unknown"),
                bbox=n.get("bbox"),
                text=n.get("text", ""),
            )
            for n in diagram.get("nodes", [])
        ],
        edges=[
            Edge(
                id=e.get("id", f"e{i}"),
                src=e.get("src"),
                dst=e.get("dst"),
                directed=bool(e.get("directed", True)),
            )
            for i, e in enumerate(diagram.get("edges", []))
        ],
    )


def score(diagram: dict, order: list[str], match: str = MATCH) -> dict[str, float]:
    """Score one emission order. **Reuses 10.2.5's `irdiff.diff` rather than a new metric.**

    Why this metric and not another:

        edge_f1        of `as_chain(order)` against the diagram's own edges. Precision is the
                       share of the order's consecutive pairs that are real flow edges - "how
                       much of what this order asserts is true"; recall is the share of the
                       diagram's edges the order got adjacent - "how much of the flow survives
                       linearisation". Every arm returns a permutation of the same node set, so
                       the chain always has exactly n-1 edges and the truth always the same edge
                       count: **the denominators are identical across arms**, which is what makes
                       this a comparison and not a size effect.
        ged_normalised the same diff's assignment-based graph edit distance over the largest
                       possible edit cost, reported as a second opinion. It is an upper bound on
                       exact GED, which 10.2.5 validated against brute force on 99.39% of
                       small-graph pairs.

    What was rejected:

        node F1        constant 1.0 for every arm by construction - all arms permute the same
                       node set - so it can never separate them. It is still returned, as the
                       assertion that the harness is comparing what it thinks it is.
        Kendall tau
        against a
        reference      there is **no ground-truth reading order** in this corpus. The nearest
                       thing is 7.3.3's DFS traversal, and scoring against that hands the DFS arm
                       a perfect 1.0 by definition. Rejected as circular.
        coverage /
        step count     every arm emits every node exactly once by contract, so both are constant.

    The honest limitation, stated because it is load-bearing: this metric rewards orders whose
    *adjacencies* are edges, so it is biased towards depth-first walking and against any arm that
    interleaves branches. That bias is why the DFS arm is expected to lead - and it means a
    learned arm that beats DFS here beats it on DFS's home ground.
    """
    result = diff(as_chain(diagram, order), as_truth(diagram), match=match)
    return {
        "edge_f1": result.edge_f1,
        "edge_precision": result.edge_precision,
        "edge_recall": result.edge_recall,
        "ged_normalised": result.ged_normalised,
        "node_f1": result.node_f1,
    }


# ------------------------------------------------------------------------------------------
# the table
# ------------------------------------------------------------------------------------------

#: Two edge F1 scores closer than this are a tie, not a win. Far below the spacing of the small
#: integer ratios an F1 over at most a few hundred edges can take, and above 0 so that arms which
#: returned *the same order* - the DFS and topological arms agree exactly on many single-path
#: pages - are never recorded as beating one another on floating-point noise.
TIE = 1e-9


@dataclass
class Table:
    """A win-rate table over one evaluation set."""

    diagrams: int
    arms: list[str]
    mean_edge_f1: dict[str, float] = field(default_factory=dict)
    mean_ged: dict[str, float] = field(default_factory=dict)
    #: outright wins: strictly the best edge F1 on that diagram, no other arm within TIE
    wins: dict[str, int] = field(default_factory=dict)
    #: shared best: tied for the best edge F1 with at least one other arm
    shared: dict[str, int] = field(default_factory=dict)
    win_rate: dict[str, float] = field(default_factory=dict)
    #: arm -> arm -> diagrams where the first strictly beat the second
    head_to_head: dict[str, dict[str, int]] = field(default_factory=dict)
    all_tied: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "diagrams": self.diagrams,
            "arms": list(self.arms),
            "mean_edge_f1": {k: round(v, 4) for k, v in self.mean_edge_f1.items()},
            "mean_ged_normalised": {k: round(v, 4) for k, v in self.mean_ged.items()},
            "wins": dict(self.wins),
            "shared_best": dict(self.shared),
            "win_rate": {k: round(v, 4) for k, v in self.win_rate.items()},
            "head_to_head": {k: dict(v) for k, v in self.head_to_head.items()},
            "all_arms_tied": self.all_tied,
        }


def compare(
    diagrams: list[dict],
    arms: dict[str, Callable[[dict], list[str]]] | None = None,
    match: str = MATCH,
) -> Table:
    """Run every arm on every diagram and build the win-rate table.

    A "win" is an outright one: strictly the best edge F1 on that diagram. Diagrams where the
    best score is shared are counted in `shared_best` and `all_arms_tied` rather than split
    between the winners, because a table whose win rates sum to more than 1 is a table hiding its
    ties - and on this corpus the ties are a large part of the result.
    """
    arms = dict(arms if arms is not None else ARMS)
    names = list(arms)
    for name, policy in arms.items():
        check_arm(name, policy, diagrams[0] if diagrams else None)
    scores: dict[str, list[dict[str, float]]] = {name: [] for name in names}
    wins = {name: 0 for name in names}
    shared = {name: 0 for name in names}
    head: dict[str, dict[str, int]] = {a: {b: 0 for b in names if b != a} for a in names}
    all_tied = 0

    for diagram in diagrams:
        expected = sorted(n["id"] for n in diagram.get("nodes", []))
        row: dict[str, dict[str, float]] = {}
        for name, policy in arms.items():
            order = policy(diagram)
            if sorted(order) != expected:
                raise ValueError(
                    f"arm {name!r} broke the permutation contract on {diagram.get('id')!r}"
                )
            row[name] = score(diagram, order, match=match)
            scores[name].append(row[name])
        best = max(row[name]["edge_f1"] for name in names)
        leaders = [n for n in names if row[n]["edge_f1"] >= best - TIE]
        if len(leaders) == 1:
            wins[leaders[0]] += 1
        else:
            for name in leaders:
                shared[name] += 1
            if len(leaders) == len(names):
                all_tied += 1
        for a in names:
            for b in names:
                if a != b and row[a]["edge_f1"] > row[b]["edge_f1"] + TIE:
                    head[a][b] += 1

    n = max(1, len(diagrams))
    return Table(
        diagrams=len(diagrams),
        arms=names,
        mean_edge_f1={
            k: statistics.fmean([s["edge_f1"] for s in v]) if v else 0.0 for k, v in scores.items()
        },
        mean_ged={
            k: statistics.fmean([s["ged_normalised"] for s in v]) if v else 0.0
            for k, v in scores.items()
        },
        wins=wins,
        shared=shared,
        win_rate={k: v / n for k, v in wins.items()},
        head_to_head=head,
        all_tied=all_tied,
    )


def sensitivity(
    diagrams: list[dict], thresholds: tuple[float, ...] = SENSITIVITY_BITS
) -> list[dict[str, Any]]:
    """The same table at several ambiguity thresholds. The justification for `AMBIGUITY_BITS`.

    A threshold is defensible only if the answer does not hinge on it. This re-runs the table
    over `total_bits >= t` across a range spanning 2 to 65,536 permitted orders; the stability of
    the ranking across that range is the argument for the particular value chosen.
    """
    measured = [(d, ambiguity(d).total_bits) for d in diagrams]
    out = []
    for bits in thresholds:
        table = compare([d for d, b in measured if b >= bits]).to_dict()
        table["bits"] = bits
        out.append(table)
    return out


def run(limit: int | None = None, write: bool = False) -> dict[str, Any]:
    """The whole 11.2.7 measurement: distribution, ambiguous-set table, full-corpus table."""
    diagrams = labelled_diagrams(limit)
    subset = ambiguous_set(diagrams)
    result: dict[str, Any] = {
        "ambiguity_bits_threshold": AMBIGUITY_BITS,
        "distribution": ambiguity_distribution(diagrams),
        "ambiguous_set": {
            "diagrams": len(subset),
            "share": round(len(subset) / max(1, len(diagrams)), 4),
            "by_source": dict(
                collections.Counter(d.get("meta", {}).get("source", "unknown") for d in subset)
            ),
        },
        "table_ambiguous": compare(subset).to_dict(),
        "table_all": compare(diagrams).to_dict(),
        "sensitivity": sensitivity(diagrams),
    }
    if write:
        RUNS.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        result["path"] = str(OUT.relative_to(ROOT))
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 11.2.7 baseline win-rate table")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--distribution", action="store_true", help="only the ambiguity distribution")
    args = ap.parse_args(argv)
    try:
        if args.distribution:
            print(json.dumps(ambiguity_distribution(labelled_diagrams(args.limit)), indent=2))
            return 0
        print(json.dumps(run(args.limit, write=args.write), indent=2))
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
