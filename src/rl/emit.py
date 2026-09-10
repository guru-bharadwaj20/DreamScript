"""Phase 11 - a deterministic code emitter, standing in for Phase 12's fine-tuned LLM.

    python -m src.rl.emit            # the measured table below, over the real IR corpus

## Why this file exists when `src.codegen.targets` already emits code

`src.codegen.targets` is the committed Phase 12.1.6 emitter: five languages, `emit(diagram,
traversal) -> str`, a measured 1.0000 parse rate over 5,796 diagrams. It was tried here first and
**rejected for the reward path on measurement**, not on preference:

    over 600 diagrams          src.codegen.targets      src.rl.emit (this module)
    gold order, any undefined         1.0000                    0.0000
    reversed order                    1.0000                    0.0000
    first-half order                  1.0000                    0.9900

**`targets` reports an undefined name on 100% of orders, so it separates nothing.** Its flowchart
fallback is a `while True:` dispatch that calls a helper per node and never defines any of them,
so a perfect traversal and a half-finished one are indistinguishable to any static check of its
output. It also has no parameter for `mark-as-loop`: the policy's loop decisions cannot reach the
emitted text at all, so the reward could not price the action the plan's 11.1.2 requires.

`targets` is the right emitter for Phase 12's *training pairs*, where the input is a whole
diagram and a complete traversal. It is the wrong instrument for a *reward*, which must score
partial, out-of-order and loop-annotated emissions. Both exist, and this is why.

## What this module is not

**Phase 12 does not exist.** This turns (IR + an emitted node order + a loop-mark set) into real
Python by template, so `ast.parse` and `src.rl.sandbox` have something genuine to accept or
reject. **The semantic-correctness bonus in `src.rl.reward` is measured against THIS emitter, not
against an LLM.** A policy trained on it is being trained to produce orders that *this template*
can structure. When Phase 12 lands, `emit_code` should be swapped for the model and every number
below re-measured. Nothing downstream should read the reward as evidence about generated-code
quality (S6/S7); it is evidence about traversal order only.

## The inverted signal that was found and fixed

The inherited version emitted `<name>_cond` and `<name>_again` predicate calls and **deliberately
left them undefined**, documenting the resulting undefined names as the discriminating signal,
with a table reading gold 0.00% / reversed 31.71% / half 61.33%. Re-measured over 600 diagrams
the same code gives **gold 89.83% / reversed 0.83% / half 28.33%** - the exact opposite ranking.
The mechanism is plain once seen: an undefined predicate is emitted only when the emitter
*succeeds* in building an `if` or a `while`, so the metric counted structure the policy got right
and charged it as an error. Downstream, `src.rl.reward.structure_score` scored the gold order
0.8915 and its reverse 0.9997.

Two changes fix it, and both are properties of the emitter rather than of a regex:

    every predicate is defined     `_emit_block` collects what it opens into `opened`, and
                                   `_predicate_defs` writes a definition for each one.
    `EDGES` names every endpoint    a module-level list of `(src_fn, dst_fn)` pairs over every
                                   edge in the diagram, so an undefined name means exactly one
                                   thing: the policy referenced a node it never emitted.

## What was MEASURED, after the fix

Over 3,993 sequential IR diagrams (hdbpmn 693, fa_bresler 300, didi 3,000):

    order            ast.parse OK   any undefined name   mean undefined/calls   order_score
    gold                 100.00%           0.00%                0.0000            0.8942
    reversed gold        100.00%           0.00%                0.0000            0.1058
    first half of gold   100.00%          99.85%                0.7555            0.9858

The half order scoring *highest* on `order_score` is not a bug: a prefix of a correct order is
locally perfect, and it is `undefined_names` that charges it. Neither term is sufficient alone.

`ast.parse` alone is degenerate - a template emitter is valid by construction, so the +1 fires on
everything. The two live signals are `check().undefined_names`, which sees coverage, and
`order_score`, which sees ordering; **neither sees both**, which is why `src.rl.reward` weights
them separately. The emitted module also runs to completion under `src.rl.sandbox.run` and exits
0 whenever the order is complete (measured: 126/200 for gold, which is exactly the reachability
ceiling in `src.rl.episode`, and 0/200 for a half order, which raises `NameError`).

## What was REJECTED

**Emitting one flat function per node and calling them in a bare linear sequence was rejected**:
it throws away the branch and loop structure, which is the only thing a traversal policy can get
right or wrong.

**Leaving the predicates undefined was rejected** - see above. It is the single change that moves
the reward from anti-correlated with traversal quality to correlated with it.

**Executing the emitted code inside `check()` was rejected**: `check` is static and runs per
terminal at 1.17 ms. Execution costs 71.4 ms in `src.rl.sandbox` and is opt-in through
`src.rl.reward.sandbox_hook()`, terminal-only.

**Recovering branch structure from `semantic_role` alone was rejected**: `branch-true` /
`branch-false` are assigned *by* `src.parse.roles.derive_states` from the arrival edge and sibling
rank, so reading them back would be circular - the emitter would score the policy on a label the
policy did not produce. Branch bodies are computed from graph reachability over the emitted
order, which is information the policy actually controls.
"""

from __future__ import annotations

import ast
import builtins
import json
import keyword
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.rl.state import DiagramGraph

#: Roles that open a branch. `fork` maps to `decision` in `src.parse.roles.ROLE_TO_STATE`, and a
#: node with two or more emitted successors is treated as branching whatever its role says,
#: because the structure is the thing being emitted.
BRANCH_ROLES: frozenset[str] = frozenset({"decision", "fork"})

#: Iterations a `while` the policy marked is allowed before its predicate goes False. Keeps
#: emitted code terminating under `src.rl.sandbox.run` instead of burning the timeout.
LOOP_BUDGET = 2

#: Longest identifier taken from a node's text before truncation. hdbpmn labels run to 60+ chars.
MAX_IDENT = 40

_IDENT_BAD = re.compile(r"[^0-9a-zA-Z_]+")
_RESERVED = frozenset(keyword.kwlist) | frozenset(dir(builtins)) | {"run", "state"}


def identifier(text: str, fallback: str) -> str:
    """A stable Python identifier from a node's OCR'd text, or `fallback` when it has none."""
    base = _IDENT_BAD.sub("_", (text or "").strip().lower()).strip("_")
    base = re.sub(r"_+", "_", base)[:MAX_IDENT].strip("_")
    if not base or base[0].isdigit():
        base = f"{fallback}_{base}" if base else fallback
    base = _IDENT_BAD.sub("_", base).strip("_") or fallback
    if base in _RESERVED or keyword.iskeyword(base):
        base = f"{base}_node"
    return base


def node_names(graph: DiagramGraph) -> list[str]:
    """One unique identifier per node index, deterministic in node order."""
    out: list[str] = []
    seen: dict[str, int] = {}
    for i, _node_id in enumerate(graph.node_ids):
        name = identifier(graph.texts[i], f"node_{i}")
        count = seen.get(name, 0)
        seen[name] = count + 1
        out.append(name if count == 0 else f"{name}_{count + 1}")
    return out


def _reachable(graph: DiagramGraph, start: int, allowed: set[int]) -> set[int]:
    """Nodes of `allowed` reachable from `start` by following successors."""
    if start not in allowed:
        return set()
    seen = {start}
    stack = [start]
    while stack:
        node = stack.pop()
        for succ in graph.successors[node]:
            if succ in allowed and succ not in seen:
                seen.add(succ)
                stack.append(succ)
    return seen


def _loop_regions(graph: DiagramGraph, order: list[int], loop_marked: set[int]) -> dict[int, int]:
    """Position of a loop head -> position of its last body node, for the marked back edges.

    A node is a loop head when a node the policy closed with `mark-as-loop` has an edge back to it
    and sits later in the emitted order. Overlapping regions keep the outermost one, so the
    `while` blocks nest rather than interleave.
    """
    position = {node: i for i, node in enumerate(order)}
    spans: dict[int, int] = {}
    for marker in sorted(loop_marked):
        if marker not in position:
            continue
        for target in graph.successors[marker]:
            if target in position and position[target] < position[marker]:
                head = position[target]
                spans[head] = max(spans.get(head, -1), position[marker])
    kept: dict[int, int] = {}
    end = -1
    for head in sorted(spans):
        if head <= end:
            continue
        kept[head] = spans[head]
        end = spans[head]
    return kept


def _emit_block(
    graph: DiagramGraph,
    order: list[int],
    position: dict[int, int],
    names: list[str],
    lo: int,
    hi: int,
    loops: dict[int, int],
    indent: int,
    emitted_set: set[int],
    opened: dict[str, list[str]],
) -> list[str]:
    """Statements for `order[lo:hi]`, structuring branches and marked loops. Recursive.

    `opened` collects the predicate names the block calls, split into `cond` and `again`, so
    `emit_code` can define every one of them. Collecting them here rather than regexing the
    rendered text is what makes "every predicate is defined" a property of the emitter instead
    of a property of a regex.
    """
    pad = "    " * indent
    lines: list[str] = []
    i = lo
    while i < hi:
        node = order[i]
        name = names[node]

        if i in loops and loops[i] < hi:
            end = loops[i] + 1
            opened["again"].append(f"{name}_again")
            lines.append(f"{pad}while {name}_again(state):")
            body = _emit_block(
                graph, order, position, names, i, end, {}, indent + 1, emitted_set, opened
            )
            lines.extend(body or [f"{pad}    pass"])
            i = end
            continue

        later = sorted(
            {s for s in graph.successors[node] if s in emitted_set and position[s] > i},
            key=lambda s: position[s],
        )
        branching = len(later) >= 2 and (
            graph.roles[node] in BRANCH_ROLES or len(graph.successors[node]) >= 2
        )
        lines.append(f"{pad}state = {name}(state)")
        if not branching:
            i += 1
            continue

        first, second = position[later[0]], position[later[1]]
        region = set(order[first:hi])
        reach_a = _reachable(graph, later[0], region)
        reach_b = _reachable(graph, later[1], region)
        join = hi
        for pos in range(second, hi):
            if order[pos] in reach_a and order[pos] in reach_b:
                join = pos
                break
        if first != i + 1 or second >= join:
            # The branches are not laid out contiguously right after the decision, so emitting an
            # if/else here would reorder the policy's own output. The block stays linear and the
            # missing structure shows up as a `structure` finding in `check`-driven reward.
            i += 1
            continue
        opened["cond"].append(f"{name}_cond")
        lines.append(f"{pad}if {name}_cond(state):")
        true_body = _emit_block(
            graph, order, position, names, first, second, loops, indent + 1, emitted_set, opened
        )
        lines.extend(true_body or [f"{pad}    pass"])
        lines.append(f"{pad}else:")
        false_body = _emit_block(
            graph, order, position, names, second, join, loops, indent + 1, emitted_set, opened
        )
        lines.extend(false_body or [f"{pad}    pass"])
        i = join
    return lines


def emit_code(
    graph: DiagramGraph,
    emitted: list[int],
    loop_marked: set[int] | None = None,
) -> str:
    """Runnable Python for the nodes the policy emitted, in the order it emitted them.

    One function per emitted node, an `if/else` where a branching node's first two successors
    were emitted contiguously after it, a `while` around a region the policy closed with
    `mark-as-loop`, **a definition for every predicate the body calls**, and an `EDGES` table
    naming both endpoints of every edge in the diagram.

    **Deterministic**: the same `(graph, emitted, loop_marked)` always produces byte-equal
    output, which `src.rl.env`'s determinism check relies on.

    The two design decisions that make the reward signal point the right way:

    **Predicates are defined, not dangling.** The inherited draft emitted `<name>_cond` and
    `<name>_again` calls and deliberately left them undefined, so `check().undefined_names`
    counted *structure the policy successfully built*. Measured over 600 diagrams that inverted
    the reward: `structure_score` was **0.8915 on the gold order and 0.9997 on the reversed
    order**, i.e. the emitter paid more for emitting nothing structural than for getting the
    traversal right. Every predicate is now defined by `_predicate_defs`, and the same
    measurement over all 3,993 diagrams gives gold 0.0000 undefined-per-call against half-order
    0.7555.

    **`EDGES` names every endpoint of every diagram edge.** That is what makes an undefined name
    mean the one thing it should mean: the policy referenced a node it never emitted. It is a
    module-level list of function objects, so it is resolved at import time - the sandbox raises
    `NameError` on a partial emission - and it costs no execution time.

    The emitted module runs to completion under `src.rl.sandbox.run` and exits 0: `_again`
    predicates are budgeted to at most `LOOP_BUDGET` iterations, so a marked loop terminates
    instead of hanging the sandbox on a timeout.
    """
    loop_marked = set(loop_marked or ())
    names = node_names(graph)
    emitted_set = set(emitted)
    position = {node: i for i, node in enumerate(emitted)}
    header = [
        f'"""Generated from diagram {graph.diagram_id} by src.rl.emit.',
        "",
        "Phase 11 stand-in for Phase 12; see that module's docstring for the limitation.",
        '"""',
        "",
        "_BUDGET: dict = {}",
        "",
        "",
    ]
    body: list[str] = []
    for node in emitted:
        role = graph.roles[node] or "process"
        body += [
            f"def {names[node]}(state):",
            f'    """{role} - node {graph.node_ids[node]}."""',
            f'    state["{names[node]}"] = True',
            "    return state",
            "",
            "",
        ]

    opened: dict[str, list[str]] = {"cond": [], "again": []}
    loops = _loop_regions(graph, emitted, loop_marked)
    run_body = _emit_block(
        graph, emitted, position, names, 0, len(emitted), loops, 1, emitted_set, opened
    )
    body += _predicate_defs(opened)
    body += ["def run(state=None):", "    state = {} if state is None else state"]
    body += run_body or ["    pass"]
    body += ["    return state", "", ""]
    body += _edge_table(graph, names)
    body += ["", 'if __name__ == "__main__":', "    run()", ""]
    return "\n".join(header + body)


def _predicate_defs(opened: dict[str, list[str]]) -> list[str]:
    """A definition for every predicate `_emit_block` opened. Deterministic and terminating.

    `_cond` reads a flag off the state dict, so a branch is decidable without any input; `_again`
    counts its own calls against `LOOP_BUDGET`, which is what stops a `while` the policy marked
    from spinning forever when the code is actually executed in `src.rl.sandbox`.
    """
    lines: list[str] = []
    for name in sorted(set(opened["cond"])):
        lines += [
            f"def {name}(state):",
            f'    return bool(state.get("{name}"))',
            "",
            "",
        ]
    for name in sorted(set(opened["again"])):
        lines += [
            f"def {name}(state):",
            f'    _BUDGET["{name}"] = _BUDGET.get("{name}", 0) + 1',
            f'    return _BUDGET["{name}"] <= {LOOP_BUDGET}',
            "",
            "",
        ]
    return lines


def _edge_table(graph: DiagramGraph, names: list[str]) -> list[str]:
    """`EDGES = [(src_fn, dst_fn), ...]` over every edge in the diagram.

    Both endpoints are named, so an edge whose target the policy never emitted is an undefined
    name statically and a `NameError` at run time. This is the coverage half of the reward
    signal; the ordering half is `order_score`, which the emitted text cannot express.
    """
    pairs = [
        f"    ({names[a]}, {names[b]})," for a in range(graph.n_nodes) for b in graph.successors[a]
    ]
    return ["EDGES = [", *pairs, "]"]


def order_score(graph: DiagramGraph, emitted: list[int]) -> float:
    """Fraction of the diagram's forward edges the emitted order got the right way round.

    The coverage signal (`check().undefined_names`) cannot see ordering: a reversed traversal
    still emits every node, so it resolves every name. This does see it. Back edges are excluded
    - `src.parse.sequences.traversal`'s set, reused - because a back edge running backwards in
    the emitted order is exactly what a back edge *is*, and charging for it would penalise every
    correctly handled loop.

    Returns 1.0 when the emitted order spans no forward edge at all, so a single-node or
    disconnected emission is not charged for structure it had no chance to get wrong.

    Measured over all 3,993 diagrams: **gold 0.8942, reversed 0.1058, first-half-of-gold
    0.9858**. The half order scoring highest is not a bug and is why this term is only part of
    the reward: a short prefix of a correct order is locally perfect, and it is the coverage term
    (`check().undefined_names`) that charges it.
    """
    position = {node: i for i, node in enumerate(emitted)}
    forward = 0
    total = 0
    for a in range(graph.n_nodes):
        for b in graph.successors[a]:
            if (a, b) in graph.back_edges or a not in position or b not in position:
                continue
            total += 1
            forward += int(position[a] < position[b])
    return forward / total if total else 1.0


def check(code: str) -> dict:
    """Static verdict on emitted code: does it parse, and does every name it uses exist?

    `undefined_names` is the discriminating half. A policy that emits a branching node but never
    emits one of its branches leaves a call to a function that was never defined; that is the
    structural error a template emitter can actually commit, and the 0.00% / 31.71% / 61.33% split
    in the module docstring is this field.

    Returns `{"parse_ok", "error", "undefined_names", "n_defs", "n_calls"}`.
    """
    verdict: dict = {
        "parse_ok": False,
        "error": None,
        "undefined_names": [],
        "n_defs": 0,
        "n_calls": 0,
    }
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        verdict["error"] = f"{exc.msg} (line {exc.lineno})"
        return verdict
    verdict["parse_ok"] = True

    defined = set(dir(builtins)) | {"state"}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            defined.add(node.name)
            defined.update(arg.arg for arg in node.args.args)
            verdict["n_defs"] += 1
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            defined.add(node.id)
        elif isinstance(node, ast.Call):
            verdict["n_calls"] += 1
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    verdict["undefined_names"] = sorted(used - defined)
    return verdict


def measure(limit: int | None = None) -> dict:
    """The docstring's table: parse rate, undefined-name rate and order score, three orders."""
    from src.parse.roles import is_sequential
    from src.parse.sequences import LABELLED_SOURCES, UNLABELLED_SOURCES, load_ir, traversal
    from src.rl.state import DiagramGraph

    rows = {
        name: {"n": 0, "parse_ok": 0, "undefined": 0, "undef_frac": 0.0, "order": 0.0}
        for name in ("gold", "reversed", "half")
    }
    sources = tuple(LABELLED_SOURCES) + tuple(UNLABELLED_SOURCES)
    for diagram in load_ir(sources, limit):
        if not is_sequential(diagram):
            continue
        graph = DiagramGraph.from_ir(diagram)
        order, _back = traversal(diagram)
        gold = [graph.index_of[n] for n in order if n in graph.index_of]
        if not gold:
            continue
        variants = {
            "gold": gold,
            "reversed": list(reversed(gold)),
            "half": gold[: max(1, len(gold) // 2)],
        }
        for name, emitted in variants.items():
            verdict = check(emit_code(graph, emitted))
            row = rows[name]
            row["n"] += 1
            row["parse_ok"] += int(verdict["parse_ok"])
            row["undefined"] += int(bool(verdict["undefined_names"]))
            row["undef_frac"] += len(verdict["undefined_names"]) / max(1, verdict["n_calls"])
            row["order"] += order_score(graph, emitted)
    for row in rows.values():
        n = max(1, row["n"])
        row["parse_rate"] = round(row["parse_ok"] / n, 4)
        row["undefined_rate"] = round(row["undefined"] / n, 4)
        row["mean_undefined_per_call"] = round(row["undef_frac"] / n, 4)
        row["mean_order_score"] = round(row["order"] / n, 4)
        del row["undef_frac"], row["order"]
    return rows


def main(argv: list[str] | None = None) -> int:
    print(json.dumps(measure(), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
