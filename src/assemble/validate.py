"""Phase 10.2.1 - type-specific semantic validators, and the corpus test of whether they are right.

    python -m src.assemble.validate                  # every validator over all 5,796 IR files
    python -m src.assemble.validate --source hdbpmn

`src/ir/qa.py` already answers "is this file well-formed" - ids unique, every `src` naming a real
node, boxes inside the frame. This answers a different question: **is this graph a legal instance
of the thing it claims to be**. A flowchart whose every id is unique and whose decision has one way
out is a perfectly valid IR file and a broken flowchart.

## Violations, not booleans

Each validator returns `list[Violation]` - `{rule, severity, refs, detail}` - because both
consumers need the *refs*. 10.2.2's repair has to know which node to attach an edge to, and Phase
16's UI has to draw a marker on it; a `bool` throws that away. Rule ids are stable, so a repair is
written against `FC.DECISION_SPLIT` rather than against a sentence.

## Evidence gating, which is the whole of this task

The plan asserts these rules. Running them over 5,796 *ground-truth* files is what tests them, and
the trap showed up on the first run: **didi (3,000) and flowchartseg (1,319) carry
`semantic_role: unknown` on every node**, and flowchartseg carries **zero edges by construction**.
Scored naively, "exactly one start" fires on all 4,319 of them - which says nothing about starts
and everything about two converters that never assigned roles.

So every rule declares the evidence it needs (`roles`, `edges`) plus an optional `guard`, a diagram
advertises what it has, and a rule that lacks its evidence is **skipped and counted separately**
rather than scored as a pass. **4,319 of 5,796 files - 74.5% of the corpus - can answer no rule at
all**, and that number is reported as `diagrams_with_no_evaluable_rule` because a validator suite
that quietly scores 74.5% of its corpus as clean is lying.

## What it measured

Ground truth, 5,796 files. `n` is the diagrams a rule could be asked:

    rule                   n    skipped   hit    rate    nodes   evaluable on
    FC.START_ONE         693      4319    160   0.2309     465   hdbpmn
    FC.END_ONE           693      4319     29   0.0418      29   hdbpmn
    FC.DECISION_BRANCH   693      4319    404   0.5830     541   hdbpmn
    FC.DECISION_SPLIT    693      4319     24   0.0346      25   hdbpmn
    SM.INITIAL_ONE       300         0     53   0.1767      53   fa_bresler
    SM.REACHABLE         247        53      1   0.0040       1   fa_bresler
    ER.ENTITY_ATTRS        0         0      0      -         0   nothing
    WF.ORPHAN_WIDGET     484         0      0   0.0000       0   sketch2code

**Two of the plan's four assertions do not survive, and both fail the same way: they encode a
textbook diagram rather than a drawn one.**

**FC.DECISION_BRANCH - "decisions >= 2 out" - fires on 58.3% of ground truth.** 404 of 693
hand-drawn BPMN pages, 541 nodes. That is not a defect rate, it is a wrong rule, and the mechanism
is visible in the in-degree: of those 541 thin decisions, **516 (95.4%) have two or more incoming
edges**. They are *merging* gateways, which have one output by design and which BPMN draws as the
same diamond. Requiring in-degree <= 1 as well - the only local evidence that separates a split
from a merge - is `FC.DECISION_SPLIT`, and it drops to **3.46%, 24 pages, 25 nodes**. **A factor of
17.** Both are registered: the unqualified rule stays as a `warning` because it is what the plan
asked for and its rate is the finding, and the qualified one is the `error` a repair should act on.
This is the rule 7.3.10 tried to repair from HMM posteriors and could not; on this evidence its
`missing_branch` was firing mostly on merges.

**FC.START_ONE is wrong in the other direction.** 23.1% of pages, but the note counters split it:
**158 pages have more than one start and only 2 have none** (533 have exactly one). A BPMN page
with three start events is a process with three triggers, which the notation permits and people
draw. **The defensible rule is "at least one start", which would fire on 2 of 693 - 0.29%.**
`FC.START_ONE` is kept at `warning` with every start in `refs` so Phase 16 can ask rather than
repair, and 10.2.2 must not delete starts.

**SM.INITIAL_ONE at 17.7% is a third case again - not a wrong rule and not a defect, but a hole in
the labels.** All 53 firings are **zero** initial states; not one automaton has two. The FA format
has an `initial` flag and 53 hand-drawn transcriptions left it unset. That matters structurally
rather than cosmetically: reachability has nowhere to start, so `SM.REACHABLE` is **guarded** to
the 247 automata with exactly one initial state. Scoring it on all 300 would have reported 0.33%
instead of 0.40% by counting 53 unanswerable diagrams as passes.

**SM.REACHABLE is the one rule that found a real defect, and it found almost none.** 1 automaton in
247, one unreachable state. The plan's reachability check is correct and this corpus is clean.

**WF.ORPHAN_WIDGET fires zero times in 484 wireframes, and that is not a pass.** Every sketch2code
IR was built by walking a DOM, so every widget has a parent *by construction* - the rule cannot
fail here any more than a rule about edges can fail on flowchartseg. It is **unfalsified, not
confirmed**, and the corpus that would test it is one where containment was *inferred* from
geometry rather than read off a tree. That corpus arrives with 10.1.

**ER.ENTITY_ATTRS has a denominator of zero.** There is not one `er_diagram` among the five
sources - 4,321 flowcharts, 484 wireframes, 300 state machines. Its `rate` is `None` rather than
`0.0`, deliberately, because a rule that was never asked and a rule that never fired must not
render as the same row. It is implemented and unit-tested and **nothing whatever is known about
whether it is right**.

## Assembled graphs

The same sweep runs over predicted graphs when `src.assemble.nodes` exposes an `assemble()`; it
does not yet, so `assembled` is `{"skipped": "src.assemble.nodes has no assemble()"}`. The
comparison is the point - a rule at 4.2% on truth and 40% on assembly is measuring the assembler,
not the diagram - and it is deferred behind a guarded import rather than dropped.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.ir.model import SUFFIX, Diagram
from src.utils.config import ROOT
from src.utils.parallel import pmap

IR_ROOT = ROOT / "data" / "processed" / "ir"
RUNS = ROOT / "experiments" / "assemble"
OUT = RUNS / "validate.json"

ERROR, WARNING = "error", "warning"

#: Evidence a diagram can offer. A rule that needs one it does not have is skipped, not passed:
#: didi and flowchartseg carry `unknown` on every node and flowchartseg carries no edges at all,
#: and scoring a role rule against them measures the converter rather than the diagram.
ROLES, EDGES = "roles", "edges"

#: `n_jobs` for the corpus sweep. Six agents share this 32-core box; four is a neighbourly share.
N_JOBS = 4

#: A `decision` with one way out is only certainly wrong when it is also a *split* - one way in.
#: A merging gateway is drawn as the same diamond and legitimately has a single output, and
#: FC.DECISION_SPLIT is the plan's rule with that qualifier applied.
MAX_SPLIT_IN_DEGREE = 1

#: Roles that count as a state for reachability. `transition` is edge-like (see
#: `src.ir.vocab.EDGE_LIKE_ROLES`) and is not a state that can be unreachable.
STATE_ROLES = frozenset({"state", "initial-state", "final-state"})

#: Roles that are widgets rather than the frame they sit in. A `container` with no parent is the
#: root of the containment tree, not an orphan, so containers are exempt.
WIDGET_ROLES = frozenset({"ui-input", "ui-button", "ui-label", "ui-image"})


@dataclass(frozen=True)
class Violation:
    """One broken rule, with the nodes to blame. `refs` is what makes this repairable."""

    rule: str
    severity: str
    refs: tuple[str, ...]
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule": self.rule,
            "severity": self.severity,
            "refs": list(self.refs),
            "detail": self.detail,
        }


@dataclass(frozen=True)
class Rule:
    """A registered rule.

    `requires` is the diagram-level evidence without which it must not be scored, and `guard` is
    the rule-specific version of the same idea: reachability needs somewhere to start from, so an
    automaton with no initial state does not *pass* SM.REACHABLE, it cannot be asked.
    """

    id: str
    diagram_type: str
    severity: str
    summary: str
    requires: tuple[str, ...] = ()
    guard: Callable[[Diagram], bool] | None = None


RULES: dict[str, Rule] = {
    rule.id: rule
    for rule in (
        Rule("FC.START_ONE", "flowchart", WARNING, "exactly one start", (ROLES,)),
        Rule("FC.END_ONE", "flowchart", WARNING, "at least one end", (ROLES,)),
        Rule(
            "FC.DECISION_BRANCH",
            "flowchart",
            WARNING,
            "a decision has >= 2 outgoing edges (the plan's rule, unqualified)",
            (ROLES, EDGES),
        ),
        Rule(
            "FC.DECISION_SPLIT",
            "flowchart",
            ERROR,
            "a decision with one way in has >= 2 outgoing edges",
            (ROLES, EDGES),
        ),
        Rule("SM.INITIAL_ONE", "state_machine", ERROR, "exactly one initial state", (ROLES,)),
        Rule(
            "SM.REACHABLE",
            "state_machine",
            ERROR,
            "every state reachable from the initial state",
            (ROLES, EDGES),
            lambda d: len(_by_role(d).get("initial-state", [])) == 1,
        ),
        Rule(
            "ER.ENTITY_ATTRS",
            "er_diagram",
            WARNING,
            "every entity has >= 1 attribute",
            (ROLES, EDGES),
        ),
        Rule(
            "WF.ORPHAN_WIDGET",
            "wireframe",
            WARNING,
            "every widget has a containment edge",
            (ROLES, EDGES),
        ),
    )
}


# ------------------------------------------------------------------------------------------
# what a diagram can be asked
# ------------------------------------------------------------------------------------------


def evidence(diagram: Diagram) -> set[str]:
    """What this diagram is able to answer questions about.

    Not a quality judgement - a diagram with every role `unknown` is a legal IR file, it simply
    cannot be asked whether it has exactly one start. Separating this from "the rule fired" is
    the difference between a corpus finding and a corpus artefact.
    """
    have = set()
    if any(node.semantic_role not in ("unknown", "") for node in diagram.nodes):
        have.add(ROLES)
    if diagram.edges:
        have.add(EDGES)
    return have


def applicable(rule: Rule, diagram: Diagram) -> bool:
    """Is `rule` scoreable on `diagram`? The type must match and every evidence need be met."""
    if rule.diagram_type != diagram.diagram_type or not set(rule.requires) <= evidence(diagram):
        return False
    return rule.guard is None or rule.guard(diagram)


def _adjacency(diagram: Diagram) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """`(out, in)` id -> ids. Built once; `Diagram.out_edges` is a scan and this is a hot loop."""
    out: dict[str, list[str]] = defaultdict(list)
    into: dict[str, list[str]] = defaultdict(list)
    for edge in diagram.edges:
        if edge.src is None or edge.dst is None:
            continue
        out[edge.src].append(edge.dst)
        into[edge.dst].append(edge.src)
    return out, into


def _by_role(diagram: Diagram) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for node in diagram.nodes:
        grouped[node.semantic_role].append(node.id)
    return grouped


# ------------------------------------------------------------------------------------------
# the validators
# ------------------------------------------------------------------------------------------


def validate_flowchart(diagram: Diagram) -> list[Violation]:
    """One start, at least one end, and a decision that actually decides."""
    roles = _by_role(diagram)
    out, into = _adjacency(diagram)
    have = evidence(diagram)
    found: list[Violation] = []

    if ROLES in have:
        starts = roles.get("start", [])
        if len(starts) != 1:
            # `refs` lists all of them when there are several, so a repair can choose and a UI
            # can ask. A BPMN page with three triggers is not defective, which is why this is a
            # warning and why the count is in the detail rather than assumed to be zero.
            found.append(
                Violation(
                    "FC.START_ONE",
                    WARNING,
                    tuple(starts),
                    f"{len(starts)} start nodes, expected 1",
                )
            )
        if not roles.get("end"):
            found.append(Violation("FC.END_ONE", WARNING, (), "no end node"))

    if {ROLES, EDGES} <= have:
        decisions = roles.get("decision", [])
        thin = [n for n in decisions if len(out.get(n, [])) < 2]
        if thin:
            found.append(
                Violation(
                    "FC.DECISION_BRANCH",
                    WARNING,
                    tuple(thin),
                    f"{len(thin)} decisions with fewer than 2 outgoing edges",
                )
            )
        splits = [n for n in thin if len(into.get(n, [])) <= MAX_SPLIT_IN_DEGREE]
        if splits:
            found.append(
                Violation(
                    "FC.DECISION_SPLIT",
                    ERROR,
                    tuple(splits),
                    f"{len(splits)} splitting decisions with fewer than 2 outgoing edges",
                )
            )
    return found


def validate_state_machine(diagram: Diagram) -> list[Violation]:
    """One initial state, and no state the machine can never enter."""
    roles = _by_role(diagram)
    out, _ = _adjacency(diagram)
    have = evidence(diagram)
    found: list[Violation] = []

    initial = roles.get("initial-state", [])
    if ROLES in have and len(initial) != 1:
        found.append(
            Violation(
                "SM.INITIAL_ONE",
                ERROR,
                tuple(initial),
                f"{len(initial)} initial states, expected 1",
            )
        )

    if {ROLES, EDGES} <= have and initial:
        seen, stack = set(initial), list(initial)
        while stack:
            for target in out.get(stack.pop(), []):
                if target not in seen:
                    seen.add(target)
                    stack.append(target)
        unreached = [
            node.id
            for node in diagram.nodes
            if node.semantic_role in STATE_ROLES and node.id not in seen
        ]
        if unreached:
            found.append(
                Violation(
                    "SM.REACHABLE",
                    ERROR,
                    tuple(unreached),
                    f"{len(unreached)} states unreachable from {initial[0]}",
                )
            )
    return found


def validate_er(diagram: Diagram) -> list[Violation]:
    """Every entity carries at least one attribute, in either edge direction."""
    roles = _by_role(diagram)
    out, into = _adjacency(diagram)
    if not {ROLES, EDGES} <= evidence(diagram):
        return []
    attributes = set(roles.get("attribute", []))
    bare = [
        node_id
        for node_id in roles.get("entity", [])
        if not attributes & set(out.get(node_id, []) + into.get(node_id, []))
    ]
    if not bare:
        return []
    return [
        Violation(
            "ER.ENTITY_ATTRS", WARNING, tuple(bare), f"{len(bare)} entities with no attribute"
        )
    ]


def validate_wireframe(diagram: Diagram) -> list[Violation]:
    """No widget floating free of the containment tree."""
    out, into = _adjacency(diagram)
    if not {ROLES, EDGES} <= evidence(diagram):
        return []
    orphans = [
        node.id
        for node in diagram.nodes
        if node.semantic_role in WIDGET_ROLES and not out.get(node.id) and not into.get(node.id)
    ]
    if not orphans:
        return []
    return [
        Violation(
            "WF.ORPHAN_WIDGET", WARNING, tuple(orphans), f"{len(orphans)} widgets with no edge"
        )
    ]


VALIDATORS = {
    "flowchart": validate_flowchart,
    "state_machine": validate_state_machine,
    "er_diagram": validate_er,
    "wireframe": validate_wireframe,
}


def validate(diagram: Diagram) -> list[Violation]:
    """Every semantic violation in one diagram, dispatched on `diagram.diagram_type`.

    An unrecognised type returns `[]` rather than raising: `unknown` is a legal `diagram_type`
    in the schema, and having no opinion is the correct answer for it.
    """
    return VALIDATORS.get(diagram.diagram_type, lambda _: [])(diagram)


# ------------------------------------------------------------------------------------------
# the corpus sweep
# ------------------------------------------------------------------------------------------


@dataclass
class Tally:
    """Per-rule counters. Diagrams hit and nodes flagged are kept apart because they differ."""

    evaluable: Counter = field(default_factory=Counter)
    hit: Counter = field(default_factory=Counter)
    refs: Counter = field(default_factory=Counter)
    skipped: Counter = field(default_factory=Counter)
    notes: Counter = field(default_factory=Counter)

    def add(self, row: dict) -> None:
        for rule_id in row["evaluable"]:
            self.evaluable[rule_id] += 1
        for rule_id in row["skipped"]:
            self.skipped[rule_id] += 1
        for rule_id, n in row["hit"].items():
            self.hit[rule_id] += 1
            self.refs[rule_id] += n
        for note in row["notes"]:
            self.notes[note] += 1

    def rows(self) -> list[dict]:
        out = []
        for rule_id, rule in RULES.items():
            n = self.evaluable[rule_id]
            out.append(
                {
                    "rule": rule_id,
                    "severity": rule.severity,
                    "summary": rule.summary,
                    "evaluable": n,
                    "skipped": self.skipped[rule_id],
                    "diagrams_hit": self.hit[rule_id],
                    # None, not 0.0: a rule with no denominator has not passed, it was not asked.
                    "rate": round(self.hit[rule_id] / n, 4) if n else None,
                    "nodes_flagged": self.refs[rule_id],
                }
            )
        return out


def score(diagram: Diagram) -> dict:
    """One diagram's contribution: which rules could be asked, and which of those fired.

    `notes` carries the diagnostics that decide whether a firing rule is finding a defect or
    encoding a wrong assumption - "no start" and "several starts" are the same violation and
    completely different findings.
    """
    hits = {v.rule: len(v.refs) or 1 for v in validate(diagram)}
    evaluable = [r.id for r in RULES.values() if applicable(r, diagram)]
    skipped = [
        r.id
        for r in RULES.values()
        if r.diagram_type == diagram.diagram_type and r.id not in evaluable
    ]
    notes: list[str] = []
    if diagram.diagram_type == "flowchart" and ROLES in evidence(diagram):
        starts = len(_by_role(diagram).get("start", []))
        notes.append("start_none" if starts == 0 else f"start_{'one' if starts == 1 else 'many'}")
    return {"evaluable": evaluable, "skipped": skipped, "hit": hits, "notes": notes}


def _score_path(path: Path) -> dict:
    row = score(Diagram.load(path))
    row["source"] = path.parent.name
    return row


def sweep(root: Path = IR_ROOT, source: str | None = None, limit: int | None = None) -> dict:
    """Every validator over every ground-truth IR file, tallied per rule and per source."""
    paths = sorted((root / source if source else root).rglob(f"*{SUFFIX}"))
    if limit:
        paths = paths[:limit]
    rows = pmap(_score_path, paths, n_jobs=N_JOBS)

    overall: Tally = Tally()
    per_source: dict[str, Tally] = defaultdict(Tally)
    files: Counter = Counter()
    unscored = 0
    for row in rows:
        overall.add(row)
        per_source[row["source"]].add(row)
        files[row["source"]] += 1
        if not row["evaluable"]:
            unscored += 1
    return {
        "files": len(paths),
        "by_source": dict(sorted(files.items())),
        # The headline caveat: how many files no rule could be scored on at all.
        "diagrams_with_no_evaluable_rule": unscored,
        "rules": overall.rows(),
        "notes": dict(sorted(overall.notes.items())),
        "per_source": {
            name: {
                "rules": [r for r in tally.rows() if r["evaluable"] or r["skipped"]],
                "notes": dict(sorted(tally.notes.items())),
            }
            for name, tally in sorted(per_source.items())
        },
    }


def assembled(limit: int | None = None) -> dict:
    """The same sweep over predicted graphs, if a sibling has built one. Guarded on purpose."""
    try:
        from src.assemble import nodes as assemble_nodes
    except ImportError:
        return {"skipped": "src.assemble.nodes not importable"}
    builder = getattr(assemble_nodes, "assemble", None)
    if builder is None:
        return {"skipped": "src.assemble.nodes has no assemble()"}
    from src.assemble.corpus import pages

    held = pages()[:limit] if limit else pages()
    tally = Tally()
    for page in held:
        tally.add(score(builder(page)))
    return {"pages": len(held), "rules": tally.rows(), "notes": dict(sorted(tally.notes.items()))}


def run(source: str | None = None, limit: int | None = None) -> dict:
    return {
        "registry": {
            r.id: {"type": r.diagram_type, "severity": r.severity, "requires": list(r.requires)}
            for r in RULES.values()
        },
        "ground_truth": sweep(source=source, limit=limit),
        "assembled": assembled(limit),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)

    result = run(args.source, args.limit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    summary = {k: v for k, v in result["ground_truth"].items() if k != "per_source"}
    print(json.dumps({"assembled": result["assembled"], **summary}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
