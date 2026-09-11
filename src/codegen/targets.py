"""Phase 12.1.6 - the five target languages, and which of them a diagram can actually reach.

    from src.codegen.targets import for_type, verify
    code = for_type("flowchart")(diagram, traversal_order)   # -> str
    ok, detail = verify(code, "flowchart")

The row names five pairs: flowchart -> Python, state machine -> Python class / `transitions`,
ER -> SQL DDL, wireframe -> React + Tailwind, circuit -> SPICE netlist. Each is a function with
the same shape, `emit(diagram, traversal) -> str`, and `for_type` is the dispatch.

## Which of the five have real diagrams behind them, and which do not

This is the first thing to say, because "all five generators specified" is not the same claim as
"all five are backed by data", and only the first is true today:

    flowchart       REAL     693 hdbpmn pages, 1,319 flowchartseg pages, 3,000 didi pages
    state machine   REAL     300 fa_bresler finite automata
    wireframe       REAL     484 sketch2code screens
    ER              NONE     synthetic only (`src.synth.graphs._er`)
    circuit         NONE     synthetic only (`src.synth.graphs._circuit`)

`data/processed/ir` holds five source directories and none of them is an ER or a circuit corpus;
2.2.6's own table already says so ("no structure exists, so no pair is produced - DIDI,
flowchartseg, ER, circuits"). So the ER and SPICE emitters below are exercised **only** against
graphs this repo generated, which means they are checked for well-formedness and not for
fidelity to anything a person drew. Two of the five languages are a specification with a
generator attached, not a measured capability.

## The flowchart emitter is the only hard one, and it has two modes

A flowchart is a graph; Python is a tree. The conversion only exists when the graph is
*reducible* - when its loops have single entry points and its branches re-converge. Synthetic
flowcharts are near-reducible by construction (`src.synth.graphs` lowers a program tree, so the
graph came *from* a tree). Hand-drawn ones frequently are not.

    structured   `_structured` rebuilds real `if` / `else` / `while` nesting by finding each
                 decision's immediate post-dominator and each back edge's loop header.
    dispatch     when that fails - out-degree > 2, a branch that never re-converges, a loop with
                 two entries, a node reached from inside two different regions - the emitter
                 falls back to an explicit `state` variable in a `while True:` loop.

**Both always parse; they are not equally good targets.** The dispatch form is faithful but flat,
and a model trained on too many of them learns to emit a switch statement for every diagram. So
the fallback is counted rather than hidden: `flowchart_mode(diagram, order)` returns
`(code, mode)`.

**Measured fallback rate** (re-run for this module, CPU only):

    real hdbpmn                495 / 693     71.4%
    real didi                  949 / 3,000   31.6%
    real flowchartseg            0 / 1,319    0.0%   <- vacuous, see below
    synthetic flowcharts         9 / 500      1.8%

Three of those four numbers need a caveat, and the caveats are the point:

**71.4%, not the 92.4% this docstring claimed before.** The old figure was inherited, could not
be reproduced by any measurement here, and is deleted rather than adjusted. The explanation it
came with survives because it is still right: a BPMN page is drawn as several pools, the
converted IR gives each pool as its own component with fork/join gateways of out-degree 3+, and
almost none of it is a structured program. That is what "flowchart -> Python" costs on real
data, and it is why 12.1.3's human-written targets exist.

**flowchartseg's 0.0% is not an achievement.** Those pages carry node polygons and no
connectors at all, so every page is a sequence of isolated statements that trivially "nests".
A structured rate over a corpus with no edges measures nothing.

**The synthetic rate is 1.8%, not 0.0%.** "Reducible by construction" is very nearly true and not
exactly true: the `disconnected` and `branching` families can put a second component's entry
inside a region the first already claimed, which `_structured` refuses (`reached from two
regions`) rather than silently dropping a node.

## What each emitter is checked with

Nothing is re-implemented here. `verify(code, diagram_type)` routes to the checkers this repo
already has and that carry their own tests, so an emitter cannot pass a private check and fail
the real gate:

    python   `src.codegen.quality.check_python` - `ast.parse` **and** `compile()`, because
             `return` outside a function parses and does not compile
    sql      `src.eval.sql.check` - executed statement by statement on in-memory SQLite, then a
             static pass that resolves every FK's parent table, parent column and parent
             uniqueness, rejects invented type names, and rejects an empty schema. Chosen over
             `quality.check_sql`, which only executes: SQLite's DDL happily accepts a foreign key
             pointing at a table nobody created.
    react    `src.codegen.quality.check_react` - a structural parse in Python (tag balance with
             the void-element set, expression braces, one root per `return`, a single default
             export), with strings masked first.
    spice    `src.codegen.quality.check_spice` - the documented structural check: known device
             letter, per-device node arity, alphanumeric net names, ground net `0` present, no
             net touched by fewer than two cards, and a closing `.end`.

**Measured pass rate, re-run for this module:**

    real hdbpmn        flowchart       693 / 693      1.0000     0.50 ms/diagram
    real flowchartseg  flowchart     1,319 / 1,319    1.0000     0.23 ms/diagram
    real didi          flowchart     3,000 / 3,000    1.0000     0.11 ms/diagram
    real fa_bresler    state machine   300 / 300      1.0000     0.32 ms/diagram
    real sketch2code   wireframe       484 / 484      1.0000     2.14 ms/diagram
    synthetic          each of five    500 / 500      1.0000     0.03-0.42 ms/diagram

5,796 real diagrams and 2,500 synthetic ones, everything CPU. The ER row has no real counterpart
by construction, and 500/500 on graphs this repo generated is a much weaker statement than
693/693 on pages people drew; they are listed apart for that reason.

The SQL number is also not just "it executed": over 500 synthetic ER diagrams the checker
resolved **1,696 foreign keys and 296 junction tables**, and no diagram fell back to the
one-table `placeholder` schema. An emitter that emitted nothing would also have scored 1.0000 on
execution alone, which is exactly the failure `sql.check`'s empty-schema rule exists to catch.

## Three bugs the inherited code shipped with, all found by running it over the corpus

    a suite of comments        `lines += [...] or ["    pass"]` guarded the *empty* branch but not
                               the branch whose only content is a comment, and `_statement`
                               returns a comment for the `start`, `fork`, `join`, `event` and
                               `container` roles. A BPMN branch holding one intermediate event
                               emitted `if cond(ctx):` followed by a comment, which Python reads
                               as an empty suite. 9 of 693 hdbpmn pages failed `ast.parse`.
                               Fixed in `_suite`.
    a SQL reserved word        `_ident` escaped Python keywords. SQL's are a different set, and
                               `order`, `group`, `table`, `index` and `key` are all ordinary
                               English words a hand-drawn entity is very likely to be called.
                               77 of 200 synthetic ER diagrams were a syntax error on `order`
                               alone. Fixed in `_sql_ident`.
    a net soldered at one end  a disconnected circuit puts parts on nets nothing else touches,
                               and a net on exactly one card is a dangling wire. 40 of 200
                               synthetic circuits were rejected. Fixed by terminating orphan
                               nets to ground with a 1meg resistor - the smallest edit that
                               makes the deck solvable rather than merely well-formed.

Two smaller ones: `ACCEPTING = {}` emitted an empty **dict** for a machine with no accepting
state, and two states whose OCR text came back identical collapsed into one row of `TABLE`,
silently merging their transitions. Both are now regression-tested.

## What was rejected

**`transitions`-library output as the state machine target.** Probed at authoring time: the
library is **not installed** in this environment (`import transitions` -> `ModuleNotFoundError`),
so emitting `from transitions import Machine` would produce targets that compile and cannot run.
It is also the weaker target on its own merits - the library form is three lines of declarative
config that moves every interesting decision into a dict literal and gives a code model nothing
to learn about control flow. The emitted class carries the machine as `STATES` / `TRANSITIONS`
class attributes in exactly the shape `transitions.Machine` accepts, so the library form is one
constructor call away if the dependency is ever added.

**The `@babel/parser` cross-validation this docstring used to claim.** It reported "agreement
1.000 on a 1,000-target sample". There is no `package.json` and no `node_modules` in this repo
and `npm ls @babel/parser` is empty, so that number cannot have been produced here and cannot be
reproduced now; it is deleted rather than re-stated. Installing babel would need the network for
a check that runs offline in well under a millisecond, so the JSX check stays a documented
structural parse, with its limits named in `quality`'s docstring: it catches unclosed tags, stray
braces and missing exports, and does **not** catch an undefined identifier.

**Emitting SQL with `PRAGMA foreign_keys = ON`.** It makes the order of `CREATE TABLE`
statements load-bearing, and a diagram has no table order. The FKs are declared and the pragma is
left to the checker, which turns it on itself.

**Re-implementing the four compile checks in this module.** They exist in `src.codegen.quality`
(12.1.7) and `src.eval.sql` (12.3.7). A second copy would drift, and the copy that matters is the
one the pairs are filtered through.

**One shared `emit` with a `diagram_type` argument.** Five separate functions plus `for_type`
keeps each emitter's assumptions in its own docstring, which is where they belong.

## Note on overlap with `src/ir/targets.py`

2.2.6 already emits Python for flowcharts and state machines. That module is the *corpus*
builder - it transcribes ground-truth structure to disk for Phase 2's report, and its flowchart
form is the flat one. This module is the *pair* builder for the fine-tune: it adds the structured
mode, the other three languages, and the traversal-order argument the 12.1.1 pair schema carries.
Neither imports the other, and both are checked by running what they emit.
"""

from __future__ import annotations

import json
import keyword
import re
from collections import Counter

#: diagram_type -> the `language` field of a training pair. The values are exactly the keys of
#: `src.codegen.quality.CHECKS`, which is 12.1.7's filter: the wireframe language is "react", not
#: "jsx", because `quality.check(record)` rejects an unknown language name loudly and a pair
#: labelled "jsx" would have been thrown out by the very gate that is supposed to check it.
LANGUAGES: dict[str, str] = {
    "flowchart": "python",
    "state_machine": "python",
    "er": "sql",
    "wireframe": "react",
    "circuit": "spice",
}

#: How the last `emit_flowchart_python` call resolved: "structured" or "dispatch", so the
#: fallback rate can be counted by a caller rather than disappearing. **Only the flowchart
#: emitter writes it** - reading it after any of the other four returns a stale value from
#: whenever a flowchart was last emitted. `flowchart_mode()` is the safe way to ask.
LAST_MODE: str = "structured"

_MAX_DEPTH = 24


class _Irreducible(Exception):
    """The graph has no structured Python form; the caller falls back to the dispatch loop."""


# ---------------------------------------------------------------------------------------------
# identifiers
# ---------------------------------------------------------------------------------------------


def _ident(text: str, fallback: str = "step") -> str:
    """A safe Python/SQL identifier from arbitrary diagram text.

    Hand-drawn labels reach here as OCR output, so this has to survive empty strings, digits,
    punctuation, non-ASCII and Python keywords. Every emitter routes label text through it.
    """
    cleaned = re.sub(r"[^0-9a-zA-Z_]+", "_", (text or "").strip().lower()).strip("_")
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"{fallback}_{cleaned}" if cleaned else fallback
    if keyword.iskeyword(cleaned) or keyword.issoftkeyword(cleaned):
        cleaned = f"{cleaned}_"
    return cleaned[:60]


#: SQLite's reserved keywords. `_ident` guards Python's, which is not the same set: `order`,
#: `group`, `table`, `index`, `key`, `select` and `default` are all ordinary English words that a
#: hand-drawn ER entity is very likely to be called, and all of them are a syntax error as a bare
#: table or column name. 77 of 200 synthetic ER diagrams failed on `order` alone before this.
_SQL_KEYWORDS = frozenset(
    """
    abort action add after all alter always analyze and as asc attach autoincrement before begin
    between by cascade case cast check collate column commit conflict constraint create cross
    current current_date current_time current_timestamp database default deferrable deferred
    delete desc detach distinct do drop each else end escape except exclude exclusive exists
    explain fail filter first following for foreign from full generated glob group groups having
    if ignore immediate in index indexed initially inner insert instead intersect into is isnull
    join key last left like limit match materialized natural no not nothing notnull null nulls of
    offset on or order others outer over partition plan pragma preceding primary query raise
    range recursive references regexp reindex release rename replace restrict returning right
    rollback row rows savepoint select set table temp temporary then ties to transaction trigger
    unbounded union unique update using vacuum values view virtual when where window with without
    """.split()
)


def _sql_ident(text: str, fallback: str = "entity") -> str:
    """`_ident`, then escape SQL's reserved words as well as Python's.

    A trailing underscore rather than double-quoting: quoting every identifier would also work,
    but the emitted DDL is a *training target*, and `"order"` teaches a model to quote
    unconditionally. `order_` is what a person writing the schema by hand would do.
    """
    name = _ident(text, fallback)
    if name.lower() in _SQL_KEYWORDS:
        name = f"{name}_"
    return name


def _literal(text: str) -> str:
    """A Python/JS string literal. `json.dumps` because it escapes quotes and controls for both."""
    return json.dumps(text or "")


def _suite(lines: list[str], indent: str = "    ") -> list[str]:
    """Indent a block, guaranteeing it holds at least one *statement*.

    `[...] or ["    pass"]` is not enough and this is the bug it hid: several `_statement` roles
    (`start`, `fork`, `join`, `event`, `container`) emit a **comment**, so a branch containing
    only a BPMN intermediate event produced a non-empty list of lines that Python still reads as
    an empty suite. 9 of the 693 hdbpmn pages failed `ast.parse` this way before this helper.
    """
    if not any(line.strip() and not line.lstrip().startswith("#") for line in lines):
        lines = [*lines, "pass"]
    return [f"{indent}{line}" for line in lines]


def _successors(diagram: dict, order: list[str]) -> dict[str, list[str]]:
    """src -> targets, ordered by position in `traversal` so emission follows reading order."""
    rank = {node_id: index for index, node_id in enumerate(order)}
    ids = {node["id"] for node in diagram.get("nodes", [])}
    out: dict[str, list[str]] = {node_id: [] for node_id in ids}
    for edge in diagram.get("edges", []):
        src, dst = edge.get("src"), edge.get("dst")
        if src in out and dst in ids and dst not in out[src]:
            out[src].append(dst)
    for src in out:
        out[src].sort(key=lambda t: rank.get(t, len(rank)))
    return out


def _back_edges(diagram: dict, order: list[str]) -> set[tuple[str, str]]:
    from src.parse.sequences import traversal

    del order
    return traversal(diagram)[1]


# ---------------------------------------------------------------------------------------------
# 1. flowchart -> Python
# ---------------------------------------------------------------------------------------------


def _statement(node: dict) -> list[str]:
    """The Python for one non-branching flowchart node."""
    role = node.get("semantic_role") or "unknown"
    attrs = node.get("attrs") or {}
    text = (node.get("text") or "").strip()
    noun = _ident(str(attrs.get("noun") or text or "value"), "value")
    if role == "start":
        return [f"# start: {text or 'begin'}"]
    if role == "end":
        return ["return ctx"]
    if role == "io":
        if str(attrs.get("verb")) == "write":
            return [f"write_{noun}(ctx[{_literal(noun)}])"]
        return [f"ctx[{_literal(noun)}] = read_{noun}()"]
    if role in ("fork", "join", "event", "container"):
        return [f"# {role}: {text or node['id']}"]
    return [f"ctx = {_ident(text, 'step')}(ctx)"]


def _condition(node: dict) -> str:
    attrs = node.get("attrs") or {}
    raw = str(attrs.get("cond") or node.get("text") or "condition")
    return f"{_ident(raw.rstrip('?'), 'condition')}(ctx)"


def _reachable(start: str, succ: dict[str, list[str]], blocked: str) -> set[str]:
    seen, stack = set(), [start]
    while stack:
        current = stack.pop()
        if current in seen or current == blocked:
            continue
        seen.add(current)
        stack.extend(succ.get(current, ()))
    return seen


def _structured(
    diagram: dict, order: list[str], succ: dict[str, list[str]], back: set[tuple[str, str]]
) -> list[str]:
    """Rebuild if/else/while nesting, or raise `_Irreducible`."""
    nodes = {node["id"]: node for node in diagram.get("nodes", [])}
    rank = {node_id: index for index, node_id in enumerate(order)}
    headers = {dst for _, dst in back}
    emitted: set[str] = set()

    def forward(node_id: str) -> list[str]:
        return [t for t in succ.get(node_id, ()) if (node_id, t) not in back]

    def block(start: str | None, stop: frozenset[str], depth: int) -> list[str]:
        if depth > _MAX_DEPTH:
            raise _Irreducible("nesting deeper than the emitter will follow")
        lines: list[str] = []
        current = start
        while current is not None and current not in stop:
            if current in emitted:
                raise _Irreducible(f"{current} reached from two regions")
            emitted.add(current)
            node = nodes[current]
            targets = forward(current)

            if current in headers:
                # a loop: one successor re-enters the body, the other leaves
                if len(targets) != 1 and len(targets) != 2:
                    raise _Irreducible("loop header with unusable out-degree")
                if len(targets) == 2:
                    body, exit_to = targets[0], targets[1]
                    if current not in _reachable(body, succ, exit_to):
                        body, exit_to = exit_to, body
                    if current not in _reachable(body, succ, exit_to):
                        raise _Irreducible("neither branch of the loop header returns to it")
                    lines.append(f"while {_condition(node)}:")
                    lines += _suite(block(body, stop | {current}, depth + 1))
                    current = exit_to
                    continue
                lines.append("while True:")
                lines += _suite(block(targets[0], stop | {current}, depth + 1))
                return lines

            if len(targets) >= 3:
                raise _Irreducible("out-degree 3+ has no if/else form")
            if len(targets) == 2:
                left, right = targets
                join = _join_point(left, right, succ, rank)
                if join is None:
                    raise _Irreducible("branches never re-converge")
                lines.append(f"if {_condition(node)}:")
                lines += _suite(block(left, stop | {join}, depth + 1))
                other = block(right, stop | {join}, depth + 1)
                lines.append("else:")
                lines += _suite(other)
                current = None if join == "__exit__" else join
                continue

            lines += _statement(node)
            current = targets[0] if targets else None
        return lines

    def _join_point(left, right, succ_map, ranking) -> str | None:
        """The first node both branches reach - the immediate post-dominator, approximated."""
        shared = _reachable(left, succ_map, "") & _reachable(right, succ_map, "")
        shared = {n for n in shared if n in ranking}
        if not shared:
            return "__exit__"
        return min(shared, key=lambda n: ranking[n])

    roots = [n for n in order if n not in emitted]
    body: list[str] = []
    for root in roots:
        if root in emitted:
            continue
        body += block(root, frozenset(), 0)
    if len(emitted) < len(nodes):
        # a node the traversal never placed in a region: refuse rather than silently drop it
        raise _Irreducible("not every node was emitted")
    return body


def _dispatch(diagram: dict, order: list[str], succ: dict[str, list[str]]) -> list[str]:
    """The always-valid fallback: an explicit state variable in a `while True:` loop."""
    nodes = {node["id"]: node for node in diagram.get("nodes", [])}
    lines = [f"state = {_literal(order[0])}", "while True:"]
    for index, node_id in enumerate(order):
        node = nodes[node_id]
        branch = "if" if index == 0 else "elif"
        lines.append(f"    {branch} state == {_literal(node_id)}:")
        targets = succ.get(node_id, [])
        body = [line for line in _statement(node) if not line.startswith("return")]
        indented = [f"        {line}" for line in body]
        if len(targets) >= 2:
            indented.append(f"        if {_condition(node)}:")
            indented.append(f"            state = {_literal(targets[0])}")
            indented.append("        else:")
            indented.append(f"            state = {_literal(targets[1])}")
        elif targets:
            indented.append(f"        state = {_literal(targets[0])}")
        else:
            indented.append("        return ctx")
        lines += indented or ["        pass"]
    lines.append("    else:")
    lines.append("        return ctx")
    return lines


def emit_flowchart_python(diagram: dict, traversal: list[str]) -> str:
    """Flowchart IR -> a Python function. Sets `LAST_MODE` to "structured" or "dispatch"."""
    global LAST_MODE
    order = [n for n in traversal if n] or [node["id"] for node in diagram.get("nodes", [])]
    name = _ident(str(diagram.get("id") or "flowchart"), "flowchart")
    if not order:
        LAST_MODE = "structured"
        return f"def run_{name}(ctx):\n    return ctx\n"

    succ = _successors(diagram, order)
    back = _back_edges(diagram, order)
    try:
        body = _structured(diagram, order, succ, back)
        LAST_MODE = "structured"
    except _Irreducible:
        body = _dispatch(diagram, order, succ)
        LAST_MODE = "dispatch"

    if not body or not any(line.strip().startswith("return") for line in body):
        body = body + ["return ctx"]
    rendered = "\n".join(f"    {line}" for line in body)
    return f"def run_{name}(ctx):\n{rendered}\n"


def flowchart_mode(diagram: dict, traversal: list[str]) -> tuple[str, str]:
    """`(code, mode)` - the race-free way to ask which mode a flowchart resolved to.

    `LAST_MODE` is module state and a caller that emits several diagram types before reading it
    gets the mode of whichever flowchart came last. This returns the pair.
    """
    code = emit_flowchart_python(diagram, traversal)
    return code, LAST_MODE


# ---------------------------------------------------------------------------------------------
# 2. state machine -> Python class
# ---------------------------------------------------------------------------------------------


def emit_state_machine_python(diagram: dict, traversal: list[str]) -> str:
    """State machine IR -> a Python class whose `STATES`/`TRANSITIONS` fit `transitions.Machine`.

    `step` raises on an undefined symbol rather than staying put: a DFA drawn without a total
    transition function is partial, and silently self-looping would invent transitions the
    diagram does not contain.
    """
    nodes = {node["id"]: node for node in diagram.get("nodes", [])}
    order = [n for n in traversal if n in nodes] or list(nodes)
    names: dict[str, str] = {}
    taken: set[str] = set()
    for node_id in order:
        # Two states whose OCR text came back identical (or empty) would otherwise collapse into
        # one row of TABLE and silently merge their transitions.
        name = _ident(nodes[node_id].get("text") or node_id, "q")
        while name in taken:
            name = f"{name}_b"
        taken.add(name)
        names[node_id] = name

    initial = next(
        (n for n in order if (nodes[n].get("attrs") or {}).get("initial")),
        next((n for n in order if nodes[n].get("semantic_role") == "initial-state"), None),
    )
    accepting = [
        n
        for n in order
        if (nodes[n].get("attrs") or {}).get("accepting")
        or nodes[n].get("semantic_role") == "final-state"
    ]

    table: dict[str, dict[str, str]] = {names[n]: {} for n in order}
    spec: list[str] = []
    for edge in diagram.get("edges", []):
        src, dst = edge.get("src"), edge.get("dst")
        if src not in names or dst not in names:
            continue
        symbol = _ident(edge.get("label") or "epsilon", "sym")
        table[names[src]].setdefault(symbol, names[dst])
        spec.append(
            "        {"
            f"'trigger': {_literal(symbol)}, 'source': {_literal(names[src])}, "
            f"'dest': {_literal(names[dst])}"
            "},"
        )

    class_name = "".join(part.title() for part in _ident(str(diagram.get("id") or "m")).split("_"))
    class_name = (class_name or "Machine") + "Machine"

    lines = [f"class {class_name}:", '    """Generated from a state-machine diagram."""', ""]
    lines.append("    STATES = [")
    lines += [f"        {_literal(names[n])}," for n in order]
    lines.append("    ]")
    lines.append(f"    INITIAL = {_literal(names[initial]) if initial else 'None'}")
    rendered_accepting = ", ".join(_literal(names[n]) for n in accepting)
    # `{}` is an empty *dict*; a machine with no accepting state needs an empty *set*.
    lines.append(
        f"    ACCEPTING = {{{rendered_accepting}}}" if accepting else "    ACCEPTING = set()"
    )
    lines.append("    TRANSITIONS = [")
    lines += spec or ["        # the diagram carries no usable transitions"]
    lines.append("    ]")
    lines.append("    TABLE = {")
    for state in table:
        pairs = ", ".join(f"{_literal(k)}: {_literal(v)}" for k, v in table[state].items())
        lines.append(f"        {_literal(state)}: {{{pairs}}},")
    lines.append("    }")
    lines += [
        "",
        "    def __init__(self):",
        "        self.state = self.INITIAL",
        "",
        "    def step(self, symbol):",
        "        row = self.TABLE.get(self.state, {})",
        "        if symbol not in row:",
        "            raise ValueError(f'no transition from {self.state!r} on {symbol!r}')",
        "        self.state = row[symbol]",
        "        return self.state",
        "",
        "    def accepts(self, symbols):",
        "        self.state = self.INITIAL",
        "        for symbol in symbols:",
        "            self.step(symbol)",
        "        return self.state in self.ACCEPTING",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------
# 3. ER -> SQL DDL
# ---------------------------------------------------------------------------------------------


def emit_er_sql(diagram: dict, traversal: list[str]) -> str:
    """ER IR -> SQLite-executable DDL. N-M relationships become join tables with a composite PK."""
    nodes = {node["id"]: node for node in diagram.get("nodes", [])}
    order = [n for n in traversal if n in nodes] or list(nodes)
    entities = [n for n in order if nodes[n].get("semantic_role") == "entity"]
    if not entities:
        entities = [n for n in order if nodes[n].get("shape") == "rectangle"]

    table_of: dict[str, str] = {}
    used: set[str] = set()
    for node_id in entities:
        name = _sql_ident(nodes[node_id].get("text") or node_id, "entity")
        while name in used:
            name = f"{name}_x"
        used.add(name)
        table_of[node_id] = name

    statements: list[str] = []
    for node_id in entities:
        table = table_of[node_id]
        columns = [f"    {table}_id INTEGER PRIMARY KEY"]
        seen = {f"{table}_id"}
        for column in (nodes[node_id].get("attrs") or {}).get("columns") or []:
            column_name = _sql_ident(str(column[0]), "col")
            if column_name in seen:
                continue
            seen.add(column_name)
            sql_type = str(column[1]).upper() if len(column) > 1 else "TEXT"
            if sql_type not in ("TEXT", "INTEGER", "REAL", "BLOB", "NUMERIC"):
                sql_type = "TEXT"
            columns.append(f"    {column_name} {sql_type}")
        statements.append(f"CREATE TABLE {table} (\n" + ",\n".join(columns) + "\n);")

    joins = 0
    for node_id in order:
        node = nodes[node_id]
        if node.get("semantic_role") != "relationship":
            continue
        left = [e["src"] for e in diagram.get("edges", []) if e.get("dst") == node_id]
        right = [e["dst"] for e in diagram.get("edges", []) if e.get("src") == node_id]
        parents = [n for n in left if n in table_of]
        children = [n for n in right if n in table_of]
        if not parents or not children:
            continue
        parent, child = table_of[parents[0]], table_of[children[0]]
        if str((node.get("attrs") or {}).get("cardinality")) == "n-m":
            joins += 1
            name = f"{parent}_{child}_link_{joins}"
            statements.append(
                f"CREATE TABLE {name} (\n"
                f"    {parent}_ref INTEGER REFERENCES {parent}({parent}_id),\n"
                f"    {child}_ref INTEGER REFERENCES {child}({child}_id),\n"
                f"    PRIMARY KEY ({parent}_ref, {child}_ref)\n);"
            )
        else:
            fk = f"{parent}_fk_{len(statements)}"
            statements.append(
                f"ALTER TABLE {child} ADD COLUMN {fk} INTEGER REFERENCES {parent}({parent}_id);"
            )

    if not statements:
        statements = ["CREATE TABLE placeholder (\n    placeholder_id INTEGER PRIMARY KEY\n);"]
    return "\n\n".join(statements) + "\n"


# ---------------------------------------------------------------------------------------------
# 4. wireframe -> React + Tailwind
# ---------------------------------------------------------------------------------------------

_TAILWIND = {
    "container": "flex flex-col gap-2 p-4",
    "ui-label": "text-sm text-slate-700",
    "ui-button": "rounded-md bg-sky-600 px-3 py-1.5 text-white hover:bg-sky-700",
    "ui-image": "h-32 w-full rounded object-cover bg-slate-200",
    "ui-input": "rounded border border-slate-300 px-2 py-1",
}
_JSX_TAG = {
    "ui-label": "span",
    "ui-button": "button",
    "ui-image": "img",
    "ui-input": "input",
}
_VOID = {"img", "input", "br", "hr"}
_SAFE_TAGS = {
    "div",
    "section",
    "nav",
    "header",
    "footer",
    "ul",
    "li",
    "span",
    "p",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "a",
    "button",
    "img",
    "input",
    "label",
    "form",
    "main",
    "aside",
    "table",
    "tr",
    "td",
    "th",
}


def emit_wireframe_react(diagram: dict, traversal: list[str]) -> str:
    """Wireframe containment tree -> a React function component with Tailwind classes.

    Only `contains` edges are followed. sketch2code's IR has no other kind, and a wireframe that
    somehow carried one would otherwise be rendered as if it were nesting.
    """
    nodes = {node["id"]: node for node in diagram.get("nodes", [])}
    order = [n for n in traversal if n in nodes] or list(nodes)
    children: dict[str, list[str]] = {node_id: [] for node_id in nodes}
    has_parent: set[str] = set()
    for edge in diagram.get("edges", []):
        kind = (edge.get("attrs") or {}).get("kind", "contains")
        src, dst = edge.get("src"), edge.get("dst")
        if kind == "contains" and src in children and dst in nodes and dst not in has_parent:
            children[src].append(dst)
            has_parent.add(dst)

    rank = {node_id: index for index, node_id in enumerate(order)}
    for node_id in children:
        children[node_id].sort(key=lambda t: rank.get(t, len(rank)))
    roots = [n for n in order if n not in has_parent] or order[:1]

    def render(node_id: str, indent: int, depth: int, seen: frozenset[str]) -> list[str]:
        if node_id in seen or depth > _MAX_DEPTH:
            return []
        node = nodes[node_id]
        role = node.get("semantic_role") or "container"
        tag = _JSX_TAG.get(role) or str((node.get("attrs") or {}).get("html_tag") or "div")
        tag = tag if tag in _SAFE_TAGS else "div"
        css = _TAILWIND.get(role, "block")
        pad = "  " * indent
        text = (node.get("text") or "").strip()

        if tag in _VOID:
            extra = f" alt={_literal(text)}" if tag == "img" else ""
            return [f'{pad}<{tag} className="{css}"{extra} />']

        inner: list[str] = []
        if text:
            inner.append(f"{pad}  {{{_literal(text)}}}")
        for child in children[node_id]:
            inner += render(child, indent + 1, depth + 1, seen | {node_id})
        if not inner:
            return [f'{pad}<{tag} className="{css}" />']
        return [f'{pad}<{tag} className="{css}">', *inner, f"{pad}</{tag}>"]

    body: list[str] = []
    for root in roots:
        body += render(root, 3, 0, frozenset())
    if not body:
        body = ['      <div className="block" />']

    name = "".join(part.title() for part in _ident(str(diagram.get("id") or "screen")).split("_"))
    name = (name or "Screen") + "Screen"
    return (
        f"export default function {name}() {{\n"
        "  return (\n"
        '    <div className="min-h-screen bg-white p-6">\n'
        + "\n".join(body)
        + "\n    </div>\n  );\n}\n"
    )


# ---------------------------------------------------------------------------------------------
# 5. circuit -> SPICE netlist
# ---------------------------------------------------------------------------------------------

_DEVICE_LETTER = {"resistor": "R", "capacitor": "C", "inductor": "L", "source": "V", "diode": "D"}


def emit_circuit_spice(diagram: dict, traversal: list[str]) -> str:
    """Circuit IR -> a SPICE netlist. Ground is net 0; every card is `<ref> <n+> <n-> <value>`."""
    nodes = {node["id"]: node for node in diagram.get("nodes", [])}
    order = [n for n in traversal if n in nodes] or list(nodes)

    title = str(diagram.get("id") or "circuit")
    lines = [f"* {title} - generated from a circuit diagram"]
    used: set[str] = set()
    nets: set[str] = set()
    uses: Counter[str] = Counter()
    has_source = False

    for index, node_id in enumerate(order):
        node = nodes[node_id]
        attrs = node.get("attrs") or {}
        role = node.get("semantic_role") or ""
        letter = str(attrs.get("component") or _DEVICE_LETTER.get(role, "R"))[:1].upper()
        if not letter.isalpha():
            letter = "R"
        ref = re.sub(r"[^0-9A-Za-z]", "", str(attrs.get("ref") or node.get("text") or "")).upper()
        if not ref or not ref[0].isalpha() or ref[0] != letter:
            ref = f"{letter}{index}"
        while ref in used:
            ref = f"{ref}A"
        used.add(ref)

        positive = re.sub(r"[^0-9A-Za-z]", "", str(attrs.get("net_pos") or index + 1)) or "1"
        negative = re.sub(r"[^0-9A-Za-z]", "", str(attrs.get("net_neg") or 0)) or "0"
        nets.update((positive, negative))
        uses[positive] += 1
        uses[negative] += 1
        value = re.sub(r"[^0-9A-Za-z.+-]", "", str(attrs.get("value") or "1k")) or "1k"
        if letter == "V":
            has_source = True
            lines.append(f"{ref} {positive} {negative} DC {value}")
        else:
            lines.append(f"{ref} {positive} {negative} {value}")

    if not has_source:
        # A deck with no independent source has nothing to solve for. Tie the auto-source across
        # an existing net so it does not introduce a dangling one of its own.
        anchor = next((n for n in sorted(nets) if n != "0"), "1")
        nets.add(anchor)
        lines.append(f"VSRC_AUTO {anchor} 0 DC 5")
        uses[anchor] += 1
        uses["0"] += 1

    if "0" not in uses:
        # Every SPICE deck needs a ground; tie the lowest-numbered net to it rather than fail.
        anchor = sorted(uses)[0] if uses else "1"
        lines.append(f"R_GND_TIE {anchor} 0 1u")
        uses[anchor] += 1
        uses["0"] += 1

    # A net that appears on exactly one card is a wire soldered at one end. `quality.check_spice`
    # rejects the netlist for it, and it is a real defect rather than a checker being fussy: a
    # disconnected synthetic circuit puts two parts on nets nothing else touches (40 of 200 before
    # this pass). Terminating them to ground is the smallest edit that makes the deck solvable.
    for index, net in enumerate(sorted(n for n, count in uses.items() if count < 2 and n != "0")):
        lines.append(f"R_TERM{index} {net} 0 1meg")
        uses[net] += 1
        uses["0"] += 1
    if uses["0"] < 2:
        other = next((n for n in sorted(uses) if n != "0"), "1")
        lines.append(f"R_TERM_GND {other} 0 1meg")
        uses["0"] += 1

    lines += [".op", ".end"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------------------------

_EMITTERS = {
    "flowchart": emit_flowchart_python,
    "state_machine": emit_state_machine_python,
    "er": emit_er_sql,
    "wireframe": emit_wireframe_react,
    "circuit": emit_circuit_spice,
}
#: bpmn is hdbpmn's own word for the same thing 12.1.6 calls a flowchart.
#: `er_diagram` is schemas/ir.schema.json's own enum value; without it `for_type` refused the
#: one spelling a schema-valid IR document can actually carry.
_ALIASES = {
    "bpmn": "flowchart",
    "state-machine": "state_machine",
    "erd": "er",
    "er_diagram": "er",
    "ui": "wireframe",
}


def for_type(diagram_type: str):
    """The `emit(diagram, traversal) -> str` generator for a diagram type.

    Raises `KeyError` rather than defaulting to the flowchart emitter: a diagram type nobody
    wrote a generator for is a gap to report, not a silent mis-emission into Python.
    """
    key = _ALIASES.get(str(diagram_type), str(diagram_type))
    if key not in _EMITTERS:
        raise KeyError(f"no target generator for diagram_type {diagram_type!r}")
    return _EMITTERS[key]


def verify(code: str, diagram_type: str) -> tuple[bool, str]:
    """`(ok, detail)` for emitted code, routed to the checker this repo already committed.

    Nothing is re-implemented here. Python/React/SPICE go to 12.1.7's `src.codegen.quality`,
    which is the filter the pairs must survive anyway, so an emitter that passes a private check
    and fails the real gate is not possible. SQL goes to `src.eval.sql.check` instead of
    `quality.check_sql`, because that one only executes the script and SQLite's DDL accepts a
    foreign key pointing at a table nobody created; `eval.sql` resolves every FK statically.
    """
    from src.codegen import quality
    from src.eval import sql

    language = language_for(diagram_type)
    if language == "sql":
        result = sql.check(code)
        return bool(result["ok"]), f"{result['kind']}: {result['detail']}"
    ok, kind, detail = quality.CHECKS[language](code)
    return ok, f"{kind}: {detail}"


def language_for(diagram_type: str) -> str:
    """The `language` field a pair of this diagram type carries."""
    return LANGUAGES[_ALIASES.get(str(diagram_type), str(diagram_type))]
