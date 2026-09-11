"""Phase 12.3.3 - functional correctness: pass@1 against per-diagram unit tests, in the sandbox.

    python -m src.llm.functional --self-test          # mutation study on the reference programs

## What a "unit test" is for a program transcribed from a drawing

The generated programs call operations the diagram only names (`evaluate_the_application(ctx)`),
so there is no oracle output to compare. What a diagram *does* specify is behaviour, and the
per-diagram test is therefore **behavioural equivalence with the reference program on inputs
derived from the diagram**, executed in 11.2.9's sandbox:

    flowchart       Every free name the program calls is bound to a recording stub (via a
                    `__builtins__` mapping with `__missing__`, so nothing needs to be declared),
                    as is every top-level function whose name resolves to a diagram node. Each
                    call or `print` that resolves to a node appends that node's id to a trace.
                    Every truth test on a stub's result consumes one bit of a decision script.
                    The **complete decision tree to depth `MAX_BITS`** is enumerated, and the
                    signature is the *set* of (node trace, how the run ended). Pass iff the
                    candidate's set equals the reference's. Using the set of all paths makes the
                    test indifferent to branch polarity (`if rejected` vs `if approved`) and to
                    how predicates are named, and sensitive to what the diagram asserts: which
                    operations happen, in which order, on which branch.
    state_machine   The program's acceptor (`accepts`/`accept`/`run`/..., top-level or on the one
                    class) is run on **every word of length 0-3** over the diagram's alphabet (the
                    edge labels, whole and split on commas). An exception on a word counts as
                    rejection, which is what a missing transition means. Pass iff every verdict
                    equals the reference's.

A name resolves to a node when its slug equals the slug of the node's text or id, or is within
difflib ratio `FUZZ` of the text slug (the model may write `create_new_bank_account` for a node
labelled `create new bank acc.`). Names that resolve to no node (`condition`, `step`, helpers)
leave no trace, so predicate naming is free and invented operations are 12.3.8's business.
The reference program passes its own test by construction, so that is not evidence. The
evidence is `mutation_study`: every reference is mutated (branch swap inside one decision is
*expected* to pass - the tests are polarity-invariant by design - while dropped statements,
reordered statements, redirected transitions and flipped accepting sets must fail).
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Any

MAX_BITS = 12
MAX_WORD = 3
MAX_SYMBOLS = 6
FUZZ = 0.85

#: Runs inside the sandbox. Reads spec.json + prog.py from its cwd, prints one JSON line.
DRIVER = r'''
import builtins, difflib, json, re, sys
sys.setrecursionlimit(400)
spec = json.load(open("spec.json", encoding="utf-8"))
source = open("prog.py", encoding="utf-8").read()
FUZZ = spec["fuzz"]

def slug(text):
    return re.sub(r"_+", "_", re.sub(r"[^0-9a-z]+", "_", str(text).lower())).strip("_")

NODES = spec["nodes"]  # [[id, text], ...]
BY_SLUG = {}
TEXT_SLUGS = []
for nid, text in NODES:
    BY_SLUG.setdefault(slug(nid), nid)
    if slug(text):
        BY_SLUG.setdefault(slug(text), nid)
        TEXT_SLUGS.append((slug(text), nid))
_cache = {}
def resolve(name):
    key = slug(name)
    if key in _cache:
        return _cache[key]
    hit = BY_SLUG.get(key)
    if hit is None and key and len(key) >= 4:
        best, score = None, 0.0
        for s, nid in TEXT_SLUGS:
            r = difflib.SequenceMatcher(None, key, s).ratio()
            if r > score:
                best, score = nid, r
        if score >= FUZZ:
            hit = best
    _cache[key] = hit
    return hit

class Budget(Exception):
    pass
class Cap(Exception):
    pass

class Run:
    def __init__(self, prefix, max_bits, max_events):
        self.prefix, self.max_bits, self.max_events = list(prefix), max_bits, max_events
        self.bits, self.trace, self.events = [], [], 0
    def bit(self):
        i = len(self.bits)
        if i >= self.max_bits:
            raise Cap()
        b = self.prefix[i] if i < len(self.prefix) else False
        self.bits.append(b)
        return b
    def event(self, nid):
        self.events += 1
        if self.events > self.max_events:
            raise Budget()
        if nid is not None and (not self.trace or self.trace[-1] != nid):
            self.trace.append(nid)

RUN = None

class P:
    """Stand-in for any value the program gets from an operation it did not define."""
    def __bool__(self): return RUN.bit()
    def __call__(self, *a, **k): RUN.event(None); return P()
    def __getattr__(self, name):
        if name.startswith("__"): raise AttributeError(name)
        return P()
    def __getitem__(self, k): return P()
    def __setitem__(self, k, v): pass
    def __setattr__(self, k, v): pass
    def __contains__(self, k): return RUN.bit()
    def __eq__(self, o): return RUN.bit()
    def __ne__(self, o): return RUN.bit()
    def __lt__(self, o): return RUN.bit()
    def __gt__(self, o): return RUN.bit()
    def __le__(self, o): return RUN.bit()
    def __ge__(self, o): return RUN.bit()
    def __hash__(self): return id(self)
    def __iter__(self): return iter(())
    def __len__(self): return 0
    def __int__(self): return 0
    def __float__(self): return 0.0
    def __index__(self): return 0
    def __str__(self): return ""
    def __repr__(self): return "P()"
    def __add__(self, o): return P()
    __radd__ = __sub__ = __rsub__ = __mul__ = __rmul__ = __truediv__ = __mod__ = __add__
    def __enter__(self): return self
    def __exit__(self, *a): return False

def stub(name):
    nid = resolve(name)
    def f(*a, **k):
        RUN.event(nid)
        return P()
    f.__name__ = name
    return f

class Builtins(dict):
    def __missing__(self, key):
        return stub(key)

def _print(*args, **kw):
    for a in args:
        if isinstance(a, str):
            RUN.event(resolve(a))
def _input(*a):
    return "yes" if RUN.bit() else "no"

def make_globals(name):
    b = Builtins(vars(builtins))
    b["print"] = _print
    b["input"] = _input
    b["exit"] = b["quit"] = lambda *a: (_ for _ in ()).throw(SystemExit(0))
    return {"__name__": name, "__builtins__": b}

def wrap(fn, nid):
    def w(*a, **k):
        RUN.event(nid)
        return fn(*a, **k)
    w.__name__ = getattr(fn, "__name__", "w")
    return w

ENTRY_NAMES = ("main", "run", "process", "execute", "start", "workflow", "flow")

def load(name="__dsprobe__"):
    g = make_globals(name)
    exec(compile(source, "prog.py", "exec"), g)
    return g

def flowchart():
    global RUN
    RUN = Run([], 10**6, 10**6)
    try:
        g = load()
        module_ok = True
    except (Budget, Cap):
        module_ok = False
        g = None
    funcs = {}
    if g is not None:
        for k, v in list(g.items()):
            if hasattr(v, "__code__") and v.__code__.co_filename == "prog.py":
                funcs[k] = v
    entry = None
    if funcs:
        names = sorted(funcs)
        runs = [n for n in names if n.lower().startswith("run")]
        mains = [n for n in names if n.lower() in ENTRY_NAMES]
        loose = [n for n in names if resolve(n) is None]
        for group in (runs, mains, loose if len(loose) == 1 else []):
            if group:
                entry = group[0]
                break
    max_events = spec["max_events"]
    paths = set()
    stack = [[]]
    runs_done = 0
    while stack:
        prefix = stack.pop()
        RUN = Run(prefix, spec["max_bits"], max_events)
        status = "return"
        try:
            if entry is None:
                load("__main__")
            else:
                g2 = load()
                for k, v in list(g2.items()):
                    if k in funcs and resolve(k) is not None:
                        g2[k] = wrap(v, resolve(k))
                fn = g2[entry]
                code = fn.__code__
                required = code.co_argcount - len(fn.__defaults__ or ())
                RUN.trace, RUN.bits, RUN.events = [], [], 0
                fn(*[P() for _ in range(max(0, required))])
        except Cap:
            status = "cap"
        except Budget:
            status = "budget"
        except SystemExit:
            status = "return"
        except RecursionError:
            status = "error:RecursionError"
        except Exception as exc:
            status = "error"
        runs_done += 1
        paths.add((tuple(RUN.trace), status))
        # branch on every default (False) bit this run took beyond its prefix
        for i in range(len(prefix), len(RUN.bits)):
            stack.append(RUN.bits[:i] + [True])
    return {"kind": "flowchart", "entry": entry, "runs": runs_done,
            "paths": sorted([list(t), s] for t, s in paths)}

ACCEPT_NAMES = ("accepts", "accept", "is_accepted", "recognize", "recognise", "matches",
                "run", "simulate", "process", "check", "evaluate")

def acceptor():
    g = load()
    for name in ACCEPT_NAMES:
        v = g.get(name)
        if callable(v) and hasattr(v, "__code__"):
            return v
    classes = [v for k, v in g.items() if isinstance(v, type) and getattr(v, "__module__", "") in ("__dsprobe__", None, "builtins") and k[:1] != "_"]
    classes = [c for c in classes if any(hasattr(c, n) for n in ACCEPT_NAMES)]
    if len(classes) >= 1:
        cls = classes[0]
        for name in ACCEPT_NAMES:
            if hasattr(cls, name):
                def call(word, cls=cls, name=name):
                    return getattr(cls(), name)(word)
                return call
    # A stepper interface: one 1-argument transition method and a 0-argument final-state test
    # (or a declared accepting collection compared against `.state`).
    import inspect
    own = [v for k, v in g.items() if isinstance(v, type) and getattr(v, "__module__", "") == "__dsprobe__"]
    STEP_WORDS = ("transition", "step", "process", "input", "feed", "consume", "read", "next", "move", "handle", "event", "trigger")
    for cls in own:
        methods = {k: v for k, v in vars(cls).items() if inspect.isfunction(v)}
        def arity(f):
            params = list(inspect.signature(f).parameters.values())[1:]
            return len([p for p in params if p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)])
        steps = [k for k, f in methods.items() if arity(f) == 1 and any(w in k.lower() for w in STEP_WORDS)]
        finals = [k for k, f in methods.items() if arity(f) == 0 and ("final" in k.lower() or "accept" in k.lower())]
        if not steps:
            continue
        try:
            init_arity = arity(cls.__init__) if inspect.isfunction(getattr(cls, "__init__", None)) else 0
        except (TypeError, ValueError):
            init_arity = 0
        if init_arity:
            continue
        def call(word, cls=cls, step=steps[0], final=(finals[0] if finals else None)):
            obj = cls()
            for sym in word:
                getattr(obj, step)(sym)
            if final is not None:
                return getattr(obj, final)()
            for attr in ("accepting", "accept_states", "final_states", "ACCEPTING", "FINAL", "accepting_states", "finals"):
                if hasattr(obj, attr):
                    return getattr(obj, "state", getattr(obj, "current_state", None)) in getattr(obj, attr)
            raise LookupError("no final-state test")
        call.__name__ = f"{cls.__name__}.{steps[0]}"
        return call
    return None

def state_machine():
    global RUN
    RUN = Run([], 10**6, 10**6)
    fn = acceptor()
    if fn is None:
        return {"kind": "state_machine", "entry": None, "verdicts": None}
    verdicts = {}
    errors = 0
    as_string = False
    for word in spec["words"]:
        RUN = Run([], 64, 10**5)
        try:
            ok = bool(fn(list(word)))
        except Exception:
            ok = False
            errors += 1
        verdicts["␟".join(word)] = ok
    if errors == len(spec["words"]):
        verdicts, errors, as_string = {}, 0, True
        for word in spec["words"]:
            RUN = Run([], 64, 10**5)
            try:
                ok = bool(fn("".join(word)))
            except Exception:
                ok = False
                errors += 1
            verdicts["␟".join(word)] = ok
    return {"kind": "state_machine", "entry": getattr(fn, "__name__", "call"),
            "as_string": as_string, "errors": errors, "verdicts": verdicts}

try:
    out = flowchart() if spec["kind"] == "flowchart" else state_machine()
    out["ok"] = True
except SyntaxError as exc:
    out = {"ok": False, "error": "syntax"}
except BaseException as exc:
    out = {"ok": False, "error": type(exc).__name__ + ": " + str(exc)[:200]}
sys.stdout.write("\n@@SIG@@" + json.dumps(out) + "\n")
'''


_SPLIT = re.compile(r"[,+\s;/.|]+")
_EPSILON = {"", "epsilon", "eps", "ε", "e", "lambda", "λ"}


def symbols_of(label: object) -> list[str]:
    """The input symbols one drawn edge label stands for: `a,b` and `0+1` are two symbols."""
    text = str(label or "").strip()
    if text.lower() in _EPSILON:
        return []
    return [piece for piece in _SPLIT.split(text) if piece]


def alphabet(diagram: dict) -> list[str]:
    symbols: set[str] = set()
    for edge in diagram.get("edges", []):
        symbols.update(symbols_of(edge.get("label")))
    return sorted(symbols)[:MAX_SYMBOLS]


def automaton(diagram: dict) -> tuple[set[str], set[str], dict[str, dict[str, set[str]]]]:
    """(initial states, accepting states, delta) read from the IR, epsilon edges under `""`."""
    nodes = {n["id"]: n for n in diagram.get("nodes", [])}
    attrs = {k: (n.get("attrs") or {}) for k, n in nodes.items()}
    initial = {k for k in nodes if attrs[k].get("initial")}
    initial |= {k for k, n in nodes.items() if n.get("semantic_role") == "initial-state"}
    accepting = {k for k in nodes if attrs[k].get("accepting")}
    accepting |= {k for k, n in nodes.items() if n.get("semantic_role") == "final-state"}
    delta: dict[str, dict[str, set[str]]] = {k: {} for k in nodes}
    for edge in diagram.get("edges", []):
        src, dst = edge.get("src"), edge.get("dst")
        if src not in nodes or dst not in nodes:
            continue
        syms = symbols_of(edge.get("label")) or [""]
        for sym in syms:
            delta[src].setdefault(sym, set()).add(dst)
    return initial, accepting, delta


def ir_verdicts(diagram: dict, word_list: list[list[str]]) -> dict[str, bool]:
    """What the drawn automaton says about each word (NFA semantics, epsilon closure)."""
    initial, accepting, delta = automaton(diagram)

    def closure(states: set[str]) -> set[str]:
        stack, seen = list(states), set(states)
        while stack:
            for nxt in delta.get(stack.pop(), {}).get("", ()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return seen

    out = {}
    for word in word_list:
        current = closure(set(initial))
        for sym in word:
            current = closure({d for s in current for d in delta.get(s, {}).get(sym, ())})
        out["␟".join(word)] = bool(current & accepting)
    return out


OPERATION_ROLES = frozenset({"process"})
_ROOT = "␀root"


def flow_graph(diagram: dict) -> dict[str, Any]:
    """Operations, the ones reachable from a start, reachability, and fork-reach, from the IR.

    Only directed edges order anything (hdbpmn's undirected edges are message flows between
    pools). Several start events are joined under a virtual root that behaves as a fork, since a
    sequential program must run every pool and may do so in either order.
    """
    nodes = {n["id"]: n for n in diagram.get("nodes", [])}
    role = {k: str(n.get("semantic_role") or "") for k, n in nodes.items()}
    succ: dict[str, set[str]] = {k: set() for k in nodes}
    incoming: dict[str, int] = {k: 0 for k in nodes}

    for edge in diagram.get("edges", []):
        a, b = edge.get("src"), edge.get("dst")
        if (
            a in nodes
            and b in nodes
            and edge.get("directed", True) is not False
            and b not in succ[a]
        ):
            succ[a].add(b)
            incoming[b] += 1

    starts = [k for k in nodes if role[k] == "start"]
    if not starts:
        starts = [k for k in nodes if incoming[k] == 0 and role[k] not in ("container", "io")]

    succ[_ROOT] = set(starts)
    reach: dict[str, set[str]] = {}

    for k in succ:
        seen: set[str] = set()
        stack = list(succ[k])
        while stack:
            x = stack.pop()
            if x not in seen:
                seen.add(x)
                stack.extend(succ.get(x, ()))
        reach[k] = seen

    ops = {k for k in nodes if role[k] in OPERATION_ROLES}
    forks = [k for k in nodes if role[k] == "fork"] + [_ROOT]

    return {"ops": ops, "required": ops & reach[_ROOT], "reach": reach, "forks": forks}


def flowchart_verdict(candidate: dict[str, Any], diagram: dict) -> tuple[bool, str]:
    """Does the program's path set respect what the drawn flow asserts? (see module docstring)"""
    graph = flow_graph(diagram)
    ops, required, reach = graph["ops"], graph["required"], graph["reach"]
    paths = [([n for n in trace if n in ops], status) for trace, status in candidate["paths"]]

    if any(status in ("error", "budget") or status.startswith("error") for _, status in paths):
        return False, "crash_or_nontermination"

    if not any(status == "return" for _, status in paths) and required:
        return False, "never_returns"

    executed = {n for trace, _ in paths for n in trace}
    if not required <= executed:
        return False, "missing_operation"

    concurrent: dict[tuple[str, str], bool] = {}
    for trace, _ in paths:
        for a, b in zip(trace, trace[1:], strict=False):
            if a in reach[b] and b not in reach[a]:
                return False, "order_violation"

        present = sorted(set(trace))
        for i, a in enumerate(present):
            for b in present[i + 1 :]:
                if a in reach[b] or b in reach[a]:
                    continue
                key = (a, b)
                if key not in concurrent:
                    concurrent[key] = any(
                        a in reach[f] and b in reach[f] and _fork_splits(f, a, b, reach, diagram)
                        for f in graph["forks"]
                    )
                if not concurrent[key]:
                    return False, "exclusive_branches_both_ran"

    return True, "equal"


def _fork_splits(fork: str, a: str, b: str, reach: dict[str, set[str]], diagram: dict) -> bool:
    """True when `a` and `b` are reached from `fork` through different outgoing branches."""
    if fork == _ROOT:
        # the start events, less any start another start already leads to
        out = [k for k in reach[_ROOT] if not any(k in reach[o] for o in reach[_ROOT] if o != k)]
    else:
        out = [
            e.get("dst")
            for e in diagram.get("edges", [])
            if e.get("src") == fork and e.get("directed", True) is not False
        ]

    via_a = {o for o in out if o == a or a in reach.get(o, set())}
    via_b = {o for o in out if o == b or b in reach.get(o, set())}
    return bool(via_a and via_b and (via_a != via_b or len(via_a) > 1))


def words(symbols: list[str], max_len: int = MAX_WORD) -> list[list[str]]:
    out: list[list[str]] = [[]]
    frontier: list[list[str]] = [[]]
    for _ in range(max_len):
        frontier = [w + [s] for w in frontier for s in symbols]
        out.extend(frontier)
    return out


def spec_for(diagram: dict) -> dict[str, Any]:
    kind = "state_machine" if diagram.get("diagram_type") == "state_machine" else "flowchart"
    nodes = [[n["id"], str(n.get("text") or "")] for n in diagram.get("nodes", [])]
    spec: dict[str, Any] = {
        "kind": kind,
        "nodes": nodes,
        "fuzz": FUZZ,
        "max_bits": MAX_BITS,
        "max_events": 4 * len(nodes) + 16,
    }
    if kind == "state_machine":
        spec["words"] = words(alphabet(diagram))
    return spec


def signature(code: str, diagram: dict, timeout_s: float = 20.0) -> dict[str, Any]:
    """Run the driver on `code` in the sandbox; the behaviour signature, or an error record."""
    from src.rl import sandbox

    result = sandbox.run(
        DRIVER,
        timeout_s=timeout_s,
        memory_limit_mb=512,
        files={"prog.py": code, "spec.json": json.dumps(spec_for(diagram))},
    )
    marker = "@@SIG@@"
    stdout = result.get("stdout") or ""
    if marker in stdout:
        return json.loads(stdout.rsplit(marker, 1)[1].strip().splitlines()[0])
    return {"ok": False, "error": f"sandbox:{result['kind']}", "detail": result.get("detail")}


def expected(diagram: dict, reference_code: str) -> dict[str, Any]:
    """The signature a correct program must reproduce.

    Flowchart: the reference program's behaviour (the diagram's gateway semantics - which
    branch of a fork runs, what a join waits for - are not recoverable from the IR alone, and
    12.1.6's emitter is the committed decision about them). State machine: the **drawn automaton
    itself**, because 12.1.6's emitter was measured to disagree with it (`reference_study`).
    """
    if diagram.get("diagram_type") == "state_machine":
        spec = spec_for(diagram)
        return {
            "ok": True,
            "kind": "state_machine",
            "verdicts": ir_verdicts(diagram, spec["words"]),
        }
    return {"ok": True, "kind": "flowchart", "diagram": diagram}


def compare(candidate: dict[str, Any], reference: dict[str, Any]) -> tuple[bool, str]:
    """(pass, reason). The reference must itself have produced a usable signature."""
    if not reference.get("ok"):
        return False, "reference_unusable"
    if not candidate.get("ok"):
        return False, "candidate_" + str(candidate.get("error", "error")).split(":")[0]
    if reference["kind"] == "flowchart":
        return flowchart_verdict(candidate, reference["diagram"])
    if candidate.get("verdicts") is None:
        return False, "no_acceptor"
    if candidate["verdicts"] == reference["verdicts"]:
        return True, "equal"
    return False, "verdicts_differ"


def is_trivial(reference: dict[str, Any]) -> bool:
    """A test that cannot fail anything non-empty: one empty path, or all verdicts equal."""
    if not reference.get("ok"):
        return True
    if reference["kind"] == "flowchart":
        return not flow_graph(reference["diagram"])["required"]
    verdicts = reference.get("verdicts") or {}
    return len(set(verdicts.values())) <= 1


# ---------------------------------------------------------------- mutation study (self-test)


def mutants(code: str) -> dict[str, str]:
    """Deliberate behaviour changes to a reference program, one per kind where applicable."""
    out: dict[str, str] = {}
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return out

    class Collect(ast.NodeVisitor):
        def __init__(self) -> None:
            self.calls: list[ast.stmt] = []
            self.bodies: list[list[ast.stmt]] = []

        def generic_visit(self, node):
            for field in ("body", "orelse"):
                seq = getattr(node, field, None)
                if isinstance(seq, list) and seq and isinstance(seq[0], ast.stmt):
                    self.bodies.append(seq)
            super().generic_visit(node)

    col = Collect()
    col.visit(tree)

    def is_op(stmt: ast.stmt) -> bool:
        return (
            isinstance(stmt, ast.Assign)
            and isinstance(stmt.value, ast.Call)
            and isinstance(stmt.value.func, ast.Name)
            and stmt.value.func.id not in ("step", "condition")
        )

    # drop the first operation statement found
    for body in col.bodies:
        idx = [i for i, s in enumerate(body) if is_op(s)]
        if idx:
            t = ast.parse(code)
            c2 = Collect()
            c2.visit(t)
            target = c2.bodies[col.bodies.index(body)]
            del target[idx[0]]
            if not target:
                target.append(ast.Pass())
            out["drop_operation"] = ast.unparse(t)
            break

    # swap two adjacent operation statements
    for bi, body in enumerate(col.bodies):
        idx = [i for i in range(len(body) - 1) if is_op(body[i]) and is_op(body[i + 1])]
        if idx:
            t = ast.parse(code)
            c2 = Collect()
            c2.visit(t)
            target = c2.bodies[bi]
            i = idx[0]
            if ast.unparse(target[i]) != ast.unparse(target[i + 1]):
                target[i], target[i + 1] = target[i + 1], target[i]
                out["swap_operations"] = ast.unparse(t)
                break

    # swap the branches of the first if/else (expected to PASS: polarity-invariant)
    t = ast.parse(code)
    for node in ast.walk(t):
        if (
            isinstance(node, ast.If)
            and node.orelse
            and isinstance(node.test, ast.Call)
            and not (len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If))
        ):
            node.body, node.orelse = node.orelse, node.body
            out["swap_branch_polarity"] = ast.unparse(t)
            break

    # state machine: redirect one transition / flip accepting set
    m = re.search(r'("(\w+)":\s*\{[^{}]*?"[^"]+":\s*")(\w+)(")', code)
    if m and "TABLE" in code:
        states = re.findall(r'^\s*"(\w+)",\s*$', code, re.M)
        other = next((s for s in states if s != m.group(3)), None)
        if other:
            out["redirect_transition"] = code[: m.start(3)] + other + code[m.end(3) :]

    acc = re.search(r"ACCEPTING = \{([^}]*)\}", code)
    if acc:
        states = re.findall(r'^\s*"(\w+)",\s*$', code, re.M)
        current = set(re.findall(r'"(\w+)"', acc.group(1)))
        flipped = sorted(set(states) - current)
        body = ", ".join(f'"{s}"' for s in flipped) if flipped else ""
        out["flip_accepting"] = (
            code[: acc.start()] + "ACCEPTING = {" + body + "}" + code[acc.end() :]
        )

    return out


EXPECTED_PASS = {"swap_branch_polarity"}


def mutation_study(pairs: list[dict], diagrams: list[dict]) -> dict[str, Any]:
    from collections import Counter

    tallies: dict[str, Counter] = {}
    trivial: Counter = Counter()

    for pair, diagram in zip(pairs, diagrams, strict=False):
        kind_of = str(diagram.get("diagram_type"))
        ref = expected(diagram, pair["target_code"])

        if is_trivial(ref):
            trivial[kind_of] += 1

        ok, _ = compare(signature(pair["target_code"], diagram), ref)
        tallies.setdefault(f"{kind_of}/reference", Counter())[ok] += 1

        for kind, code in mutants(pair["target_code"]).items():
            passed, _ = compare(signature(code, diagram), ref)
            tallies.setdefault(f"{kind_of}/{kind}", Counter())[passed] += 1

    table = {}
    for kind, c in tallies.items():
        n = c[True] + c[False]
        table[kind] = {
            "n": n,
            "pass": c[True],
            "pass_rate": round(c[True] / max(1, n), 4),
        }

    return {"diagrams": len(pairs), "trivial_expected": dict(trivial), "mutants": table}


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description="Phase 12.3.3 - functional tests")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/llm/functional_mutation.json"),
    )
    args = parser.parse_args(argv)
    from src.llm import pairs as pairs_mod

    chosen = pairs_mod.by_split(pairs_mod.real_pairs(), args.split)
    if args.limit:
        chosen = chosen[: args.limit]

    report = mutation_study(chosen, [pairs_mod.diagram_for(p) for p in chosen])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
