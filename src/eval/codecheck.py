"""Phase 12.3.1 / 12.3.2 helpers - syntactic validity and executability of generated programs.

    from src.eval.codecheck import syntactic_many, executable_many
    syntactic_many([(code, language), ...])   -> [{ok, kind, detail}]   12.3.1: does it parse?
    executable_many([(code, record), ...])    -> [{ok, kind, detail}]   12.3.2: does it run?

These score *model outputs*; the rows they serve are owned and flipped by the fine-tuning work.
They are validated here on the 12.1 reference targets and on deliberately broken programs.

## Syntactic validity (12.3.1) - parse, never execute

    python   `ast.parse`, then `compile()` (which rejects `return` outside a function)
    sql      every statement passes `sqlite3.complete_statement` and `EXPLAIN` prepares it
             against the schema built so far - SQLite's parser, without trusting execution
    react    esbuild's JSX transform (build only, no render)
    spice    ngspice parses the deck: a `spice.parse_error` is invalid, while
             `spice.no_convergence` is valid syntax that does not solve (that is 12.3.2)

## Executability (12.3.2) - run it

    python   `src.rl.sandbox.run` (11.2.9): a fresh interpreter under a timeout and memory cap.
             A generated flowchart calls functions the diagram only names (`validate_order(ctx)`),
             so the harness defines each free name the program calls as a stub before running it:
             a name called inside an `if`/`while` test returns True twice and then False, any other
             returns its first argument - so a correct `while` terminates and a `while True:` with
             no exit times out. Builtins are never stubbed: a target that calls `open(ctx)` because
             a node was labelled "open" fails, as it would for real. The entry point is invoked
             with `ctx = defaultdict(int)` (an io step reads `ctx["file"]` nothing wrote); for a state
             machine the class is instantiated and `accepts([])` plus one `step` per row of `TABLE`
             from its own source state are run.
    sql      `src.eval.sql.check` - executes, resolves FKs, rejects an empty schema
    react    `src.eval.react` build + `renderToString`
    spice    ngspice `.op` must solve
"""

from __future__ import annotations

import ast
import builtins
import json
import sqlite3
from typing import Any

_BUILTINS = frozenset(dir(builtins))

HARNESS = """
import collections as _collections
_budget = {}
def _cond(name):
    def cond(*args, **kwargs):
        _budget[name] = _budget.get(name, 0) + 1
        return _budget[name] <= 2
    return cond
def _stmt(name):
    def stmt(*args, **kwargs):
        return args[0] if args else 0
    return stmt
"""


def _test_calls(tree: ast.AST) -> set[str]:
    """Names called inside an `if`/`while` test - conditions; every other free call is a step."""
    return {
        sub.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.If | ast.While)
        for sub in ast.walk(node.test)
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)
    }


def _free_calls(tree: ast.AST) -> set[str]:
    defined = {
        n.name
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
    }
    defined |= {
        t.id
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        for t in n.targets
        if isinstance(t, ast.Name)
    }
    return {
        n.func.id
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id not in _BUILTINS
        and n.func.id not in defined
    }


# -- 12.3.1 -------------------------------------------------------------------------------------


def _python_syntax(code: str) -> dict:
    try:
        compile(ast.parse(code), "<generated>", "exec")
    except (SyntaxError, ValueError) as exc:
        return {"ok": False, "kind": "syntax.python", "detail": str(exc)[:300]}
    return {"ok": True, "kind": "syntax.ok", "detail": "parsed and compiled"}


def _sql_syntax(code: str) -> dict:
    from src.eval.sql import split_statements

    statements = split_statements(code)
    if not statements:
        return {"ok": False, "kind": "syntax.sql", "detail": "no statements"}
    con = sqlite3.connect(":memory:")
    try:
        for statement in statements:
            if not sqlite3.complete_statement(statement):
                return {
                    "ok": False,
                    "kind": "syntax.sql",
                    "detail": f"incomplete: {statement[:80]}",
                }
            try:
                con.execute(f"EXPLAIN {statement}")
                con.execute(statement)  # later statements are prepared against earlier tables
            except sqlite3.Error as exc:
                if "syntax error" in str(exc) or "incomplete input" in str(exc):
                    return {"ok": False, "kind": "syntax.sql", "detail": str(exc)[:300]}
    finally:
        con.close()
    return {"ok": True, "kind": "syntax.ok", "detail": f"{len(statements)} statement(s) parsed"}


def syntactic_many(items: list[tuple[str, str]]) -> list[dict]:
    """12.3.1 verdict per `(code, language)`. `None` code (no extractable block) is invalid."""
    from src.eval import react, spice

    out: list[dict] = [{} for _ in items]
    react_todo, spice_todo = [], []
    for i, (code, language) in enumerate(items):
        if not isinstance(code, str) or not code.strip():
            out[i] = {"ok": False, "kind": "syntax.empty", "detail": "no code"}
        elif language == "python":
            out[i] = _python_syntax(code)
        elif language == "sql":
            out[i] = _sql_syntax(code)
        elif language == "react":
            react_todo.append(i)
        elif language == "spice":
            spice_todo.append(i)
        else:
            out[i] = {"ok": False, "kind": "syntax.unsupported", "detail": str(language)}
    for i, verdict in zip(
        react_todo, react.check_many([items[i][0] for i in react_todo], render=False), strict=True
    ):
        out[i] = {
            "ok": verdict["ok"],
            "kind": "syntax.ok" if verdict["ok"] else verdict["kind"],
            "detail": verdict["detail"],
        }
    for i, verdict in zip(
        spice_todo, spice.check_many([items[i][0] for i in spice_todo]), strict=True
    ):
        parsed = verdict["kind"] in ("spice.ok", "spice.no_convergence")
        out[i] = {
            "ok": parsed,
            "kind": "syntax.ok" if parsed else verdict["kind"],
            "detail": verdict["detail"],
        }
    return out


# -- 12.3.2 -------------------------------------------------------------------------------------


def python_program(code: str, diagram_type: str) -> str | None:
    """The sandboxed script for one Python target: stubs + the program + an entry-point call."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    tests = _test_calls(tree)
    stubs = "".join(
        f"{name} = {'_cond' if name in tests else '_stmt'}({name!r})\n"
        for name in sorted(_free_calls(tree))
    )
    if diagram_type == "state_machine":
        classes = [n.name for n in tree.body if isinstance(n, ast.ClassDef)]
        if not classes:
            return None
        entry = (
            f"_m = {classes[0]}()\n"
            "_m.accepts([])\n"
            "for _src, _row in getattr(_m, 'TABLE', {}).items():\n"
            "    for _sym in _row:\n"
            "        _m.state = _src\n"
            "        _m.step(_sym)\n"
        )
    else:
        functions = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
        if not functions:
            return None
        entry = f"{functions[0]}(_collections.defaultdict(int))\n"
    return f"{HARNESS}\n{stubs}\n{code}\n\n{entry}print('__ran__')\n"


def executable_many(items: list[tuple[str, dict]], timeout_s: float = 5.0) -> list[dict]:
    """12.3.2 verdict per `(code, record)`; `record` supplies `language` and `diagram_type`."""
    from concurrent.futures import ThreadPoolExecutor

    from src.eval import react, spice, sql
    from src.rl import sandbox

    out: list[dict] = [{} for _ in items]
    python_jobs: list[tuple[int, str]] = []
    react_todo, spice_todo = [], []
    for i, (code, record) in enumerate(items):
        language = record["language"]
        if not isinstance(code, str) or not code.strip():
            out[i] = {"ok": False, "kind": "exec.empty", "detail": "no code"}
        elif language == "python":
            program = python_program(code, record["diagram_type"])
            if program is None:
                out[i] = {
                    "ok": False,
                    "kind": "exec.no_entry",
                    "detail": "no parseable entry point",
                }
            else:
                python_jobs.append((i, program))
        elif language == "sql":
            verdict = sql.check(code)
            out[i] = {
                "ok": bool(verdict["ok"]),
                "kind": verdict["kind"],
                "detail": verdict["detail"],
            }
        elif language == "react":
            react_todo.append(i)
        elif language == "spice":
            spice_todo.append(i)
        else:
            out[i] = {"ok": False, "kind": "exec.unsupported", "detail": str(language)}

    def run(job: tuple[int, str]) -> tuple[int, dict]:
        index, program = job
        result = sandbox.run(program, timeout_s=timeout_s, memory_limit_mb=256)
        ran = result["ok"] and "__ran__" in (result.get("stdout") or "")
        kind = "exec.ok" if ran else f"exec.{result['kind']}"
        return index, {"ok": ran, "kind": kind, "detail": (result.get("detail") or "")[:300]}

    with ThreadPoolExecutor(8) as pool:
        for index, verdict in pool.map(run, python_jobs):
            out[index] = verdict
    for i, verdict in zip(
        react_todo, react.check_many([items[i][0] for i in react_todo]), strict=True
    ):
        out[i] = verdict
    for i, verdict in zip(
        spice_todo, spice.check_many([items[i][0] for i in spice_todo]), strict=True
    ):
        out[i] = verdict
    return out


def summarise(verdicts: list[dict]) -> dict[str, Any]:
    from collections import Counter

    return {
        "n": len(verdicts),
        "ok": sum(v["ok"] for v in verdicts),
        "rate": round(sum(v["ok"] for v in verdicts) / max(1, len(verdicts)), 4),
        "kinds": dict(Counter(v["kind"] for v in verdicts)),
    }


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps({"doc": __doc__.splitlines()[0]}))
