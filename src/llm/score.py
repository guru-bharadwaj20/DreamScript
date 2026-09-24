"""Phase 12.3.1 / 12.3.2 / 12.3.3 / 12.3.8 - scoring generated programs, one denominator for all.

    python -m src.llm.score experiments/llm/benchmark/qwen7b_validation.jsonl --split validation

Every metric is over **all** generations for the split, never over the subset that happened to
parse, so a model cannot raise its executability by failing to produce code more often:

    contract      the reply is exactly one fenced block and nothing else (12.2.4's `extract_code`,
                  strict). Reported, but not a gate for the rest - see `code_of`.
    syntax        12.3.1: the extracted program passes `ast.parse` and `compile()`.
    executes      12.3.2: the program runs in 11.2.9's sandbox as `__main__` (5 s, 256 MB) with
                  `kind == ok`, **and** its entry point runs under the functional driver without
                  raising. Module-level execution alone is not enough: a flowchart program is a
                  function definition, so "it imported" would pass `def run(ctx): undefined()`.
    functional    12.3.3: pass@1 against the per-diagram tests of `src.llm.functional`.
    hallucination 12.3.8: drawn nodes the program never names (dropped) and operations it calls
                  that name no drawn node (invented), from `hallucination`.

`code_of` extracts leniently (the first fenced block, else the whole reply) so that 12.3.1-12.3.3
measure the program and the contract is measured separately; scoring only strictly-compliant
replies would conflate "wrote prose around correct code" with "wrote wrong code".
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

_FENCE = re.compile(r"```[ \t]*([\w+-]*)[ \t]*\n(.*?)(?:```|\Z)", re.S)

#: Called names that are scaffolding, not operations: calling them invents nothing.
HELPERS = frozenset(
    {
        "condition",
        "step",
        "main",
        "print",
        "input",
        "len",
        "range",
        "str",
        "int",
        "float",
        "bool",
        "list",
        "dict",
        "set",
        "tuple",
        "isinstance",
        "super",
        "enumerate",
        "zip",
        "sorted",
        "min",
        "max",
        "sum",
        "any",
        "all",
        "getattr",
        "setattr",
        "hasattr",
        "type",
        "ValueError",
        "KeyError",
        "TypeError",
        "RuntimeError",
        "Exception",
        "NotImplementedError",
        "append",
        "get",
        "setdefault",
        "update",
        "pop",
        "add",
        "join",
        "format",
        "lower",
        "upper",
        "strip",
        "split",
        "items",
        "keys",
        "values",
        "copy",
        "open",
        "repr",
        "iter",
        "next",
        "object",
        "property",
        "staticmethod",
        "classmethod",
        "exit",
        "quit",
        "abs",
        "round",
        "accepts",
        "accept",
        "transition",
        "reset",
        "run",
        "process",
        "is_accepted",
    }
)


def code_of(reply: str) -> str:
    """The program in a reply: the first fenced block, else the reply itself."""
    match = _FENCE.search(reply)
    return match.group(2) if match else reply


def strict_contract(reply: str) -> bool:
    from src.codegen import prompt as template

    return template.extract_code(reply) is not None


def syntax_ok(code: str) -> tuple[bool, str]:
    try:
        compile(ast.parse(code), "<generated>", "exec")
        return True, "ok"
    except SyntaxError as exc:
        return False, f"SyntaxError: {exc.msg} (line {exc.lineno})"
    except (ValueError, RecursionError) as exc:
        return False, f"{type(exc).__name__}: {exc}"


def module_runs(code: str) -> dict[str, Any]:
    from src.rl import sandbox

    result = sandbox.run(code, timeout_s=5.0, memory_limit_mb=256)
    return {"kind": result["kind"], "detail": (result.get("detail") or "")[:300]}


def called_names(code: str) -> set[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name):
                out.add(fn.id)
        elif isinstance(node, ast.FunctionDef):
            out.add(node.name)
    return out


def predicate_names(code: str) -> set[str]:
    """Names called inside an `if`/`while`/ternary condition: questions, not operations."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If | ast.While | ast.IfExp):
            for sub in ast.walk(node.test):
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
                    out.add(sub.func.id)
    return out


def string_literals(code: str) -> set[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    return {
        n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)
    }


def hallucination(code: str, diagram: dict) -> dict[str, Any]:
    """Dropped and invented nodes (12.3.8).

    flowchart: expected = operation nodes (role `process`); a node is *named* when some called or
    defined function name resolves to it under the functional driver's resolver. Invented =
    called/defined names that resolve to no drawn node and are not scaffolding (`HELPERS`).
    state_machine: expected = states by their drawn text (or id when the text is empty); a state
    is named when that text is a string literal or identifier in the program. Invented = string
    literals shaped like state names (`q\\d+`/`s\\d+`) that are not drawn states.
    """
    from src.llm import functional

    kind = diagram.get("diagram_type")
    nodes = diagram.get("nodes", [])
    if kind == "state_machine":
        labels = {str(n.get("text") or n["id"]).strip() for n in nodes}
        present = string_literals(code) | set(re.findall(r"\b\w+\b", code))
        dropped = sorted(lbl for lbl in labels if lbl and lbl not in present)
        shaped = {s for s in string_literals(code) if re.fullmatch(r"[qQsS]_?\d+", s)}
        invented = sorted(shaped - labels)
        return {"expected": len(labels), "dropped": dropped, "invented": invented}
    graph = functional.flow_graph(diagram)
    ops = graph["ops"]
    resolver = _resolver(diagram)
    names = called_names(code)
    named = {resolver(n) for n in names} - {None}
    names -= {n for n in predicate_names(code) if resolver(n) is None}
    dropped = sorted(ops - named)
    invented = sorted(
        n
        for n in names
        if resolver(n) is None
        and n not in HELPERS
        and not n.startswith("_")
        and not n.lower().startswith("run")
    )
    return {"expected": len(ops), "dropped": dropped, "invented": invented}


def _resolver(diagram: dict):
    """The functional driver's name -> node resolver, run in-process."""
    import difflib

    from src.llm.functional import FUZZ

    def slug(text: object) -> str:
        return re.sub(r"_+", "_", re.sub(r"[^0-9a-z]+", "_", str(text).lower())).strip("_")

    by_slug: dict[str, str] = {}
    texts: list[tuple[str, str]] = []
    for node in diagram.get("nodes", []):
        by_slug.setdefault(slug(node["id"]), node["id"])
        if slug(node.get("text") or ""):
            by_slug.setdefault(slug(node.get("text")), node["id"])
            texts.append((slug(node.get("text")), node["id"]))

    def resolve(name: str) -> str | None:
        key = slug(name)
        if key in by_slug:
            return by_slug[key]
        if len(key) < 4:
            return None
        best, score = None, 0.0
        for s, nid in texts:
            r = difflib.SequenceMatcher(None, key, s).ratio()
            if r > score:
                best, score = nid, r
        return best if score >= FUZZ else None

    return resolve


def score_one(row: dict, pair: dict, diagram: dict, expected: dict | None = None) -> dict[str, Any]:
    from src.llm import functional

    reply = row["text"]
    code = code_of(reply)
    out: dict[str, Any] = {
        "diagram_id": pair["diagram_id"],
        "source": pair["source"],
        "contract": strict_contract(reply),
        "hit_limit": bool(row.get("hit_limit")),
    }
    ok, detail = syntax_ok(code)
    out["syntax"] = ok
    out["syntax_detail"] = detail
    if not ok:
        out.update(
            executes=False, functional=False, functional_reason="syntax", exec_kind="syntax_error"
        )
        out["hallucination"] = hallucination("", diagram)
        return out
    module = module_runs(code)
    sig = functional.signature(code, diagram)
    expected = expected or functional.expected(diagram)
    passed, reason = functional.compare(sig, expected)
    entry_ok = bool(sig.get("ok")) and not any(
        str(status).startswith("error") for _, status in sig.get("paths", [])
    )
    if sig.get("kind") == "state_machine":
        entry_ok = bool(sig.get("ok")) and sig.get("verdicts") is not None
    out["exec_kind"] = module["kind"]
    out["exec_detail"] = module["detail"]
    out["entry_ok"] = entry_ok
    out["executes"] = module["kind"] == "ok" and entry_ok
    out["functional"] = passed
    out["functional_reason"] = reason
    out["hallucination"] = hallucination(code, diagram)
    return out


def score_rows(rows: list[dict], split: str, workers: int = 16) -> list[dict]:
    from src.llm import pairs as pairs_mod

    index = {p["diagram_id"]: p for p in pairs_mod.load(split)}

    def job(row: dict) -> dict:
        pair = index[row["diagram_id"]]
        return score_one(row, pair, pairs_mod.diagram_for(pair))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(job, rows))


def summarise(scored: list[dict]) -> dict[str, Any]:
    n = len(scored)

    def rate(key: str, subset: list[dict] | None = None) -> float:
        items = scored if subset is None else subset
        return round(sum(bool(s[key]) for s in items) / max(1, len(items)), 4)

    expected = sum(s["hallucination"]["expected"] for s in scored)
    dropped = sum(len(s["hallucination"]["dropped"]) for s in scored)
    invented = sum(len(s["hallucination"]["invented"]) for s in scored)
    out: dict[str, Any] = {
        "n": n,
        "contract": rate("contract"),
        "syntax": rate("syntax"),
        "executes": rate("executes"),
        "functional": rate("functional"),
        "hit_limit": sum(bool(s["hit_limit"]) for s in scored),
        "dropped_node_rate": round(dropped / max(1, expected), 4),
        "invented_per_program": round(invented / max(1, n), 3),
        "programs_with_invented": rate_programs(scored, "invented"),
        "programs_with_dropped": rate_programs(scored, "dropped"),
        "functional_reasons": dict(Counter(s["functional_reason"] for s in scored)),
        "exec_kinds": dict(Counter(s["exec_kind"] for s in scored)),
    }
    for source in sorted({s["source"] for s in scored}):
        sub = [s for s in scored if s["source"] == source]
        out[f"by_source/{source}"] = {
            "n": len(sub),
            "syntax": rate("syntax", sub),
            "executes": rate("executes", sub),
            "functional": rate("functional", sub),
        }
    return out


def rate_programs(scored: list[dict], key: str) -> float:
    return round(sum(bool(s["hallucination"][key]) for s in scored) / max(1, len(scored)), 4)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description="Phase 12.3 - score generations")
    parser.add_argument("generations", type=Path)
    parser.add_argument("--split", default="validation")
    args = parser.parse_args(argv)
    from src.llm.generate import read_jsonl, write_jsonl

    scored = score_rows(read_jsonl(args.generations), args.split)
    write_jsonl(scored, args.generations.with_suffix(".scored.jsonl"))
    summary = summarise(scored)
    args.generations.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
