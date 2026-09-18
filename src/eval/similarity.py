"""Phase 12.3.5 - similarity of a generated program to its reference: CodeBLEU, exact match, edit distance.

    from src.eval import similarity
    similarity.score(candidate, reference, language="python")   # one pair
    similarity.score_many([(candidate, reference), ...])        # corpus + per-row

    python -m src.eval.similarity --study        # what the metric does to known perturbations

## Why this is the weakest of the 12.3 metrics, and is reported anyway

12.3.1-12.3.4 ask whether a program parses, runs, passes its tests and matches the drawn
structure. This one asks whether it *looks like* the reference, which is a different and much
weaker question: 12.1.6's emitter is one of many correct transcriptions of a diagram, so a model
that writes the same behaviour with different names scores badly here and perfectly there. The
plan asks for it, and the honest use is comparative - arm against arm on one corpus - never as a
claim about correctness. Where the two disagree, `functional pass@1` is the one that means
something.

## CodeBLEU, and which of its four parts this corpus can actually support

CodeBLEU (Ren et al., 2020) is the weighted mean of four components. Two are language-agnostic
and two need a parser for the target language:

    ngram_match           BLEU-4 over tokens, with the standard brevity penalty
    weighted_ngram_match  the same, but a token that is a language keyword counts `KEYWORD_WEIGHT`
                          times, so getting `while` wrong costs more than getting a name wrong
    syntax_match          multiset F1 over AST subtree signatures        (python only)
    dataflow_match        multiset F1 over variable def-use edges        (python only)

This corpus is five languages - Python (flowchart, state machine), React/JSX (wireframe), SQL
(ER) and SPICE (circuit) - and only Python has a parser in the standard library. Rather than
invent a half-parser for JSX or quietly score the missing components as 0 (which would make a
perfect SQL answer look like a 0.5), an undefined component is **dropped and the remaining
weights renormalised**. Every result therefore carries `components` - which parts were defined -
so a CodeBLEU over SQL is never silently compared against one over Python. A pair whose candidate
does not parse keeps its two token components and loses the two tree ones, which is the intended
reading: unparseable code is already condemned by 12.3.1.

## Measured behaviour (`--study`, the 162 test references against perturbations of themselves)

| perturbation                      | codebleu | exact | edit_sim |
| :--- | ---: | ---: | ---: |
| identity                          |  1.000   | 1.000 | 1.000    |
| reformat (blank lines, whitespace)|  1.000   | 1.000 | 1.000    |
| rename every local variable       |  0.578   | 0.000 | 0.828    |
| reverse the order of statements   |  0.797   | 0.000 | 0.375    |
| drop one statement                |  0.976   | 0.000 | 0.974    |

The last three rows are the point. **Dropping a statement - a program that now does less than it
was asked to - costs 0.024 of CodeBLEU**, while renaming variables, which changes no behaviour at
all, costs 0.422. The metric is ordered almost exactly against correctness: the harmless edit is
punished seventeen times harder than the harmful one. Reordering every statement, which in these
targets means the control flow is destroyed, still scores 0.797, because a bag of n-grams and an
order-blind AST multiset cannot see order; only edit similarity (0.375) notices, and it is the
component with no claim to semantics.

So a high CodeBLEU is not evidence that a program runs, and these three rows are why
12.3.2/12.3.3 are where that question is settled. Exact match is 0.000 for every perturbation
including the semantically identical rename, which is what makes it useless alone and worth
reporting only beside the other two.

Normalisation before exact match is deliberate and narrow: trailing whitespace per line, blank
lines collapsed, and a single trailing newline. Comments are **not** stripped - 12.1.6's targets
carry a generated docstring, and a model that omits it has produced a different program than the
one it was trained on.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any

#: Weight multiplier for a keyword token in `weighted_ngram_match`.
KEYWORD_WEIGHT = 4.0

#: The four CodeBLEU components, in the paper's order, at equal weight.
WEIGHTS = {
    "ngram_match": 0.25,
    "weighted_ngram_match": 0.25,
    "syntax_match": 0.25,
    "dataflow_match": 0.25,
}

MAX_N = 4

PYTHON_KEYWORDS = frozenset(
    """False None True and as assert async await break class continue def del elif else except
    finally for from global if import in is lambda nonlocal not or pass raise return try while
    with yield self""".split()
)
SQL_KEYWORDS = frozenset(
    """create table primary key foreign references integer text real not null unique default
    constraint on delete cascade insert into values alter add index view select from where join""".upper().split()
)
JSX_KEYWORDS = frozenset(
    """import export default function const let var return div span button input img className
    onClick useState useEffect props children React""".split()
)
SPICE_KEYWORDS = frozenset(""".subckt .ends .model .tran .dc .ac .end V I R L C D Q M X""".split())

KEYWORDS = {
    "python": PYTHON_KEYWORDS,
    "sql": SQL_KEYWORDS,
    "react": JSX_KEYWORDS,
    "jsx": JSX_KEYWORDS,
    "javascript": JSX_KEYWORDS,
    "html": JSX_KEYWORDS,
    "spice": SPICE_KEYWORDS,
}

#: Languages whose AST this module can actually read.
PARSEABLE = frozenset({"python"})

_TOKEN = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|\d+\.\d+|\d+|[^\sA-Za-z_0-9]")


# -- normalisation and tokens ---------------------------------------------------------------


def normalise(code: str) -> str:
    """Trailing whitespace per line, no blank lines, one trailing newline. Comments are kept."""
    lines = [line.rstrip() for line in (code or "").replace("\r\n", "\n").split("\n")]
    return "\n".join(line for line in lines if line.strip()) + "\n"


def tokenize(code: str) -> list[str]:
    """Identifier / number / punctuation tokens. Language-agnostic on purpose - the languages
    here differ in what an identifier *means*, not in what one looks like."""
    return _TOKEN.findall(code or "")


def _ngrams(tokens: Sequence[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


# -- the two token components ---------------------------------------------------------------


def _modified_precision(cand: Sequence[str], ref: Sequence[str], n: int, weights: dict | None):
    """Clipped n-gram precision; `weights` maps a token to its multiplier, or None for plain."""
    c_grams, r_grams = _ngrams(cand, n), _ngrams(ref, n)
    if not c_grams:
        return 0.0, 0.0

    def w(gram: tuple[str, ...]) -> float:
        if weights is None:
            return 1.0
        return max(weights.get(t, 1.0) for t in gram)

    overlap = sum(w(g) * min(c, r_grams[g]) for g, c in c_grams.items() if g in r_grams)
    total = sum(w(g) * c for g, c in c_grams.items())
    return overlap, total


def _bleu(cand: Sequence[str], ref: Sequence[str], weights: dict | None = None) -> float:
    """BLEU-MAX_N with the standard brevity penalty; 0 if the candidate is too short to score."""
    if not cand or not ref:
        return float(cand == ref)
    logs = []
    for n in range(1, MAX_N + 1):
        if len(cand) < n or len(ref) < n:
            continue
        overlap, total = _modified_precision(cand, ref, n, weights)
        if total == 0:
            continue
        # A zero-overlap order would send the geometric mean to 0 and throw away the orders that
        # did match, so it is smoothed rather than dropped (Chen & Cherry smoothing 1).
        logs.append(math.log((overlap or 1e-9) / total))
    if not logs:
        return 0.0
    bp = 1.0 if len(cand) > len(ref) else math.exp(1 - len(ref) / max(len(cand), 1))
    return bp * math.exp(sum(logs) / len(logs))


# -- the two tree components ----------------------------------------------------------------


def _subtrees(tree: ast.AST) -> Counter:
    """Multiset of `parent(child, child, ...)` signatures - the paper's subtree match, one level
    deep, which is what makes it survive a rename and not survive a restructure."""
    out: Counter = Counter()
    for node in ast.walk(tree):
        kids = [type(k).__name__ for k in ast.iter_child_nodes(node)]
        out[f"{type(node).__name__}({','.join(kids)})"] += 1
    return out


def _dataflow(tree: ast.AST) -> Counter:
    """Multiset of def-use edges: (variable, the node type that produced its value).

    Names are deliberately kept. CodeBLEU's dataflow match normalises variable names away, but
    every target here is generated from a diagram whose node text *is* the variable name, so a
    renamed variable is a real difference: it means the model read the box differently.
    """
    edges: Counter = Counter()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                for name in _names(target):
                    edges[(name, type(node.value).__name__)] += 1
        elif isinstance(node, ast.AugAssign | ast.AnnAssign) and node.value is not None:
            for name in _names(node.target):
                edges[(name, type(node.value).__name__)] += 1
        elif isinstance(node, ast.For):
            for name in _names(node.target):
                edges[(name, "iter:" + type(node.iter).__name__)] += 1
        elif isinstance(node, ast.arg):
            edges[(node.arg, "arg")] += 1
    return edges


def _names(target: ast.AST) -> list[str]:
    return [n.id for n in ast.walk(target) if isinstance(n, ast.Name)]


def _multiset_f1(cand: Counter, ref: Counter) -> float | None:
    if not cand and not ref:
        return 1.0
    if not cand or not ref:
        return 0.0
    overlap = sum((cand & ref).values())
    precision = overlap / sum(cand.values())
    recall = overlap / sum(ref.values())
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


# -- edit distance --------------------------------------------------------------------------


def edit_distance(a: Sequence[Any], b: Sequence[Any]) -> int:
    """Levenshtein, two rows of memory."""
    if a == b:
        return 0
    if not a or not b:
        return len(a) or len(b)
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(
                previous[j - 1] if x == y else 1 + min(previous[j - 1], previous[j], current[j - 1])
            )
        previous = current
    return previous[-1]


# -- the metric -----------------------------------------------------------------------------


def score(candidate: str, reference: str, language: str = "python") -> dict[str, Any]:
    """CodeBLEU, exact match and edit similarity for one (candidate, reference) pair.

    `components` names the CodeBLEU parts that were defined; the undefined ones are dropped and
    the remaining weights renormalised, never scored as zero.
    """
    lang = (language or "python").lower()
    cand_n, ref_n = normalise(candidate), normalise(reference)
    cand_t, ref_t = tokenize(cand_n), tokenize(ref_n)
    keywords = KEYWORDS.get(lang, PYTHON_KEYWORDS)
    weights = {k: KEYWORD_WEIGHT for k in keywords}

    parts: dict[str, float] = {
        "ngram_match": _bleu(cand_t, ref_t),
        "weighted_ngram_match": _bleu(cand_t, ref_t, weights),
    }
    syntax_ok = True
    if lang in PARSEABLE:
        try:
            cand_ast, ref_ast = ast.parse(cand_n), ast.parse(ref_n)
        except SyntaxError:
            syntax_ok = False
        else:
            parts["syntax_match"] = _multiset_f1(_subtrees(cand_ast), _subtrees(ref_ast))
            parts["dataflow_match"] = _multiset_f1(_dataflow(cand_ast), _dataflow(ref_ast))

    total_w = sum(WEIGHTS[k] for k in parts)
    codebleu = sum(WEIGHTS[k] * v for k, v in parts.items()) / total_w if total_w else 0.0

    distance = edit_distance(cand_t, ref_t)
    longest = max(len(cand_t), len(ref_t))
    return {
        "codebleu": round(codebleu, 6),
        **{k: round(v, 6) for k, v in parts.items()},
        "components": sorted(parts),
        "language": lang,
        "parsed": syntax_ok if lang in PARSEABLE else None,
        "exact_match": cand_n == ref_n,
        "edit_distance": distance,
        "edit_similarity": round(1.0 - distance / longest, 6) if longest else 1.0,
        "candidate_tokens": len(cand_t),
        "reference_tokens": len(ref_t),
    }


def score_many(
    pairs: Iterable[tuple[str, str] | tuple[str, str, str]],
) -> dict[str, Any]:
    """Score `(candidate, reference[, language])` pairs; corpus means plus every row.

    Means are macro (per row), not corpus-level BLEU: the rows are different languages and
    different lengths, and a corpus BLEU would let the longest programs speak for the rest.
    """
    rows = [
        score(p[0], p[1], p[2] if len(p) > 2 else "python") for p in pairs  # type: ignore[misc]
    ]
    if not rows:
        return {"n": 0, "per_row": []}

    def mean(key: str, where=lambda r: True) -> float | None:
        vals = [r[key] for r in rows if where(r) and r.get(key) is not None]
        return round(sum(vals) / len(vals), 6) if vals else None

    by_language: dict[str, Any] = {}
    for lang in sorted({r["language"] for r in rows}):
        sel = [r for r in rows if r["language"] == lang]
        by_language[lang] = {
            "n": len(sel),
            "codebleu": round(sum(r["codebleu"] for r in sel) / len(sel), 6),
            "exact_match": round(sum(r["exact_match"] for r in sel) / len(sel), 6),
            "edit_similarity": round(sum(r["edit_similarity"] for r in sel) / len(sel), 6),
        }
    return {
        "n": len(rows),
        "codebleu": mean("codebleu"),
        "ngram_match": mean("ngram_match"),
        "weighted_ngram_match": mean("weighted_ngram_match"),
        "syntax_match": mean("syntax_match"),
        "dataflow_match": mean("dataflow_match"),
        "exact_match": mean("exact_match"),
        "edit_similarity": mean("edit_similarity"),
        "by_language": by_language,
        "per_row": rows,
    }


# -- the study ------------------------------------------------------------------------------


def _rename_locals(code: str) -> str | None:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    names = {
        n.id
        for n in ast.walk(tree)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store | ast.Load)
    }
    out = code
    for i, name in enumerate(sorted(names)):
        if name in PYTHON_KEYWORDS:
            continue
        out = re.sub(rf"\b{re.escape(name)}\b", f"v{i}", out)
    return out


def _reverse_statements(code: str) -> str | None:
    lines = [ln for ln in code.split("\n") if ln.strip()]
    return "\n".join(reversed(lines)) if lines else None


def _drop_one(code: str) -> str | None:
    lines = [ln for ln in code.split("\n") if ln.strip()]
    if len(lines) < 3:
        return None
    del lines[len(lines) // 2]
    return "\n".join(lines)


def _reformat(code: str) -> str:
    return "\n\n".join(code.split("\n")) + "\n   \n"


PERTURBATIONS = {
    "identity": lambda c: c,
    "reformat": _reformat,
    "rename_locals": _rename_locals,
    "reverse_statements": _reverse_statements,
    "drop_one_statement": _drop_one,
}


def study(split: str = "test", limit: int | None = None) -> dict[str, Any]:
    """Score every reference target against perturbations of itself, per 12.1.8's split."""
    from src.codegen import pairs as pair_mod

    records = list(pair_mod.load_pairs(split))
    if limit:
        records = records[:limit]
    out: dict[str, Any] = {"split": split, "n": len(records), "perturbations": {}}
    for name, fn in PERTURBATIONS.items():
        scored = []
        for record in records:
            target = record["target_code"]
            mutated = fn(target)
            if mutated is None:
                continue
            scored.append(score(mutated, target, record.get("language", "python")))
        if not scored:
            continue
        out["perturbations"][name] = {
            "n": len(scored),
            "codebleu": round(sum(r["codebleu"] for r in scored) / len(scored), 4),
            "exact_match": round(sum(r["exact_match"] for r in scored) / len(scored), 4),
            "edit_similarity": round(sum(r["edit_similarity"] for r in scored) / len(scored), 4),
        }
    return out


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description="Phase 12.3.5 similarity metrics")
    parser.add_argument("--study", action="store_true")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)
    if args.study:
        print(json.dumps(study(args.split, args.limit), indent=2))
    else:
        parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
