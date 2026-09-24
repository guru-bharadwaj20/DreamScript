"""Top-level functions in `src/` that nothing calls.

An AST sweep over `src/`, `tests/` and `scripts/` found 17 top-level `def`s whose name appears
exactly once in the whole repo - 263 lines that cannot be reached from anywhere, including from
the `main()` of the module that defines them.

Eight were deleted: `_trace`, `_corner_positions` (private, with no caller), `seed_worker` (a
DataLoader `worker_init_fn` idiom that no DataLoader in this repo passes), and five undocumented
helpers - `code_files`, `iter_records`, `file_sha256`, `ink_mask`, `messages_sha`.

The rest are kept, and the point of this file is that keeping one is now a decision with a
reason attached rather than an oversight nobody can distinguish from a decision. Each entry says
why. The test fails when the set changes in either direction - a new uncalled function appears,
or one of these gains a caller and the note goes stale.
"""

from __future__ import annotations

import ast
import re
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: name -> why it has no caller and is kept anyway.
KEPT = {
    "packed_probe": "12.2.5's packing probe, run by hand against a model id; its output is a "
    "table in reports/llm_quant.md, not a value any stage consumes",
    "write_sourced_registry": "builds data/raw/chaos/scribes.csv once, from the sourced corpus; "
    "a corpus builder, invoked when the corpus is built",
    "resolve_page": "`direction`'s documented convenience wrapper around `resolve`, for a caller "
    "holding a page rather than an edge",
    "classify_page": "the same shape in `crossings`: the page-level pass over `classify`",
    "trace_page": "the same shape in `tracing`: `trace` with the boxes pulled from the cache",
    "headroom_gib": "free VRAM right now. A diagnostic for a human deciding a batch size, which "
    "is the one number the GPU rules are written around",
    "to_mask": "renders proposed text boxes onto a page-sized mask, for looking at them",
}


def _uncalled() -> dict[str, tuple[str, int]]:
    """`name -> (file, lines)` for every top-level def named exactly once in the repo."""
    texts = {
        path: path.read_text(encoding="utf-8", errors="ignore")
        for base in ("src", "tests", "scripts")
        for path in (ROOT / base).rglob("*.py")
        # This file names every one of them in order to account for them, which would make each
        # look called. Excluded for the same reason a linter does not lint its own test data.
        if path != Path(__file__).resolve()
    }
    counts = Counter(re.findall(r"\b\w+\b", "\n".join(texts.values())))
    found: dict[str, tuple[str, int]] = {}
    for path, text in sorted(texts.items()):
        if not path.is_relative_to(ROOT / "src"):
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and counts[node.name] == 1:
                lines = (node.end_lineno or node.lineno) - node.lineno + 1
                found[node.name] = (path.relative_to(ROOT).as_posix(), lines)
    return found


def test_every_uncalled_function_is_one_this_file_accounts_for():
    """A new one appearing is the finding; it is caught here rather than in an audit."""
    unaccounted = {name: where for name, where in _uncalled().items() if name not in KEPT}
    assert not unaccounted, (
        "top-level functions in src/ that nothing in the repo calls, and that are not listed in "
        f"KEPT with a reason: {unaccounted}"
    )


@pytest.mark.parametrize("name", sorted(KEPT))
def test_every_kept_entry_is_still_uncalled(name: str):
    """The other direction: a note that has stopped being true is worse than no note."""
    assert name in _uncalled(), f"{name} has a caller now - drop it from KEPT"


@pytest.mark.parametrize("name", sorted(KEPT))
def test_every_reason_is_a_reason(name: str):
    assert len(KEPT[name]) > 30, f"{name}'s entry does not say why it is kept"


@pytest.mark.parametrize(
    "name",
    ["_trace", "_corner_positions", "seed_worker", "code_files", "iter_records", "file_sha256"],
)
def test_the_deleted_ones_are_gone(name: str):
    text = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in sorted((ROOT / "src").rglob("*.py"))
    )
    assert f"def {name}(" not in text
