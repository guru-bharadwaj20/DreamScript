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


# --- unused arguments cannot appear in a new file (audit 45) ----------------------------------


def test_arg_is_selected_so_a_new_unused_argument_fails():
    import tomllib

    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert "ARG" in config["tool"]["ruff"]["lint"]["select"]


def test_the_per_file_ignores_are_an_inventory_not_a_blanket():
    """31 files carry the 50 unused arguments that already existed, listed individually. A
    blanket `src/*` ignore would have the same effect today and no effect at all on the next
    file, which is the whole difference."""
    import tomllib

    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    ignores = config["tool"]["ruff"]["lint"]["per-file-ignores"]
    listed = [pattern for pattern, codes in ignores.items() if "ARG" in codes]
    assert "src/*" not in listed and "src/**" not in listed
    assert len([p for p in listed if p.startswith("src/")]) >= 25
    for pattern in listed:
        if pattern.startswith("src/"):
            assert (ROOT / pattern).is_file(), f"{pattern} is listed and does not exist"


@pytest.mark.parametrize(
    ("module", "function", "argument"),
    [
        ("src/llm/functional.py", "expected", "reference_code"),
        ("src/classify/fastpath.py", "run", "n_jobs"),
        ("src/classify/optimizers.py", "curves", "n_jobs"),
        ("src/ocr/ownership.py", "container_cost", "page_w"),
        ("src/parse/repair.py", "roles_and_confidence", "diagram"),
    ],
)
def test_the_misleading_arguments_are_gone(module: str, function: str, argument: str):
    """Not merely unused - misleading: a function called `expected(diagram, reference_code)`
    whose docstring calls the flowchart expectation "the reference program's behaviour" and
    never reads the code; an `n_jobs` that reaches no estimator; a `page_w` on a rule measured
    entirely in the container's own box."""
    tree = ast.parse((ROOT / module).read_text(encoding="utf-8"))
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == function)
    names = {a.arg for a in node.args.args} | {a.arg for a in node.args.kwonlyargs}
    assert argument not in names


def test_the_one_that_was_kept_is_reported_instead_of_dropped():
    """`svm.support_vector_count`'s `loss` is a LinearSVC setting with no SVC equivalent, so the
    function cannot honour it - it echoes it into the result rather than swallowing it."""
    source = (ROOT / "src" / "classify" / "svm.py").read_text(encoding="utf-8")
    assert '"proxy_for_loss": loss' in source


# --- one definition per measured rule (audit 46) ----------------------------------------------


def test_inset_box_has_one_definition():
    """9.3.1's node crop lived in `labelcrops` and `ownership`, byte for byte. The inset is a
    measured value, and two copies of a measured value is one place for it to be re-tuned and
    one place for it to stay behind."""
    from src.ocr import labelcrops, ownership

    assert ownership.inset_box is labelcrops.inset_box


def test_folds_has_one_definition():
    """Identical bodies in `s4` and `viterbi`, with *different defaults* - so the two could have
    been re-tuned apart while still producing "the same" split."""
    from src.parse import s4, viterbi

    assert s4.folds is viterbi.folds


@pytest.mark.parametrize(
    ("module", "name"),
    [("src/ocr/ownership.py", "inset_box"), ("src/parse/s4.py", "folds")],
)
def test_the_copy_is_gone_and_not_merely_shadowed(module: str, name: str):
    tree = ast.parse((ROOT / module).read_text(encoding="utf-8"))
    defined = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert name not in defined
