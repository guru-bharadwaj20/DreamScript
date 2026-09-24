"""The skip lists in `tests/conftest.py` describe the suite that exists.

Two counts were written into that file's comments and both went stale: "40 tests across 16
modules" against a `NEEDS_PAYLOAD` holding 15 names, and "the 20 failures that remain with the
payload present" against a run that had 3 and now has none. A number in a comment is a
measurement with no way to be re-taken.

So the counts live here, where they are assertions. Each one fails if the thing it counts moves,
which is the whole difference between a record and a claim.
"""

from __future__ import annotations

from pathlib import Path

TESTS = Path(__file__).parent


def _conftest():
    """`tests/conftest.py` is a pytest plugin, not a module on the import path - pytest loads it
    under a generated name. Loaded by location so this file can read its own suite's skip lists
    without depending on how that happened."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("_dreamscript_conftest", TESTS / "conftest.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ct = _conftest()


def test_every_named_module_exists():
    """A name that no longer matches a file silently stops skipping anything, and the module it
    was protecting starts failing for want of the corpus again."""
    for name in sorted(ct.NEEDS_PAYLOAD | ct.NEEDS_WINDOWS):
        assert (TESTS / f"{name}.py").is_file(), f"{name} is listed and does not exist"


def test_the_comment_block_states_the_module_count_it_has():
    text = (TESTS / "conftest.py").read_text(encoding="utf-8")
    assert f"across {len(ct.NEEDS_PAYLOAD)} modules" in text


def test_windows_only_names_are_split_between_modules_and_tests():
    """`NEEDS_WINDOWS` holds module stems and `NEEDS_WINDOWS_TESTS` holds test names; mixing the
    two skips nothing and reports nothing."""
    for name in ct.NEEDS_WINDOWS:
        assert not name.startswith("test_") or (TESTS / f"{name}.py").is_file()
    for name in ct.NEEDS_WINDOWS_TESTS:
        assert not (TESTS / f"{name}.py").is_file(), f"{name} is a module, not a test"
