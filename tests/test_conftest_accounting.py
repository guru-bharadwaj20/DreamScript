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

import pytest

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


def test_gpu_skipping_has_one_mechanism():
    """conftest's `pytest_collection_modifyitems` hook governs `@pytest.mark.gpu` and nothing
    else. A standalone `skipif(not torch.cuda.is_available())` skips the same test on the same
    machines - which is why this went unnoticed - but is invisible to the hook, so a change to
    how this project decides "has a GPU" reaches every marked test and misses that one."""
    offenders = []
    for path in sorted(TESTS.glob("test_*.py")):
        if path.name == Path(__file__).name:
            continue  # this file names the pattern in order to forbid it
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "skipif" in line and "cuda.is_available" in line:
                offenders.append(f"{path.name}:{number}")
    assert not offenders, offenders


def test_the_gpu_marker_is_registered():
    """An unregistered marker is a typo away from silently marking nothing."""
    text = (TESTS / "conftest.py").read_text(encoding="utf-8")
    assert 'addinivalue_line("markers", "gpu:' in text


@pytest.mark.parametrize(
    "name", ["flowchart", "wireframe", "state_machine", "er_diagram", "circuit"]
)
def test_every_named_fixture_image_asserts_it_was_read(name, request):
    """`cv2.imread` returns None rather than raising, and these five returned it unchecked."""
    image = request.getfixturevalue(f"{name}_image")
    assert image is not None
    assert image.ndim == 2


def test_the_single_factory_carries_the_assert_sample_image_has():
    source = (TESTS / "conftest.py").read_text(encoding="utf-8")
    block = source.split("def _single(", 1)[1].split("\nflowchart_image", 1)[0]
    assert "assert img is not None" in block
    assert "make_fixtures.py" in block, "the message should say how to fix it"


# --- nothing claims a remote that does not exist (audit 61) -----------------------------------


def test_the_data_remote_is_documented_as_machine_local():
    """`.dvc/config` defines four remotes and all four are directories on one Windows machine.
    A fresh clone - CI above all - gets the pointers in git and cannot fetch a byte behind
    them."""
    root = TESTS.parent
    doc = (root / "docs" / "data_remote.md").read_text(encoding="utf-8")
    assert "There is no shared remote" in doc
    config = (root / ".dvc" / "config").read_text(encoding="utf-8")
    for line in config.splitlines():
        if "url" in line:
            assert "dreamscript-dvc-store" in line, f"a remote moved: {line.strip()}"


def test_nothing_tells_a_reader_to_run_dvc_pull_without_saying_where():
    """The skip reason said "run `dvc pull`", which cannot work anywhere but one machine."""
    root = TESTS.parent
    for path in (TESTS / "conftest.py", root / ".github" / "workflows" / "ci.yml"):
        text = path.read_text(encoding="utf-8")
        if "dvc pull" in text or "content in a" in text:
            assert "data_remote.md" in text or "one Windows machine" in text, path.name
