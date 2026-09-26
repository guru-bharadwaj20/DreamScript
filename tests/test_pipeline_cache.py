"""Phase 13.6 - what belongs in the stage cache key.

`cache.py`'s docstring promises "a stale entry from a different configuration is a miss rather
than a wrong answer". `code_key` made that true of the *code*, after it had been false. It was
still false of the *environment*: `assemble.s5` takes three switches from `os.environ` rather
than from arguments - deliberately, because `run` fans pages out to joblib workers and a module
constant set in the parent is re-imported at its default in every child - and three more
variables name the checkpoints that produce the text and the edges. None of the six was in the
key, so flipping one and re-running served the previous configuration's answer from a warm cache.
"""

from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"


def test_env_key_changes_when_a_stage_deciding_variable_changes(monkeypatch):
    from src.pipeline.cache import ENV_KEYS, env_key

    monkeypatch.delenv("DREAMSCRIPT_ARROW_EDGES", raising=False)
    before = env_key()
    monkeypatch.setenv("DREAMSCRIPT_ARROW_EDGES", "0")
    assert env_key() != before
    monkeypatch.delenv("DREAMSCRIPT_ARROW_EDGES", raising=False)
    assert env_key() == before
    assert "DREAMSCRIPT_ARROW_EDGES" in ENV_KEYS


def test_every_env_key_names_a_variable_src_actually_reads():
    """A name in ENV_KEYS that nothing reads invalidates the cache for no reason. Matched as a
    string literal rather than at the `os.environ.get` call, because `s5` names its three through
    `EDGE_TEXT_ENV`-style constants."""
    from src.pipeline.cache import ENV_KEYS

    text = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore") for path in sorted(SRC.rglob("*.py"))
    )
    missing = [name for name in ENV_KEYS if f'"{name}"' not in text]
    assert not missing, missing


#: Environment variables `src/` reads that deliberately do *not* enter the cache key, each with
#: the reason it does not. The point of naming them is that the test below then covers every
#: `os.environ` read in the tree: a new one is either keyed or listed here, and neither happens
#: by accident.
NOT_IN_KEY = {
    # Process identity, not stage configuration. `utils.seed` owns it.
    "PYTHONHASHSEED",
    # Where a subprocess is found, not what it answers.
    "PATH",
    "DREAMSCRIPT_NODE",
    # Throughput. `ownlearn` reads in batches of this size and concatenates the same result.
    "OWNLEARN_READ_BATCH",
    # *Where* the stage cache lives, not what a stage answers. A cache in a different directory
    # is a cold cache, not a different result - and folding it into the key would mean a
    # container and a checkout could never share one.
    "DREAMSCRIPT_CACHE_DIR",
    # Training-time augmentation. What it changes is the checkpoint, and the checkpoint's
    # identity is `S3_CHECKPOINT`, which *is* keyed.
    "S3_AUGMENT",
    # *Which device* answers, read by 16.1.5's profiler. Deliberately not keyed, and this is the
    # one exemption that is a load-bearing claim rather than a convenience: a GPU and a CPU must
    # produce the same answer, so the device cannot be part of a stage's identity.
    #
    # The consequence is recorded where it bites. Because this is not in the key, a warm cache
    # would hand the CPU arm of that profile the GPU arm's stored outputs and the CPU arm would
    # come back instantaneous - so `src/serve/profile.py` runs both arms with the cache disabled,
    # and `tests/test_serve_profile.py` asserts `ENV_KEYS` still has no CUDA entry. If that ever
    # changes, both places are wrong together and both say so.
    "CUDA_VISIBLE_DEVICES",
}


def test_no_environ_read_in_src_is_missing_from_env_keys():
    """The other direction, which is the bug itself."""
    from src.pipeline.cache import ENV_KEYS

    read: set[str] = set()
    for path in sorted(SRC.rglob("*.py")):
        source = path.read_text(encoding="utf-8", errors="ignore")
        read |= set(re.findall(r"""environ(?:\.get)?[(\[]\s*["']([A-Z0-9_]+)["']""", source))
        read |= {
            value
            for _, value in re.findall(r"""([A-Z0-9_]*ENV)\s*=\s*["']([A-Z0-9_]+)["']""", source)
        }
    unaccounted = read - NOT_IN_KEY - set(ENV_KEYS)
    assert not unaccounted, (
        "reads the environment but is neither in `cache.ENV_KEYS` nor named in `NOT_IN_KEY` "
        f"with a reason: {sorted(unaccounted)}"
    )


def test_the_pipeline_config_key_moves_with_the_environment(monkeypatch):
    from src.pipeline.core import DreamScriptPipeline

    monkeypatch.delenv("DREAMSCRIPT_PAGE_TEXT", raising=False)
    before = DreamScriptPipeline(generate=False).config_key
    monkeypatch.setenv("DREAMSCRIPT_PAGE_TEXT", "0")
    assert DreamScriptPipeline(generate=False).config_key != before


# --- the code key covers the code that decides an answer (audit 5) ----------------------------


def test_code_key_changes_when_any_package_under_src_changes(tmp_path, monkeypatch):
    """It covered five packages: pipeline, assemble, codegen, ocr, parse. Editing
    `preprocess/binarize.py` changed the ink mask, the trace and the IR, and left the key alone."""
    from src.pipeline import cache

    root = tmp_path / "src"
    for package in ("pipeline", "preprocess", "detect", "ir", "llm", "utils", "features"):
        (root / package).mkdir(parents=True)
        (root / package / "m.py").write_text("x = 1\n", encoding="utf-8")

    monkeypatch.setattr(cache, "CODE_ROOT", root)
    for package in ("pipeline", "preprocess", "detect", "ir", "llm", "utils", "features"):
        cache.code_key.cache_clear()
        before = cache.code_key()
        (root / package / "m.py").write_text("x = 2\n", encoding="utf-8")
        cache.code_key.cache_clear()
        assert cache.code_key() != before, f"editing src/{package} left the key unchanged"


def test_code_key_changes_when_a_module_moves_without_being_edited(tmp_path, monkeypatch):
    from src.pipeline import cache

    root = tmp_path / "src"
    (root / "a").mkdir(parents=True)
    (root / "b").mkdir(parents=True)
    (root / "a" / "m.py").write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setattr(cache, "CODE_ROOT", root)
    cache.code_key.cache_clear()
    before = cache.code_key()
    (root / "a" / "m.py").rename(root / "b" / "m.py")
    cache.code_key.cache_clear()
    assert cache.code_key() != before


# --- one class vocabulary (audit 12) ----------------------------------------------------------


def test_routing_and_the_detector_share_one_class_vocabulary():
    """`routing.CLASSES` was a verbatim copy of `detect.classes.CLASSES`, whose own comment says
    it must never be reordered because a checkpoint stores integers, not names. A copy of that
    list is a list that can be reordered in one place, and the result would be a prior fitted on
    one histogram scoring another with its columns shuffled."""
    from src.detect.classes import CLASSES as DETECTOR
    from src.pipeline.routing import CLASSES as ROUTER

    assert ROUTER is DETECTOR


# --- the router says which types it cannot learn (audit 29) -----------------------------------


def test_uncovered_sources_are_exactly_the_ones_the_detect_index_lacks():
    """`SOURCE_TYPE` maps sketch2code -> wireframe and cghd -> circuit, and
    `data/processed/detect/index.json` contains no page from either, so `fit` never reaches those
    entries and the router can never learn or return those two types."""
    import json

    from src.pipeline.routing import SOURCE_TYPE, UNCOVERED_SOURCES
    from src.utils.config import ROOT

    index_path = ROOT / "data" / "processed" / "detect" / "index.json"
    if not index_path.is_file():
        import pytest

        pytest.skip("needs the DVC payload (data/processed/detect/index.json)")
    present = {row["source"] for row in json.loads(index_path.read_text(encoding="utf-8"))}
    assert set(UNCOVERED_SOURCES) == set(SOURCE_TYPE) - present


def test_uncovered_sources_are_a_subset_of_the_map_they_annotate():
    from src.pipeline.routing import SOURCE_TYPE, UNCOVERED_SOURCES

    assert set(UNCOVERED_SOURCES) < set(SOURCE_TYPE)


# --- a cache that cannot write says so (audit 59) ---------------------------------------------


def test_the_cache_directory_comes_from_the_environment(monkeypatch, tmp_path):
    """The default is `data/interim/pipeline_cache`, which is inside the read-only `./data`
    mount in docker-compose.yml - so every `put()` in the container raised OSError into a
    `contextlib.suppress` and the cache silently never worked, in the deployment it is for."""
    import importlib

    from src.pipeline import cache as cache_module

    monkeypatch.setenv(cache_module.CACHE_DIR_ENV, str(tmp_path / "elsewhere"))
    reloaded = importlib.reload(cache_module)
    try:
        elsewhere = tmp_path / "elsewhere"
        assert elsewhere == reloaded.CACHE_DIR
        assert elsewhere == reloaded.StageCache().directory
    finally:
        monkeypatch.delenv(cache_module.CACHE_DIR_ENV, raising=False)
        importlib.reload(cache_module)


def test_a_write_that_cannot_land_is_counted_not_swallowed(tmp_path):
    from src.pipeline.cache import StageCache

    blocked = tmp_path / "file-not-a-directory"
    blocked.write_text("", encoding="utf-8")
    store = StageCache(directory=blocked)
    assert store.put("detect", "k", [1, 2, 3]) == [1, 2, 3]
    stats = store.stats()
    assert stats["write_failures"] == 1
    assert stats["write_error"]
    assert store.get("detect", "k") is None


def test_a_value_that_will_not_serialise_is_still_only_a_miss(tmp_path):
    """The original reason for the suppress, kept: an unserialisable stage output is not cached
    and is not an error."""
    from src.pipeline.cache import StageCache

    store = StageCache(directory=tmp_path)
    circular: dict = {}
    circular["self"] = circular
    assert store.put("detect", "k", circular) is circular
    assert store.stats()["write_failures"] == 1
    assert store.get("detect", "k") is None


def test_compose_points_the_container_at_a_writable_volume():
    import yaml

    from src.pipeline.cache import CACHE_DIR_ENV
    from src.utils.config import ROOT

    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    model = compose["services"]["model"]
    target = model["environment"][CACHE_DIR_ENV]
    mounts = {entry.split(":")[1]: entry for entry in model["volumes"]}
    assert target in mounts, f"{target} is not mounted"
    assert not mounts[target].endswith(":ro"), "the cache is mounted read-only"
    assert mounts[target].split(":")[0] in (compose.get("volumes") or {})
