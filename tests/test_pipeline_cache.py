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
    # Training-time augmentation. What it changes is the checkpoint, and the checkpoint's
    # identity is `S3_CHECKPOINT`, which *is* keyed.
    "S3_AUGMENT",
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
