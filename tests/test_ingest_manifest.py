"""Phase 1.1 - the corpus manifest, and the dispatch table that builds it.

`SOURCES` maps a source name to its loader and `build` iterates it. It used to iterate it with a
special case - `fn(limit=didi_limit) if name == "didi" else fn()` - because `from_didi` was the
only loader that took a row cap. A dispatch table whose entries have different signatures, worked
around by comparing the key to a string literal, breaks silently the moment the key is renamed,
and mypy had been reporting the call as `Unexpected keyword argument "limit"` throughout.
"""

from __future__ import annotations

# --- one dispatch table, one signature (audit 14) ---------------------------------------------


def test_every_source_loader_takes_the_same_arguments():
    """`build` dispatched with `fn(limit=didi_limit) if name == "didi" else fn()`, because
    `from_didi` was the only loader that took a cap. A table whose entries have different
    signatures, worked around by comparing the key to a string literal, breaks the moment the key
    is renamed - and mypy had been reporting the call as `Unexpected keyword argument "limit"`."""
    import inspect

    from src.ingest.manifest import SOURCES

    signatures = {str(inspect.signature(fn)) for fn in SOURCES.values()}
    assert len(signatures) == 1, signatures
    assert "limit" in signatures.pop()


def test_build_no_longer_names_a_source_to_decide_how_to_call_it():
    import inspect

    from src.ingest.manifest import build

    source = inspect.getsource(build)
    assert '"didi"' not in source
    assert "fn(limit=limit)" in source


def test_didi_limit_still_means_what_the_cli_has_always_meant_by_it():
    import inspect

    from src.ingest.manifest import build

    assert "didi_limit" in inspect.signature(build).parameters
