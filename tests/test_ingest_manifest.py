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


# --- the three corpora reconcile (audit 66) ---------------------------------------------------


def test_every_difference_between_the_three_corpora_is_accounted_for():
    """`manifest.parquet` says 3,054, `data/processed/ir/` says 2,796 and `detect/index.json`
    says 2,312. Nothing reconciled them, so each number has been quoted somewhere as "the
    corpus" and a reader had no way to tell a legitimate filter from data loss."""
    import pytest

    from src.ingest.reconcile import MANIFEST, problems, reconcile

    if not MANIFEST.is_file():
        pytest.skip("needs the DVC payload (data/processed/manifest.parquet)")
    found = problems(reconcile())
    assert not found, found


def test_the_sketch2code_relationship_is_exact_not_approximate():
    """The manifest row is a sketch and the IR document is the page it depicts - 731 sketches
    over 484 pages. 484 is the count of distinct page ids, exactly."""
    import pytest

    from src.ingest.reconcile import MANIFEST, manifest_units, reconcile

    if not MANIFEST.is_file():
        pytest.skip("needs the DVC payload (data/processed/manifest.parquet)")
    result = reconcile()["by_source"]["sketch2code"]
    assert len(manifest_units()["sketch2code"]) == result["ir"]
    assert result["detect"] == 0, "sketch2code has no detection annotations in this repo"


def test_the_hdbpmn_gap_is_named_not_counted():
    import pytest

    from src.ingest.reconcile import MANIFEST, unconverted

    if not MANIFEST.is_file():
        pytest.skip("needs the DVC payload (data/processed/manifest.parquet)")
    gap = unconverted().get("hdbpmn", [])
    assert gap, "the eleven unconverted pages are no longer detected"
    assert all(name.startswith("ex") for name in gap), gap


# --- lost is not the same as not pulled (audit 63) --------------------------------------------


def test_the_lost_payloads_are_named_and_the_document_is_current():
    """`dvc status -c` prints the same "missing from remote and local" line for a source you
    have not pulled and one whose content exists nowhere. Four of the eight raw sources are in
    the second state - cghd_extracted, chaos, didi, iam_line - and nothing said so."""
    from src.ingest.losses import lost, main

    assert set(lost()) == {"cghd_extracted", "chaos", "didi", "iam_line"}
    assert main(["--check"]) == 0, "docs/data_losses.md is stale"


def test_every_source_is_in_exactly_one_state():
    """A pointer with no md5 at all would read as lost and would actually be a broken pointer."""
    from src.ingest.losses import survey

    state = survey()
    assert len(state) == 8
    for name, row in state.items():
        assert row["md5"], f"{name} has a .dvc pointer with no hash"
        assert row["files"], f"{name} records no file count"


def test_the_survey_needs_no_dvc_process():
    """It reads the pointers, so it answers while a `dvc repro` holds the lock and on a machine
    with no DVC installed - which is the situation the document describes."""
    import inspect

    from src.ingest import losses

    source = inspect.getsource(losses)
    assert "subprocess" not in source
    assert "import dvc" not in source
