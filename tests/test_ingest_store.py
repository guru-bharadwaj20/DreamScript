"""Phase 1.1.9 - the store survey distinguishes "cache-only" from "stored" from "gone".

`dvc status -c` reported 1,507 objects as `new:` - in `.dvc/cache`, absent from the store. The
word for that state is not "new". `.dvc/cache` is a working directory and the store is the copy
that is meant to outlive it, so every one of those objects - the whole Phase 12.1 target corpus,
the arrow pose checkpoint, the TrOCR fine-tune - was one `dvc gc` away from joining the four
sources in `docs/data_losses.md`.

`src.ingest.store` answers that question from the on-disk layout, so these tests build stores on
`tmp_path` rather than touching the real one. The state is checked against the real store in
exactly one test, which skips when the store is not on this machine.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.ingest import store as mod
from src.utils.config import ROOT

# A hash and a directory hash, in the shape DVC writes them.
FILE_MD5 = "a" * 32
DIR_MD5 = "b" * 32 + ".dir"
MEMBER = ["c" * 32, "d" * 32]


def _write(store: Path, md5: str, subtree: str, payload: str = "x") -> Path:
    path = mod.object_path(md5, store, subtree)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return path


# --- where an object is looked for ------------------------------------------------------------


def test_an_object_is_found_in_the_subtree_its_pointer_declares(tmp_path: Path):
    _write(tmp_path, FILE_MD5, "processed")
    assert mod.in_store(FILE_MD5, tmp_path, "processed") == "processed"


def test_an_object_is_still_found_when_it_is_not_where_the_pointer_says(tmp_path: Path):
    """`data/processed/targets.dvc` declares no remote and lands in `<store>/files/md5` while
    its four siblings declare `remote: processed`. All four remotes in `.dvc/config` are
    subdirectories of one store on one disk, so for "does a durable copy exist" they are the
    same place - and the answer says which subtree it was actually in."""
    _write(tmp_path, FILE_MD5, "")
    assert mod.in_store(FILE_MD5, tmp_path, "processed") == "."


def test_an_object_in_no_subtree_is_not_in_the_store(tmp_path: Path):
    assert mod.in_store(FILE_MD5, tmp_path, "raw") is None


def test_the_declared_remote_is_tried_first(tmp_path: Path):
    """Two copies is not an error, but the reported location must be the declared one, or the
    survey would name a subtree the pointer does not use."""
    _write(tmp_path, FILE_MD5, "")
    _write(tmp_path, FILE_MD5, "raw")
    assert mod.in_store(FILE_MD5, tmp_path, "raw") == "raw"


# --- a directory is its manifest and every member ---------------------------------------------


def test_a_directory_expands_into_its_members(tmp_path: Path):
    """The failure this guards is the one an interrupted push leaves: the `.dir` manifest in the
    store and half the files it names still only in the cache. Counting the manifest as one
    object would report that directory as fully stored."""
    entries = [{"md5": md5, "relpath": f"{i}.txt"} for i, md5 in enumerate(MEMBER)]
    _write(tmp_path, DIR_MD5, "", json.dumps(entries))
    assert mod._members(DIR_MD5, tmp_path, "") == MEMBER


def test_a_directory_with_no_manifest_anywhere_expands_to_nothing(tmp_path: Path):
    """The four lost sources. There is nothing to expand and nothing to push; `losses` is where
    that state is reported, and this must not raise trying to read a file that is gone."""
    assert mod._members(DIR_MD5, tmp_path, "raw") == []


# --- uncached outputs are not missing ---------------------------------------------------------


def test_the_cache_false_report_outputs_are_excluded():
    """`dvc.yaml` declares the twelve report artefacts `cache: false` - small text files, in git,
    and `dvc push` correctly never sends them. Counting them as absent would make `--check` fail
    on every clean run, which is the fastest way to teach somebody to ignore it."""
    skip = mod.uncached()
    assert "reports/master_results.md" in skip
    assert "reports/stagewise.json" in skip
    paths = {row["path"] for row in mod.tracked()}
    assert not (paths & skip)


def test_every_cached_pipeline_output_is_tracked():
    paths = {row["path"] for row in mod.tracked()}
    assert "data/features/handcrafted.parquet" in paths
    assert "data/processed/targets" in paths
    assert "data/raw/didi" in paths
    assert "experiments/ocr/trocr_large" in paths


def test_a_tracked_row_says_where_it_was_read_from():
    """A survey naming an output and not the file that declares it leaves the reader grepping."""
    origins = {row["path"]: row["from"] for row in mod.tracked()}
    assert origins["data/processed/targets"] == "data/processed/targets.dvc"
    assert origins["data/features/handcrafted.parquet"] == "dvc.lock:features"


# --- the three states -------------------------------------------------------------------------


def _row(**kwargs) -> dict:
    base = {"path": "x", "md5": FILE_MD5, "objects": 1, "cached": 1, "stored": 1, "subtree": "."}
    return {**base, **kwargs}


def test_cache_only_is_reported_as_unpushed():
    rows = [_row(path="behind", cached=4, stored=1, objects=4)]
    assert [row["path"] for row in mod.unpushed(rows)] == ["behind"]
    assert not mod.absent(rows)


def test_gone_is_not_reported_as_unpushed():
    """`dvc push` cannot fix a payload that is in neither place, and listing the four lost
    sources under "run dvc push" would send somebody after a command that does nothing."""
    rows = [_row(path="gone", cached=0, stored=0, subtree=None)]
    assert not mod.unpushed(rows)
    assert [row["path"] for row in mod.absent(rows)] == ["gone"]


def test_fully_stored_is_neither():
    rows = [_row(cached=9, stored=9, objects=9)]
    assert not mod.unpushed(rows)
    assert not mod.absent(rows)


def test_render_names_the_count_that_is_cache_only():
    text = mod.render([_row(path="data/x", cached=4, stored=1, objects=4)])
    assert "NOT PUSHED" in text
    assert "3 of 4" in text


# --- against the real store -------------------------------------------------------------------


@pytest.mark.skipif(not mod.config_store().is_dir(), reason="the store is on one machine only")
def test_nothing_tracked_exists_only_in_this_machines_cache():
    rows = mod.survey()
    behind = mod.unpushed(rows)
    assert not behind, f"cache-only: {[row['path'] for row in behind]} - run `dvc push`"


@pytest.mark.skipif(not mod.config_store().is_dir(), reason="the store is on one machine only")
def test_the_gone_outputs_are_the_four_data_losses_names():
    """Two modules answer "where is this payload" from the same layout. If they disagreed, one
    of `docs/data_losses.md` and this survey would be wrong and nothing would say which."""
    from src.ingest.losses import lost

    gone = {Path(row["path"]).name for row in mod.absent(mod.survey())}
    assert gone == set(lost())


def test_losses_and_store_share_one_definition_of_the_store():
    """They did not: `losses` carried a private copy of the path logic. Two copies of "where
    would the store keep this" is how they drift."""
    from src.ingest import losses

    assert losses.config_store is mod.config_store
    assert losses.in_store is mod.in_store
    assert "_in_store" not in (ROOT / "src" / "ingest" / "losses.py").read_text(encoding="utf-8")


# --- the commands that do it ------------------------------------------------------------------


def test_both_task_runners_offer_a_push():
    """The store was 1.2 GB behind the cache partly because no target named the command. `make
    repro` and `make dag` existed; the one that makes an output durable did not."""
    assert "-m dvc push" in (ROOT / "Makefile").read_text(encoding="utf-8")
    assert '"-m", "dvc", "push"' in (ROOT / "tasks.ps1").read_text(encoding="utf-8")
