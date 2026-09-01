"""Phase 6.1.2 - the embedding cache."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.embed import cache


@pytest.fixture
def written(tmp_path):
    """A small cache on disk, written through the real `write`."""
    matrix = np.arange(12, dtype=np.float32).reshape(4, 3)
    index = pd.DataFrame(
        {
            "row": [0, 1, 2, 3],
            "id": ["a/1", "a/2", "b/1", "s/1"],
            "source": ["a", "a", "b", "synthetic"],
            "diagram_type": ["flowchart", "circuit", "flowchart", "circuit"],
            "split": ["train"] * 4,
            "scribe_id": [None, "w1", "w2", None],
            "adverse": [False] * 4,
            "synthetic": [False, False, False, True],
        }
    )
    built = {
        "matrix": matrix,
        "index": index,
        "meta": {"backbone": "resnet18", "input": "gray", "dimension": 3, "rows": 4},
    }
    paths = (
        tmp_path / "embeddings.npy",
        tmp_path / "embeddings_index.parquet",
        tmp_path / "embeddings_meta.json",
    )
    cache.write(built, *paths)
    return paths


def test_all_three_files_are_written(written):
    for path in written:
        assert path.is_file()
    assert json.loads(written[2].read_text(encoding="utf-8"))["backbone"] == "resnet18"


def test_loading_returns_the_matrix_the_index_and_the_header(written):
    embeddings, frame, header = cache.load(*written)
    assert embeddings.shape == (4, 3)
    assert list(frame["id"]) == ["a/1", "a/2", "b/1", "s/1"]
    assert header["input"] == "gray"


def test_a_missing_cache_names_the_command_that_builds_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="src.embed.cache --build"):
        cache.load(tmp_path / "nothing.npy", tmp_path / "nothing.parquet")


def test_a_matrix_and_index_of_different_lengths_is_refused(written, tmp_path):
    """Two files written separately can drift, and a silent join would be off by one."""
    short = pd.read_parquet(written[1]).iloc[:2]
    short.to_parquet(tmp_path / "short.parquet", index=False)
    with pytest.raises(ValueError, match="inconsistent"):
        cache.load(written[0], tmp_path / "short.parquet")


def test_aligned_returns_rows_in_the_callers_order(written):
    embeddings, mask = cache.aligned(["b/1", "a/1"], written[0], written[1])
    assert mask.all()
    assert np.array_equal(embeddings[0], np.array([6, 7, 8], np.float32))
    assert np.array_equal(embeddings[1], np.array([0, 1, 2], np.float32))


def test_aligned_refuses_a_missing_id_rather_than_shortening_the_matrix(written):
    with pytest.raises(KeyError, match="not in the embedding cache"):
        cache.aligned(["a/1", "nope"], written[0], written[1])


def test_aligned_can_be_told_to_drop_missing_ids(written):
    embeddings, mask = cache.aligned(
        ["a/1", "nope", "s/1"], written[0], written[1], drop_missing=True
    )
    assert list(mask) == [True, False, True]
    assert embeddings.shape == (2, 3)


def test_a_failed_rebuild_leaves_the_previous_cache_intact(written):
    """The matrix must not be replaced when the index write fails - that pairing is the cache."""
    before = np.load(written[0])
    broken = {
        "matrix": np.zeros((2, 3), np.float32),
        "index": object(),  # has no `to_parquet`
        "meta": {"backbone": "wrong"},
    }
    with pytest.raises(AttributeError):
        cache.write(broken, *written)

    embeddings, frame, header = cache.load(*written)
    assert np.array_equal(embeddings, before)
    assert len(frame) == 4
    assert header["backbone"] == "resnet18"


def test_a_failed_rebuild_leaves_no_temporary_files(written):
    broken = {"matrix": np.zeros((2, 3), np.float32), "index": object(), "meta": {}}
    with pytest.raises(AttributeError):
        cache.write(broken, *written)
    assert not list(written[0].parent.glob("*.tmp"))


def test_the_summary_counts_both_corpora(written):
    result = cache.summary(*cache.load(*written))
    assert result["by_corpus"] == {"real": 3, "synthetic": 1}
    assert result["by_type"] == {"circuit": 2, "flowchart": 2}
    assert result["duplicate_ids"] == 0
    assert result["usable_rows"] == 4


def test_the_summary_notices_unreadable_rows(written):
    embeddings, frame, header = cache.load(*written)
    embeddings[1] = np.nan
    assert cache.summary(embeddings, frame, header)["usable_rows"] == 3


def test_the_summary_carries_the_provenance(written):
    result = cache.summary(*cache.load(*written))
    assert result["backbone"] == "resnet18"
    assert result["input"] == "gray"
