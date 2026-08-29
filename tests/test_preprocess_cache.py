"""Phase 3.2.9 - the primitive cache, and the two ways it is allowed to miss."""

from __future__ import annotations

import pickle

import cv2
import numpy as np
import pytest

from src.preprocess import cache
from src.preprocess.primitives import text as tx


@pytest.fixture
def page(tmp_path):
    """A small synthetic page written to disk, since the cache keys on file content."""
    built = tx.synthetic_page(2, size=500)
    path = tmp_path / "page.png"
    cv2.imwrite(str(path), built["gray"])
    return path


def test_cold_then_warm(page, tmp_path):
    first, hit = cache.load_or_compute("page", page, tmp_path)
    assert hit is False
    second, hit = cache.load_or_compute("page", page, tmp_path)
    assert hit is True
    assert second.id == first.id
    assert second.components == first.components
    assert second.text_boxes == first.text_boxes


def test_the_entry_lands_where_the_plan_says(page, tmp_path):
    cache.load_or_compute("page", page, tmp_path)
    assert (tmp_path / "page.pkl").is_file()


def test_editing_the_image_is_a_miss(page, tmp_path):
    cache.load_or_compute("page", page, tmp_path)
    image = cv2.imread(str(page), cv2.IMREAD_GRAYSCALE)
    cv2.line(image, (10, 10), (480, 480), 0, 5)
    cv2.imwrite(str(page), image)
    _, hit = cache.load_or_compute("page", page, tmp_path)
    assert hit is False, "a changed image must not return the old primitives"


def test_changing_a_tuned_constant_is_a_miss(page, tmp_path, monkeypatch):
    cache.load_or_compute("page", page, tmp_path)
    monkeypatch.setattr(tx, "MIN_GROUP_SIZE", tx.MIN_GROUP_SIZE + 1)
    _, hit = cache.load_or_compute("page", page, tmp_path)
    assert hit is False, "a retuned threshold must invalidate the entry"


def test_touching_the_file_without_changing_it_is_still_a_hit(page, tmp_path):
    cache.load_or_compute("page", page, tmp_path)
    page.touch()
    _, hit = cache.load_or_compute("page", page, tmp_path)
    assert hit is True, "the cache keys on content, not on the clock"


def test_a_corrupt_entry_is_a_miss_not_a_crash(page, tmp_path):
    cache.load_or_compute("page", page, tmp_path)
    (tmp_path / "page.pkl").write_bytes(b"not a pickle")
    primitives, hit = cache.load_or_compute("page", page, tmp_path)
    assert hit is False
    assert primitives.id == "page"


def test_an_entry_of_the_wrong_type_is_a_miss(tmp_path):
    with (tmp_path / "other.pkl").open("wb") as handle:
        pickle.dump({"not": "primitives"}, handle)
    assert cache.load("other", tmp_path) is None


def test_missing_entry_loads_as_none(tmp_path):
    assert cache.load("never-computed", tmp_path) is None


def test_fingerprint_is_stable_within_a_process():
    assert cache.fingerprint() == cache.fingerprint()


def test_fingerprint_covers_a_constant_from_every_stage(monkeypatch):
    from src.preprocess import binarize
    from src.preprocess.primitives import arrowheads

    before = cache.fingerprint()
    monkeypatch.setattr(arrowheads, "MIN_BARB_ANGLE", arrowheads.MIN_BARB_ANGLE + 1.0)
    assert cache.fingerprint() != before
    monkeypatch.undo()
    monkeypatch.setattr(binarize, "DEFAULT_K", binarize.DEFAULT_K + 0.01)
    assert cache.fingerprint() != before


def test_primitives_are_extracted_not_empty(page, tmp_path):
    primitives, _ = cache.load_or_compute("page", page, tmp_path)
    assert primitives.shape == (500, 500)
    assert primitives.components and primitives.contours and primitives.segments
    assert primitives.layers["ink_pixels"] > 0
    assert primitives.skeleton["pixels"] > 0
    assert np.all([len(box) == 4 for box in primitives.text_boxes])
