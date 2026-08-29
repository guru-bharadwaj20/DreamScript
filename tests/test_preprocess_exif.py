"""Phase 3.1.1 - EXIF orientation.

The test that matters is the round trip: write a rotated image with the tag that says how to
un-rotate it, read it back through `load`, and get the original.
"""

from __future__ import annotations

import numpy as np
import piexif
import pytest
from PIL import Image

from src.preprocess import exif


def _landmarked() -> Image.Image:
    """An image with an unambiguous top-left: only a rotation-correct read reproduces it."""
    array = np.full((60, 40, 3), 240, np.uint8)
    array[0:15, 0:10] = (255, 0, 0)  # red block, top-left
    array[-10:, -8:] = (0, 0, 255)  # blue block, bottom-right
    return Image.fromarray(array)


def _write(path, image, orientation):
    payload = piexif.dump({"0th": {piexif.ImageIFD.Orientation: orientation}})
    image.save(path, exif=payload)


def test_upright_image_is_unchanged(tmp_path):
    original = _landmarked()
    path = tmp_path / "a.jpg"
    _write(path, original, 1)
    assert exif.orientation(path) == 1
    assert not exif.was_rotated(path)
    loaded = exif.load(path)
    assert loaded.shape == (60, 40, 3)


@pytest.mark.parametrize(("stored", "value"), [(6, 6), (8, 8), (3, 3)])
def test_rotated_image_comes_back_upright(tmp_path, stored, value):
    original = _landmarked()
    # Rotate the pixels the way a camera would, and record the tag that undoes it.
    # Verified against Pillow's own exif_transpose rather than reasoned about: these are the
    # transforms a camera applies for each tag, so applying them and then loading must undo.
    rotations = {6: Image.ROTATE_90, 8: Image.ROTATE_270, 3: Image.ROTATE_180}
    path = tmp_path / f"r{stored}.jpg"
    _write(path, original.transpose(rotations[stored]), stored)

    assert exif.orientation(path) == value
    assert exif.was_rotated(path)

    loaded = exif.load(path)
    assert loaded.shape[:2] == (60, 40), "portrait shape not restored"
    # Top-left must be red again. BGR, and JPEG is lossy, so compare loosely.
    b, g, r = loaded[5, 3]
    assert r > 150 and b < 100


def test_mirrored_orientation_is_handled_not_just_rotated(tmp_path):
    """Values 2, 4, 5 and 7 involve a flip; rotating by the tag's angle alone gets them wrong."""
    original = _landmarked()
    path = tmp_path / "m.jpg"
    _write(path, original.transpose(Image.FLIP_LEFT_RIGHT), 2)
    loaded = exif.load(path)
    b, g, r = loaded[5, 3]
    assert r > 150 and b < 100


def test_grayscale_read_is_single_channel(tmp_path):
    path = tmp_path / "g.jpg"
    _write(path, _landmarked(), 1)
    assert exif.load(path, grayscale=True).ndim == 2


def test_missing_exif_is_treated_as_upright(tmp_path):
    path = tmp_path / "plain.png"
    _landmarked().save(path)
    assert exif.orientation(path) == 1
    assert exif.load(path).shape == (60, 40, 3)


def test_corpus_orientation_survey_runs():
    from src.utils.config import ROOT

    directory = ROOT / "data" / "raw" / "hdbpmn" / "data" / "images" / "ex00"
    if not directory.is_dir():
        pytest.skip("hdBPMN not present")
    assert exif.main([str(directory)]) == 0
