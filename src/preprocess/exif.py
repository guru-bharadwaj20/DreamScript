"""Phase 3.1.1 - EXIF orientation.

A phone photograph is very often stored rotated, with an EXIF tag saying which way up it
really is. Decoders disagree about whether to honour that tag: PIL does not by default, and
`cv2.imread` never does. So the same file read two ways gives two different images, and a
pipeline that mixes the two silently trains on sideways diagrams.

This module is the single place the project reads an image from disk. It applies the tag once,
records what it did, and everything downstream can assume upright pixels.

Surveyed across the corpus, **all 693 hdBPMN photographs carry orientation 1**: the dataset
authors already normalised them. So this module changes nothing for training, and everything
for Phase 16, where the input is a photo straight off someone's phone. It is cheap insurance
against a class of bug that is invisible until a user reports that their diagram came out
sideways.

    python -m src.preprocess.exif data/raw/hdbpmn/data/images/ex00
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import ExifTags, Image, ImageOps

#: The EXIF tag id for orientation, looked up by name so it survives a Pillow change.
ORIENTATION_TAG = next(
    (k for k, v in ExifTags.TAGS.items() if v == "Orientation"),
    274,
)

#: What each of the eight orientation values means, for the log line.
ORIENTATIONS = {
    1: "upright",
    2: "mirrored",
    3: "rotated 180",
    4: "mirrored, rotated 180",
    5: "mirrored, rotated 90 CW",
    6: "rotated 90 CCW",
    7: "mirrored, rotated 90 CCW",
    8: "rotated 90 CW",
}


def orientation(path: str | Path) -> int:
    """The stored orientation, 1 (upright) when the file does not say."""
    try:
        with Image.open(path) as image:
            exif = image.getexif()
    except Exception:  # noqa: BLE001 - a file without readable EXIF is simply upright
        return 1
    value = exif.get(ORIENTATION_TAG, 1) if exif else 1
    return int(value) if value in ORIENTATIONS else 1


def load(path: str | Path, *, grayscale: bool = False) -> np.ndarray:
    """Read an image with its EXIF orientation applied, as a numpy array.

    Returns BGR to match OpenCV, or a single channel when `grayscale` is set. Every other
    module in Phase 3 reads through this function rather than calling `cv2.imread`.
    """
    with Image.open(path) as image:
        # `exif_transpose` handles all eight cases including the mirrored ones, which a plain
        # rotation by the tag's angle would get wrong for values 2, 4, 5 and 7.
        upright = ImageOps.exif_transpose(image)
        upright = upright.convert("L" if grayscale else "RGB")
        array = np.asarray(upright)
    return array if grayscale else cv2.cvtColor(array, cv2.COLOR_RGB2BGR)


def was_rotated(path: str | Path) -> bool:
    return orientation(path) != 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+", type=Path, help="images or directories")
    args = ap.parse_args(argv)

    files: list[Path] = []
    for path in args.paths:
        files.extend(sorted(path.rglob("*.jpg")) if path.is_dir() else [path])
    if not files:
        print("no images found", file=sys.stderr)
        return 1

    counts: dict[int, int] = {}
    for path in files:
        value = orientation(path)
        counts[value] = counts.get(value, 0) + 1
    for value, count in sorted(counts.items()):
        print(f"  {count:5d}  orientation {value} ({ORIENTATIONS[value]})")
    rotated = sum(c for v, c in counts.items() if v != 1)
    print(f"{rotated} of {len(files)} images carry a non-upright orientation tag")
    return 0


if __name__ == "__main__":
    sys.exit(main())
