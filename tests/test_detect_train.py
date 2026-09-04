"""Phase 9.1.3 - the anchor analysis, which is the part of the training run that is a measurement."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.detect import train
from src.detect.classes import CLASSES


@pytest.fixture
def export(tmp_path: Path):
    """Pages whose boxes have a known size distribution: big rectangles and tiny arrowheads."""
    import cv2

    root = tmp_path / "detect"
    (root / "images" / "train").mkdir(parents=True)
    (root / "labels" / "train").mkdir(parents=True)
    for index in range(6):
        name = f"hdbpmn__p{index}"
        cv2.imwrite(
            str(root / "images" / "train" / f"{name}.png"), np.full((1000, 1000, 3), 240, np.uint8)
        )
        lines = ["0 0.5 0.5 0.2 0.2"]  # rectangle, 200px
        lines += [f"7 0.{i}1 0.31 0.012 0.012" for i in range(1, 5)]  # arrowheads, 12px
        (root / "labels" / "train" / f"{name}.txt").write_text("\n".join(lines) + "\n", "utf-8")
    return root


def test_the_boxes_are_read_in_pixels(export):
    sizes = train.training_boxes(export)
    assert len(sizes) == 30
    assert sizes.max() == pytest.approx(200.0)


def test_the_analysis_names_the_strides_yolov8_actually_has():
    assert train.STRIDES == (8, 16, 32)


def test_the_anchor_clusters_separate_the_two_populations(export):
    result = train.anchor_analysis(export, k=2, imgsz=1000)
    smallest, largest = sorted(result["kmeans_anchors"], key=lambda a: a[0])
    assert smallest[0] < 20
    assert largest[0] > 150


def test_the_small_class_is_reported_as_living_below_a_coarse_cell(export):
    result = train.anchor_analysis(export, k=2, imgsz=1000)
    assert result["per_class"]["arrowhead"]["under_p4_cell"] == 1.0
    assert result["per_class"]["rectangle"]["under_p4_cell"] == 0.0


def test_the_analysis_says_out_loud_that_the_head_is_anchor_free(export):
    assert "anchor-free" in train.anchor_analysis(export, k=2, imgsz=1000)["note"]


def test_an_empty_export_is_reported_rather_than_crashing(tmp_path):
    root = tmp_path / "empty"
    (root / "images" / "train").mkdir(parents=True)
    (root / "labels" / "train").mkdir(parents=True)
    assert train.anchor_analysis(root, k=2)["boxes"] == 0


def test_the_class_buckets_cover_the_frozen_vocabulary(export):
    buckets = train._boxes_by_class(export, 1000)
    assert set(buckets) == set(CLASSES)


def test_the_sweep_sizes_bracket_the_export_resolution():
    """One below the exported long side, one at it, one above - so the sweep can show a peak."""
    assert min(train.SWEEP_SIZES) < 1280 <= max(train.SWEEP_SIZES)
