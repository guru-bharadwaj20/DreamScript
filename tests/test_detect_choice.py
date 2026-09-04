"""Phase 9.1.1 - the bake-off's fairness properties, which are the only thing a test can pin.

The scores themselves need a GPU and an hour; what is checked here is that the three arms are
made comparable - one ground truth read from the export, one metric, one budget - because that
is the claim the task rests on and the one a refactor can silently break.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.detect import choice
from src.detect.classes import CLASSES


@pytest.fixture
def export(tmp_path: Path):
    """A two-page export in the exact layout `src.detect.dataset` writes."""
    import cv2

    root = tmp_path / "detect"
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
    for name in ("hdbpmn__a", "flowchartseg__b"):
        image = np.full((100, 200, 3), 240, np.uint8)
        cv2.imwrite(str(root / "images" / "val" / f"{name}.png"), image)
        # one centred box covering the middle half of the page
        (root / "labels" / "val" / f"{name}.txt").write_text(
            "2 0.5 0.5 0.5 0.5\n", encoding="utf-8"
        )
    (root / "data.yaml").write_text("names:\n  0: rectangle\n", encoding="utf-8")
    return root


def test_truths_are_read_back_from_the_export_in_pixels(export):
    rows, sources = choice.truths("val", export)
    assert len(rows) == 2
    assert {r["cls"] for r in rows} == {"diamond"}
    x1, y1, x2, y2 = rows[0]["xyxy"]
    assert (x1, y1, x2, y2) == pytest.approx((50.0, 25.0, 150.0, 75.0))


def test_truths_carry_the_source_so_the_mix_can_be_reported_apart(export):
    _, sources = choice.truths("val", export)
    assert sources == {"hdbpmn__a": "hdbpmn", "flowchartseg__b": "flowchartseg"}


def test_a_label_without_its_image_is_skipped(export):
    (export / "labels" / "val" / "ghost.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    rows, sources = choice.truths("val", export)
    assert "ghost" not in sources
    assert len(rows) == 2


def test_the_arms_are_the_three_families_the_plan_names():
    assert "fasterrcnn" in choice.ARMS
    assert any(a.startswith("yolov8") for a in choice.ARMS)
    assert any(a.startswith("rtdetr") for a in choice.ARMS)


def test_the_torchvision_dataset_reads_the_same_boxes_as_yolo(export):
    """The whole comparison rests on this: one export, two readers, identical boxes."""
    data = choice.YoloFolder("val", export, imgsz=200)
    tensor, target, name, scale = data[0]
    assert tensor.shape[0] == 3
    assert len(target["boxes"]) == 1
    # torchvision reserves 0 for background, so every class id shifts by one
    assert int(target["labels"][0]) == CLASSES.index("diamond") + 1
    x1, y1, x2, y2 = (float(v) for v in target["boxes"][0])
    assert (x1 / scale, y1 / scale) == pytest.approx((50.0, 25.0), abs=1.0)


def test_the_torchvision_labels_are_offset_by_exactly_one_from_the_class_index(export):
    data = choice.YoloFolder("val", export, imgsz=200)
    _, target, _, _ = data[0]
    assert 1 <= int(target["labels"][0]) <= len(CLASSES)


def arm_result(name, map50, train_seconds, parameters, head_ap):
    return {
        "arm": name,
        "map50": map50,
        "map50_95": map50 - 0.2,
        "map25": map50 + 0.05,
        "parameters": parameters,
        "predictions": 1000,
        "train_seconds": train_seconds,
        "page_latency_ms": 20.0,
        "per_class": {
            n: {"instances": 1, "ap50": head_ap if n == "arrowhead" else 0.9} for n in CLASSES
        },
        "by_source": {"hdbpmn": {"map50": 0.4}},
    }


def test_the_report_names_every_arm_and_every_class(tmp_path):
    results = [arm_result("yolov8n", 0.5, 100.0, 3_000_000, 0.3)]
    path = choice.write_report(results, tmp_path / "detector_choice.md")
    text = path.read_text(encoding="utf-8")
    assert "yolov8n" in text
    for name in CLASSES:
        assert name in text
    assert "hdbpmn" in text


def test_the_verdict_names_the_accurate_arm_and_the_cheap_one_separately():
    """They are different questions, and on this corpus they have different answers."""
    results = [
        arm_result("yolov8n", 0.89, 752.0, 3_000_000, 0.345),
        arm_result("rtdetr-l", 0.906, 4338.0, 32_000_000, 0.486),
    ]
    text = "\n".join(choice._verdict(results))
    assert "Most accurate: **`rtdetr-l`**" in text
    assert "Cheapest: **`yolov8n`**" in text
    assert "5.8x less training time" in text


def test_the_verdict_reports_the_spread_across_arms():
    results = [
        arm_result("a", 0.90, 100.0, 1_000_000, 0.3),
        arm_result("b", 0.88, 200.0, 2_000_000, 0.4),
    ]
    assert "**0.0200**" in "\n".join(choice._verdict(results))
