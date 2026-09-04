"""Phase 9.1.7 - the synthesized damage, and the rule that the eval splits are left alone."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.detect import negatives


@pytest.fixture
def export(tmp_path: Path):
    import cv2

    root = tmp_path / "detect"
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
    for split in ("train", "val", "test"):
        for index in range(4):
            name = f"hdbpmn__p{split}{index}"
            cv2.imwrite(
                str(root / "images" / split / f"{name}.png"), np.full((200, 200, 3), 240, np.uint8)
            )
            (root / "labels" / split / f"{name}.txt").write_text(
                "0 0.5 0.5 0.4 0.4\n1 0.2 0.2 0.1 0.1\n", encoding="utf-8"
            )
    (root / "data.yaml").write_text(f"path: {root.as_posix()}\nnames:\n  0: rectangle\n", "utf-8")
    (root / "index.json").write_text("[]", encoding="utf-8")
    return root


def test_a_strike_puts_ink_inside_the_box():
    image = np.full((100, 100, 3), 240, np.uint8)
    negatives.strike_out(image, (20, 20, 80, 80), np.random.default_rng(0))
    assert image[20:80, 20:80].min() < 100


def test_a_strike_is_confined_to_the_box_corners_it_was_given():
    image = np.full((100, 100, 3), 240, np.uint8)
    negatives.strike_out(image, (40, 40, 60, 60), np.random.default_rng(1))
    assert image[0:35, 0:35].min() > 200


def test_a_doodle_marks_the_page():
    image = np.full((300, 300, 3), 240, np.uint8)
    negatives.doodle(image, np.random.default_rng(3))
    assert image.min() < 200


def test_the_build_leaves_val_and_test_untouched(export):
    negatives.build(export, export.parent / "hard", hard=None, seed=1)
    out = export.parent / "hard"
    for split in ("val", "test"):
        assert sorted(p.name for p in (out / "images" / split).glob("*.png")) == sorted(
            p.name for p in (export / "images" / split).glob("*.png")
        )


def test_the_build_adds_pages_to_train_only(export):
    result = negatives.build(export, export.parent / "hard", hard=None, seed=1)
    out = export.parent / "hard"
    assert result["train_images"] > len(list((export / "images" / "train").glob("*.png")))
    assert len(list((out / "images" / "val").glob("*.png"))) == 4


def test_a_struck_out_node_loses_its_label(export):
    negatives.build(export, export.parent / "hard", hard=None, seed=1)
    out = export.parent / "hard"
    hard_labels = list((out / "labels" / "train").glob("*__hard.txt"))
    assert hard_labels, "the seed should have struck at least one page"
    for label in hard_labels:
        lines = [ln for ln in label.read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert len(lines) == 1  # started with two, one was struck out


def test_a_mined_false_positive_becomes_an_empty_label_file(export):
    hard = [
        {
            "image": "hdbpmn__ptrain0",
            "cls": "circle",
            "xyxy": [40.0, 40.0, 90.0, 90.0],
            "score": 0.9,
        }
    ]
    result = negatives.build(export, export.parent / "hard", hard=hard, seed=2)
    out = export.parent / "hard"
    assert result["mined_negative_crops"] == 1
    (crop,) = list((out / "labels" / "train").glob("*__neg0.txt"))
    assert crop.read_text(encoding="utf-8") == ""


def test_the_yaml_points_at_the_augmented_root(export):
    negatives.build(export, export.parent / "hard", hard=None, seed=1)
    out = export.parent / "hard"
    assert out.as_posix() in (out / "data.yaml").read_text(encoding="utf-8")


def test_a_hard_negative_must_overlap_nothing_and_be_confident():
    assert negatives.BACKGROUND_IOU < 0.2
    assert negatives.CONFIDENT >= 0.5
