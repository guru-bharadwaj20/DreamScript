"""S2's headline artifact gate."""

from __future__ import annotations

import json

import pytest

from src.detect import s2


def artifact(tmp_path, *, map50=0.921, split="val"):
    path = tmp_path / "report.json"
    path.write_text(
        json.dumps(
            {
                "split": split,
                "pages": 308,
                "predictions": 39548,
                "map50": map50,
                "views": {
                    "pooled": {"map50": map50},
                    "hand_drawn_only": {"map50": 0.8907, "pages": 128},
                },
                "per_class": {"rectangle": {"ap50": 0.99}},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_s2_accepts_a_validation_artifact_that_clears_the_target(tmp_path):
    result = s2.evaluate_artifact(artifact(tmp_path))
    assert result["map50"] == 0.921
    assert result["passes"] is True


def test_s2_rejects_a_non_validation_artifact(tmp_path):
    with pytest.raises(ValueError, match="validation"):
        s2.evaluate_artifact(artifact(tmp_path, split="train"))


def test_s2_rejects_a_disagreed_pooled_headline(tmp_path):
    path = artifact(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["views"]["pooled"]["map50"] = 0.8
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="disagrees"):
        s2.evaluate_artifact(path)
