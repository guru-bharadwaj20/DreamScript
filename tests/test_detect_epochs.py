"""Phase 9.1.3 - the epoch axis, assembled from runs the phase already paid for."""

from __future__ import annotations

import json

from src.detect import train


def test_the_points_are_sorted_and_labelled_by_where_they_came_from(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "RUNS", tmp_path)
    result = {
        "sweep": {
            "epochs": 10,
            "rows": [
                {"imgsz": 896, "amp": True, "map50": 0.90, "train_seconds": 600.0},
                {"imgsz": 640, "amp": True, "map50": 0.85, "train_seconds": 400.0},
            ],
        },
        "final": {"epochs": 40, "map50": 0.93, "train_seconds": 2400.0},
    }
    effect = train._epochs_effect(result, 896)
    assert [p["epochs"] for p in effect["points"]] == [10, 40]
    assert effect["points"][0]["from"] == "9.1.3 sweep"
    assert effect["map50_gain_over_range"] == 0.03


def test_the_bakeoff_arm_is_picked_up_when_it_shares_the_input_size(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "RUNS", tmp_path)
    (tmp_path / "choice.json").write_text(
        json.dumps(
            [
                {
                    "arm": "yolov8n",
                    "imgsz": 896,
                    "epochs": 12,
                    "map50": 0.8913,
                    "train_seconds": 752.2,
                },
                {
                    "arm": "rtdetr-l",
                    "imgsz": 896,
                    "epochs": 12,
                    "map50": 0.9055,
                    "train_seconds": 4338.0,
                },
            ]
        ),
        encoding="utf-8",
    )
    result = {"final": {"epochs": 40, "map50": 0.93, "train_seconds": 2400.0}}
    effect = train._epochs_effect(result, 896)
    # only the yolov8n arm, because the final run is a yolov8n and the others are not comparable
    assert [p["from"] for p in effect["points"]] == ["9.1.1 bake-off", "9.1.3 final"]


def test_a_bakeoff_arm_at_a_different_size_is_not_mixed_in(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "RUNS", tmp_path)
    (tmp_path / "choice.json").write_text(
        json.dumps(
            [{"arm": "yolov8n", "imgsz": 640, "epochs": 12, "map50": 0.85, "train_seconds": 400.0}]
        ),
        encoding="utf-8",
    )
    result = {"final": {"epochs": 40, "map50": 0.93, "train_seconds": 2400.0}}
    assert len(train._epochs_effect(result, 1280)["points"]) == 1


def test_the_note_says_it_is_not_a_controlled_sweep(tmp_path, monkeypatch):
    """It is three runs that differ in more than epochs, and the write-up must not pretend."""
    monkeypatch.setattr(train, "RUNS", tmp_path)
    result = {"final": {"epochs": 40, "map50": 0.93, "train_seconds": 2400.0}}
    assert "not a controlled sweep" in train._epochs_effect(result, 896)["note"]


def test_a_single_point_reports_no_gain(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "RUNS", tmp_path)
    result = {"final": {"epochs": 40, "map50": 0.93, "train_seconds": 2400.0}}
    assert train._epochs_effect(result, 896)["map50_gain_over_range"] is None
