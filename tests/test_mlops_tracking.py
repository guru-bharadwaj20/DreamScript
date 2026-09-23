"""Phase 15.1 - a tracking store is only useful if what is in it is what happened.

Two properties matter more than the plumbing: a backfilled run must be distinguishable from a
live one (otherwise the store quietly claims to have watched runs it only read about), and a
metric the artefact never carried must be absent rather than zero.
"""

from __future__ import annotations

import json

import pytest

from src.mlops import tracking


@pytest.fixture()
def store(tmp_path):
    return f"file:{(tmp_path / 'mlruns').as_posix()}"


def test_dig_walks_a_dotted_path():
    payload = {"overall": {"median_ged": 13.0}}
    assert tracking._dig(payload, "overall.median_ged") == 13.0
    assert tracking._dig(payload, "overall.absent") is None
    assert tracking._dig(payload, "nothing.here") is None


def test_a_metric_that_is_none_is_not_logged_as_zero():
    recorder = tracking.Recorder()
    recorder.log_metric("cer", None)
    recorder.log_metric("wer", 0.0)
    assert "cer" not in recorder.metrics
    assert recorder.metrics["wer"] == 0.0


def test_a_param_that_is_none_is_not_logged():
    recorder = tracking.Recorder()
    recorder.log_param("checkpoint", None)
    recorder.log_param("split", "val")
    assert recorder.params == {"split": "val"}


def test_an_artifact_that_does_not_exist_is_not_logged(tmp_path):
    recorder = tracking.Recorder()
    recorder.log_artifact(tmp_path / "absent.json")
    assert recorder.artifacts == []


def test_track_without_mlflow_is_a_recorder_and_never_raises(monkeypatch):
    monkeypatch.setattr(tracking, "available", lambda: False)
    with tracking.track("detect", "run", params={"epochs": 40}) as run:
        run.log_metric("map50", 0.91)
    assert run.metrics == {"map50": 0.91}
    assert run.params == {"epochs": 40}
    assert run.run_id is None


@pytest.mark.skipif(not tracking.available(), reason="mlflow is not installed")
def test_a_tracked_run_lands_in_the_store(store, tmp_path):
    artefact = tmp_path / "report.json"
    artefact.write_text(json.dumps({"cer": 0.2}), encoding="utf-8")
    with tracking.track("ocr", "a_run", params={"model": "trocr"}, store_uri=store) as run:
        run.log_metric("cer", 0.2006)
        run.log_artifact(artefact)

    rows = tracking.listing(store)
    assert [row["run"] for row in rows] == ["a_run"]
    assert rows[0]["metrics"]["cer"] == pytest.approx(0.2006)
    assert rows[0]["status"] == "FINISHED"
    assert rows[0]["backfilled"] is False


@pytest.mark.skipif(not tracking.available(), reason="mlflow is not installed")
def test_a_failing_run_is_recorded_as_failed_rather_than_lost(store):
    with (
        pytest.raises(ValueError),
        tracking.track("ocr", "bad_run", store_uri=store) as run,
    ):
        run.log_metric("cer", 0.9)
        raise ValueError("training diverged")

    rows = tracking.listing(store)
    assert rows[0]["status"] == "FAILED"
    assert rows[0]["metrics"]["cer"] == pytest.approx(0.9)


@pytest.mark.skipif(not tracking.available(), reason="mlflow is not installed")
def test_backfilled_runs_are_tagged_and_carry_their_source(store, tmp_path, monkeypatch):
    artefact = tmp_path / "s3.json"
    artefact.write_text(json.dumps({"cer": 0.2006, "model": "trocr_large"}), encoding="utf-8")
    monkeypatch.setattr(tracking, "ROOT", tmp_path)
    rows = (("ocr", "s3", "s3.json", ("cer", "absent_metric"), ("model",)),)

    result = tracking.backfill(store, rows)

    assert result["runs_created"] == 1
    assert result["runs"][0]["metrics_logged"] == 1
    assert result["runs"][0]["metrics_absent"] == 1  # the missing metric is counted, not zeroed
    listed = tracking.listing(store)[0]
    assert listed["backfilled"] is True
    assert listed["source_file"] == "s3.json"
    assert "absent_metric" not in listed["metrics"]


def test_backfill_reports_an_artefact_it_could_not_find(store, tmp_path, monkeypatch):
    monkeypatch.setattr(tracking, "ROOT", tmp_path)
    result = tracking.backfill(store, (("ocr", "gone", "nowhere.json", ("cer",), ()),))
    assert result["runs_created"] == 0
    assert result["artefacts_missing"] == ["nowhere.json"]


def test_the_backfill_registry_covers_every_stage_the_plan_names():
    experiments = {row[0] for row in tracking.BACKFILL}
    for stage in ("detection", "classification", "parsing", "generation"):
        assert stage in experiments
