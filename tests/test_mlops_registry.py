"""Phase 15.3 - a registry is only worth having if its stages cannot lie.

The property that matters is not that versions get created: it is that a component which misses
its criterion cannot end up at `prod`, however convenient that would be, and that the stage
follows the artefact rather than whatever was typed in last time.
"""

from __future__ import annotations

import json

import pytest

from src.mlops import registry


@pytest.fixture()
def store(tmp_path):
    return f"file:{(tmp_path / 'mlruns').as_posix()}"


def test_meets_respects_the_direction():
    # higher-is-better
    assert registry.meets(0.91, 0.80, "higher") is True
    assert registry.meets(0.79, 0.80, "higher") is False
    # lower-is-better: the same numbers invert
    assert registry.meets(0.10, 0.15, "lower") is True
    assert registry.meets(0.26, 0.15, "lower") is False
    # the boundary is inclusive both ways - a criterion of ">= 0.80" is met by exactly 0.80
    assert registry.meets(0.80, 0.80, "higher") is True
    assert registry.meets(0.15, 0.15, "lower") is True


def test_a_missing_metric_is_unknown_not_a_failure():
    assert registry.meets(None, 0.80, "higher") is None


def test_missing_artefact_reads_as_none(tmp_path):
    assert registry._metric("reports/does_not_exist_at_all.json", "accuracy") is None


def test_a_component_that_misses_its_bar_is_never_prod(monkeypatch):
    """The row's whole point: `staging` exists so a miss is not promoted."""
    monkeypatch.setattr(
        registry,
        "SHIPPED",
        (
            {
                "name": "probe",
                "component": "ocr",
                "run": "probe_run",
                "artefact": "reports/probe.json",
                "metric": "cer",
                "criterion": 0.15,
                "direction": "lower",
                "bar": "probe bar",
            },
        ),
    )
    monkeypatch.setattr(registry, "SUPERSEDED", ())
    monkeypatch.setattr(registry, "_metric", lambda *_: 0.257)
    (row,) = registry.resolve()
    assert row["meets_criterion"] is False
    assert row["stage"] == "staging"
    assert registry.MLFLOW_STAGE[row["stage"]] == "Staging"


def test_a_component_that_meets_its_bar_is_prod(monkeypatch):
    monkeypatch.setattr(
        registry,
        "SHIPPED",
        (
            {
                "name": "probe",
                "component": "detection",
                "run": "probe_run",
                "artefact": "reports/probe.json",
                "metric": "map50",
                "criterion": 0.80,
                "direction": "higher",
                "bar": "probe bar",
            },
        ),
    )
    monkeypatch.setattr(registry, "SUPERSEDED", ())
    monkeypatch.setattr(registry, "_metric", lambda *_: 0.9107)
    (row,) = registry.resolve()
    assert row["meets_criterion"] is True
    assert row["stage"] == "prod"


def test_a_missing_artefact_lands_in_dev_rather_than_claiming_anything(monkeypatch):
    monkeypatch.setattr(
        registry,
        "SHIPPED",
        (
            {
                "name": "probe",
                "component": "parsing",
                "run": "probe_run",
                "artefact": "reports/absent.json",
                "metric": "macro_f1",
                "criterion": 0.80,
                "direction": "higher",
                "bar": "probe bar",
            },
        ),
    )
    monkeypatch.setattr(registry, "SUPERSEDED", ())
    (row,) = registry.resolve()
    assert row["value"] is None
    assert row["meets_criterion"] is None
    assert row["stage"] == "dev"


def test_superseded_candidates_never_rise_above_dev():
    rows = registry.resolve()
    for row in rows:
        if row["role"] == "superseded":
            assert row["stage"] == "dev"


def test_the_real_table_stages_every_shipped_component_on_its_own_evidence():
    """No hand-set stages: each shipped row's stage must follow from its own numbers."""
    for row in registry.resolve():
        if row["role"] != "shipped":
            continue
        if row["meets_criterion"] is True:
            assert row["stage"] == "prod", row
        elif row["meets_criterion"] is False:
            assert row["stage"] == "staging", row
        else:
            assert row["stage"] == "dev", row


def test_populate_is_idempotent_and_check_agrees_with_it(store):
    pytest.importorskip("mlflow")
    rows = [
        {
            "name": "probe-model",
            "component": "detection",
            "run": "probe_run",
            "artefact": "contributing.md",  # any real file; the registry stores a path
            "metric": "map50",
            "bar": "probe bar",
            "value": 0.9107,
            "criterion": 0.80,
            "direction": "higher",
            "meets_criterion": True,
            "stage": "prod",
            "stage_reason": "meets its criterion",
            "on_pipeline_path": True,
            "role": "shipped",
        }
    ]
    first = registry.populate(store, rows)
    assert not first["errors"], first["errors"]
    assert first["registered"][0]["mlflow_stage"] == "Production"

    second = registry.populate(store, rows)
    assert not second["errors"], second["errors"]
    # A second run adds a version rather than failing, and the newest is still Production.
    held = registry.listing(store)
    assert len(held) == 2
    assert held[-1]["stage"] == "Production"


def test_render_names_the_components_that_miss_their_bar():
    result = {
        "store": "file:probe",
        "versions": [
            {
                "component": "ocr",
                "run": "r",
                "metric": "cer",
                "value": 0.257,
                "criterion": 0.15,
                "meets_criterion": False,
                "stage": "staging",
                "on_pipeline_path": True,
                "role": "shipped",
                "stage_reason": "x",
            }
        ],
        "errors": [],
    }
    text = registry.render(result)
    assert "ocr" in text
    assert "`staging`" in text
    assert "miss their own bar" in text


def test_report_is_json_serialisable():
    json.dumps(registry.resolve(), default=str)
