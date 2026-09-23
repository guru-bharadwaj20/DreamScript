"""Phase 14.11 - a compute table is only honest if the gaps are visible in the total.

The failure mode is a tidy sum that silently omits every run whose artefact was lost. These
tests pin that a missing duration stays missing (never zero, never guessed), that the arithmetic
of hours to kWh to dollars follows the printed parameters, and that the report says out loud
that its total is a lower bound.
"""

from __future__ import annotations

import json

from src.eval import compute


def test_a_missing_artefact_is_not_read_as_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(compute, "ROOT", tmp_path)
    assert compute._read("nowhere.json", ("seconds",)) is None


def test_a_missing_key_is_not_read_as_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(compute, "ROOT", tmp_path)
    (tmp_path / "x.json").write_text(json.dumps({"other": 5}), encoding="utf-8")
    assert compute._read("x.json", ("seconds",)) is None


def test_a_duration_is_read_from_a_key_path(tmp_path, monkeypatch):
    monkeypatch.setattr(compute, "ROOT", tmp_path)
    (tmp_path / "x.json").write_text(json.dumps({"a": {"seconds": 7200}}), encoding="utf-8")
    assert compute._read("x.json", ("a", "seconds")) == 7200.0


def _tree(tmp_path, seconds):
    (tmp_path / "one.json").write_text(json.dumps({"seconds": seconds}), encoding="utf-8")
    return (
        ("9 - detection", "a run", "one.json", ("seconds",)),
        ("9 - OCR", "lost", "gone.json", ("seconds",)),
    )


def test_energy_and_cost_follow_the_printed_parameters(tmp_path, monkeypatch):
    monkeypatch.setattr(compute, "ROOT", tmp_path)
    monkeypatch.setattr(compute, "SOURCES", _tree(tmp_path, 3600))
    result = compute.collect(rate=2.0, utilisation=0.5, pue=1.0, watts=1000.0)
    assert result["totals"]["gpu_hours"] == 1.0
    assert result["totals"]["kwh"] == 0.5  # 1 h x 1000 W x 0.5 x 1.0 / 1000
    assert result["totals"]["usd"] == 2.0
    assert result["model"]["utilisation"] == 0.5


def test_runs_without_a_duration_are_counted_and_excluded(tmp_path, monkeypatch):
    monkeypatch.setattr(compute, "ROOT", tmp_path)
    monkeypatch.setattr(compute, "SOURCES", _tree(tmp_path, 7200))
    result = compute.collect()
    assert result["totals"]["runs_recorded"] == 1
    assert result["totals"]["runs_missing_a_duration"] == 1
    assert result["totals"]["gpu_hours"] == 2.0
    lost = [run for run in result["runs"] if not run["recorded"]][0]
    assert lost["gpu_hours"] is None


def test_the_total_declares_itself_a_lower_bound(tmp_path, monkeypatch):
    monkeypatch.setattr(compute, "ROOT", tmp_path)
    monkeypatch.setattr(compute, "SOURCES", _tree(tmp_path, 3600))
    result = compute.collect()
    assert result["totals"]["is_lower_bound"] is True
    text = compute.render(result)
    assert "lower bound" in text


def test_the_report_lists_the_phases_it_cannot_account_for(tmp_path, monkeypatch):
    monkeypatch.setattr(compute, "ROOT", tmp_path)
    monkeypatch.setattr(compute, "SOURCES", _tree(tmp_path, 3600))
    text = compute.render(compute.collect())
    assert "cannot account for" in text
    for phase, _ in compute.UNRECORDED:
        assert phase in text


def test_the_energy_sensitivity_scales_with_utilisation(tmp_path, monkeypatch):
    monkeypatch.setattr(compute, "ROOT", tmp_path)
    monkeypatch.setattr(compute, "SOURCES", _tree(tmp_path, 3600))
    arms = compute.sensitivity(compute.collect())
    values = [arm["kwh"] for arm in arms]
    assert values == sorted(values)
    assert arms[-1]["utilisation"] == 1.0


def test_cost_is_labelled_as_a_rental_equivalent(tmp_path, monkeypatch):
    monkeypatch.setattr(compute, "ROOT", tmp_path)
    monkeypatch.setattr(compute, "SOURCES", _tree(tmp_path, 3600))
    text = compute.render(compute.collect())
    assert "rental equivalent" in text
    assert "modelled, not metered" in text


def test_every_declared_source_names_a_file_and_a_phase():
    for phase, label, path, keys in compute.SOURCES:
        assert phase and label and path.endswith(".json") and keys
