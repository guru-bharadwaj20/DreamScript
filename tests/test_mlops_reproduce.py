"""Phase 15.12 - the reproducibility check, and the incident it was written for.

The shadowing check is the part that matters and the part that is hardest to trust, because on a
healthy machine it finds nothing and a broken implementation looks identical to a working one. So
it is tested against a reconstruction of the real incident rather than only against the current
environment.
"""

from __future__ import annotations

import json

import pytest

from src.mlops import reproduce as R


def test_a_local_build_is_not_drift():
    """`torch==2.5.1` installed as `2.5.1+cu124` is that exact version built for CUDA."""
    assert R._public("2.5.1+cu124") == "2.5.1"
    assert R._public("2.5.1") == "2.5.1"
    assert R._public("0.20.1+cu124") == R._public("0.20.1")


def test_pins_are_read_from_every_requirements_layer():
    pins = R._pins()
    assert pins, "no pins parsed at all"
    # Spread across layers: base, cv and genai each contribute.
    assert "numpy" in pins
    assert "opencv-contrib-python" in pins
    assert pins["opencv-contrib-python"] == "4.10.0.84"


def test_a_commented_pin_is_not_a_pin(tmp_path, monkeypatch):
    (tmp_path / "x.txt").write_text(
        "# torch==9.9.9 is not installed\nreal-package==1.2.3\nother==4.5  # trailing\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(R, "REQUIREMENTS", tmp_path)
    pins = R._pins()
    assert "torch" not in pins
    assert pins["real-package"] == "1.2.3"
    assert pins["other"] == "4.5"


def test_the_shadowing_check_catches_the_incident_it_was_written_for(monkeypatch):
    """Three OpenCV distributions at once, the undeclared 5.0 shadowing the pinned 4.10.

    This is the state the machine was actually in. Every pin was satisfied - the audit's version
    comparison would have said nothing - while `import cv2` resolved to a distribution that
    appears in no requirements file and whose HoughLinesP and MSER behave differently.
    """
    monkeypatch.setattr(R, "_pins", lambda: {"opencv-contrib-python": "4.10.0.84"})
    monkeypatch.setattr(
        R,
        "_installed",
        lambda: {
            "opencv-contrib-python": "4.10.0.84",
            "opencv-python": "5.0.0.93",
            "opencv-python-headless": "5.0.0.93",
        },
    )
    audit = R.audit_environment()

    # The version comparison alone sees nothing wrong - which is the whole point.
    assert audit["mismatched"] == []
    # The shadowing check is what fails it.
    assert audit["ok"] is False
    assert len(audit["shadowed_imports"]) == 1
    row = audit["shadowed_imports"][0]
    assert row["import"] == "cv2"
    assert set(row["undeclared"]) == {"opencv-python", "opencv-python-headless"}
    assert row["pinned"] == ["opencv-contrib-python"]


def test_one_distribution_per_import_is_not_shadowed(monkeypatch):
    monkeypatch.setattr(R, "_pins", lambda: {"opencv-contrib-python": "4.10.0.84"})
    monkeypatch.setattr(R, "_installed", lambda: {"opencv-contrib-python": "4.10.0.84"})
    audit = R.audit_environment()
    assert audit["shadowed_imports"] == []
    assert audit["ok"] is True


def test_a_version_mismatch_fails_the_audit(monkeypatch):
    monkeypatch.setattr(R, "_pins", lambda: {"numpy": "2.1.3"})
    monkeypatch.setattr(R, "_installed", lambda: {"numpy": "2.4.6"})
    audit = R.audit_environment()
    assert audit["ok"] is False
    assert audit["mismatched"][0]["installed"] == "2.4.6"


def test_a_local_build_does_not_fail_the_audit(monkeypatch):
    monkeypatch.setattr(R, "_pins", lambda: {"torch": "2.5.1"})
    monkeypatch.setattr(R, "_installed", lambda: {"torch": "2.5.1+cu124"})
    audit = R.audit_environment()
    assert audit["mismatched"] == []
    assert audit["local_builds"][0]["installed"] == "2.5.1+cu124"
    assert audit["ok"] is True


def test_a_pin_that_is_not_installed_is_reported_but_is_not_drift(monkeypatch):
    """An uninstalled optional layer is a different thing from a wrong version."""
    monkeypatch.setattr(R, "_pins", lambda: {"absent-package": "1.0"})
    monkeypatch.setattr(R, "_installed", lambda: {})
    audit = R.audit_environment()
    assert audit["missing"][0]["package"] == "absent-package"
    assert audit["mismatched"] == []


def test_metrics_are_compared_against_the_recorded_values():
    result = R.check_metrics()
    assert result["checked"], "no metrics checked"
    for row in result["checked"]:
        assert row["status"] in ("matches", "DIVERGES", "artefact missing", "not a number") or row[
            "status"
        ].startswith("unreadable")


def test_a_diverged_metric_is_reported_as_diverged(monkeypatch):
    monkeypatch.setattr(
        R,
        "CHEAP_METRICS",
        (
            {
                "name": "probe",
                "artefact": "reports/s1_heldout_scribes.json",
                "key": "accuracy",
                "expected": 0.1,
                "tolerance": 0.0005,
            },
        ),
    )
    result = R.check_metrics()
    if result["checked"][0]["status"] == "artefact missing":
        pytest.skip("needs reports/s1_heldout_scribes.json")
    assert result["checked"][0]["status"] == "DIVERGES"
    assert result["ok"] is False


def test_a_missing_artefact_is_unavailable_not_a_match(monkeypatch):
    monkeypatch.setattr(
        R,
        "CHEAP_METRICS",
        (
            {
                "name": "probe",
                "artefact": "reports/definitely_absent.json",
                "key": "x",
                "expected": 1.0,
                "tolerance": 0.1,
            },
        ),
    )
    result = R.check_metrics()
    assert result["checked"][0]["status"] == "artefact missing"
    assert result["ok"] is False


def test_what_is_not_attempted_is_named():
    """A partial pass reported as a pass is the failure this row exists to avoid."""
    result = R.collect(env_only=True)
    assert result["not_attempted"]
    for row in result["not_attempted"]:
        assert row["criterion"] and row["why"]


def test_env_only_does_not_claim_the_metrics_were_checked():
    result = R.collect(env_only=True)
    assert result["verdict"]["metrics_checked"] is False
    assert "reproducible" not in result["verdict"]


def test_render_names_the_shadowed_import(monkeypatch):
    monkeypatch.setattr(R, "_pins", lambda: {"opencv-contrib-python": "4.10.0.84"})
    monkeypatch.setattr(
        R,
        "_installed",
        lambda: {"opencv-contrib-python": "4.10.0.84", "opencv-python": "5.0.0.93"},
    )
    text = R.render(R.collect(env_only=True))
    assert "Shadowed imports" in text
    assert "opencv-python" in text


def test_report_is_json_serialisable():
    json.dumps(R.collect(), default=str)
