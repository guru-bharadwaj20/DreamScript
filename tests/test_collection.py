"""Phase 1.2 acceptance tests — the collection protocol, registry and capture tool.

The corpus itself cannot be tested until it exists, so these test the machinery that will
accept it: the assignment satisfies every plan target, filenames round-trip to metadata, and
the capture tool refuses everything that would corrupt the corpus.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.ingest import capture, scribes
from src.ingest.collection import (
    ADVERSE_CONDITIONS,
    CONDITIONS,
    DIAGRAM_TYPES,
    MEDIA,
    MIN_ADVERSE_FRACTION,
    MIN_SCRIBES,
    SCENARIOS,
    TARGETS,
    assignments,
    parse_filename,
    progress,
)


def test_targets_match_the_plan():
    assert TARGETS == {
        "flowchart": 60,
        "wireframe": 60,
        "state_machine": 50,
        "er_diagram": 50,
        "circuit": 40,
    }
    assert sum(TARGETS.values()) == 260


def test_every_type_has_scenarios():
    for dtype in DIAGRAM_TYPES:
        assert len(SCENARIOS[dtype]) >= 10, dtype
        slugs = [s.slug for s in SCENARIOS[dtype]]
        assert len(slugs) == len(set(slugs)), f"duplicate scenario slug in {dtype}"
        for s in SCENARIOS[dtype]:
            assert s.brief and s.must_contain, f"{s.slug} is underspecified"


def test_assignment_hits_every_target():
    rows = assignments()
    assert len(rows) == 260
    for dtype, target in TARGETS.items():
        assert sum(1 for r in rows if r["diagram_type"] == dtype) == target


def test_assignment_spreads_across_scribes():
    """No diagram type may be drawn by fewer than the minimum number of people."""
    rows = assignments()
    for dtype in DIAGRAM_TYPES:
        scribes_for_type = {r["scribe_id"] for r in rows if r["diagram_type"] == dtype}
        assert len(scribes_for_type) >= MIN_SCRIBES, dtype


def test_assignment_uses_every_medium():
    assert {r["medium"] for r in assignments()} == set(MEDIA)


def test_assignment_meets_the_adverse_target():
    rows = assignments()
    adverse = sum(1 for r in rows if r["adverse"])
    assert adverse / len(rows) >= MIN_ADVERSE_FRACTION


def test_assignment_spreads_adverse_conditions():
    """Every adverse condition must appear, or a mitigation goes untested."""
    used = {r["condition"] for r in assignments() if r["adverse"]}
    assert used == set(ADVERSE_CONDITIONS)


def test_filenames_round_trip(tmp_path: Path):
    for row in assignments()[:20]:
        p = tmp_path / row["diagram_type"] / row["scribe_id"] / row["filename"]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
        meta = parse_filename(p)
        assert meta is not None, row["filename"]
        assert meta["diagram_type"] == row["diagram_type"]
        assert meta["scribe_id"] == row["scribe_id"]
        assert meta["scenario"] == row["scenario"]
        assert meta["medium"] == row["medium"]
        assert meta["condition"] == row["condition"]
        assert meta["adverse"] == row["adverse"]


def test_malformed_filenames_are_rejected(tmp_path: Path):
    bad = ["nometadata.jpg", "a__b.jpg", "scenario__pencil__clean.txt", "UPPER__pencil__clean.jpg"]
    for name in bad:
        p = tmp_path / "flowchart" / "scribe01" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
        assert parse_filename(p) is None, name


def test_progress_reports_pending_on_an_empty_corpus():
    r = progress()
    assert r["total_target"] == 260
    assert set(r["checks"]) == {
        "1.2.1_flowcharts",
        "1.2.2_wireframes",
        "1.2.3_state_machines",
        "1.2.4_er_diagrams",
        "1.2.5_circuits",
        "1.2.6_min_scribes",
        "1.2.7_adverse_fraction",
        "1.2.8_all_media",
    }


# --- capture tool -----------------------------------------------------------------------


def _sketch(blur: int = 0, scale: float = 1.0) -> np.ndarray:
    img = np.full((240, 320, 3), 235, np.uint8)
    cv2.rectangle(img, (30, 60), (150, 130), (40, 40, 40), 3)
    cv2.arrowedLine(img, (152, 95), (250, 95), (40, 40, 40), 3)
    if blur:
        img = cv2.GaussianBlur(img, (blur, blur), 0)
    if scale != 1.0:
        img = np.clip(img.astype(np.float32) * scale, 0, 255).astype(np.uint8)
    return img


def test_quality_accepts_a_normal_sketch():
    q = capture.quality(_sketch())
    assert q["usable"], q["reasons"]


def test_quality_rejects_a_smear():
    q = capture.quality(_sketch(blur=31))
    assert not q["usable"]
    assert any("focus" in r for r in q["reasons"])


def test_quality_rejects_a_black_frame():
    q = capture.quality(np.zeros((240, 320, 3), np.uint8))
    assert not q["usable"]


@pytest.mark.parametrize(
    "dtype,scenario,medium,condition",
    [
        ("flowchart", "nope", "pencil", "clean"),
        ("not_a_type", "login_check", "pencil", "clean"),
        ("flowchart", "login_check", "crayon", "clean"),
        ("flowchart", "login_check", "pencil", "underwater"),
    ],
)
def test_capture_rejects_bad_metadata(dtype, scenario, medium, condition):
    assert capture.validate_metadata(dtype, scenario, medium, condition)


def test_capture_accepts_valid_metadata():
    assert capture.validate_metadata("flowchart", "login_check", "pencil", "shadow") == []


def test_capture_files_an_image_with_metadata_in_the_path(tmp_path, monkeypatch):
    monkeypatch.setattr(capture, "CHAOS", tmp_path)
    src = tmp_path / "photo.jpg"
    cv2.imwrite(str(src), _sketch())

    out = capture.capture("flowchart", "scribe02", "age_gate", "ballpoint", "shadow", source=src)
    assert out is not None and out.is_file()
    assert out.parent.name == "scribe02"
    assert out.parent.parent.name == "flowchart"
    assert out.name.startswith("age_gate__ballpoint__shadow__")


def test_capture_never_overwrites(tmp_path, monkeypatch):
    monkeypatch.setattr(capture, "CHAOS", tmp_path)
    src = tmp_path / "photo.jpg"
    cv2.imwrite(str(src), _sketch())
    a = capture.capture("circuit", "scribe01", "rc_filter", "pencil", "clean", source=src)
    b = capture.capture("circuit", "scribe01", "rc_filter", "pencil", "clean", source=src)
    assert a != b and a.is_file() and b.is_file()


# --- scribe registry --------------------------------------------------------------------


def test_scribe_template_covers_the_style_range():
    rows = scribes.template_rows()
    assert len(rows) >= MIN_SCRIBES
    styles = {r.style for r in rows}
    assert styles == set(scribes.STYLES), "the template must span neat, average and messy"
    # The self-collection template must state handedness for real; `unknown` exists only for
    # writers sourced from public datasets, which never publish it.
    assert {r.handedness for r in rows} == {"left", "right"}
    assert all(r.origin == "self" for r in rows)


def test_scribe_validation_catches_bad_rows():
    bad = scribes.Scribe("nope01", "beautiful", "sideways", "crayon", "maybe")
    problems = bad.validate()
    # id, style, handedness, medium and consent are all wrong: five problems.
    assert len(problems) == 5


def test_scribe_without_consent_is_invalid():
    s = scribes.Scribe("scribe01", "neat", "right", "pencil", consent="")
    assert any("consent" in p for p in s.validate())


def test_scribe_report_matches_the_corpus_on_disk():
    """Whatever is collected, every drawing must belong to a registered, consenting writer."""
    r = scribes.report()
    assert r["checks"]["no_unregistered_drawings"] is True
    assert r["checks"]["no_validation_problems"] is True
    # The registry and the corpus must agree about who exists.
    assert set(r["scribes_with_drawings"]) <= {s.scribe_id for s in scribes.load()}


def test_condition_vocabulary_is_closed():
    assert "clean" in CONDITIONS
    assert set(ADVERSE_CONDITIONS) == set(CONDITIONS) - {"clean"}
    assert len(ADVERSE_CONDITIONS) >= 6
