"""Phase 14.4 - an ablation row is a claim about two arms and about the pipeline's real shape.

Both are testable without a GPU: the delta must be the subtraction it says it is (including in
the direction that matters for an error rate), and `on_deployed_path` must come from the
pipeline's own source rather than from the plan's intentions - a marker that only appears in a
docstring must not count as a call site.
"""

from __future__ import annotations

import json

import pytest

from src.eval import ablate


def test_every_component_names_what_it_was_replaced_by():
    result = ablate.collect(fit=False)
    assert len(result["rows"]) == 5
    for row in result["rows"]:
        assert row["replaced_by"], row["component"]
        assert row["metric"], row["component"]
        assert "on_deployed_path" in row


def test_path_detection_reads_the_pipeline_package(tmp_path, monkeypatch):
    package = tmp_path / "pipeline"
    package.mkdir()
    (package / "core.py").write_text('"""mentions an adapter in prose only."""\n', encoding="utf-8")
    (package / "generate.py").write_text("from src.pipeline.generate import x\n", encoding="utf-8")
    monkeypatch.setattr(ablate, "PIPELINE_PACKAGE", package)

    lora = ablate.on_deployed_path("lora")
    assert lora["on_deployed_path"] is True
    assert any("generate.py" in hit for hit in lora["evidence"])

    rl = ablate.on_deployed_path("rl")
    assert rl["on_deployed_path"] is False
    assert "no rl call site" in rl["evidence"]


def test_a_prose_only_mention_is_not_a_call_site(tmp_path, monkeypatch):
    package = tmp_path / "pipeline"
    package.mkdir()
    (package / "core.py").write_text(
        '"""stage: the reinforcement learning agent of src.rl is not used here."""\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(ablate, "PIPELINE_PACKAGE", package)
    assert ablate.on_deployed_path("rl")["on_deployed_path"] is False


def test_the_rl_row_subtracts_the_heuristic_from_the_learned_arm():
    row = ablate.rl_row()
    if row.get("delta") is None:
        pytest.skip("reports/rl_ablation.md is not present in this tree")
    assert row["delta"] == pytest.approx(row["with_component"] - row["without_component"], abs=1e-9)


def test_the_lora_row_compares_against_the_same_base_model():
    row = ablate.lora_row()
    if row.get("delta") is None:
        pytest.skip("the Phase 12 artefacts are not present in this tree")
    assert row["delta"] == pytest.approx(row["with_component"] - row["without_component"], abs=1e-9)
    assert row["delta"] > 0
    assert "zero-shot" in row["replaced_by"]


def test_the_style_row_is_declared_as_an_error_rate():
    row = ablate.style_row()
    assert row["lower_is_better"] is True
    if row.get("delta") is None:
        pytest.skip("reports/ocr.md is not present in this tree")
    # A negative delta on an error rate means the component helped; the sign must not be flipped
    # silently to make the table look uniform.
    assert row["delta"] == pytest.approx(row["with_component"] - row["without_component"], abs=1e-9)


def test_render_marks_which_rows_are_off_the_path():
    result = ablate.collect(fit=False)
    text = ablate.render(result)
    assert "on the deployed path" in text
    assert text.count("| no |") + text.count("| yes |") == 5


def test_a_missing_artefact_leaves_the_row_without_a_delta(monkeypatch, tmp_path):
    monkeypatch.setattr(ablate, "ROOT", tmp_path)
    row = ablate.lora_row()
    assert row.get("delta") is None
    assert "absent" in row["note"]


def test_the_report_is_valid_json_and_names_every_component():
    path = ablate.ROOT / "reports" / "ablation_matrix.json"
    if not path.is_file():
        pytest.skip("run python -m src.eval.ablate first")
    result = json.loads(path.read_text(encoding="utf-8"))
    names = " ".join(row["component"] for row in result["rows"]).lower()
    for component in ("hmm", "gmm", "rl", "lora", "style"):
        assert component in names
