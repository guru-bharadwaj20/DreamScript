"""Phase 15.4 - the DAG is a claim about the project, so the file is checked against the project.

A `dvc.yaml` that names a module which does not exist, or an output nothing writes, is worse
than no DAG: it looks like a specification and reproduces nothing. These tests read the real
file and check it against the real tree - every dependency exists or is itself produced by a
stage, every command names an importable module, and every expensive stage is frozen.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.utils.config import ROOT

DVC_YAML = ROOT / "dvc.yaml"

#: Stages whose cost is GPU-hours: 14.11 prices the LLM sweeps at 10.8 h and the arrows at 3.4.
MUST_BE_FROZEN = {"detector", "arrows", "ocr", "lora"}


def outs(stage) -> list[str]:
    """An out is either a path or a one-key mapping of path to options (`cache: false`)."""
    return [next(iter(o)) if isinstance(o, dict) else o for o in stage["outs"]]


@pytest.fixture(scope="module")
def pipeline():
    return yaml.safe_load(DVC_YAML.read_text(encoding="utf-8"))["stages"]


def test_the_dag_file_exists_and_parses(pipeline):
    assert pipeline
    assert "evaluate" in pipeline


def test_every_stage_declares_deps_and_outs(pipeline):
    for name, stage in pipeline.items():
        assert stage.get("cmd"), name
        assert stage.get("deps"), name
        assert stage.get("outs"), name
        assert stage.get("desc"), f"{name} has no description"


def test_every_command_names_a_module_that_exists(pipeline):
    for name, stage in pipeline.items():
        parts = stage["cmd"].split()
        # The interpreter is the project venv, not a bare `python`: `dvc repro` does not
        # activate the environment, and a bare `python` finds the Windows Store stub here.
        assert parts[:2] == ["python", "-m"], name
        module = parts[2]
        path = ROOT / Path(module.replace(".", "/") + ".py")
        package = ROOT / Path(module.replace(".", "/")) / "__init__.py"
        assert path.is_file() or package.is_file(), f"{name}: {module} does not exist"


def test_every_dependency_is_a_file_or_another_stages_output(pipeline):
    produced = {out for stage in pipeline.values() for out in outs(stage)}
    for name, stage in pipeline.items():
        for dep in stage["deps"]:
            if dep in produced:
                continue
            assert (ROOT / dep).exists(), f"{name} depends on missing {dep}"


def test_the_expensive_stages_are_frozen(pipeline):
    for name in MUST_BE_FROZEN:
        assert pipeline[name].get("frozen") is True, f"{name} would re-run on a bare dvc repro"


def test_the_cheap_stages_are_not_frozen(pipeline):
    for name in ("features", "stagewise", "ablation", "evaluate"):
        assert not pipeline[name].get("frozen"), f"{name} should really run"


def test_no_output_is_claimed_by_two_stages(pipeline):
    seen: dict[str, str] = {}
    for name, stage in pipeline.items():
        for out in outs(stage):
            assert out not in seen, f"{out} is claimed by {seen[out]} and {name}"
            seen[out] = name


def test_no_stage_output_is_also_a_standalone_dvc_pointer(pipeline):
    """`dvc add` and a stage cannot both own a path - DVC refuses to build the graph at all."""
    for stage in pipeline.values():
        for out in outs(stage):
            pointer = ROOT / (out + ".dvc")
            assert not pointer.is_file(), f"{out} is both a stage output and a dvc add pointer"


def test_no_command_hardcodes_an_interpreter_path(pipeline):
    """A path like `.venv/Scripts/python.exe` runs here and fails on Linux; the DAG stays portable
    and the file states that the environment must be active instead."""
    for name, stage in pipeline.items():
        assert not stage["cmd"].startswith("."), name
        assert ".venv" not in stage["cmd"], name


def test_the_file_says_the_environment_must_be_active():
    text = DVC_YAML.read_text(encoding="utf-8")
    assert "project environment active" in text


def test_the_evaluation_stage_depends_on_the_reports_it_reads(pipeline):
    deps = set(pipeline["evaluate"]["deps"])
    assert "reports/stagewise.json" in deps
    assert "reports/ablation_matrix.json" in deps


def test_the_lock_file_records_what_ran():
    lock = ROOT / "dvc.lock"
    if not lock.is_file():
        pytest.skip("dvc repro has not run in this tree")
    stages = yaml.safe_load(lock.read_text(encoding="utf-8"))["stages"]
    assert "evaluate" in stages
    assert stages["evaluate"]["outs"]


def test_the_report_outputs_are_left_to_git_rather_than_the_dvc_cache():
    """A path cannot be owned by git and by DVC's cache at once - DVC refuses to build the graph.
    The reports are the project's evidence and belong in history, so they are `cache: false`."""
    pipeline = yaml.safe_load(DVC_YAML.read_text(encoding="utf-8"))["stages"]
    for name, stage in pipeline.items():
        for out in stage["outs"]:
            path = next(iter(out)) if isinstance(out, dict) else out
            if path.startswith("reports/"):
                assert isinstance(out, dict), f"{name}: {path} would collide with git"
                assert out[path]["cache"] is False, path
            else:
                assert not isinstance(out, dict) or out[path].get("cache") is not False, path
