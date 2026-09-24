"""Phase 0.2.4 acceptance test — run directories, log sinks and metrics."""

from __future__ import annotations

import json
from pathlib import Path

from src.utils.config import load_config
from src.utils.logging import start_run


def test_run_directory_is_created_and_named(tmp_path: Path):
    run = start_run(name="p0-test", root=tmp_path)
    try:
        assert run.dir.parent == tmp_path
        assert run.dir.name.endswith("_p0-test")
        stamp = run.dir.name.split("_")[0]
        assert len(stamp) == 15 and stamp[8] == "-"  # YYYYmmdd-HHMMSS
    finally:
        run.finish()


def test_two_runs_never_collide(tmp_path: Path):
    a = start_run(name="same", root=tmp_path)
    b = start_run(name="same", root=tmp_path)
    try:
        assert a.dir != b.dir
    finally:
        a.finish()
        b.finish()


def test_log_and_jsonl_sinks_both_receive(tmp_path: Path):
    run = start_run(name="sinks", root=tmp_path)
    run.log.info("hello %s", "world")
    run.finish()

    text = (run.dir / "run.log").read_text(encoding="utf-8")
    assert "hello world" in text

    records = [json.loads(line) for line in (run.dir / "run.jsonl").read_text().splitlines()]
    assert any(r["message"] == "hello world" and r["level"] == "INFO" for r in records)


def test_metrics_accumulate_and_persist(tmp_path: Path):
    run = start_run(name="metrics", root=tmp_path)
    run.log_metrics({"acc": 0.5}, step=1)
    run.log_metrics({"acc": 0.9}, step=2)
    run.log_metrics({"final_acc": 0.9})
    run.finish("ok")

    metrics = json.loads((run.dir / "metrics.json").read_text(encoding="utf-8"))
    assert [s["acc"] for s in metrics["steps"]] == [0.5, 0.9]
    assert metrics["final_acc"] == 0.9
    assert metrics["status"] == "ok"


def test_config_and_env_are_captured(tmp_path: Path):
    cfg = load_config("base", ["logging.run_name=captured", "seed=7"])
    run = start_run(cfg, root=tmp_path)
    run.finish()

    saved = (run.dir / "config.yaml").read_text(encoding="utf-8")
    assert "seed: 7" in saved

    env = json.loads((run.dir / "env.json").read_text(encoding="utf-8"))
    assert env["python"].startswith("3.11")
    assert "git" in env and "commit" in env["git"]


def test_run_name_comes_from_config(tmp_path: Path):
    cfg = load_config("classify")  # run_name: p5-logreg-baseline-s42
    run = start_run(cfg, root=tmp_path)
    try:
        assert run.dir.name.endswith("_p5-logreg-baseline-s42")
    finally:
        run.finish()


def test_artifact_creates_parent_dirs(tmp_path: Path):
    run = start_run(name="artifacts", root=tmp_path)
    p = run.artifact("figures", "nested", "plot.png")
    p.write_bytes(b"x")
    run.finish()
    assert p.is_file() and p.parent.is_dir()


# --- a stage that refuses leaves nothing behind (audit 17) ------------------------------------


def test_discard_removes_a_run_that_produced_nothing(tmp_path):
    """`start_run` creates the directory before `run(cfg, active)` is called, which is right for
    a stage that fails halfway - it keeps its log. It is wrong for eleven entry points that
    raised on their first line, which wrote a timestamped directory on every invocation."""
    from src.utils.logging import start_run

    run = start_run(name="refused", root=tmp_path)
    assert run.dir.is_dir()
    assert run.discard() is True
    assert not run.dir.exists()


def test_discard_refuses_to_remove_a_run_that_produced_something(tmp_path):
    from src.utils.logging import start_run

    run = start_run(name="did-work", root=tmp_path)
    run.artifact("model.json").write_text("{}", encoding="utf-8")
    assert run.discard() is False
    assert run.dir.is_dir()


def test_startup_files_is_exactly_what_start_run_writes(tmp_path):
    """If `start_run` gains a file and this set does not, `discard` silently stops discarding."""
    from src.utils.logging import STARTUP_FILES, start_run

    run = start_run(name="empty", root=tmp_path)
    written = {path.name for path in run.dir.iterdir()}
    assert written <= STARTUP_FILES, written - STARTUP_FILES
    run.discard()


def test_the_cli_discards_rather_than_finishing_a_refused_stage():
    import inspect

    from src.utils import cli

    source = inspect.getsource(cli.main)
    assert "active.discard()" in source
    assert 'active.finish("not-implemented")' not in source
