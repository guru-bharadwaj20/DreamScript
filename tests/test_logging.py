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
    cfg = load_config("classify")  # run_name: p5-classical-baseline
    run = start_run(cfg, root=tmp_path)
    try:
        assert run.dir.name.endswith("_p5-classical-baseline")
    finally:
        run.finish()


def test_artifact_creates_parent_dirs(tmp_path: Path):
    run = start_run(name="artifacts", root=tmp_path)
    p = run.artifact("figures", "nested", "plot.png")
    p.write_bytes(b"x")
    run.finish()
    assert p.is_file() and p.parent.is_dir()
