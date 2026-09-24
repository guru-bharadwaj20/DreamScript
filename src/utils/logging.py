"""Phase 0.2.4 — run logging.

Every execution of a pipeline stage gets its own directory:

    experiments/<YYYYmmdd-HHMMSS>_<run_name>/
        config.yaml     the fully resolved config the run actually used
        run.log         human-readable log
        run.jsonl       one JSON object per record, for machine parsing
        metrics.json    whatever the stage called `log_metrics` with
        env.json        python/torch/cuda/git provenance

Nothing is ever overwritten: a second run with the same name gets a new timestamp. The
directory returned by `start_run` is the only place a stage may write.

    from src.utils.logging import start_run
    run = start_run(cfg)
    run.log.info("preprocessing %d images", n)
    run.log_metrics({"stroke_iou": 0.83})
"""

from __future__ import annotations

import json
import logging
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from src.utils.config import ROOT, save_config, to_dict

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class JsonLinesHandler(logging.Handler):
    """Mirror every record into run.jsonl so runs can be diffed and aggregated."""

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path

    def emit(self, record: logging.LogRecord) -> None:
        payload = {
            "ts": datetime.fromtimestamp(record.created).isoformat(timespec="seconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.format(record)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload) + "\n")


def _git_provenance() -> dict[str, Any]:
    def git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10, check=False
            )
            return out.stdout.strip() if out.returncode == 0 else None
        except Exception:  # noqa: BLE001 - provenance must never break a run
            return None

    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
    }


def _environment() -> dict[str, Any]:
    env: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "argv": sys.argv,
        "git": _git_provenance(),
    }
    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return env


#: Exactly the files `start_run` writes before a stage has done anything. `Run.discard` refuses
#: to remove a directory holding anything else, so a run that produced a result is never deleted.
STARTUP_FILES = frozenset({"config.yaml", "env.json", "run.log", "run.jsonl", "metrics.json"})


@dataclass
class Run:
    """A single stage execution and the directory that belongs to it."""

    name: str
    dir: Path
    log: logging.Logger
    cfg: DictConfig | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    #: The 15.1 tracking recorder for this run, when tracking is on. `Recorder` degrades to an
    #: in-memory object with no MLflow installed, so this is never None once `start_run` has set
    #: it and callers never have to test for it.
    recorder: Any = None
    _tracking: Any = None

    def log_metrics(self, metrics: dict[str, Any], *, step: int | None = None) -> None:
        """Record metrics to metrics.json and the log. Repeated calls merge."""
        if step is not None:
            self.metrics.setdefault("steps", []).append({"step": step, **metrics})
        else:
            self.metrics.update(metrics)
        (self.dir / "metrics.json").write_text(
            json.dumps(self.metrics, indent=2, default=str) + "\n", encoding="utf-8"
        )
        if self.recorder is not None:
            # 15.1's store, from the convention every stage already uses. `src.mlops` was 3,250
            # lines with one caller anywhere in the repo, and `tracking.py` - the module the plan
            # asks for "every run logged with params, metrics, artifacts" - had none at all.
            for key, value in metrics.items():
                if isinstance(value, int | float) and not isinstance(value, bool):
                    self.recorder.log_metric(key, float(value), step=step)
        pretty = ", ".join(f"{k}={v}" for k, v in metrics.items())
        self.log.info("metrics%s: %s", f" @ step {step}" if step is not None else "", pretty)

    def artifact(self, *parts: str) -> Path:
        """Path inside this run's directory, with parent directories created."""
        p = self.dir.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def _close_tracking(self, status: str) -> None:
        if self._tracking is None:
            return
        stack, self._tracking = self._tracking, None
        if self.recorder is not None:
            self.recorder.set_tag("status", status)
            for name in ("metrics.json", "config.yaml", "env.json"):
                self.recorder.log_artifact(self.dir / name)
        stack.close()

    def discard(self) -> bool:
        """Close the log and remove this run's directory, if the run produced nothing.

        For the case `utils.cli.main` has: a stage that refuses before it starts. `start_run`
        creates the directory *before* `run(cfg, active)` is called, because a stage that fails
        halfway should keep its log - but a stage that raises `StageNotImplemented` on its first
        line leaves a timestamped directory holding a config and a two-line log, and eleven
        commands did exactly that on every invocation.

        Refuses to remove a directory that holds anything besides the files `start_run` itself
        wrote, so a run that did produce something is never deleted by this path. Returns whether
        it removed anything.
        """
        import shutil

        self._close_tracking("discarded")
        for handler in list(self.log.handlers):
            handler.close()
            self.log.removeHandler(handler)
        written = {path.name for path in self.dir.iterdir()} if self.dir.is_dir() else set()
        if not written <= STARTUP_FILES:
            return False
        shutil.rmtree(self.dir, ignore_errors=True)
        return not self.dir.exists()

    def finish(self, status: str = "ok") -> None:
        self.metrics["status"] = status
        (self.dir / "metrics.json").write_text(
            json.dumps(self.metrics, indent=2, default=str) + "\n", encoding="utf-8"
        )
        self._close_tracking(status)
        self.log.info("run finished (%s): %s", status, self.dir)
        for handler in list(self.log.handlers):
            handler.close()
            self.log.removeHandler(handler)


def start_run(
    cfg: DictConfig | None = None,
    *,
    name: str | None = None,
    root: str | Path | None = None,
    level: str | None = None,
    to_file: bool = True,
    track: bool = False,
) -> Run:
    """Create experiments/<timestamp>_<name>/ and a logger writing into it.

    `track` opens a 15.1 MLflow run alongside the directory, so everything passed to
    `log_metrics` reaches the store and the run's own files are attached to it when it finishes.
    It defaults **off** for a bare `start_run(name=...)` - a library call should not write to a
    tracking store nobody asked for - and **on** for a config-driven run, where
    `logging.track: false` turns it off again. `src.mlops` was 3,250 lines with exactly one
    caller in the repo, and `tracking.py`, the module that exists so that "every run is logged
    with params, metrics and artifacts", had none; this is the convention every stage already
    goes through, so it is where the two meet.
    """
    if cfg is not None:
        name = name or cfg.get("logging", {}).get("run_name") or "run"
        root = root or cfg.get("paths", {}).get("experiments", "experiments")
        level = level or cfg.get("logging", {}).get("level", "INFO")
        to_file = cfg.get("logging", {}).get("to_file", to_file)
        track = bool(cfg.get("logging", {}).get("track", True))
    name = name or "run"
    root = Path(root or "experiments")
    if not root.is_absolute():
        root = ROOT / root

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = root / f"{stamp}_{name}"
    suffix = 1
    while run_dir.exists():  # two runs inside the same second
        run_dir = root / f"{stamp}-{suffix}_{name}"
        suffix += 1
    run_dir.mkdir(parents=True)

    logger = logging.getLogger(f"dreamscript.{name}")
    logger.setLevel(getattr(logging, str(level or "INFO").upper(), logging.INFO))
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if to_file:
        file_handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(JsonLinesHandler(run_dir / "run.jsonl"))

    (run_dir / "env.json").write_text(json.dumps(_environment(), indent=2) + "\n", encoding="utf-8")
    if cfg is not None:
        save_config(cfg, run_dir / "config.yaml")

    run = Run(name=name, dir=run_dir, log=logger, cfg=cfg)
    if track:
        # The experiment is the phase prefix of the run name (`p5-knn-k7-s42` -> `p5`), which is
        # the grouping docs/conventions.md section 1 already defines. A tracking store that
        # cannot be opened is not a reason a run fails, so this is best-effort by construction:
        # `track()` yields an in-memory recorder when MLflow is absent.
        import contextlib as _contextlib

        from src.mlops.tracking import track as _track

        stack = _contextlib.ExitStack()
        with _contextlib.suppress(Exception):
            run.recorder = stack.enter_context(
                _track(
                    name.split("-")[0] or "run",
                    name=name,
                    params=_environment(),
                    tags={"run_dir": str(run_dir)},
                )
            )
            run._tracking = stack  # its own attribute
    logger.info("run started: %s", run_dir)
    if cfg is not None:
        logger.debug("config: %s", json.dumps(to_dict(cfg), default=str))
    return run
