"""Phase 15.1 - experiment tracking: every run in one store, with its params, metrics and files.

    python -m src.mlops.tracking --backfill        # ingest the runs this project already did
    python -m src.mlops.tracking --list            # what is in the store
    mlflow ui --backend-store-uri file:mlruns      # browse it

    from src.mlops.tracking import track
    with track("detect", params={"epochs": 40}) as run:
        run.log_metric("map50", 0.91)
        run.log_artifact(report_path)

The plan asks for MLflow across detection, classification, parsing and generation, with every run
logged. Two design decisions make that achievable on a project whose runs mostly already
happened.

**The store is a local file backend, not a server.** `file:mlruns` needs nothing running, works
offline, and survives being committed to nothing - the directory is gitignored because it is
regenerable from the artefacts. A tracking server is a deployment decision and this row does not
need one to satisfy "every run logged with params, metrics, artifacts".

**The history is backfilled from the artefacts rather than invented.** Fourteen phases of runs
finished before this module existed, and re-running them to populate a tracking store would cost
days of GPU time to learn nothing. Every one of them wrote a JSON report, so `--backfill` reads
those reports and creates one MLflow run per artefact, with the metrics it recorded, the
parameters it declared, and the file itself attached. Each backfilled run is tagged
`backfilled=true` and carries `source_file` and the artefact's modification time, so nothing in
the store can be mistaken for a live run that this module watched.

**What a backfilled run cannot have is recorded as absent**: a duration the artefact never kept
is not logged as zero, and a metric the artefact never carried is not invented. The listing
prints how many of each a run has, so a sparse run looks sparse.

`track()` is the interface for runs from here on - a context manager that opens an MLflow run,
logs parameters, takes metrics as they arrive and attaches artifacts at the end. It degrades to a
no-op recorder if MLflow is not installed, because a missing tracking library must never be the
reason a training run fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.utils.config import ROOT

STORE = ROOT / "mlruns"
STORE_URI = f"file:{STORE.as_posix()}"

#: (experiment, run name, artefact, metric keys, param keys). One row per artefact worth a run.
BACKFILL: tuple[tuple[str, str, str, tuple[str, ...], tuple[str, ...]], ...] = (
    (
        "classification",
        "s1_clip_svm",
        "reports/s1_heldout_scribes.json",
        ("accuracy", "accuracy_std", "macro_f1", "minimum_repeat_accuracy"),
        ("model", "protocol", "rows", "target_accuracy"),
    ),
    (
        "detection",
        "s2_component_detector",
        "reports/s2_component_detection.json",
        ("map50", "hand_drawn_map50", "pages", "predictions"),
        ("model", "protocol", "split", "target_map50"),
    ),
    (
        "ocr",
        "s3_trocr_large_selected",
        "reports/s3_label_ocr.json",
        ("cer", "wer", "exact_match", "mean_crop_cer", "crops"),
        ("model", "checkpoint", "split", "protocol", "target_cer"),
    ),
    (
        "ocr",
        "s3_trocr_large_16_epochs",
        "reports/s3_label_ocr_large.json",
        ("cer", "wer", "exact_match"),
        ("model", "checkpoint", "split"),
    ),
    (
        "ocr",
        "s3_trocr_large_28_epochs",
        "reports/s3_label_ocr_large28.json",
        ("cer", "wer", "exact_match"),
        ("model", "checkpoint", "split"),
    ),
    (
        "ocr",
        "s3_trocr_strong_augmentation",
        "reports/s3_label_ocr_aug.json",
        ("cer", "wer", "exact_match"),
        ("model", "checkpoint", "split"),
    ),
    (
        "ocr",
        "s3_retrained_on_reselection",
        "reports/s3_label_ocr_resel_retrain.json",
        ("cer", "wer", "exact_match"),
        ("model", "checkpoint", "split"),
    ),
    (
        "parsing",
        "s4_role_hmm",
        "reports/s4_role_labelling.json",
        ("macro_f1", "macro_f1_std", "accuracy", "sequences", "nodes"),
        ("protocol", "target_macro_f1", "seeds"),
    ),
    (
        "assembly",
        "s5_test_trocr_large",
        "reports/s5_test_large.json",
        ("overall.median_ged", "overall.mean_ged", "overall.pass_share"),
        ("criterion", "split", "match", "target_median_ged"),
    ),
    (
        "assembly",
        "s5_val",
        "reports/s5_graph_ged.json",
        ("overall.median_ged", "overall.pass_share"),
        ("criterion", "split", "match"),
    ),
    (
        "generation",
        "lora_r64_quality",
        "experiments/llm/quality/done.json",
        ("lora.functional", "lora.executes", "lora.syntax", "lora.contract", "lora.n"),
        (),
    ),
    (
        "generation",
        "structural_fidelity",
        "experiments/llm/structural/done.json",
        ("lora.structural", "reference.structural"),
        (),
    ),
    (
        "generation",
        "similarity",
        "experiments/llm/similarity/done.json",
        ("lora.codebleu", "lora.exact_match", "lora.edit_similarity"),
        (),
    ),
    (
        "generation",
        "repair_loop",
        "experiments/llm/repair/done.json",
        ("pass_before", "pass_after", "success_after_repair", "fixed"),
        (),
    ),
    (
        "pipeline",
        "p13_golden",
        "runs/p13_golden/summary.json",
        ("pages", "produced_code", "rate", "seconds.median", "seconds.p90"),
        (),
    ),
    (
        "evaluation",
        "error_propagation",
        "reports/error_propagation.json",
        (
            "pass_rate.gold",
            "pass_rate.gold_structure",
            "pass_rate.gold_text",
            "pass_rate.predicted",
            "seconds",
        ),
        ("split", "generator", "test"),
    ),
    (
        "evaluation",
        "robustness",
        "reports/robustness_pipeline.json",
        ("seconds",),
        ("split",),
    ),
    (
        "evaluation",
        "unseen_type",
        "reports/unseen_type.json",
        ("unseen_pages", "unseen_confident_wrong", "unseen_asked", "unseen_stopped"),
        ("corpus", "confidence_floor"),
    ),
)


def _dig(payload: Any, dotted: str) -> Any:
    value = payload
    for key in dotted.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def available() -> bool:
    try:
        import mlflow  # noqa: F401
    except ImportError:
        return False
    return True


class Recorder:
    """What a caller talks to inside `track`. Holds what was logged so tests can read it back."""

    def __init__(self, client=None, run_id: str | None = None) -> None:
        self._client = client
        self.run_id = run_id
        self.metrics: dict[str, float] = {}
        self.params: dict[str, Any] = {}
        self.artifacts: list[str] = []

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        if value is None:
            return  # a metric the run did not produce is absent, not zero
        self.metrics[key] = float(value)
        if self._client and self.run_id:
            self._client.log_metric(self.run_id, key, float(value), step=step or 0)

    def log_param(self, key: str, value: Any) -> None:
        if value is None:
            return
        self.params[key] = value
        if self._client and self.run_id:
            self._client.log_param(self.run_id, key, str(value)[:500])

    def log_artifact(self, path: Path | str) -> None:
        path = Path(path)
        if not path.is_file():
            return
        self.artifacts.append(str(path))
        if self._client and self.run_id:
            self._client.log_artifact(self.run_id, str(path))

    def set_tag(self, key: str, value: Any) -> None:
        if self._client and self.run_id:
            self._client.set_tag(self.run_id, key, str(value)[:500])


@contextmanager
def track(
    experiment: str,
    name: str | None = None,
    params: dict[str, Any] | None = None,
    tags: dict[str, Any] | None = None,
    store_uri: str = STORE_URI,
) -> Iterator[Recorder]:
    """Open a tracked run. A missing MLflow degrades to an in-memory recorder, never an error."""
    if not available():
        recorder = Recorder()
        for key, value in (params or {}).items():
            recorder.log_param(key, value)
        yield recorder
        return

    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(store_uri)
    client = MlflowClient(tracking_uri=store_uri)
    existing = client.get_experiment_by_name(experiment)
    experiment_id = existing.experiment_id if existing else client.create_experiment(experiment)
    run = client.create_run(experiment_id, run_name=name or experiment)
    recorder = Recorder(client, run.info.run_id)
    for key, value in (params or {}).items():
        recorder.log_param(key, value)
    for key, value in (tags or {}).items():
        recorder.set_tag(key, value)
    status = "FINISHED"
    try:
        yield recorder
    except Exception:
        status = "FAILED"
        raise
    finally:
        client.set_terminated(recorder.run_id, status)


def backfill(store_uri: str = STORE_URI, rows=BACKFILL) -> dict:
    """One run per surviving artefact, tagged as backfilled and carrying the file itself."""
    created, missing = [], []
    for experiment, name, path, metrics, params in rows:
        artefact = ROOT / path
        if not artefact.is_file():
            missing.append(path)
            continue
        payload = json.loads(artefact.read_text(encoding="utf-8"))
        modified = datetime.fromtimestamp(artefact.stat().st_mtime, tz=UTC).isoformat()
        with track(
            experiment,
            name,
            params={key: _dig(payload, key) for key in params},
            tags={
                "backfilled": "true",
                "source_file": path,
                "artefact_modified": modified,
            },
            store_uri=store_uri,
        ) as run:
            for key in metrics:
                run.log_metric(key.replace(".", "_"), _dig(payload, key))
            run.log_artifact(artefact)
            created.append(
                {
                    "experiment": experiment,
                    "run": name,
                    "source_file": path,
                    "metrics_logged": len(run.metrics),
                    "metrics_absent": len(metrics) - len(run.metrics),
                    "params_logged": len(run.params),
                    "artifacts": len(run.artifacts),
                }
            )
    return {
        "store": store_uri,
        "mlflow_available": available(),
        "runs_created": len(created),
        "artefacts_missing": missing,
        "runs": created,
    }


def listing(store_uri: str = STORE_URI) -> list[dict]:
    """Every run in the store, newest first - what `mlflow ui` would show, as text."""
    if not available():
        return []
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=store_uri)
    out = []
    for experiment in client.search_experiments():
        for run in client.search_runs([experiment.experiment_id], max_results=200):
            out.append(
                {
                    "experiment": experiment.name,
                    "run": run.info.run_name,
                    "status": run.info.status,
                    "backfilled": run.data.tags.get("backfilled") == "true",
                    "source_file": run.data.tags.get("source_file"),
                    "metrics": dict(sorted(run.data.metrics.items())),
                }
            )
    return sorted(out, key=lambda row: (row["experiment"], row["run"]))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backfill", action="store_true", help="ingest the artefacts already on disk")
    ap.add_argument("--list", dest="show", action="store_true", help="print the store's runs")
    ap.add_argument("--store", default=STORE_URI)
    args = ap.parse_args(argv)

    if not available():
        print("mlflow is not installed; install it with: uv pip install mlflow", file=sys.stderr)
        return 1
    if args.backfill:
        result = backfill(args.store)
        print(json.dumps({k: v for k, v in result.items() if k != "runs"}, indent=2))
        print(f"{result['runs_created']} runs -> {args.store}")
    if args.show or not args.backfill:
        for row in listing(args.store):
            flag = "backfilled" if row["backfilled"] else "live"
            print(f"{row['experiment']:<16} {row['run']:<32} {flag:<11} {row['metrics']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
