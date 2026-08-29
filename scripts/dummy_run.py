"""Phase 0.2.4 acceptance check — a dummy stage that writes a complete run directory.

Exercises the logging contract end to end without depending on any pipeline code:

    .venv/Scripts/python.exe scripts/dummy_run.py

Prints the run directory it created and lists the files inside it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config  # noqa: E402
from src.utils.logging import start_run  # noqa: E402


def main() -> int:
    cfg = load_config("base", ["logging.run_name=p0-dummy"])
    run = start_run(cfg)

    run.log.info("dummy stage: pretending to process %d images", 5)
    for step, loss in enumerate([0.91, 0.62, 0.44, 0.31, 0.27], start=1):
        run.log_metrics({"loss": loss}, step=step)
    run.log.debug("this debug line is filtered out at INFO level")
    run.log.warning("dummy warning, to prove warnings reach both sinks")

    out = run.artifact("figures", "dummy.txt")
    out.write_text("placeholder artifact\n", encoding="utf-8")

    run.log_metrics({"final_loss": 0.27, "n_images": 5})
    run.finish("ok")

    expected = ["config.yaml", "env.json", "metrics.json", "run.jsonl", "run.log"]
    present = sorted(p.name for p in run.dir.iterdir())
    print("-" * 60)
    print(f"run directory : {run.dir.relative_to(ROOT)}")
    print(f"files         : {present}")

    metrics = json.loads((run.dir / "metrics.json").read_text(encoding="utf-8"))
    jsonl = (run.dir / "run.jsonl").read_text(encoding="utf-8").strip().splitlines()
    print(f"metrics       : {metrics}")
    print(f"jsonl records : {len(jsonl)}")

    checks = {
        "all_files_written": all(f in present for f in expected),
        "artifact_written": out.is_file(),
        "steps_recorded": len(metrics.get("steps", [])) == 5,
        "final_metrics_recorded": metrics.get("final_loss") == 0.27,
        "status_ok": metrics.get("status") == "ok",
        "jsonl_mirrors_log": len(jsonl) >= 8,
    }
    for name, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    ok = all(checks.values())
    print("RESULT:", "logging ready" if ok else "logging NOT ready")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
