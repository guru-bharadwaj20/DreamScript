"""Headline S2: validate the component-detection result against its target.

    python -m src.detect.s2

The detector evaluation itself is deliberately owned by :mod:`src.detect.report`, which runs the
selected YOLO model and computes AP from its raw predictions. This small headline gate consumes
that immutable run artifact rather than reinterpreting framework metrics, rejects incomplete or
mislabelled artifacts, and writes the exact S2 verdict as a durable report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.detect.report import TARGET
from src.utils.config import ROOT

ARTIFACT = ROOT / "experiments" / "detect" / "report.json"
REPORT = ROOT / "reports" / "s2_component_detection.json"


def evaluate_artifact(artifact: Path = ARTIFACT) -> dict:
    """Validate the 9.1.4 prediction-based validation artifact and return S2's verdict."""
    if not artifact.is_file():
        raise FileNotFoundError(f"no detector evaluation artifact at {artifact}")
    data = json.loads(artifact.read_text(encoding="utf-8"))
    required = ("split", "pages", "predictions", "map50", "views", "per_class")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"detector artifact is missing required fields: {', '.join(missing)}")
    if data["split"] != "val":
        raise ValueError(f"S2 requires validation results; got split={data['split']!r}")
    pooled = data["views"].get("pooled", {})
    hand_drawn = data["views"].get("hand_drawn_only", {})
    if pooled.get("map50") != data["map50"]:
        raise ValueError("pooled mAP@0.5 disagrees with the artifact headline")
    if not isinstance(data["map50"], int | float):
        raise ValueError("mAP@0.5 must be numeric")
    if not isinstance(hand_drawn.get("map50"), int | float):
        raise ValueError("artifact lacks a numeric hand-drawn mAP@0.5")
    return {
        "criterion": "S2",
        "target_map50": TARGET,
        "protocol": "raw-prediction AP evaluation at IoU 0.5 on the held-out validation split",
        "model": "YOLO detector selected by Phase 9.1",
        "split": data["split"],
        "pages": data["pages"],
        "predictions": data["predictions"],
        "map50": round(float(data["map50"]), 4),
        "hand_drawn_map50": round(float(hand_drawn["map50"]), 4),
        "hand_drawn_pages": hand_drawn.get("pages"),
        "classes": len(data["per_class"]),
        "passes": bool(data["map50"] >= TARGET),
        "artifact": (
            str(artifact.relative_to(ROOT)) if artifact.is_relative_to(ROOT) else str(artifact)
        ),
    }


def write_report(result: dict, path: Path = REPORT) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifact", type=Path, default=ARTIFACT)
    ap.add_argument("--out", type=Path, default=REPORT)
    args = ap.parse_args(argv)
    try:
        result = evaluate_artifact(args.artifact)
        write_report(result, args.out)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
        print(error, file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0 if result["passes"] else 2


if __name__ == "__main__":
    sys.exit(main())
