"""Phase 1.2.7 / 1.2.8 — the capture tool.

Files a photograph into the chaos corpus with its metadata baked into the path, and refuses
anything that would corrupt the corpus: unknown scenario, unknown medium, unknown condition,
a scribe who is not in the registry, or an image too blurred or too dark to be usable.

    python -m src.ingest.capture --webcam --scribe scribe03 --type flowchart \
        --scenario atm_withdraw --medium ballpoint --condition shadow
    python -m src.ingest.capture --file photo.jpg --scribe scribe03 --type flowchart \
        --scenario atm_withdraw --medium ballpoint --condition shadow

The quality gate is deliberately permissive. Adverse capture is the *point* of Phase 1.2.7:
shadows, glare and odd angles must get through. It rejects only images that carry no
recoverable signal at all - a lens cap, a motion smear, a black frame.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

from src.ingest.collection import (
    ADVERSE_CONDITIONS,
    CHAOS,
    CONDITIONS,
    DIAGRAM_TYPES,
    MEDIA,
    SCENARIOS,
)
from src.utils.config import ROOT

# Thresholds tuned to reject only unusable frames, not merely difficult ones.
MIN_LAPLACIAN_VAR = 40.0  # focus: below this the strokes are a smear
MIN_MEAN_INTENSITY = 25  # a near-black frame (lens cap, no light)
MAX_MEAN_INTENSITY = 250  # a blown-out white frame
MIN_INK_FRACTION = 0.001  # there must be *something* drawn


def quality(image: np.ndarray) -> dict:
    """Measure whether a frame carries recoverable strokes. Returns metrics plus a verdict."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    focus = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean = float(gray.mean())
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10
    )
    ink = float((binary > 0).mean())

    reasons = []
    if focus < MIN_LAPLACIAN_VAR:
        reasons.append(f"out of focus (laplacian var {focus:.0f} < {MIN_LAPLACIAN_VAR})")
    if mean < MIN_MEAN_INTENSITY:
        reasons.append(f"too dark (mean {mean:.0f})")
    if mean > MAX_MEAN_INTENSITY:
        reasons.append(f"blown out (mean {mean:.0f})")
    if ink < MIN_INK_FRACTION:
        reasons.append(f"no strokes found (ink fraction {ink:.4f})")

    return {
        "focus": round(focus, 1),
        "mean_intensity": round(mean, 1),
        "ink_fraction": round(ink, 4),
        "width": int(gray.shape[1]),
        "height": int(gray.shape[0]),
        "usable": not reasons,
        "reasons": reasons,
    }


def validate_metadata(dtype: str, scenario: str, medium: str, condition: str) -> list[str]:
    problems = []
    if dtype not in DIAGRAM_TYPES:
        problems.append(f"unknown diagram type {dtype!r}; expected one of {DIAGRAM_TYPES}")
    elif scenario not in {s.slug for s in SCENARIOS[dtype]}:
        valid = sorted(s.slug for s in SCENARIOS[dtype])
        problems.append(f"unknown scenario {scenario!r} for {dtype}; expected one of {valid}")
    if medium not in MEDIA:
        problems.append(f"unknown medium {medium!r}; expected one of {MEDIA}")
    if condition not in CONDITIONS:
        problems.append(f"unknown condition {condition!r}; expected one of {CONDITIONS}")
    return problems


def target_path(dtype: str, scribe: str, scenario: str, medium: str, condition: str) -> Path:
    folder = CHAOS / dtype / scribe
    folder.mkdir(parents=True, exist_ok=True)
    n = 1
    while (p := folder / f"{scenario}__{medium}__{condition}__{n:03d}.jpg").exists():
        n += 1
    return p


def grab_webcam(device: int = 0, warmup: int = 10) -> np.ndarray:
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        raise SystemExit(f"could not open webcam device {device}")
    try:
        frame = None
        for _ in range(warmup):  # let auto-exposure and focus settle
            ok, frame = cap.read()
            if not ok:
                raise SystemExit("webcam returned no frame")
        return frame
    finally:
        cap.release()


def capture(
    dtype: str,
    scribe: str,
    scenario: str,
    medium: str,
    condition: str,
    *,
    source: Path | None = None,
    device: int = 0,
    force: bool = False,
) -> Path | None:
    problems = validate_metadata(dtype, scenario, medium, condition)
    if problems:
        for p in problems:
            print(f"  REJECT: {p}", file=sys.stderr)
        return None

    image = cv2.imread(str(source)) if source else grab_webcam(device)
    if image is None:
        print(f"  REJECT: could not read {source}", file=sys.stderr)
        return None

    q = quality(image)
    print(
        f"  quality: focus={q['focus']} mean={q['mean_intensity']} ink={q['ink_fraction']} "
        f"size={q['width']}x{q['height']}"
    )
    if not q["usable"] and not force:
        for r in q["reasons"]:
            print(f"  REJECT: {r}", file=sys.stderr)
        print("  (pass --force to keep it anyway)", file=sys.stderr)
        return None

    out = target_path(dtype, scribe, scenario, medium, condition)
    if source:
        shutil.copy2(source, out)
    else:
        cv2.imwrite(str(out), image)

    adverse = condition in ADVERSE_CONDITIONS
    try:
        shown = out.relative_to(ROOT)
    except ValueError:  # tests file into a temporary corpus outside the repo
        shown = out
    print(f"  saved {shown}  ({'adverse' if adverse else 'clean'})")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--file", type=Path, help="an existing photograph to file")
    src.add_argument("--webcam", action="store_true", help="grab a frame from the webcam")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--scribe", required=True)
    ap.add_argument("--type", dest="dtype", required=True, choices=DIAGRAM_TYPES)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--medium", required=True, choices=MEDIA)
    ap.add_argument("--condition", required=True, choices=CONDITIONS)
    ap.add_argument("--force", action="store_true", help="keep the image despite quality problems")
    args = ap.parse_args(argv)

    out = capture(
        args.dtype,
        args.scribe,
        args.scenario,
        args.medium,
        args.condition,
        source=args.file,
        device=args.device,
        force=args.force,
    )
    return 0 if out else 1


if __name__ == "__main__":
    sys.exit(main())
