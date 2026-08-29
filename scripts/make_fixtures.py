"""Phase 0.2.7 — generate the five tiny fixture images committed under tests/fixtures/.

One synthetic sketch per diagram type the pipeline must handle. They are deliberately small
(320x240, a few KB each) so the test suite stays fast and the repository stays light, but
each carries the geometry its type is recognised by: diamonds for flowcharts, a grid of
widgets for wireframes, circles with a self-loop for state machines, rectangles with ovals
for ER diagrams, and straight rails with component symbols for circuits.

Regenerate with:  .venv/Scripts/python.exe scripts/make_fixtures.py
The images are deterministic - regenerating produces byte-identical files.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
W, H = 320, 240
INK = 40
PAPER = 235


def _canvas() -> np.ndarray:
    return np.full((H, W), PAPER, np.uint8)


def _jitter(img: np.ndarray, seed: int) -> np.ndarray:
    """Sparse speckle, so fixtures are not unrealistically clean.

    Deliberately sparse rather than per-pixel Gaussian noise: full-frame noise defeats PNG
    compression and would make each fixture ~50 KB instead of ~2 KB.
    """
    rng = np.random.default_rng(seed)
    out = img.copy()
    n = int(0.004 * img.size)
    ys = rng.integers(0, img.shape[0], n)
    xs = rng.integers(0, img.shape[1], n)
    out[ys, xs] = rng.choice(np.array([160, 200, 255], np.uint8), n)
    return out


def flowchart() -> np.ndarray:
    img = _canvas()
    cv2.ellipse(img, (60, 30), (34, 16), 0, 0, 360, INK, 2)  # start terminator
    cv2.rectangle(img, (26, 80), (94, 118), INK, 2)  # process
    pts = np.array([[60, 140], [104, 170], [60, 200], [16, 170]], np.int32)
    cv2.polylines(img, [pts], True, INK, 2)  # decision diamond
    cv2.rectangle(img, (200, 150), (280, 190), INK, 2)  # branch process
    cv2.arrowedLine(img, (60, 46), (60, 78), INK, 2, tipLength=0.3)
    cv2.arrowedLine(img, (60, 120), (60, 138), INK, 2, tipLength=0.3)
    cv2.arrowedLine(img, (106, 170), (198, 170), INK, 2, tipLength=0.15)
    return _jitter(img, 1)


def wireframe() -> np.ndarray:
    img = _canvas()
    cv2.rectangle(img, (20, 16), (300, 224), INK, 2)  # page frame
    cv2.rectangle(img, (20, 16), (300, 50), INK, 2)  # header bar
    for i, y in enumerate((70, 108, 146)):  # stacked input fields
        cv2.rectangle(img, (40, y), (280, y + 26), INK, 2)
        cv2.line(img, (50, y + 13), (50 + 40 + i * 20, y + 13), INK, 1)
    cv2.rectangle(img, (200, 186), (280, 212), INK, 2)  # button
    cv2.line(img, (215, 199), (265, 199), INK, 2)
    return _jitter(img, 2)


def state_machine() -> np.ndarray:
    img = _canvas()
    cv2.circle(img, (70, 80), 30, INK, 2)
    cv2.circle(img, (230, 80), 30, INK, 2)
    cv2.circle(img, (150, 190), 28, INK, 2)
    cv2.circle(img, (150, 190), 23, INK, 2)  # double circle = accepting state
    cv2.arrowedLine(img, (101, 80), (198, 80), INK, 2, tipLength=0.12)
    cv2.arrowedLine(img, (212, 108), (170, 168), INK, 2, tipLength=0.15)
    cv2.ellipse(img, (70, 36), (22, 16), 0, 20, 320, INK, 2)  # self-loop
    return _jitter(img, 3)


def er_diagram() -> np.ndarray:
    img = _canvas()
    cv2.rectangle(img, (24, 96), (108, 140), INK, 2)  # entity
    cv2.rectangle(img, (212, 96), (296, 140), INK, 2)  # entity
    pts = np.array([[160, 96], [196, 118], [160, 140], [124, 118]], np.int32)
    cv2.polylines(img, [pts], True, INK, 2)  # relationship diamond
    cv2.line(img, (108, 118), (124, 118), INK, 2)
    cv2.line(img, (196, 118), (212, 118), INK, 2)
    for cx in (44, 92):  # attributes hanging off the left entity
        cv2.ellipse(img, (cx, 46), (26, 14), 0, 0, 360, INK, 2)
        cv2.line(img, (cx, 60), (cx, 94), INK, 1)
    cv2.ellipse(img, (254, 196), (30, 14), 0, 0, 360, INK, 2)
    cv2.line(img, (254, 182), (254, 142), INK, 1)
    return _jitter(img, 4)


def circuit() -> np.ndarray:
    img = _canvas()
    cv2.rectangle(img, (40, 50), (280, 190), INK, 2)  # the loop of wire
    cv2.rectangle(img, (120, 40), (170, 60), PAPER, -1)  # resistor body on the top rail
    cv2.rectangle(img, (120, 40), (170, 60), INK, 2)
    cv2.line(img, (40, 110), (40, 120), PAPER, 4)  # battery gap on the left rail
    cv2.line(img, (26, 106), (54, 106), INK, 3)
    cv2.line(img, (32, 122), (48, 122), INK, 2)
    cv2.circle(img, (200, 190), 16, INK, 2)  # lamp
    cv2.line(img, (189, 179), (211, 201), INK, 1)
    cv2.line(img, (211, 179), (189, 201), INK, 1)
    return _jitter(img, 5)


BUILDERS = {
    "flowchart": flowchart,
    "wireframe": wireframe,
    "state_machine": state_machine,
    "er_diagram": er_diagram,
    "circuit": circuit,
}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name, build in BUILDERS.items():
        img = build()
        path = OUT / f"{name}.png"
        cv2.imwrite(str(path), img)
        manifest.append(
            {
                "id": name,
                "file": path.name,
                "diagram_type": name,
                "width": int(img.shape[1]),
                "height": int(img.shape[0]),
                "synthetic": True,
                "bytes": path.stat().st_size,
            }
        )
        print(f"  wrote {path.name:<20} {path.stat().st_size:>6} bytes")

    # Trailing newline: the end-of-file-fixer pre-commit hook rewrites files without one,
    # which would break the byte-identical regeneration test.
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    total = sum(m["bytes"] for m in manifest)
    print(f"{len(manifest)} fixtures, {total} bytes total -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
