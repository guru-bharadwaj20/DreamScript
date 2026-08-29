"""Phase 0.1.3 acceptance check — the CV stack loads and thresholds an image.

Generates a synthetic "hand-drawn" flowchart tile (two boxes joined by an arrow), writes it
to disk, then round-trips it through Pillow, OpenCV and scikit-image to prove every library
in the CV stack can read pixels and binarize them. This is exactly the first operation the
Phase 3 preprocessing pipeline performs, so a pass here means Phase 3 can start.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage.filters import threshold_otsu, threshold_sauvola

OUT = Path("experiments/phase0/smoke_cv")


def make_sketch(path: Path) -> None:
    """A grey-ish page with two rectangles, a connecting arrow, and a little noise."""
    img = np.full((240, 400), 235, np.uint8)
    cv2.rectangle(img, (30, 90), (150, 160), 40, 3)
    cv2.rectangle(img, (250, 90), (370, 160), 40, 3)
    cv2.arrowedLine(img, (152, 125), (248, 125), 40, 3, tipLength=0.25)
    rng = np.random.default_rng(0)
    img = np.clip(img.astype(np.int16) + rng.normal(0, 8, img.shape), 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def main() -> int:
    src = OUT / "sketch.png"
    make_sketch(src)

    checks: dict[str, bool] = {}

    # --- Pillow ---------------------------------------------------------------
    pil = Image.open(src)
    checks["pillow_loads"] = pil.size == (400, 240)
    print(f"pillow      : {pil.size} mode={pil.mode}")

    # --- OpenCV: adaptive threshold ------------------------------------------
    gray = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
    checks["opencv_loads"] = gray is not None and gray.shape == (240, 400)
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10
    )
    ink_adaptive = float((adaptive > 0).mean())
    print(f"opencv      : shape={gray.shape} adaptive ink fraction={ink_adaptive:.4f}")

    # --- scikit-image: Otsu and Sauvola --------------------------------------
    otsu = gray < threshold_otsu(gray)
    sauvola = gray < threshold_sauvola(gray, window_size=25)
    print(f"skimage     : otsu ink={otsu.mean():.4f} sauvola ink={sauvola.mean():.4f}")

    # Strokes are thin: ink should be a small but non-trivial fraction of the page.
    checks["adaptive_binarizes"] = 0.005 < ink_adaptive < 0.30
    checks["otsu_binarizes"] = 0.005 < float(otsu.mean()) < 0.30
    checks["sauvola_binarizes"] = 0.005 < float(sauvola.mean()) < 0.30

    # Connected components: 2 boxes + 1 arrow should survive binarization.
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(adaptive, connectivity=8)
    big = int((stats[1:, cv2.CC_STAT_AREA] > 100).sum())
    print(f"components  : {n_labels - 1} total, {big} larger than 100 px")
    checks["components_found"] = big >= 3

    # --- pdf2image (import-level; poppler binary is only needed for real PDFs) --
    try:
        import pdf2image  # noqa: F401

        checks["pdf2image_imports"] = True
    except Exception as exc:  # noqa: BLE001
        print(f"pdf2image   : {exc}")
        checks["pdf2image_imports"] = False

    cv2.imwrite(str(OUT / "adaptive.png"), adaptive)
    cv2.imwrite(str(OUT / "otsu.png"), (otsu * 255).astype(np.uint8))
    cv2.imwrite(str(OUT / "sauvola.png"), (sauvola * 255).astype(np.uint8))
    print(f"artifacts   : {OUT}")

    print("-" * 60)
    for name, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    ok = all(checks.values())
    print("RESULT:", "CV stack ready" if ok else "CV stack NOT ready")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
