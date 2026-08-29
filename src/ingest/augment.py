"""Phase 1.3.6 — the augmentation policy.

Augmentation here is not generic image jitter. Each transform reproduces a **specific real
failure** the pipeline must survive, and each one is traceable to a row of the capture
protocol (`docs/collection/capture_protocol.md`) or a risk in `docs/risks.md`:

| transform | reproduces | risk |
| :--- | :--- | :--- |
| `rotate` | a page photographed slightly turned | V2 |
| `perspective` | an oblique phone shot | V2 |
| `shadow` | a hand shadow across the page | V1 |
| `glare` | a lamp reflecting off the paper | V1 |
| `brightness` | a dim or over-lit room | V1 |
| `blur` | camera shake and misfocus | V1 |
| `jpeg` | recompression by a phone or messaging app | V1 |
| `ink_thickness` | a different pen: erode/dilate the strokes | media (1.2.8) |
| `paper_texture` | ruled or textured paper under the drawing | V3 |
| `stain` | a coffee ring or smudge occluding part of the drawing | V5 |

**The strokes are never invented.** Every transform degrades an existing real drawing; none
of them draw a shape that a person did not draw.

    python -m src.ingest.augment --demo    # write a visual grid to reports/figures/
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import cv2
import numpy as np

from src.utils.config import ROOT
from src.utils.seed import set_seed

FIGURE = ROOT / "reports" / "figures" / "p1_augmentation_grid.png"


@dataclass(frozen=True)
class Policy:
    """Ranges are deliberately conservative: augmentation must not destroy legibility.

    An augmented image a human could not read is not a hard example, it is a wrong label.
    """

    rotate_deg: float = 15.0
    perspective: float = 0.06
    shadow_strength: float = 0.45
    glare_strength: float = 0.55
    brightness: tuple[float, float] = (0.65, 1.35)
    blur_max: int = 5
    jpeg_quality: tuple[int, int] = (35, 95)
    thickness_iter: int = 1
    texture_strength: float = 0.20
    stain_radius: tuple[float, float] = (0.08, 0.22)


DEFAULT = Policy()


def rotate(img: np.ndarray, rng: np.random.Generator, p: Policy = DEFAULT) -> np.ndarray:
    angle = float(rng.uniform(-p.rotate_deg, p.rotate_deg))
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)


def perspective(img: np.ndarray, rng: np.random.Generator, p: Policy = DEFAULT) -> np.ndarray:
    h, w = img.shape[:2]
    d = p.perspective
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    # float32 is required by getPerspectiveTransform; the multiply promotes to float64.
    dst = (src + rng.uniform(-d, d, src.shape).astype(np.float32) * [w, h]).astype(np.float32)
    m = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)


def shadow(img: np.ndarray, rng: np.random.Generator, p: Policy = DEFAULT) -> np.ndarray:
    """A soft linear gradient, like a hand or body blocking the light."""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ang = float(rng.uniform(0, np.pi))
    proj = xx * np.cos(ang) + yy * np.sin(ang)
    proj = (proj - proj.min()) / max(float(np.ptp(proj)), 1e-6)
    strength = float(rng.uniform(0.2, p.shadow_strength))
    mask = 1.0 - strength * np.clip((proj - rng.uniform(0.2, 0.6)) * 2.5, 0, 1)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=w * 0.05)
    out = img.astype(np.float32) * (mask[..., None] if img.ndim == 3 else mask)
    return np.clip(out, 0, 255).astype(np.uint8)


def glare(img: np.ndarray, rng: np.random.Generator, p: Policy = DEFAULT) -> np.ndarray:
    """A bright elliptical hotspot that washes strokes out, as on glossy paper."""
    h, w = img.shape[:2]
    cx, cy = rng.uniform(0.2, 0.8) * w, rng.uniform(0.2, 0.8) * h
    rx, ry = rng.uniform(0.15, 0.35) * w, rng.uniform(0.15, 0.35) * h
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
    hotspot = np.clip(1.0 - d, 0, 1) ** 2 * float(rng.uniform(0.3, p.glare_strength))
    hotspot = cv2.GaussianBlur(hotspot, (0, 0), sigmaX=w * 0.02)
    base = img.astype(np.float32)
    out = base + (255 - base) * (hotspot[..., None] if img.ndim == 3 else hotspot)
    return np.clip(out, 0, 255).astype(np.uint8)


def brightness(img: np.ndarray, rng: np.random.Generator, p: Policy = DEFAULT) -> np.ndarray:
    f = float(rng.uniform(*p.brightness))
    return np.clip(img.astype(np.float32) * f, 0, 255).astype(np.uint8)


def blur(img: np.ndarray, rng: np.random.Generator, p: Policy = DEFAULT) -> np.ndarray:
    k = int(rng.integers(1, p.blur_max + 1)) * 2 + 1
    return cv2.GaussianBlur(img, (k, k), 0)


def jpeg(img: np.ndarray, rng: np.random.Generator, p: Policy = DEFAULT) -> np.ndarray:
    q = int(rng.integers(*p.jpeg_quality))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(buf, cv2.IMREAD_UNCHANGED) if ok else img


def ink_thickness(img: np.ndarray, rng: np.random.Generator, p: Policy = DEFAULT) -> np.ndarray:
    """Erode or dilate the *ink*, simulating a finer or fatter pen.

    Ink is dark, so a morphological erode on the image thickens strokes and dilate thins
    them - the operations are inverted relative to the usual binary-mask intuition.
    """
    k = np.ones((3, 3), np.uint8)
    it = int(p.thickness_iter)
    return (
        cv2.erode(img, k, iterations=it)
        if rng.random() < 0.5
        else cv2.dilate(img, k, iterations=it)
    )


def paper_texture(img: np.ndarray, rng: np.random.Generator, p: Policy = DEFAULT) -> np.ndarray:
    """Overlay ruled lines or a grid, the false structure of notebook paper (risk V3)."""
    h, w = img.shape[:2]
    overlay = np.full((h, w), 255, np.float32)
    spacing = int(rng.integers(h // 30, max(h // 12, h // 30 + 2)))
    darkness = 255 * float(rng.uniform(0.05, p.texture_strength))
    for y in range(spacing, h, spacing):
        overlay[y : y + 1, :] -= darkness
    if rng.random() < 0.4:  # squared paper as well as ruled
        for x in range(spacing, w, spacing):
            overlay[:, x : x + 1] -= darkness
    overlay = cv2.GaussianBlur(overlay, (3, 3), 0) / 255.0
    out = img.astype(np.float32) * (overlay[..., None] if img.ndim == 3 else overlay)
    return np.clip(out, 0, 255).astype(np.uint8)


def stain(img: np.ndarray, rng: np.random.Generator, p: Policy = DEFAULT) -> np.ndarray:
    """A translucent blot occluding part of the drawing (risk V5)."""
    h, w = img.shape[:2]
    out = img.astype(np.float32)
    r = float(rng.uniform(*p.stain_radius)) * min(h, w)
    cx, cy = rng.uniform(0.15, 0.85) * w, rng.uniform(0.15, 0.85) * h
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    ring = np.clip(1 - (d / r) ** 3, 0, 1) * 0.55
    ring = cv2.GaussianBlur(ring, (0, 0), sigmaX=r * 0.15)
    tint = np.array([120.0, 150.0, 190.0]) if img.ndim == 3 else 150.0
    if img.ndim == 3:
        for c in range(3):
            out[..., c] = out[..., c] * (1 - ring) + tint[c] * ring
    else:
        out = out * (1 - ring) + tint * ring
    return np.clip(out, 0, 255).astype(np.uint8)


TRANSFORMS = {
    "rotate": rotate,
    "perspective": perspective,
    "shadow": shadow,
    "glare": glare,
    "brightness": brightness,
    "blur": blur,
    "jpeg": jpeg,
    "ink_thickness": ink_thickness,
    "paper_texture": paper_texture,
    "stain": stain,
}

# Probability of applying each transform in a random pipeline. Geometry and lighting are
# common; stains and heavy texture are rare, matching how often they actually occur.
PROBABILITIES = {
    "rotate": 0.5,
    "perspective": 0.4,
    "shadow": 0.35,
    "glare": 0.2,
    "brightness": 0.5,
    "blur": 0.3,
    "jpeg": 0.4,
    "ink_thickness": 0.3,
    "paper_texture": 0.2,
    "stain": 0.1,
}


def augment(img: np.ndarray, rng: np.random.Generator | None = None, p: Policy = DEFAULT):
    """Apply a random subset of the policy. Returns the image and the transforms applied."""
    rng = rng or np.random.default_rng()
    applied = []
    out = img
    for name, fn in TRANSFORMS.items():
        if rng.random() < PROBABILITIES[name]:
            out = fn(out, rng, p)
            applied.append(name)
    return out, applied


def demo_grid(source: str | None = None, seed: int = 42):
    """Write a visual grid: original, each transform alone, then random combinations."""
    set_seed(seed)
    rng = np.random.default_rng(seed)

    if source is None:
        from src.ingest.collection import collected

        rows = collected()
        source = str(ROOT / rows[0]["path"]) if rows else str(ROOT / "tests/fixtures/flowchart.png")
    img = cv2.imread(str(source))
    if img is None:
        raise SystemExit(f"could not read {source}")
    scale = 420 / max(img.shape[:2])
    img = cv2.resize(img, None, fx=scale, fy=scale)

    panels = [("original", img)]
    panels += [(name, fn(img, rng, DEFAULT)) for name, fn in TRANSFORMS.items()]
    for i in range(3):
        out, applied = augment(img, rng)
        panels.append((f"combo {i + 1}: {'+'.join(applied) or 'none'}", out))

    cols = 4
    rows_n = (len(panels) + cols - 1) // cols
    h, w = img.shape[:2]
    pad = 28
    canvas = np.full((rows_n * (h + pad), cols * w, 3), 255, np.uint8)
    for i, (name, panel) in enumerate(panels):
        r, c = divmod(i, cols)
        y, x = r * (h + pad), c * w
        panel = cv2.resize(panel, (w, h))
        if panel.ndim == 2:
            panel = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)
        canvas[y + pad : y + pad + h, x : x + w] = panel
        cv2.putText(canvas, name[:46], (x + 4, y + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(FIGURE), canvas)
    return FIGURE, [n for n, _ in panels]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--demo", action="store_true", help="write the visual grid")
    ap.add_argument("--source", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    path, names = demo_grid(args.source, args.seed)
    print(f"wrote {path.relative_to(ROOT)}")
    print(f"panels: {len(names)}")
    for n in names:
        print(f"  {n}")

    checks = {
        "all_transforms_demonstrated": len(names) >= len(TRANSFORMS) + 1,
        "figure_written": path.is_file(),
        "every_transform_has_a_probability": set(TRANSFORMS) == set(PROBABILITIES),
    }
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
