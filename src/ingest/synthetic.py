"""Phase 1.3.7 — synthetic diagram generator with a hand-drawn appearance.

The corpus is 487:1 imbalanced (`reports/class_balance.md`): 24k flowchart-like images, and
40-60 each of state machines, ER diagrams and circuits. No amount of augmentation invents a
new *structure*, so the rare types need generated diagrams — with a known graph attached,
which also gives Phase 12 free `(graph, code)` pairs.

**These are not passed off as real drawings.** Everything lands in `data/processed/synthetic/`
with `source="synthetic"` in the manifest, never in `data/raw/chaos/`, and Phase 14 reports
any metric that uses them separately.

The "hand-drawn" look comes from wobbling every stroke: each line is subdivided and its
points displaced by low-frequency noise, so edges bend the way a hand bends them, corners
overshoot, and circles are never quite round. That is a cosmetic imitation of handwriting,
not a claim to be handwriting.

    python -m src.ingest.synthetic --count 200 --types state_machine er_diagram circuit
"""

from __future__ import annotations

import argparse
import json
import math
import sys

import cv2
import numpy as np

from src.utils.config import ROOT
from src.utils.seed import set_seed

OUT = ROOT / "data" / "processed" / "synthetic"
INK = (35, 35, 40)
PAPER = 246


# --------------------------------------------------------------------------------------
# Wobbly primitives
# --------------------------------------------------------------------------------------


def _wobble(points: np.ndarray, rng: np.random.Generator, amp: float) -> np.ndarray:
    """Displace a polyline by smooth low-frequency noise - the hand-drawn look."""
    n = len(points)
    if n < 2:
        return points
    noise = rng.normal(0, amp, (n, 2))
    k = max(3, n // 4 | 1)
    kernel = np.ones(k) / k
    for c in range(2):
        noise[:, c] = np.convolve(noise[:, c], kernel, mode="same")
    return points + noise


def _line(img, p0, p1, rng, amp=2.4, thickness=2, overshoot=True):
    p0 = np.array(p0, float)
    p1 = np.array(p1, float)
    if overshoot and rng.random() < 0.35:  # hands overshoot corners
        p1 = p1 + (p1 - p0) / max(np.linalg.norm(p1 - p0), 1e-6) * rng.uniform(2, 7)
    n = max(8, int(np.linalg.norm(p1 - p0) / 12))
    t = np.linspace(0, 1, n)[:, None]
    pts = _wobble(p0 + (p1 - p0) * t, rng, amp)
    cv2.polylines(img, [pts.astype(np.int32)], False, INK, thickness, cv2.LINE_AA)


def _rect(img, x, y, w, h, rng, amp=2.4, thickness=2):
    corners = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    for i in range(4):
        _line(img, corners[i], corners[(i + 1) % 4], rng, amp, thickness)


def _ellipse(img, cx, cy, rx, ry, rng, amp=2.2, thickness=2):
    t = np.linspace(0, 2 * math.pi, 60)
    pts = np.stack([cx + rx * np.cos(t), cy + ry * np.sin(t)], axis=1)
    pts = _wobble(pts, rng, amp)
    cv2.polylines(img, [pts.astype(np.int32)], True, INK, thickness, cv2.LINE_AA)


def _diamond(img, cx, cy, w, h, rng, amp=2.4, thickness=2):
    pts = [(cx, cy - h / 2), (cx + w / 2, cy), (cx, cy + h / 2), (cx - w / 2, cy)]
    for i in range(4):
        _line(img, pts[i], pts[(i + 1) % 4], rng, amp, thickness)


def _arrow(img, p0, p1, rng, amp=2.2, thickness=2, head=11.0):
    _line(img, p0, p1, rng, amp, thickness, overshoot=False)
    p0, p1 = np.array(p0, float), np.array(p1, float)
    d = p1 - p0
    norm = np.linalg.norm(d)
    if norm < 1e-6:
        return
    d /= norm
    # A real arrowhead is rarely symmetric, and often does not quite meet the shape.
    gap = rng.uniform(0, 4)
    tip = p1 - d * gap
    for sign in (1, -1):
        ang = math.radians(rng.uniform(20, 34) * sign)
        r = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
        back = tip - r @ d * rng.uniform(head * 0.8, head * 1.2)
        _line(img, tip, back, rng, amp * 0.5, thickness, overshoot=False)


def _text(img, x, y, label, rng, scale=0.5):
    """Labels are rendered as a font, then wobbled by a small affine jitter."""
    ang = rng.uniform(-3, 3)
    patch = np.full((40, 240, 3), PAPER, np.uint8)
    cv2.putText(patch, label[:18], (4, 26), cv2.FONT_HERSHEY_SIMPLEX, scale, INK, 1, cv2.LINE_AA)
    m = cv2.getRotationMatrix2D((120, 20), ang, 1.0)
    patch = cv2.warpAffine(patch, m, (240, 40), borderValue=(PAPER, PAPER, PAPER))
    h, w = patch.shape[:2]
    x, y = int(x), int(y)
    if 0 <= y < img.shape[0] - h and 0 <= x < img.shape[1] - w:
        roi = img[y : y + h, x : x + w]
        img[y : y + h, x : x + w] = np.minimum(roi, patch)


def _canvas(w=900, h=700):
    return np.full((h, w, 3), PAPER, np.uint8)


# --------------------------------------------------------------------------------------
# Generators - each returns (image, graph)
# --------------------------------------------------------------------------------------


def gen_state_machine(rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    img = _canvas()
    n = int(rng.integers(3, 6))
    centres, nodes = [], []
    for i in range(n):
        cx = 150 + (i % 3) * 280 + rng.integers(-25, 25)
        cy = 180 + (i // 3) * 260 + rng.integers(-25, 25)
        r = rng.integers(48, 62)
        _ellipse(img, cx, cy, r, r * rng.uniform(0.85, 1.0), rng)
        accepting = i == n - 1
        if accepting:  # double circle
            _ellipse(img, cx, cy, r - 7, (r - 7) * 0.9, rng)
        _text(img, cx - 22, cy - 12, f"q{i}", rng)
        centres.append((cx, cy))
        nodes.append(
            {"id": f"q{i}", "role": "start" if i == 0 else ("accept" if accepting else "state")}
        )

    edges = []
    for i in range(n - 1):
        (x0, y0), (x1, y1) = centres[i], centres[i + 1]
        d = np.array([x1 - x0, y1 - y0], float)
        d /= max(np.linalg.norm(d), 1e-6)
        _arrow(img, (x0 + d[0] * 58, y0 + d[1] * 58), (x1 - d[0] * 60, y1 - d[1] * 60), rng)
        label = rng.choice(["a", "b", "0", "1"])
        _text(img, (x0 + x1) / 2 - 14, (y0 + y1) / 2 - 34, str(label), rng, 0.45)
        edges.append({"src": f"q{i}", "dst": f"q{i + 1}", "label": str(label)})

    if rng.random() < 0.8:  # a self-loop: the signature of a state machine
        cx, cy = centres[int(rng.integers(0, n))]
        idx = centres.index((cx, cy))
        t = np.linspace(0.35 * math.pi, 2.3 * math.pi, 40)
        pts = np.stack([cx + 34 * np.cos(t), cy - 66 + 26 * np.sin(t)], axis=1)
        cv2.polylines(img, [_wobble(pts, rng, 1.4).astype(np.int32)], False, INK, 2, cv2.LINE_AA)
        edges.append({"src": f"q{idx}", "dst": f"q{idx}", "label": "loop"})

    return img, {"diagram_type": "state_machine", "nodes": nodes, "edges": edges}


def gen_er_diagram(rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    img = _canvas()
    n_ent = int(rng.integers(2, 4))
    nodes, edges, boxes = [], [], []
    for i in range(n_ent):
        x = 90 + i * 330 + rng.integers(-20, 20)
        y = 280 + rng.integers(-30, 30)
        w, h = rng.integers(150, 190), rng.integers(70, 90)
        _rect(img, x, y, w, h, rng)
        name = ["Customer", "Order", "Product", "Invoice"][i]
        _text(img, x + 14, y + h / 2 - 18, name, rng)
        boxes.append((x, y, w, h))
        nodes.append({"id": name, "role": "entity"})

        for a in range(int(rng.integers(2, 4))):  # attribute ovals
            ax = x + 20 + a * 70
            ay = y - rng.integers(95, 130)
            _ellipse(img, ax, ay, 46, 24, rng)
            attr = ["id", "name", "date", "qty", "price"][(i + a) % 5]
            _text(img, ax - 26, ay - 12, attr, rng, 0.42)
            _line(img, (ax, ay + 24), (ax, y), rng, 1.2, 1)
            nodes.append({"id": f"{name}.{attr}", "role": "attribute"})
            edges.append({"src": name, "dst": f"{name}.{attr}", "label": ""})

    for i in range(n_ent - 1):
        (x0, y0, w0, h0), (x1, y1, _, _) = boxes[i], boxes[i + 1]
        mx = (x0 + w0 + x1) / 2
        my = (y0 + y1) / 2 + 40
        _diamond(img, mx, my, 110, 70, rng)
        rel = ["places", "contains", "billed"][i % 3]
        _text(img, mx - 34, my - 12, rel, rng, 0.42)
        _line(img, (x0 + w0, y0 + h0 / 2), (mx - 55, my), rng, 1.4)
        _line(img, (mx + 55, my), (x1, y1 + h0 / 2), rng, 1.4)
        _text(img, x0 + w0 + 8, y0 + h0 / 2 - 30, "1", rng, 0.5)
        _text(img, x1 - 26, y1 + h0 / 2 - 30, "N", rng, 0.5)
        nodes.append({"id": rel, "role": "relationship"})
        edges.append({"src": nodes[0]["id"], "dst": rel, "label": "1"})
        edges.append({"src": rel, "dst": nodes[-2]["id"], "label": "N"})

    return img, {"diagram_type": "er_diagram", "nodes": nodes, "edges": edges}


def gen_circuit(rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    img = _canvas(820, 620)
    x0, y0, x1, y1 = 120, 130, 700, 500
    nodes, edges = [], []

    # The loop of wire, drawn as four hand-wobbled rails.
    _line(img, (x0, y0), (x1, y0), rng, 1.5)
    _line(img, (x1, y0), (x1, y1), rng, 1.5)
    _line(img, (x1, y1), (x0, y1), rng, 1.5)
    _line(img, (x0, y1), (x0, y0), rng, 1.5)

    # Battery: break the left rail and draw the long/short plate pair.
    by = (y0 + y1) / 2
    cv2.rectangle(img, (x0 - 6, int(by - 16)), (x0 + 6, int(by + 16)), (PAPER, PAPER, PAPER), -1)
    _line(img, (x0 - 22, by - 14), (x0 + 22, by - 14), rng, 0.8, 3)
    _line(img, (x0 - 12, by + 14), (x0 + 12, by + 14), rng, 0.8, 2)
    nodes.append({"id": "V1", "role": "source"})
    _text(img, x0 - 70, by - 12, "9V", rng, 0.45)

    # Resistors on the top rail: a zigzag or a box, both are drawn in practice.
    n_r = int(rng.integers(1, 4))
    for i in range(n_r):
        rx = x0 + 140 + i * 180
        cv2.rectangle(img, (rx - 45, y0 - 14), (rx + 45, y0 + 14), (PAPER, PAPER, PAPER), -1)
        if rng.random() < 0.5:
            _rect(img, rx - 42, y0 - 16, 84, 32, rng)
        else:
            zig = [(rx - 42, y0)]
            for k in range(6):
                zig.append((rx - 42 + k * 14 + 7, y0 + (-18 if k % 2 == 0 else 18)))
            zig.append((rx + 42, y0))
            for a, b in zip(zig, zig[1:], strict=False):
                _line(img, a, b, rng, 0.9, 2, overshoot=False)
        _text(img, rx - 26, y0 - 58, f"R{i + 1}", rng, 0.45)
        nodes.append({"id": f"R{i + 1}", "role": "resistor"})
        edges.append({"src": "V1" if i == 0 else f"R{i}", "dst": f"R{i + 1}", "label": ""})

    if rng.random() < 0.6:  # a lamp or capacitor on the bottom rail
        lx = (x0 + x1) / 2
        cv2.circle(img, (int(lx), int(y1)), 22, (PAPER, PAPER, PAPER), -1)
        _ellipse(img, lx, y1, 22, 22, rng)
        _line(img, (lx - 15, y1 - 15), (lx + 15, y1 + 15), rng, 0.8, 1)
        _line(img, (lx + 15, y1 - 15), (lx - 15, y1 + 15), rng, 0.8, 1)
        nodes.append({"id": "L1", "role": "lamp"})
        edges.append({"src": f"R{n_r}", "dst": "L1", "label": ""})

    # Ground symbol on the bottom rail.
    gx = x0 + 90
    _line(img, (gx, y1), (gx, y1 + 34), rng, 0.8)
    for k, half in enumerate((22, 14, 7)):
        _line(img, (gx - half, y1 + 34 + k * 8), (gx + half, y1 + 34 + k * 8), rng, 0.6, 2, False)
    nodes.append({"id": "GND", "role": "ground"})

    return img, {"diagram_type": "circuit", "nodes": nodes, "edges": edges}


def gen_flowchart(rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    # 960 wide, not 800: the "yes" branch box extends to cx+430 and was being clipped.
    img = _canvas(960, 900)
    nodes, edges = [], []
    cx = 380
    y = 90
    _ellipse(img, cx, y, 80, 34, rng)
    _text(img, cx - 34, y - 12, "start", rng, 0.5)
    nodes.append({"id": "start", "role": "start"})
    prev, prev_y = "start", y + 34

    for i in range(int(rng.integers(1, 3))):
        y = prev_y + 90
        _arrow(img, (cx, prev_y), (cx, y - 40), rng)
        _rect(img, cx - 105, y - 40, 210, 80, rng)
        label = ["read x", "compute", "update", "save"][i % 4]
        _text(img, cx - 60, y - 16, label, rng, 0.5)
        nodes.append({"id": label, "role": "process"})
        edges.append({"src": prev, "dst": label, "label": ""})
        prev, prev_y = label, y + 40

    y = prev_y + 110
    _arrow(img, (cx, prev_y), (cx, y - 62), rng)
    _diamond(img, cx, y, 240, 124, rng)
    cond = rng.choice(["x > 0?", "age >= 18?", "found?"])
    _text(img, cx - 58, y - 14, str(cond), rng, 0.5)
    nodes.append({"id": str(cond), "role": "decision"})
    edges.append({"src": prev, "dst": str(cond), "label": ""})

    _arrow(img, (cx + 120, y), (cx + 250, y), rng)
    _text(img, cx + 150, y - 42, "yes", rng, 0.45)
    _rect(img, cx + 250, y - 38, 180, 76, rng)
    _text(img, cx + 274, y - 14, "print ok", rng, 0.5)
    nodes.append({"id": "print ok", "role": "process"})
    edges.append({"src": str(cond), "dst": "print ok", "label": "yes"})

    ey = y + 190
    _arrow(img, (cx, y + 62), (cx, ey - 34), rng)
    _text(img, cx + 14, y + 96, "no", rng, 0.45)
    _ellipse(img, cx, ey, 78, 34, rng)
    _text(img, cx - 26, ey - 12, "end", rng, 0.5)
    nodes.append({"id": "end", "role": "end"})
    edges.append({"src": str(cond), "dst": "end", "label": "no"})

    return img, {"diagram_type": "flowchart", "nodes": nodes, "edges": edges}


def gen_wireframe(rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    img = _canvas(760, 940)
    nodes = []
    _rect(img, 40, 40, 680, 860, rng)
    _rect(img, 40, 40, 680, 90, rng)
    _text(img, 70, 62, "HEADER", rng, 0.55)
    nodes.append({"id": "header", "role": "container"})

    y = 190
    for i in range(int(rng.integers(2, 5))):
        _rect(img, 90, y, 580, 74, rng)
        _line(img, (110, y + 40), (110 + rng.integers(120, 320), y + 40), rng, 1.0, 1)
        nodes.append({"id": f"input{i}", "role": "ui-input"})
        y += 110

    _rect(img, 430, y + 20, 230, 78, rng)
    _line(img, (470, y + 60), (620, y + 60), rng, 1.0, 2)
    nodes.append({"id": "submit", "role": "ui-button"})

    if rng.random() < 0.5:  # image placeholder: a box with a cross
        _rect(img, 90, y + 150, 240, 180, rng)
        _line(img, (90, y + 150), (330, y + 330), rng, 1.2, 1)
        _line(img, (330, y + 150), (90, y + 330), rng, 1.2, 1)
        nodes.append({"id": "image", "role": "ui-image"})

    return img, {"diagram_type": "wireframe", "nodes": nodes, "edges": []}


GENERATORS = {
    "state_machine": gen_state_machine,
    "er_diagram": gen_er_diagram,
    "circuit": gen_circuit,
    "flowchart": gen_flowchart,
    "wireframe": gen_wireframe,
}


def generate(count: int, types: list[str], seed: int = 42, augment_fraction: float = 0.6) -> dict:
    """Generate `count` diagrams per type, applying the Phase 1.3.6 augmentations to most."""
    from src.ingest.augment import augment as augment_fn

    set_seed(seed)
    rng = np.random.default_rng(seed)
    OUT.mkdir(parents=True, exist_ok=True)
    index = []

    for dtype in types:
        gen = GENERATORS[dtype]
        d = OUT / dtype
        d.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            img, graph = gen(rng)
            applied: list[str] = []
            if rng.random() < augment_fraction:
                img, applied = augment_fn(img, rng)
            name = f"{dtype}_{i:05d}.jpg"
            cv2.imwrite(str(d / name), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            graph.update(
                {
                    "id": f"synthetic/{dtype}/{i:05d}",
                    "file": f"data/processed/synthetic/{dtype}/{name}",
                    "augmentations": applied,
                    "synthetic": True,
                }
            )
            index.append(graph)

    with (OUT / "index.json").open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(index, fh, indent=2)

    per_type: dict[str, int] = {}
    for g in index:
        per_type[g["diagram_type"]] = per_type.get(g["diagram_type"], 0) + 1
    return {
        "generated": len(index),
        "per_type": per_type,
        "with_graph": sum(1 for g in index if g["nodes"]),
        "augmented": sum(1 for g in index if g["augmentations"]),
        "index": str((OUT / "index.json").relative_to(ROOT)),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--count", type=int, default=200, help="diagrams per type")
    ap.add_argument("--types", nargs="*", default=list(GENERATORS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grid", action="store_true", help="also write a preview grid")
    args = ap.parse_args(argv)

    r = generate(args.count, args.types, args.seed)
    print(json.dumps(r, indent=2))

    if args.grid:
        rng = np.random.default_rng(args.seed)
        panels = [(t, GENERATORS[t](rng)[0]) for t in args.types]
        hh = 420
        tiles = []
        for _, im in panels:
            s = hh / im.shape[0]
            tiles.append(cv2.resize(im, None, fx=s, fy=s))
        wmax = sum(t.shape[1] for t in tiles)
        canvas = np.full((hh, wmax, 3), 255, np.uint8)
        x = 0
        for t in tiles:
            canvas[:, x : x + t.shape[1]] = t
            x += t.shape[1]
        grid = ROOT / "reports" / "figures" / "p1_synthetic_grid.png"
        grid.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(grid), canvas)
        print(f"wrote {grid.relative_to(ROOT)}")

    checks = {
        "generated_requested_count": r["generated"] == args.count * len(args.types),
        "every_diagram_has_a_graph": r["with_graph"] == r["generated"],
        "some_are_augmented": r["augmented"] > 0,
        "index_written": (OUT / "index.json").is_file(),
    }
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
