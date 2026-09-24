"""Phase 10 shared foundation - the pages assembly is measured on, and what is known about them.

Every module in Phase 10 needs the same three things about a page: the image, what the detector
says is on it, and what is actually on it. Collecting that in one place is not tidiness - it is
what makes the eight assembly tasks comparable, because a binding accuracy and a tracing recall
quoted on different page sets are two numbers that cannot be put in the same sentence.

## The page set, and why it is the detector's val and test splits

9.1.2 exported 2,312 pages into a YOLO layout and held out 308 val and 162 test. Those pages are
the ones where a *predicted* box can be compared against a *ground-truth* graph, because the IR
file that produced the label is still on disk. The 1,842 training pages are excluded: the
detector has seen them, so assembly measured on them would be measuring memorisation.

## Coordinates: everything is in detector-image pixels

Three coordinate systems meet here and mixing them silently is the obvious way to get a number
that looks plausible and is wrong:

    native      the IR's own bboxes, in the original photograph's pixels (up to 2837 wide)
    detect      9.1.2's resized page, `native * page.scale`, which is what the model sees
    normalised  YOLO's cx/cy/w/h in [0, 1]

`detect` is the common frame, because it is the only one in which a predicted box and a stroke
skeleton are both measurable without a second conversion. `truth()` scales the IR into it on the
way out and records the scale it used, so anything reading a bbox from this module is reading
detector pixels and nothing else.

## Detections are cached, once

Inference over the held-out pages at 896px is a pure function of the weights, so it is run once
into `experiments/assemble/detections.json` and read from there. The cache records the weights'
content hash; a different checkpoint is a miss rather than a stale answer, which is the lesson
3.2.9's primitives cache paid for already.

    python -m src.assemble.corpus            # build the detection cache and summarise it
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.detect.classes import CLASSES
from src.detect.train import WEIGHTS
from src.ir.model import Diagram
from src.utils.config import ROOT
from src.utils.splits import normalise

DATA = ROOT / "data" / "processed" / "detect"
IR = ROOT / "data" / "processed" / "ir"
RUNS = ROOT / "experiments" / "assemble"
CACHE = RUNS / "detections.json"

#: The splits the detector did not train on. Assembly is only measured here.
HELD_OUT = ("val", "test")

#: 9.1.1's chosen inference size, kept identical so the cached boxes are the reported ones.
IMGSZ = 896

#: Floored well below the 0.25 default: assembly's job includes deciding which weak boxes to
#: keep, and a filter applied before assembly sees them makes that decision unmeasurable.
CONF_FLOOR = 0.05


@dataclass(frozen=True)
class Page:
    """One held-out page: the pixels, the detector's frame, and where the truth lives."""

    name: str
    id: str
    source: str
    split: str
    scale: float
    native_size: tuple[int, int]

    @property
    def image(self) -> Path:
        return DATA / "images" / self.split / f"{self.name}.png"

    @property
    def ir_path(self) -> Path:
        return IR / self.source / f"{self.id}.ir.json"

    @property
    def size(self) -> tuple[float, float]:
        """The page's width and height in detector pixels."""
        w, h = self.native_size
        return (w * self.scale, h * self.scale)

    @property
    def diagonal(self) -> float:
        w, h = self.size
        return float((w * w + h * h) ** 0.5)


@lru_cache(maxsize=1)
def index() -> list[dict[str, Any]]:
    return json.loads((DATA / "index.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=8)
def pages(splits: tuple[str, ...] = HELD_OUT, source: str | None = None) -> tuple[Page, ...]:
    """The held-out pages, in a fixed order so any two runs enumerate them identically.

    `splits` is normalised rather than compared as text. This index spells the middle split
    `val` and the manifest spells it `validation`, so `pages(("validation",))` used to match
    nothing and return an empty tuple - a typo and the other half of the project's own
    vocabulary both produced "no pages" rather than an error. `utils.splits.normalise` raises on
    a spelling that is neither.
    """
    wanted = {normalise(name) for name in splits}
    out = []
    for row in index():
        if normalise(row["split"]) not in wanted or (source and row["source"] != source):
            continue
        page = Page(
            name=row["name"],
            id=row["id"],
            source=row["source"],
            split=row["split"],
            scale=float(row["scale"]),
            native_size=tuple(row["native_size"]),
        )
        if page.ir_path.is_file() and page.image.is_file():
            out.append(page)
    return tuple(sorted(out, key=lambda p: p.name))


def scale_box(bbox: list[float], scale: float) -> list[float]:
    """`[x, y, w, h]` from native pixels into detector pixels."""
    return [float(v) * scale for v in bbox]


def truth(page: Page) -> Diagram:
    """The ground-truth graph, with every geometry scaled into detector pixels.

    The IR is loaded through the dataclass rather than as a dict because the consumer's loader is
    the contract - 9.3.7 shipped a bug that round-tripped through `json` perfectly and broke
    `Diagram.from_dict`, and going through `Diagram` here means Phase 10 cannot repeat it.
    """
    diagram = Diagram.load(page.ir_path)
    s = page.scale
    for node in diagram.nodes:
        if node.bbox is not None:
            node.bbox = scale_box(node.bbox, s)
    for edge in diagram.edges:
        if edge.polyline is not None:
            edge.polyline = [[x * s, y * s] for x, y in edge.polyline]
    diagram.meta["assemble_scale"] = s
    diagram.meta["assemble_frame"] = "detect-pixels"
    return diagram


def _weights_hash(weights: Path) -> str:
    digest = hashlib.sha1()
    with weights.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:16]


def predict(pages_: tuple[Page, ...], weights: Path = WEIGHTS, imgsz: int = IMGSZ) -> dict:
    """Run 9.1's detector over the held-out pages. Boxes come back as `[x, y, w, h]`."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    out: dict[str, list[dict]] = {}
    for page in pages_:
        result = model.predict(
            source=str(page.image),
            imgsz=imgsz,
            conf=CONF_FLOOR,
            iou=0.7,
            verbose=False,
            device=0,
        )[0]
        boxes = result.boxes
        rows: list[dict] = []
        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), score, index_ in zip(xyxy, conf, cls, strict=True):
                rows.append(
                    {
                        "cls": CLASSES[int(index_)],
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "score": round(float(score), 4),
                    }
                )
        rows.sort(key=lambda r: -r["score"])
        out[page.name] = rows
    return {
        "weights": _weights_hash(weights),
        "imgsz": imgsz,
        "conf_floor": CONF_FLOOR,
        "pages": out,
    }


def detections(page: Page | None = None, *, weights: Path = WEIGHTS, rebuild: bool = False) -> Any:
    """Cached detector output. With no page, the whole `{name: [box, ...]}` mapping."""
    payload = _cache(weights, rebuild)
    if page is None:
        return payload["pages"]
    return payload["pages"].get(page.name, [])


@lru_cache(maxsize=2)
def _cache(weights: Path = WEIGHTS, rebuild: bool = False) -> dict:
    if CACHE.is_file() and not rebuild:
        payload = json.loads(CACHE.read_text(encoding="utf-8"))
        if payload.get("weights") == _weights_hash(weights):
            return payload
    payload = predict(pages(), weights)
    RUNS.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(payload), encoding="utf-8")
    return payload


# -- geometry every assembly task needs ---------------------------------------------------


def centre(bbox: list[float]) -> tuple[float, float]:
    x, y, w, h = bbox
    return (x + w / 2.0, y + h / 2.0)


def corners(bbox: list[float]) -> tuple[float, float, float, float]:
    """`(x1, y1, x2, y2)`."""
    x, y, w, h = bbox
    return (x, y, x + w, y + h)


def iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = corners(a)
    bx1, by1, bx2, by2 = corners(b)
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    union = max(a[2] * a[3], 0.0) + max(b[2] * b[3], 0.0) - inter
    return float(inter / union) if union > 0 else 0.0


def contains_fraction(inner: list[float], outer: list[float]) -> float:
    """Share of `inner`'s area that falls inside `outer`. Containment, not overlap."""
    ix1, iy1, ix2, iy2 = corners(inner)
    ox1, oy1, ox2, oy2 = corners(outer)
    ix = max(0.0, min(ix2, ox2) - max(ix1, ox1))
    iy = max(0.0, min(iy2, oy2) - max(iy1, oy1))
    area = max(inner[2] * inner[3], 0.0)
    return float(ix * iy / area) if area > 0 else 0.0


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args(argv)

    held = pages()
    if not WEIGHTS.is_file():
        print(f"no detector weights at {WEIGHTS}", file=sys.stderr)
        return 1
    boxes = detections(rebuild=args.rebuild)
    counts = {s: sum(1 for p in held if p.source == s) for s in sorted({p.source for p in held})}
    print(
        json.dumps(
            {
                "pages": len(held),
                "by_source": counts,
                "detections": sum(len(v) for v in boxes.values()),
                "cache": str(CACHE.relative_to(ROOT)),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
