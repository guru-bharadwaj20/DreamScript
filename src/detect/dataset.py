"""Phase 9.1.2 - the detection corpus on disk, in the format every detector in 9.1.1 can read.

    python -m src.detect.dataset                    # build data/processed/detect
    python -m src.detect.dataset --summary          # what was built, without rebuilding

Turns 9.1.2's frozen class set into a YOLO-layout dataset: one `.txt` of normalised
`cls cx cy w h` per image, `images/{train,val,test}` beside `labels/{train,val,test}`, and a
`data.yaml` naming them. torchvision's Faster R-CNN cannot read that layout, so `records()`
returns the same boxes as plain Python and 9.1.1's third arm consumes those - one export, two
readers, no chance of the two detectors being compared on different boxes.

## Three decisions the export makes

**Splits are inherited, never re-derived.** hdbpmn and flowchartseg carry a `split` in 1.3.3's
manifest, which is writer-disjoint where a writer identity exists; fa_bresler is not in the
manifest at all, so it is split **by scribe** here, 25 writers assigned largest-first into a
70/15/15 quota. Re-deriving hdbpmn's split would silently break the property every score in
Phases 5-8 depends on.

**Pages are resized to a 1280px long side and the boxes with them.** An hdbpmn photograph is
2837x2000, and decoding 693 of those every epoch costs more than the training step does. The
scale factor is recorded per page in `index.json` so 9.1.5 can work from the originals when it
tests higher input resolution. The cost is stated rather than hidden: the export measures its
own median arrowhead and it is **15.6 px** at 1280, against a median node of some hundreds -
which is where the small-object problem 9.1.5 exists to measure comes from, and it is a
consequence of this resize as much as of how people draw.

**fa_bresler is rendered here, deterministically.** Its IR coordinates already live in the
1024px canvas `chaos_builder.render_inkml` produces, so rendering is what makes those
coordinates mean anything. The renders are written once to `data/processed/fa_render/`.

## The mix, and the caveat that travels with it

Roughly 2,300 pages, of which **693 are photographs of paper and 1,619 are not**. Only the
hdbpmn rows answer the deployment question, so `records()` keeps `source` on every page and
9.1.4 reports hdbpmn separately rather than quoting a pooled mAP that is three-quarters
rendered pages.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

from src.detect.classes import CLASS_INDEX, CLASSES, SOURCES, boxes_for, pages
from src.utils.config import ROOT
from src.utils.splits import CANONICAL, DETECT, detect_name

OUT = ROOT / "data" / "processed" / "detect"
FA_RENDER = ROOT / "data" / "processed" / "fa_render"
MANIFEST = ROOT / "data" / "processed" / "manifest.parquet"

#: Long-side pixels for the exported copy. 1280 is ultralytics' habitual training size, so the
#: export and the trainer agree and no page is resized twice.
LONG_SIDE = 1280

#: The manifest spells it `validation`; YOLO's data.yaml spells it `val`. Both tables come from
#: `src.utils.splits`, which is the one place the two vocabularies are related.
SPLITS = DETECT
SPLIT_ALIAS = {name: detect_name(name) for name in (*CANONICAL, *DETECT)}

RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


# ------------------------------------------------------------------------------------------
# images
# ------------------------------------------------------------------------------------------


def render_fa(diagram: dict[str, Any]) -> Path | None:
    """Rasterize one fa_bresler page into the canvas its IR coordinates already assume."""
    import cv2

    from src.ingest.chaos_builder import render_inkml

    name = diagram["id"]
    out = FA_RENDER / f"{name}.png"
    if out.is_file():
        return out
    source_file = _fa_inkml(name)
    if source_file is None:
        return None
    canvas = render_inkml(source_file)
    if canvas is None:
        return None
    FA_RENDER.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), canvas)
    return out


def _fa_inkml(diagram_id: str) -> Path | None:
    """`writer000_fa_001` -> the InkML file under data/raw/fa_bresler.

    The IR id is the InkML stem verbatim, which is why this is a lookup rather than a search.
    """
    root = ROOT / "data" / "raw" / "fa_bresler"
    if not root.is_dir():
        return None
    direct = root / "FA_1.0" / f"{diagram_id}.inkml"
    if direct.is_file():
        return direct
    return next((p for p in root.rglob(f"{diagram_id}.inkml") if p.is_file()), None)


def image_for(diagram: dict[str, Any]) -> Path | None:
    """The page image, rendering fa_bresler on demand and returning None when there is none."""
    declared = diagram["meta"].get("image")
    if declared:
        path = ROOT / declared
        return path if path.is_file() else None
    if diagram["meta"].get("source") == "fa_bresler":
        return render_fa(diagram)
    return None


# ------------------------------------------------------------------------------------------
# splits
# ------------------------------------------------------------------------------------------


def manifest_splits() -> dict[str, str]:
    """`(source, key) -> split` from 1.3.3, for the two sources that appear in the manifest."""
    if not MANIFEST.is_file():
        return {}
    import pandas as pd

    frame = pd.read_parquet(MANIFEST, columns=["id", "source", "split"])
    out: dict[str, str] = {}
    for identifier, source, split in zip(frame["id"], frame["source"], frame["split"], strict=True):
        if source == "hdbpmn":
            out[f"hdbpmn:{str(identifier).split('/')[-1]}"] = SPLIT_ALIAS.get(str(split), "train")
    # flowchartseg's IR ids are a running counter over the source parquets in name order, so
    # the manifest's per-native-split counts recover the boundary without guessing at it.
    counts = (
        frame[frame["source"] == "flowchartseg"]["id"]
        .str.split("/")
        .str[1]
        .value_counts()
        .to_dict()
    )
    index = 0
    for native in ("train", "test", "validation"):  # parquet file order, by name
        for _ in range(int(counts.get(native, 0))):
            out[f"flowchartseg:fcseg_{index:05d}"] = SPLIT_ALIAS[native]
            index += 1
    return out


def fa_splits(diagrams: list[dict[str, Any]]) -> dict[str, str]:
    """Writer-disjoint 70/15/15 over fa_bresler's 25 scribes, largest unit first.

    The same rule 1.3.3 applies, restated here because fa_bresler never entered the manifest -
    it is an IR-only source with no page image until this module renders one.
    """
    import collections

    by_writer: dict[str, list[str]] = collections.defaultdict(list)
    for diagram in diagrams:
        by_writer[str(diagram["meta"].get("scribe_id"))].append(diagram["id"])
    total = sum(len(v) for v in by_writer.values())
    quota = {k: v * total for k, v in RATIOS.items()}
    filled = dict.fromkeys(RATIOS, 0)
    out: dict[str, str] = {}
    for writer in sorted(by_writer, key=lambda w: (-len(by_writer[w]), w)):
        target = max(RATIOS, key=lambda k: quota[k] - filled[k])
        for identifier in by_writer[writer]:
            out[f"fa_bresler:{identifier}"] = target
        filled[target] += len(by_writer[writer])
    return out


# ------------------------------------------------------------------------------------------
# records
# ------------------------------------------------------------------------------------------


def records(sources=tuple(SOURCES), limit: int | None = None) -> list[dict[str, Any]]:
    """Every usable page: its image, its split, and its boxes in native image pixels."""
    inherited = manifest_splits()
    out: list[dict[str, Any]] = []
    for source in sources:
        loaded = []
        for path in pages(source):
            diagram = json.loads(path.read_text(encoding="utf-8"))
            loaded.append(diagram)
        assigned = fa_splits(loaded) if source == "fa_bresler" else {}
        for diagram in loaded:
            key = f"{source}:{diagram['id']}"
            image = image_for(diagram)
            if image is None:
                continue
            boxes = boxes_for(diagram)
            if not boxes:
                continue
            width, height = diagram["meta"]["image_size"]
            out.append(
                {
                    "id": diagram["id"],
                    "source": source,
                    "split": assigned.get(key, inherited.get(key, "train")),
                    "image": str(image),
                    "width": int(width),
                    "height": int(height),
                    "boxes": boxes,
                }
            )
            if limit and len(out) >= limit:
                break
        if limit and len(out) >= limit:
            break
    return out


def yolo_lines(record: dict[str, Any]) -> list[str]:
    """`cls cx cy w h`, normalised - the only format all three of 9.1.1's arms agree on."""
    width, height = record["width"], record["height"]
    lines = []
    for box in record["boxes"]:
        x, y, w, h = box["bbox"]
        cx, cy = (x + w / 2) / width, (y + h / 2) / height
        nw, nh = w / width, h / height
        if not (0 <= cx <= 1 and 0 <= cy <= 1) or nw <= 0 or nh <= 0:
            continue
        lines.append(
            f"{CLASS_INDEX[box['cls']]} {cx:.6f} {cy:.6f} {min(nw, 1.0):.6f} {min(nh, 1.0):.6f}"
        )
    return lines


# ------------------------------------------------------------------------------------------
# export
# ------------------------------------------------------------------------------------------


def export(
    out: Path = OUT, sources=tuple(SOURCES), long_side: int = LONG_SIDE, limit: int | None = None
) -> dict:
    """Write the YOLO-layout dataset and return what was written.

    Boxes are normalised, so resizing the page does not touch the label file - which is why
    the scale factor is recorded rather than applied twice.
    """
    import cv2

    rows = records(sources, limit)
    if out.exists():
        shutil.rmtree(out)
    for split in SPLITS:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    index: list[dict[str, Any]] = []
    counts = {split: 0 for split in SPLITS}
    per_class = {name: 0 for name in CLASSES}
    skipped = 0
    for record in rows:
        image = cv2.imread(record["image"], cv2.IMREAD_COLOR)
        if image is None:
            skipped += 1
            continue
        height, width = image.shape[:2]
        scale = min(1.0, long_side / max(height, width))
        if scale < 1.0:
            image = cv2.resize(
                image,
                (int(round(width * scale)), int(round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        split = record["split"]
        name = f"{record['source']}__{record['id']}"
        cv2.imwrite(str(out / "images" / split / f"{name}.png"), image)
        lines = yolo_lines(record)
        (out / "labels" / split / f"{name}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
        counts[split] += 1
        for box in record["boxes"]:
            per_class[box["cls"]] += 1
        index.append(
            {
                "name": name,
                "id": record["id"],
                "source": record["source"],
                "split": split,
                "native_size": [record["width"], record["height"]],
                "scale": round(float(scale), 5),
                "boxes": len(lines),
            }
        )

    yaml = "\n".join(
        [
            "# Phase 9.1.2 - generated by python -m src.detect.dataset. Do not hand-edit.",
            f"path: {out.as_posix()}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "names:",
            *[f"  {i}: {name}" for i, name in enumerate(CLASSES)],
            "",
        ]
    )
    (out / "data.yaml").write_text(yaml, encoding="utf-8")
    (out / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")

    by_source = {}
    for record in index:
        by_source.setdefault(record["source"], {s: 0 for s in SPLITS})[record["split"]] += 1
    return {
        "root": str(out.relative_to(ROOT)),
        "pages": len(index),
        "unreadable_images": skipped,
        "per_split": counts,
        "per_source": by_source,
        "per_class": per_class,
        "long_side": long_side,
        "median_arrowhead_px_at_export": _median_arrowhead(index, rows),
    }


def _median_arrowhead(index: list[dict], rows: list[dict]) -> float | None:
    """How big a derived head actually is after the resize - the number 9.1.5 is about."""
    sizes = []
    scale_by_id = {r["name"].split("__", 1)[1]: r["scale"] for r in index}
    for record in rows:
        scale = scale_by_id.get(record["id"])
        if scale is None:
            continue
        for box in record["boxes"]:
            if box["cls"] == "arrowhead":
                sizes.append(box["bbox"][2] * scale)
    return round(float(np.median(sizes)), 1) if sizes else None


def summary(out: Path = OUT) -> dict:
    index_path = out / "index.json"
    if not index_path.is_file():
        raise SystemExit(f"not built: {index_path} (run `python -m src.detect.dataset`)")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    by_source: dict[str, dict[str, int]] = {}
    for record in index:
        by_source.setdefault(record["source"], {s: 0 for s in SPLITS})[record["split"]] += 1
    return {"pages": len(index), "per_source": by_source}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--sources", nargs="*", default=list(SOURCES))
    ap.add_argument("--long-side", type=int, default=LONG_SIDE)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args(argv)

    if args.summary:
        print(json.dumps(summary(args.out), indent=2))
        return 0
    print(json.dumps(export(args.out, tuple(args.sources), args.long_side, args.limit), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
