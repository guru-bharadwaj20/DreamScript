"""Phase 3.2.8 - shape and text as two separate layers.

3.2.7 proposes where the writing is. This splits the page on that proposal and writes the two
halves out, because every later phase wants one of them and not the other:

    text_layer.png     what Phase 9's OCR reads
    shape_layer.png    what Phases 4 and 10 measure geometry on

The split is a **partition**, and that is deliberate. Every ink pixel goes to exactly one layer,
the two never overlap, and their union is the input mask. It is easy to write a separator whose
layers quietly lose a few percent of the ink to neither side, and the loss is then invisible
until a downstream phase is missing an edge. `check_partition` asserts the property, and the
test suite pins it.

Components move whole, and that is what keeps the drawing intact: lifting a word out of a box
does not touch the box. The case it cannot handle is a *label written across a connector*,
which binarization leaves as one component with that
connector: it stays in the shape layer rather than being torn in half. Text that touches
drawing is therefore not separated, and it lands with the drawing. That is the safer direction
- a stray word among the shapes costs Phase 4 a little noise, while a piece of a connector
missing from the shape layer costs Phase 10 an edge.

## What the layers look like on real pages

Over 40 hdBPMN photographs the text layer takes a **median 33% of the ink pixels** but 2,882 of
the 5,252 components - and the ratio between those two numbers is the point. Handwriting is
many small components; drawing is a few large ones. A split that got this backwards would be
obviously wrong, and this one does not.

The absolute share is higher than a page of BPMN looks, and 3.2.7 measured why: 16% of
annotated connector ink is claimed by the text proposal. The layers are a useful separation,
not a clean one, and Phase 9 should expect drawing debris in what it OCRs. The per-page numbers
are in `reports/layers.md`.

    python -m src.preprocess.layers --limit 40
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.utils.config import ROOT

OUT = ROOT / "data" / "interim" / "layers"
REPORT = ROOT / "reports" / "layers.md"

#: Working width every page is resized to before the split, so thresholds expressed in pixels
#: mean the same thing on a 1000px scan and a 4000px photograph.
WORKING_WIDTH = 1400


@dataclass
class Layers:
    text: np.ndarray  # bool
    shape: np.ndarray  # bool

    @property
    def ink(self) -> np.ndarray:
        return self.text | self.shape

    def stats(self) -> dict:
        from src.preprocess.denoise import count_components

        total = int(self.ink.sum())
        return {
            "ink_pixels": total,
            "text_pixel_share": round(float(self.text.sum() / total), 4) if total else 0.0,
            "text_components": count_components(self.text),
            "shape_components": count_components(self.shape),
        }

    def write(self, stem: str, directory: Path = OUT) -> dict[str, Path]:
        directory.mkdir(parents=True, exist_ok=True)
        paths = {
            "text": directory / f"{stem}_text_layer.png",
            "shape": directory / f"{stem}_shape_layer.png",
        }
        # Ink black on white, the same polarity as the photograph it came from, so the files can
        # be looked at without inverting them in one's head.
        cv2.imwrite(str(paths["text"]), np.where(self.text, 0, 255).astype(np.uint8))
        cv2.imwrite(str(paths["shape"]), np.where(self.shape, 0, 255).astype(np.uint8))
        return paths


def separate(mask: np.ndarray, gray: np.ndarray | None = None) -> Layers:
    """Split an ink mask into writing and drawing."""
    from src.preprocess.primitives import text as tx

    text = tx.refine(mask, tx.propose(mask, gray))
    return Layers(text=text & mask, shape=mask & ~text)


def check_partition(layers: Layers, mask: np.ndarray) -> dict[str, bool]:
    """The three properties that make this a split rather than two filters."""
    return {
        "layers_disjoint": not bool((layers.text & layers.shape).any()),
        "union_is_the_input": bool(np.array_equal(layers.ink, mask.astype(bool))),
        "nothing_invented": not bool((layers.ink & ~mask.astype(bool)).any()),
    }


def prepare(image_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(grayscale, ink mask) at the working width, through the 3.1 pipeline."""
    from src.preprocess.binarize import binarize
    from src.preprocess.denoise import denoise, median
    from src.preprocess.exif import load
    from src.preprocess.rules import suppress

    gray = load(image_path, grayscale=True)
    scale = WORKING_WIDTH / max(gray.shape)
    if scale < 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return gray, denoise(suppress(binarize(median(gray))))


def run(limit: int = 40, *, write: bool = True) -> list[dict]:
    """Split `limit` real pages and write both layers for each."""
    from src.ir.model import SUFFIX, Diagram
    from src.utils.parallel import pmap

    paths = sorted((ROOT / "data" / "processed" / "ir" / "hdbpmn").glob(f"*{SUFFIX}"))[:limit]

    def one(path):
        diagram = Diagram.load(path)
        image_path = ROOT / diagram.meta["image"]
        if not image_path.is_file():
            return None
        gray, mask = prepare(image_path)
        layers = separate(mask, gray)
        row = {"id": path.stem, **layers.stats(), **check_partition(layers, mask)}
        if write:
            written = layers.write(path.stem)
            row["files"] = [str(p.relative_to(ROOT)) for p in written.values()]
            row["both_written"] = all(p.is_file() for p in written.values())
        return row

    return [r for r in pmap(one, paths, prefer="threads") if r]


def report(rows: list[dict]) -> Path:
    shares = [r["text_pixel_share"] for r in rows]
    text_components = sum(r["text_components"] for r in rows)
    shape_components = sum(r["shape_components"] for r in rows)
    lines = [
        "# Phase 3.2.8 — shape and text layers",
        "",
        f"{len(rows)} hdBPMN photographs, split by `src/preprocess/layers.py`. Both layers are ",
        f"written per page to `{OUT.relative_to(ROOT).as_posix()}` as PNG, ink black on white.",
        "",
        "| | |",
        "| :--- | ---: |",
        f"| pages | {len(rows)} |",
        f"| median text share of ink | {np.median(shares):.1%} |",
        f"| text components | {text_components} |",
        f"| shape components | {shape_components} |",
        f"| pages where both files were written | {sum(r['both_written'] for r in rows)} |",
        f"| pages where the split is a partition | {sum(r['union_is_the_input'] for r in rows)} |",
        "",
        "The text layer holds more than half the *components* and a third of the *pixels*: ",
        "handwriting is many small marks, drawing is a few large ones, and a split that got ",
        "that ratio backwards would be visibly wrong. The absolute share is an over-claim — ",
        "Phase 3.2.7 measured 16% of annotated connector ink landing in the text proposal — so ",
        "read the text layer as *enriched for writing*, not as writing. Where the share is far ",
        "below the median, the page's writing touches its shapes and stayed with them; the ",
        "module docstring says why that direction was chosen.",
        "",
        "| page | text share | text components | shape components |",
        "| :--- | ---: | ---: | ---: |",
    ]
    for row in sorted(rows, key=lambda r: -r["text_pixel_share"])[:20]:
        lines.append(
            f"| `{row['id']}` | {row['text_pixel_share']:.1%} | "
            f"{row['text_components']} | {row['shape_components']} |"
        )
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return REPORT


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=40)
    args = ap.parse_args(argv)

    rows = run(args.limit)
    if not rows:
        print("no pages; run the hdBPMN converter first", file=sys.stderr)
        return 1
    path = report(rows)
    print(json.dumps({"pages": len(rows), "report": str(path.relative_to(ROOT))}, indent=2))

    checks = {
        "both_layers_written_for_every_page": all(r["both_written"] for r in rows),
        "every_split_is_a_partition": all(
            r["layers_disjoint"] and r["union_is_the_input"] and r["nothing_invented"] for r in rows
        ),
        "text_layer_is_never_the_whole_page": all(r["text_pixel_share"] < 0.9 for r in rows),
    }
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
