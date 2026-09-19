"""Phase 9.3.1 support - cache CRAFT's line boxes per page, so the geometry can be re-run cheaply.

    python -m src.ocr.craftcache --build          # every page, hdbpmn + fa_bresler
    python -m src.ocr.craftcache --build --limit 50

`labelcrops.cached_lines` reads `experiments/ocr/craft/boxes3.json`, and `vertical.py` builds that
from `boxes2.json`, but **nothing in the tree writes `boxes2.json`** - the detection pass that
produced it did not survive the rebuild that followed the empty DVC store. This is that pass.

## Why it matters, and what it does not change

Detection itself is not lost: `labelcrops.text_boxes` is CRAFT (easyocr) and still works, and
`labelcrops.page_labels` falls back to calling it live when the cache is absent. So the corpus
builder is *slow* without this, not wrong - about 2.2 s of detection per page, repaid on every
later pass over the same corpus.

`ownlearn.build` is the one that genuinely needs it: it reads `cached_lines` and `continue`s when
a page has none, so with no cache it skips **every** page, recognises 0 candidate crops and dies
zipping empty columns. That is the failure this module exists to remove.

The output is deliberately `boxes2.json`, not `boxes3.json`: `vertical --build` reads the former
and writes the latter, adding the rotated container titles CRAFT cannot see on its own (42.6% of
Participants and 28.4% of pools had no candidate line in their own strip). Writing `boxes3.json`
here directly would silently drop those.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from src.utils.config import ROOT

OUT = ROOT / "experiments" / "ocr" / "craft" / "boxes2.json"

#: The corpora whose pages carry labels worth detecting. flowchartseg annotates no text at all.
SOURCES = ("hdbpmn", "fa_bresler")


def build(limit: int | None = None, out: Path = OUT) -> dict:
    """Run CRAFT over every page once and store the line boxes it finds."""
    from src.detect.dataset import image_for
    from src.ocr.labelcrops import text_boxes
    from src.parse.sequences import load_ir
    from src.preprocess.exif import load

    out.parent.mkdir(parents=True, exist_ok=True)
    cache: dict[str, dict] = {}
    if out.is_file():
        try:
            cache = json.loads(out.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cache = {}

    started = time.perf_counter()
    done = skipped = 0
    for diagram in load_ir(list(SOURCES), limit=limit):
        page = diagram["id"]
        if page in cache:
            continue
        path = image_for(diagram)
        if not path or not Path(path).is_file():
            skipped += 1
            continue
        image = load(Path(path), grayscale=True)
        if image is None:
            skipped += 1
            continue
        cache[page] = {"boxes": [list(map(int, b)) for b in text_boxes(image)]}
        done += 1
        if done % 25 == 0:
            rate = (time.perf_counter() - started) / done
            print(f"[craftcache] {done} pages, {rate:.2f}s/page", flush=True)
            out.write_text(json.dumps(cache), encoding="utf-8")

    out.write_text(json.dumps(cache), encoding="utf-8")
    boxes = sum(len(v["boxes"]) for v in cache.values())
    result = {
        "pages": len(cache),
        "detected_now": done,
        "skipped_no_image": skipped,
        "boxes": boxes,
        "boxes_per_page": round(boxes / max(1, len(cache)), 2),
        "seconds": round(time.perf_counter() - started, 1),
        "out": str(out),
    }
    print(json.dumps(result, indent=2))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    if not args.build:
        parser.error("nothing to do: pass --build")
    build(args.limit, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
