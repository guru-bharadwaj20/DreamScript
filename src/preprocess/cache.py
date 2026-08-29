"""Phase 3.2.9 - a cache for the primitives of a page.

Everything 3.2 extracts - components, contours, polygons, segments, arrowheads, text boxes, the
two layers' statistics - is a pure function of one image and a set of tuned constants. It is
also slow: several seconds a page, most of it in thinning and MSER. Phase 4 will read these
primitives once per feature experiment and Phase 5 once per training epoch, so recomputing them
is the difference between an experiment that takes a minute and one that takes an hour.

The cache is a pickle per page under `data/interim/primitives/`.

## The only interesting part: knowing when the cache is wrong

A cache that returns a stale answer is worse than no cache, because the staleness shows up as an
unexplained result three phases later. Two things are therefore recorded with every entry and
checked on every read:

* **The image's content hash.** Not its path, not its modification time - a file copied,
  re-exported or re-downloaded has a new mtime and the same pixels, and the reverse also
  happens.
* **The extractor's fingerprint**: the module version plus every tuned constant that goes into
  the primitives, hashed. Change `MIN_BARB_ANGLE` or the binarization method and the fingerprint
  changes, so the entry is recomputed rather than silently reused. This is what makes the cache
  safe to keep across a phase that re-tunes a threshold - which is exactly what 3.3.2 does.

A mismatch on either is a miss, not an error. `load_or_compute` returns the primitives and
whether they came from disk, so the caller can report a hit rate rather than guess at one.

Measured over 30 hdBPMN pages, 32 threads: **cold 153.9 s, warm 0.14 s, 30/30 hits** - three
orders of magnitude, and the cached primitives compare equal to the computed ones field by
field, which is checked rather than assumed. 12 MB of pickle for 30 pages, most of it the
116,733 line segments.

    python -m src.preprocess.cache --limit 30
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from src.utils.config import ROOT

OUT = ROOT / "data" / "interim" / "primitives"

#: Bumped by hand when the *structure* of what is cached changes, as opposed to a constant.
VERSION = 1


@dataclass
class Primitives:
    """Everything Phase 3.2 extracts from one page, plus what it was extracted from."""

    id: str
    image: str
    image_hash: str
    fingerprint: str
    shape: tuple[int, int]
    components: list[dict] = field(default_factory=list)
    contours: list[dict] = field(default_factory=list)
    segments: list[dict] = field(default_factory=list)
    arrowheads: list[dict] = field(default_factory=list)
    text_boxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    layers: dict = field(default_factory=dict)
    skeleton: dict = field(default_factory=dict)
    seconds: float = 0.0

    def summary(self) -> dict:
        return {
            "id": self.id,
            "components": len(self.components),
            "contours": len(self.contours),
            "segments": len(self.segments),
            "arrowheads": len(self.arrowheads),
            "text_boxes": len(self.text_boxes),
        }


def image_hash(path: Path) -> str:
    """SHA-1 of the file's bytes: the identity of the pixels, not of the path or the clock."""
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint() -> str:
    """A hash of every tuned constant that affects the primitives.

    Collected by reading the modules' own module-level constants rather than by maintaining a
    list here, so a constant added later is covered without anyone remembering to add it.
    """
    from src.preprocess import binarize, denoise, rules, thinning
    from src.preprocess.primitives import arrowheads, contours, polygons, segments
    from src.preprocess.primitives import text as text_module

    modules = [
        binarize,
        denoise,
        rules,
        thinning,
        contours,
        polygons,
        segments,
        arrowheads,
        text_module,
    ]
    parts = [f"v{VERSION}"]
    for module in modules:
        for name in sorted(dir(module)):
            if not name.isupper() or name.startswith("_"):
                continue
            value = getattr(module, name)
            if isinstance(value, int | float | str | bool | tuple | list):
                parts.append(f"{module.__name__}.{name}={value}")
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


def path_for(page_id: str, directory: Path = OUT) -> Path:
    return directory / f"{page_id}.pkl"


def compute(page_id: str, image_path: Path) -> Primitives:
    """Extract every 3.2 primitive from one page. Slow, by design; this is what is cached."""
    from src.preprocess import layers as ly
    from src.preprocess.primitives import arrowheads as ah
    from src.preprocess.primitives import components as cc
    from src.preprocess.primitives import contours as ct
    from src.preprocess.primitives import segments as sg
    from src.preprocess.primitives import text as tx
    from src.preprocess.thinning import branch_points, end_points, thin

    started = time.perf_counter()
    gray, mask = ly.prepare(image_path)
    split = ly.separate(mask, gray)
    skeleton = thin(mask)
    # Geometry is read off the *shape* layer: that is the whole point of having split it.
    found = ct.drawn_outlines(ct.extract(split.shape))
    return Primitives(
        id=page_id,
        image=(
            str(image_path.relative_to(ROOT))
            if image_path.is_relative_to(ROOT)
            else str(image_path)
        ),
        image_hash=image_hash(image_path),
        fingerprint=fingerprint(),
        shape=tuple(int(v) for v in mask.shape),
        components=[c.to_dict() for c in cc.extract(split.shape)],
        contours=[c.to_dict() for c in found],
        segments=[s.to_dict() for s in sg.detect_both(gray, split.shape)],
        arrowheads=[a.to_dict() for a in ah.detect(split.shape)],
        text_boxes=[tuple(int(v) for v in box) for box in tx.propose(mask, gray)],
        layers=split.stats(),
        skeleton={
            "pixels": int(skeleton.sum()),
            "branch_points": int(branch_points(skeleton).sum()),
            "end_points": int(end_points(skeleton).sum()),
        },
        seconds=round(time.perf_counter() - started, 3),
    )


def save(primitives: Primitives, directory: Path = OUT) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    target = path_for(primitives.id, directory)
    # Written to a neighbouring file and moved into place, so an interrupted run leaves either
    # the old entry or the new one and never a half-written pickle that unpickles as garbage.
    temporary = target.with_suffix(".pkl.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(primitives, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(target)
    return target


def load(page_id: str, directory: Path = OUT) -> Primitives | None:
    target = path_for(page_id, directory)
    if not target.is_file():
        return None
    try:
        with target.open("rb") as handle:
            loaded = pickle.load(handle)
    except (pickle.UnpicklingError, EOFError, AttributeError, ImportError):
        # A pickle written by an older layout is a miss, not a crash.
        return None
    return loaded if isinstance(loaded, Primitives) else None


def is_current(primitives: Primitives, image_path: Path) -> bool:
    return (
        primitives.fingerprint == fingerprint()
        and image_path.is_file()
        and primitives.image_hash == image_hash(image_path)
    )


def load_or_compute(
    page_id: str, image_path: Path, directory: Path = OUT
) -> tuple[Primitives, bool]:
    """(primitives, came_from_cache). A stale or unreadable entry is a miss, not an error."""
    cached = load(page_id, directory)
    if cached is not None and is_current(cached, image_path):
        return cached, True
    fresh = compute(page_id, image_path)
    save(fresh, directory)
    return fresh, False


def _pages(limit: int) -> list[tuple[str, Path]]:
    from src.ir.model import SUFFIX, Diagram

    out = []
    for path in sorted((ROOT / "data" / "processed" / "ir" / "hdbpmn").glob(f"*{SUFFIX}"))[:limit]:
        image = ROOT / Diagram.load(path).meta["image"]
        if image.is_file():
            out.append((path.stem, image))
    return out


def warm(limit: int = 30) -> dict:
    """Two passes over the same pages: the second must be all hits, and much faster."""
    from src.utils.parallel import pstarmap

    pages = _pages(limit)
    if not pages:
        return {"pages": 0}

    def pass_over() -> tuple[float, int, list[Primitives]]:
        started = time.perf_counter()
        results = pstarmap(load_or_compute, pages, prefer="threads")
        elapsed = time.perf_counter() - started
        return elapsed, sum(1 for _, hit in results if hit), [p for p, _ in results]

    cold_seconds, cold_hits, cold = pass_over()
    warm_seconds, warm_hits, warm_results = pass_over()
    return {
        "pages": len(pages),
        "cold_seconds": round(cold_seconds, 2),
        "cold_hits": cold_hits,
        "warm_seconds": round(warm_seconds, 2),
        "warm_hits": warm_hits,
        "speedup": round(cold_seconds / warm_seconds, 1) if warm_seconds else 0.0,
        "identical": all(asdict(a) == asdict(b) for a, b in zip(cold, warm_results, strict=True)),
        "primitives": {
            key: int(np.sum([len(getattr(p, key)) for p in warm_results]))
            for key in ("components", "contours", "segments", "arrowheads", "text_boxes")
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--clear", action="store_true", help="delete the cache before running")
    args = ap.parse_args(argv)

    if args.clear and OUT.is_dir():
        for stale in OUT.glob("*.pkl"):
            stale.unlink()

    result = warm(args.limit)
    if not result["pages"]:
        print("no pages; run the hdBPMN converter first", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))

    checks = {
        "second_pass_is_all_hits": result["warm_hits"] == result["pages"],
        "cache_is_faster_than_recomputing": result["speedup"] > 5.0,
        "cached_primitives_equal_computed_ones": result["identical"],
    }
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
