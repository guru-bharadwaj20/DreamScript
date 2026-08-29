"""Phase 1.1.1 — hdBPMN: 700+ hand-drawn BPMN diagrams with full shape/edge/label annotation.

Why it is the primary source: it is the only public set that gives *both* component
annotations and the connections between them, on genuinely hand-drawn images, with an
official writer-disjoint split. Everything Phase 10 needs to learn about edges comes from
here.

    python -m src.ingest.datasets.hdbpmn        # clone into data/raw/hdbpmn
    python -m src.ingest.datasets.hdbpmn --check # verify an existing copy
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from src.ingest.registry import Availability, Dataset, Redistribution, register, write_provenance

REPO = "https://github.com/dwslab/hdBPMN.git"

DATASET = register(
    Dataset(
        slug="hdbpmn",
        name="hdBPMN",
        url="https://github.com/dwslab/hdBPMN",
        license="CC-BY-4.0",
        availability=Availability.OPEN,
        redistribution=Redistribution.ATTRIBUTION,
        diagram_types=("flowchart",),  # BPMN maps onto our flowchart class
        phases=("1.1.1", "9.1", "10"),
        data_card="docs/data_cards/hdbpmn.md",
        notes=(
            "Shallow git clone, ~550 MB. Ships data/writer_split.csv, which is a "
            "writer-disjoint split we adopt directly rather than inventing our own."
        ),
    )
)

EXPECTED = ["data/images", "data/annotations", "data/words", "data/writer_split.csv"]


def download(force: bool = False) -> Path:
    target = DATASET.path
    if target.exists() and any(target.iterdir()) and not force:
        print(f"already present: {target}")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"cloning {REPO} -> {target} (~550 MB, shallow)")
    subprocess.run(["git", "clone", "--depth", "1", REPO, str(target)], check=True)
    return target


def check() -> dict:
    """Inventory the local copy: counts, formats and the official writer split."""
    root = DATASET.path
    if not root.is_dir():
        raise SystemExit(f"not downloaded: {root} (run without --check first)")

    missing = [p for p in EXPECTED if not (root / p).exists()]
    images = sorted((root / "data" / "images").rglob("*"))
    images = [p for p in images if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    # Diagram structure is stored as BPMN XML (.bpmn); word-level transcripts are separate
    # .xml files under data/words, which feed the Phase 9.3 OCR sub-pipeline.
    annotations = sorted((root / "data" / "annotations").rglob("*.bpmn"))
    words = sorted((root / "data" / "words").rglob("*.xml"))

    writers: set[str] = set()
    split_counts: dict[str, int] = {}
    split_csv = root / "data" / "writer_split.csv"
    if split_csv.is_file():
        lines = split_csv.read_text(encoding="utf-8").strip().splitlines()
        header = [h.strip() for h in lines[0].split(",")]
        for line in lines[1:]:
            cells = [c.strip() for c in line.split(",")]
            row = dict(zip(header, cells, strict=False))
            writers.add(row.get("writer") or cells[0])
            split = row.get("split") or (cells[1] if len(cells) > 1 else "?")
            split_counts[split] = split_counts.get(split, 0) + 1

    report = {
        "images": len(images),
        "annotations_bpmn": len(annotations),
        "word_annotations": len(words),
        "writers": len(writers),
        "official_split_counts": split_counts,
        "missing_paths": missing,
        "bytes": sum(p.stat().st_size for p in images),
    }
    print(json.dumps(report, indent=2))

    checks = {
        "layout_complete": not missing,
        "has_700_plus_images": len(images) >= 700,
        "annotations_present": len(annotations) >= 700,
        "word_annotations_present": len(words) >= 700,
        "annotations_match_images": abs(len(annotations) - len(images)) <= 5,
        "writer_split_present": len(writers) > 1,
    }
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if all(checks.values()):
        write_provenance(DATASET, {"inventory": report})
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="verify an existing copy only")
    ap.add_argument("--force", action="store_true", help="re-clone even if present")
    args = ap.parse_args(argv)

    if not args.check:
        download(force=args.force)
    check()
    return 0


if __name__ == "__main__":
    sys.exit(main())
