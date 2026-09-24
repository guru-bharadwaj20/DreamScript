"""Phase 1.1.3 — DIDI: digital-ink diagrams with stroke-level drawing data.

DIDI is the only source in the corpus that records **how** a diagram was drawn, not just how
it ended up: every diagram is a sequence of pen strokes with point-by-point coordinates. That
matters for Phase 7.3, whose whole premise is that reading (and drawing) a diagram is a
sequence — DIDI is where the "nodes first, then edges, then labels" prior can actually be
measured instead of assumed.

Each record also carries a `label_id` pointing at the Graphviz `dot` prompt the writer was
asked to reproduce, which gives ground-truth structure (nodes and edges) for free.

    python -m src.ingest.datasets.didi           # download (~1.3 GB)
    python -m src.ingest.datasets.didi --check   # verify an existing copy
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

from src.ingest.registry import Availability, Dataset, Redistribution, register, write_provenance

BUCKET = "https://storage.googleapis.com/digital_ink_diagram_data"
FILES = {
    "diagrams_20200131.ndjson": f"{BUCKET}/diagrams_20200131.ndjson",
    "diagrams_20200131_prompts.tgz": f"{BUCKET}/diagrams_20200131_prompts.tgz",
}

DATASET = register(
    Dataset(
        slug="didi",
        name="DIDI (Digital Ink Diagram data)",
        url="https://github.com/google-research/google-research/tree/master/didi_dataset",
        license="CC BY 4.0 (Google LLC)",
        availability=Availability.OPEN,
        redistribution=Redistribution.ATTRIBUTION,
        diagram_types=("flowchart", "state_machine"),  # dot-graph prompts cover both shapes
        phases=("1.1.3", "7.3", "5"),
        data_card="docs/data_cards/didi.md",
        notes=(
            "1.2 GB NDJSON of stroke sequences plus an 82 MB prompt archive (png/dot/xdot). "
            "The only source with drawing order; used for the Phase 7.3 sequence prior."
        ),
    )
)


def download(force: bool = False) -> Path:
    target = DATASET.path
    target.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        out = target / name
        if out.exists() and out.stat().st_size > 0 and not force:
            print(f"already present: {out.name} ({out.stat().st_size:,} bytes)")
            continue
        print(f"downloading {name} ...")
        urllib.request.urlretrieve(url, out)  # fixed https bucket
        print(f"  {out.stat().st_size:,} bytes")
    return target


def extract_prompts() -> Path:
    """Unpack the dot/xdot/png prompts, which carry the ground-truth graph structure."""
    root = DATASET.path
    tgz = root / "diagrams_20200131_prompts.tgz"
    out = root / "prompts"
    if out.is_dir() and any(out.iterdir()):
        return out
    out.mkdir(parents=True, exist_ok=True)
    print(f"extracting {tgz.name} ...")
    with tarfile.open(tgz) as tf:
        tf.extractall(out, filter="data")
    return out


def check(sample: int = 5000) -> dict:
    root = DATASET.path
    ndjson = root / "diagrams_20200131.ndjson"
    if not ndjson.is_file():
        raise SystemExit(f"not downloaded: {ndjson}")

    splits: dict[str, int] = {}
    total = 0
    stroke_counts: list[int] = []
    point_counts: list[int] = []
    keys: set[str] = set()

    with ndjson.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            total += 1
            if i >= sample:
                continue
            rec = json.loads(line)
            splits[rec.get("split", "?")] = splits.get(rec.get("split", "?"), 0) + 1
            keys.add(rec["key"])
            drawing = rec.get("drawing", [])
            stroke_counts.append(len(drawing))
            point_counts.append(sum(len(s[0]) for s in drawing if s))

    prompts = root / "prompts"
    dot_files = list(prompts.rglob("*.dot")) if prompts.is_dir() else []
    png_files = list(prompts.rglob("*.png")) if prompts.is_dir() else []

    report = {
        "total_diagrams": total,
        "sampled": min(sample, total),
        "split_counts_in_sample": splits,
        "unique_keys_in_sample": len(keys),
        "mean_strokes_per_diagram": round(sum(stroke_counts) / max(len(stroke_counts), 1), 1),
        "mean_points_per_diagram": round(sum(point_counts) / max(len(point_counts), 1), 1),
        "prompt_dot_files": len(dot_files),
        "prompt_png_files": len(png_files),
        "bytes": sum(p.stat().st_size for p in root.rglob("*") if p.is_file()),
    }
    print(json.dumps(report, indent=2))

    checks = {
        "has_20k_diagrams": total >= 20_000,
        "records_parse": len(keys) == min(sample, total),
        "has_stroke_data": report["mean_strokes_per_diagram"] > 1,
        "prompts_extracted": len(dot_files) > 0,
        "splits_declared": bool(splits),
    }
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if all(checks.values()):
        write_provenance(DATASET, {"inventory": report})
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--sample", type=int, default=5000, help="records to parse for the inventory")
    args = ap.parse_args(argv)
    if not args.check:
        download(force=args.force)
    extract_prompts()
    check(sample=args.sample)
    return 0


if __name__ == "__main__":
    sys.exit(main())
