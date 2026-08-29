"""Phase 1.3.1 — the unified corpus manifest.

Five datasets with five different layouts become one table. Everything downstream — features,
splits, training, evaluation — reads `data/processed/manifest.parquet` and never touches a
dataset's native format again.

    python -m src.ingest.manifest              # build it
    python -m src.ingest.manifest --summary    # describe what was built

Columns:

| column | meaning |
| :--- | :--- |
| `id` | stable, unique: `<source>/<relative path without extension>` |
| `source` | which dataset it came from |
| `path` | repo-relative path to the image (or to the record's container) |
| `diagram_type` | flowchart / wireframe / state_machine / er_diagram / circuit / text |
| `scribe_id` | writer identity where the source publishes one, else null |
| `medium`, `condition`, `adverse` | chaos-corpus capture metadata, else null |
| `has_structure` | does this record carry node/edge annotation? |
| `has_text` | does it carry transcribed labels? |
| `native_split` | the split the source published, if any |
| `split` | the split DreamScript assigns (Phase 1.3.3) |

`scribe_id` is the important one: sources that publish writer identity can participate in
scribe-disjoint splitting, and sources that do not cannot. That distinction has to survive
into every downstream query, so it lives in the manifest rather than in someone's memory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from src.ingest.collection import collected
from src.utils.config import ROOT

OUT = ROOT / "data" / "processed" / "manifest.parquet"
RAW = ROOT / "data" / "raw"

COLUMNS = [
    "id",
    "source",
    "path",
    "diagram_type",
    "scribe_id",
    "medium",
    "condition",
    "adverse",
    "has_structure",
    "has_text",
    "native_split",
    "split",
]


def _row(**kw) -> dict:
    row = dict.fromkeys(COLUMNS)
    row.update(kw)
    return row


# --- per-source collectors ---------------------------------------------------------------


def from_hdbpmn() -> list[dict]:
    root = RAW / "hdbpmn"
    if not root.is_dir():
        return []

    # writer_split.csv is writer-disjoint and published by the authors; we adopt it.
    split_by_writer: dict[str, str] = {}
    csv_path = root / "data" / "writer_split.csv"
    if csv_path.is_file():
        lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
        for line in lines[1:]:
            cells = [c.strip() for c in line.split(",")]
            if len(cells) >= 2:
                split_by_writer[cells[0]] = cells[1]

    rows = []
    for img in sorted((root / "data" / "images").rglob("*")):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        # filenames look like ex00_writer0001.jpg
        stem = img.stem
        writer = stem.split("_")[-1] if "_" in stem else None
        rel = img.relative_to(ROOT)
        annotation = root / "data" / "annotations" / img.parent.name / f"{stem}.bpmn"
        words = root / "data" / "words" / img.parent.name / f"{stem}.xml"
        # hdBPMN writes "val"; DIDI writes "valid"; we normalise both to "validation" so
        # native_split is comparable across sources.
        native = {"val": "validation"}.get(
            split_by_writer.get(writer or ""), split_by_writer.get(writer or "")
        )
        rows.append(
            _row(
                id=f"hdbpmn/{img.parent.name}/{stem}",
                source="hdbpmn",
                path=str(rel).replace("\\", "/"),
                diagram_type="flowchart",
                scribe_id=f"hdbpmn:{writer}" if writer else None,
                has_structure=annotation.is_file(),
                has_text=words.is_file(),
                native_split=native,
                adverse=False,
            )
        )
    return rows


def from_flowchartseg() -> list[dict]:
    """Parquet-packed; one manifest row per record, addressed by shard and row index."""
    root = RAW / "flowchartseg"
    if not root.is_dir():
        return []
    import pyarrow.parquet as pq

    rows = []
    for shard in sorted(root.rglob("*.parquet")):
        split = "train" if "train" in shard.name else "validation"
        n = pq.read_metadata(shard).num_rows
        rel = str(shard.relative_to(ROOT)).replace("\\", "/")
        for i in range(n):
            rows.append(
                _row(
                    id=f"flowchartseg/{split}/{i:05d}",
                    source="flowchartseg",
                    path=f"{rel}#{i}",
                    diagram_type="flowchart",
                    scribe_id=None,  # writer identity not published - cannot be split by scribe
                    has_structure=True,  # node masks, but no edges
                    has_text=False,
                    native_split=split,
                    adverse=False,
                )
            )
    return rows


def from_didi(limit: int | None = None) -> list[dict]:
    root = RAW / "didi"
    ndjson = root / "diagrams_20200131.ndjson"
    if not ndjson.is_file():
        return []
    rel = str(ndjson.relative_to(ROOT)).replace("\\", "/")
    rows = []
    with ndjson.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if limit and i >= limit:
                break
            # Parse only the fields we need; the drawing array is large and unused here.
            rec = json.loads(line)
            rows.append(
                _row(
                    id=f"didi/{rec['key']}",
                    source="didi",
                    path=f"{rel}#{i}",
                    diagram_type="flowchart",  # graphviz-style node-edge diagrams
                    scribe_id=None,  # anonymous contributors, no writer id published
                    has_structure=True,  # via the .dot prompt
                    has_text=False,
                    native_split={"valid": "validation"}.get(rec.get("split"), rec.get("split")),
                    adverse=False,
                )
            )
    return rows


def from_iam() -> list[dict]:
    root = RAW / "iam_line"
    if not root.is_dir():
        return []
    import pyarrow.parquet as pq

    rows = []
    for shard in sorted(root.rglob("*.parquet")):
        split = shard.stem
        n = pq.read_metadata(shard).num_rows
        rel = str(shard.relative_to(ROOT)).replace("\\", "/")
        for i in range(n):
            rows.append(
                _row(
                    id=f"iam/{split}/{i:05d}",
                    source="iam_line",
                    path=f"{rel}#{i}",
                    diagram_type="text",  # not a diagram: OCR pretraining only
                    scribe_id=None,
                    has_structure=False,
                    has_text=True,
                    native_split=split,
                    adverse=False,
                )
            )
    return rows


def from_sketch2code() -> list[dict]:
    root = RAW / "sketch2code"
    sketches = root / "sketches"
    if not sketches.is_dir():
        return []
    rows = []
    for img in sorted(sketches.glob("*.png")):
        page = img.stem.rsplit("_", 1)[0]
        html = root / "webpages" / f"{page}.html"
        rows.append(
            _row(
                id=f"sketch2code/{img.stem}",
                source="sketch2code",
                path=str(img.relative_to(ROOT)).replace("\\", "/"),
                diagram_type="wireframe",
                scribe_id=None,  # 1-3 annotators, identities not published
                has_structure=html.is_file(),  # the HTML is the structure
                has_text=True,
                native_split=None,
                adverse=False,
            )
        )
    return rows


def from_chaos() -> list[dict]:
    rows = []
    for r in collected():
        p = Path(r["path"])
        rows.append(
            _row(
                id=f"chaos/{r['diagram_type']}/{r['scribe_id']}/{p.stem}",
                source="chaos",
                path=str(r["path"]).replace("\\", "/"),
                diagram_type=r["diagram_type"],
                scribe_id=f"chaos:{r['scribe_id']}",
                medium=r["medium"],
                condition=r["condition"],
                adverse=r["adverse"],
                has_structure=False,  # filled in by Phase 2 annotation
                has_text=False,
                native_split=None,
            )
        )
    return rows


SOURCES = {
    "hdbpmn": from_hdbpmn,
    "flowchartseg": from_flowchartseg,
    "didi": from_didi,
    "iam_line": from_iam,
    "sketch2code": from_sketch2code,
    "chaos": from_chaos,
}


def build(didi_limit: int | None = None) -> pd.DataFrame:
    frames = []
    for name, fn in SOURCES.items():
        rows = fn(limit=didi_limit) if name == "didi" else fn()
        print(f"  {name:<14} {len(rows):>7} rows")
        if rows:
            frames.append(pd.DataFrame(rows, columns=COLUMNS))
    if not frames:
        return pd.DataFrame(columns=COLUMNS)

    df = pd.concat(frames, ignore_index=True)
    df["adverse"] = df["adverse"].fillna(False).astype(bool)
    df["has_structure"] = df["has_structure"].fillna(False).astype(bool)
    df["has_text"] = df["has_text"].fillna(False).astype(bool)
    return df


def save(df: pd.DataFrame) -> Path:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    return OUT


def load() -> pd.DataFrame:
    if not OUT.is_file():
        raise SystemExit(f"manifest not built: {OUT} (run `python -m src.ingest.manifest`)")
    return pd.read_parquet(OUT)


def summary(df: pd.DataFrame) -> dict:
    return {
        "rows": len(df),
        "unique_ids": int(df["id"].nunique()),
        "by_source": df["source"].value_counts().to_dict(),
        "by_diagram_type": df["diagram_type"].value_counts().to_dict(),
        "with_scribe_id": int(df["scribe_id"].notna().sum()),
        "distinct_scribes": int(df["scribe_id"].nunique()),
        "with_structure": int(df["has_structure"].sum()),
        "with_text": int(df["has_text"].sum()),
        "adverse": int(df["adverse"].sum()),
        "native_splits": df["native_split"].value_counts(dropna=False).to_dict(),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", action="store_true", help="describe the existing manifest")
    ap.add_argument("--didi-limit", type=int, default=None, help="cap DIDI rows (for quick runs)")
    args = ap.parse_args(argv)

    df = load() if args.summary else build(didi_limit=args.didi_limit)
    if not args.summary:
        path = save(df)
        print(f"wrote {path.relative_to(ROOT)}")

    s = summary(df)
    print(json.dumps(s, indent=2, default=str))

    checks = {
        "ids_unique": s["rows"] == s["unique_ids"],
        "all_sources_present": len(s["by_source"]) >= 5,
        "has_structure_annotations": s["with_structure"] > 0,
        "has_scribe_identities": s["distinct_scribes"] > 1,
        "no_null_diagram_type": int(df["diagram_type"].isna().sum()) == 0,
    }
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
