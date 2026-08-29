"""Phase 1.1.4 — IAM Handwriting: line-level handwritten text for the OCR sub-pipeline.

The original IAM database (fki.tic.heia-fr.ch) requires registration and manual acceptance of
its terms, which cannot be automated. `Teklia/IAM-line` is a public, MIT-licensed
line-segmented mirror of the same corpus, so the OCR work in Phase 9.3 can start without a
registration gate blocking it.

    python -m src.ingest.datasets.iam
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.ingest.registry import Availability, Dataset, Redistribution, register, write_provenance

ORIGINAL_URL = "https://fki.tic.heia-fr.ch/databases/iam-handwriting-database"
HF_REPO = "Teklia/IAM-line"

DATASET = register(
    Dataset(
        slug="iam_line",
        name="IAM Handwriting (line-level, Teklia mirror)",
        url=f"https://huggingface.co/datasets/{HF_REPO}",
        license="MIT (mirror); original IAM is research-use with registration",
        availability=Availability.OPEN,
        redistribution=Redistribution.ATTRIBUTION,
        diagram_types=(),  # text only; not a diagram source
        phases=("1.1.4", "9.3"),
        data_card="docs/data_cards/iam.md",
        notes=(
            "Line-segmented mirror of IAM. The canonical source at fki.tic.heia-fr.ch needs "
            "an account, so it is recorded as ACCOUNT and this mirror is used instead."
        ),
    )
)

EXPECTED_SPLITS = {"train", "validation", "test"}


def download(force: bool = False) -> Path:
    from huggingface_hub import snapshot_download

    target = DATASET.path
    if target.exists() and any(target.iterdir()) and not force:
        print(f"already present: {target}")
        return target
    target.mkdir(parents=True, exist_ok=True)
    print(f"downloading {HF_REPO} -> {target}")
    snapshot_download(repo_id=HF_REPO, repo_type="dataset", local_dir=str(target), max_workers=4)
    return target


def check() -> dict:
    import pyarrow.parquet as pq

    root = DATASET.path
    files = sorted(root.rglob("*.parquet"))
    if not files:
        raise SystemExit(f"not downloaded: {root}")

    per_split: dict[str, int] = {}
    columns: list[str] = []
    sample_texts: list[str] = []
    char_set: set[str] = set()

    for f in files:
        table = pq.read_table(f)
        per_split[f.stem] = table.num_rows
        columns = list(table.schema.names)
        if "text" in columns:
            texts = table.column("text").to_pylist()[:2000]
            sample_texts = texts[:3]
            for t in texts:
                char_set.update(t)

    report = {
        "rows_per_split": per_split,
        "total_lines": sum(per_split.values()),
        "columns": columns,
        "bytes": sum(f.stat().st_size for f in files),
        "charset_size": len(char_set),
        "sample_transcripts": sample_texts,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))

    checks = {
        "all_splits_present": set(per_split) >= EXPECTED_SPLITS,
        "has_10k_lines": sum(per_split.values()) >= 10_000,
        "has_image_column": "image" in columns,
        "has_text_column": "text" in columns,
        "charset_reasonable": 40 <= len(char_set) <= 200,
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
    args = ap.parse_args(argv)
    if not args.check:
        download(force=args.force)
    check()
    return 0


if __name__ == "__main__":
    sys.exit(main())
