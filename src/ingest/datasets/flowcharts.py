"""Phase 1.1.2 — hand-drawn flowcharts with component-level annotation.

**The plan named FC-A / FC-B (Bresler et al., CMP CTU Prague). Those are no longer
downloadable.** The dataset pages are still up at
`https://cmp.felk.cvut.cz/~breslmar/flowcharts/`, but the archive page's only download link
(`version1.0.html`) returns 404, and no mirror was found on HuggingFace, Kaggle or the
Internet Archive. This module records that verified fact rather than pretending otherwise,
and acquires a substitute that serves the same purpose.

**Substitute: `MananSuri27/flowchartseg`** — 1,319 hand-drawn flowchart images with per-node
segmentation masks. FC-A/FC-B offered component bounding boxes; masks are strictly richer,
and boxes are recovered from them by connected-component analysis
(`masks_to_boxes`), which is what Phase 9.1 actually consumes.

    python -m src.ingest.datasets.flowcharts          # fetch the substitute
    python -m src.ingest.datasets.flowcharts --probe  # re-verify FC-A/FC-B availability
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

from src.ingest.registry import Availability, Dataset, Redistribution, register, write_provenance

# The canonical source named in plan.md, kept so availability can be re-probed later.
FC_ORIGINAL_URLS = {
    "landing": "https://cmp.felk.cvut.cz/~breslmar/flowcharts/",
    "archive": "https://cmp.felk.cvut.cz/~breslmar/flowcharts/archive.html",
    "download": "https://cmp.felk.cvut.cz/~breslmar/flowcharts/version1.0.html",
    "offline_variant": "https://cmp.felk.cvut.cz/~breslmar/flowcharts_offline/",
}

FC_ORIGINAL = register(
    Dataset(
        slug="fc_bresler",
        name="FC-A / FC-B (Bresler flowchart database)",
        url=FC_ORIGINAL_URLS["landing"],
        license="unknown (research use, per the authors' pages)",
        availability=Availability.UNAVAILABLE,
        redistribution=Redistribution.UNKNOWN,
        diagram_types=("flowchart",),
        phases=("1.1.2",),
        data_card="docs/data_cards/flowchart_fc.md",
        notes=(
            "Landing and archive pages return 200 but the archive's download link "
            "(version1.0.html) returns 404; no mirror found. Substituted by flowchartseg. "
            "Re-probe with `python -m src.ingest.datasets.flowcharts --probe`."
        ),
    )
)

HF_REPO = "MananSuri27/flowchartseg"

FLOWCHARTSEG = register(
    Dataset(
        slug="flowchartseg",
        name="flowchartseg (FC-A/FC-B substitute)",
        url=f"https://huggingface.co/datasets/{HF_REPO}",
        license="see dataset card on the Hub",
        availability=Availability.OPEN,
        redistribution=Redistribution.UNKNOWN,
        diagram_types=("flowchart",),
        phases=("1.1.2", "9.1"),
        data_card="docs/data_cards/flowchart_fc.md",
        notes=(
            "1,187 train + 132 validation hand-drawn flowcharts with per-node segmentation "
            "masks; ~110 MB parquet. Boxes for the Phase 9.1 detector are derived from the "
            "masks by connected components."
        ),
    )
)


def probe_original(timeout: int = 20) -> dict[str, int | str]:
    """Re-check whether FC-A/FC-B became reachable again. Records the HTTP status of each URL."""
    results: dict[str, int | str] = {}
    for name, url in FC_ORIGINAL_URLS.items():
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
                results[name] = resp.status
        except urllib.error.HTTPError as exc:
            results[name] = exc.code
        except Exception as exc:  # noqa: BLE001 - network failures are data here
            results[name] = f"{type(exc).__name__}"
    return results


def download(force: bool = False) -> Path:
    from huggingface_hub import snapshot_download

    target = FLOWCHARTSEG.path
    if target.exists() and any(target.iterdir()) and not force:
        print(f"already present: {target}")
        return target
    target.mkdir(parents=True, exist_ok=True)
    print(f"downloading {HF_REPO} -> {target} (~110 MB)")
    snapshot_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        local_dir=str(target),
        max_workers=4,
    )
    return target


def check() -> dict:
    """Inventory the substitute: row counts, image sizes, and mask label values."""
    import pyarrow.parquet as pq

    root = FLOWCHARTSEG.path
    files = sorted(root.rglob("*.parquet"))
    if not files:
        raise SystemExit(f"not downloaded: {root}")

    per_split = {}
    total = 0
    for f in files:
        table = pq.read_file(f) if hasattr(pq, "read_file") else pq.read_table(f)
        split = "train" if "train" in f.name else "validation"
        per_split[split] = table.num_rows
        total += table.num_rows

    report = {
        "parquet_files": [str(f.relative_to(root)) for f in files],
        "rows_per_split": per_split,
        "total_images": total,
        "bytes": sum(f.stat().st_size for f in files),
        "columns": list(table.schema.names),
        "fc_original_probe": probe_original(),
    }
    print(json.dumps(report, indent=2))

    checks = {
        "both_splits_present": set(per_split) == {"train", "validation"},
        "has_1000_plus_images": total >= 1000,
        "has_image_column": "image" in report["columns"],
        "has_annotation_column": "annotation" in report["columns"],
        "fc_original_still_unavailable": report["fc_original_probe"].get("download") == 404,
    }
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if checks["both_splits_present"] and checks["has_image_column"]:
        write_provenance(FLOWCHARTSEG, {"inventory": report})
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe", action="store_true", help="only re-check FC-A/FC-B availability")
    ap.add_argument("--check", action="store_true", help="verify an existing copy only")
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args(argv)

    if args.probe:
        print(json.dumps(probe_original(), indent=2))
        return 0
    if not args.check:
        download(force=args.force)
    check()
    return 0


if __name__ == "__main__":
    sys.exit(main())
