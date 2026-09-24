"""Phase 1.1.5 — Sketch2Code: hand-drawn UI wireframes paired with real HTML.

The plan named Microsoft's Sketch2Code. Its GitHub repository still exists, but the hosted
demo and the dataset download behind it are gone (`sketch2code.azurewebsites.net` does not
resolve). **`SALT-NLP/Sketch2Code` is used instead, and it is a better fit**: 731 human-drawn
sketches paired with 484 *real-world* webpages, each with source HTML and a rendered
screenshot — exactly the `(sketch, code)` supervision Phase 12 needs for the wireframe →
React path, rather than the coarse widget labels the Microsoft set provided.

    python -m src.ingest.datasets.sketch2code
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

from src.ingest.registry import Availability, Dataset, Redistribution, register, write_provenance

MS_URLS = {
    "github": "https://github.com/microsoft/ailab/tree/master/Sketch2Code",
    "demo": "https://sketch2code.azurewebsites.net",
}

MS_SKETCH2CODE = register(
    Dataset(
        slug="sketch2code_ms",
        name="Sketch2Code (Microsoft)",
        url=MS_URLS["github"],
        license="MIT (code); dataset terms unclear",
        availability=Availability.UNAVAILABLE,
        redistribution=Redistribution.UNKNOWN,
        diagram_types=("wireframe",),
        phases=("1.1.5",),
        data_card="docs/data_cards/sketch2code.md",
        notes=(
            "Repository is live but the hosted demo and its dataset endpoint do not resolve. "
            "Substituted by SALT-NLP/Sketch2Code."
        ),
    )
)

HF_REPO = "SALT-NLP/Sketch2Code"

DATASET = register(
    Dataset(
        slug="sketch2code",
        name="Sketch2Code (SALT-NLP)",
        url=f"https://huggingface.co/datasets/{HF_REPO}",
        license="ODC-BY",
        availability=Availability.OPEN,
        redistribution=Redistribution.ATTRIBUTION,
        diagram_types=("wireframe",),
        phases=("1.1.5", "12.1.5"),
        data_card="docs/data_cards/sketch2code.md",
        notes=(
            "731 sketches over 484 webpages; each webpage ships source HTML and a screenshot. "
            "Images inside the HTML are replaced by a placeholder (rick.jpg) by the authors."
        ),
    )
)


def probe_microsoft(timeout: int = 15) -> dict[str, int | str]:
    out: dict[str, int | str] = {}
    for name, url in MS_URLS.items():
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                out[name] = resp.status
        except urllib.error.HTTPError as exc:
            out[name] = exc.code
        except Exception as exc:  # noqa: BLE001
            out[name] = type(exc).__name__
    return out


def download(force: bool = False) -> Path:
    from huggingface_hub import snapshot_download

    target = DATASET.path
    if target.exists() and any(target.iterdir()) and not force:
        print(f"already present: {target}")
        return target
    target.mkdir(parents=True, exist_ok=True)
    print(f"downloading {HF_REPO} -> {target}")
    snapshot_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        local_dir=str(target),
        max_workers=4,
        ignore_patterns=["*.zip"],  # the zip duplicates the flat sketches/ and webpages/ dirs
    )
    return target


def check() -> dict:
    root = DATASET.path
    sketches = sorted((root / "sketches").glob("*.png"))
    webpages = sorted((root / "webpages").glob("*.html"))
    screenshots = sorted((root / "webpages").glob("*.png"))

    # Each sketch is named <webpage_id>_<sketch_id>.png; several sketches may share a webpage.
    covered = {p.stem.rsplit("_", 1)[0] for p in sketches}
    page_ids = {p.stem for p in webpages}
    orphans = sorted(covered - page_ids)[:5]

    report = {
        "sketches": len(sketches),
        "webpages_html": len(webpages),
        "webpage_screenshots": len(screenshots),
        "distinct_pages_with_sketches": len(covered),
        "sketches_per_page": round(len(sketches) / max(len(covered), 1), 2),
        "orphan_sketch_pages_sample": orphans,
        "bytes": sum(p.stat().st_size for p in root.rglob("*") if p.is_file()),
        "microsoft_probe": probe_microsoft(),
    }
    print(json.dumps(report, indent=2))

    checks = {
        "sketches_present": len(sketches) >= 700,
        "html_present": len(webpages) >= 480,
        "screenshots_present": len(screenshots) >= 480,
        "every_sketch_has_its_page": not orphans,
        "microsoft_demo_gone": report["microsoft_probe"].get("demo") != 200,
    }
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if checks["sketches_present"] and checks["html_present"]:
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
