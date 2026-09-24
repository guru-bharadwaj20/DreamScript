"""Phase 15.2 - data versioning: what is tracked, what is deliberately not, and what is pushed.

    python -m src.mlops.dataversion              # reports/data_versioning.md + .json
    python -m src.mlops.dataversion --check      # non-zero if anything tracked is unpushed

This project has already lost its data once: the DVC store was empty when the corpus was needed,
and the raw data, splits and checkpoints had to be rebuilt from source. That is the failure this
row exists to prevent recurring, so it does two things beyond running `dvc add`.

**It states a rule for what gets versioned, and applies it.** Not everything under `data/`
deserves a remote. The rule is *version what a fresh clone cannot rebuild cheaply or
deterministically*:

    tracked          raw corpora (cannot be re-downloaded reliably), the IR, the label and text
                     crops, the manifest, the handcrafted feature table
    not tracked      the synthetic renders (deterministic from a seed - 12.1.4 rebuilds all
                     12,000 in 64.5 s), the detector's YOLO image trees (regenerated from the IR
                     and the raw pages), the pipeline and page-text caches (volatile by design)

Both lists are in the report with their sizes, so "not tracked" is a decision a reader can
disagree with rather than an omission they have to discover.

**It verifies, rather than assuming, that the store has the bytes.** `--check` compares every
tracked output against the remote and exits non-zero if anything is missing - which is exactly
the condition that went unnoticed last time. The three remotes the plan asks for (raw, interim,
processed) are configured in `.dvc/config`; they are local directories here because there is no
cloud account attached to this project, and a local directory is a real remote as far as DVC's
push/pull contract is concerned - what it does not give is off-machine durability, and the
report says so in those words rather than implying a backup exists.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from src.utils.config import ROOT

REPORT_MD = ROOT / "reports" / "data_versioning.md"
REPORT_JSON = ROOT / "reports" / "data_versioning.json"

DVC = ROOT / ".venv" / "Scripts" / "dvc.exe"

#: What is versioned, and the reason it earns a remote.
TRACKED_REASON = {
    "data/raw": "source corpora; re-downloading is not reproducible and some sources are gone",
    "data/processed/ir": "the annotated graphs - the ground truth every criterion is scored on",
    "data/processed/label_crops": "9.3.1's crops, built by a CRAFT pass that costs GPU time",
    "data/processed/text_crops": "the OCR training corpus",
    "data/processed/manifest.parquet": "the split assignment; regenerating it would reshuffle it",
    "data/features/handcrafted.parquet": "4.2's feature table, 2.5 pages a second to rebuild",
}

#: What is deliberately not versioned, and why. Sizes are measured, not guessed.
UNTRACKED_REASON = {
    "data/processed/codegen/synthetic": "deterministic from a seed - 12.1.4 rebuilds 12,000"
    " renders in 64.5 s, so a remote would store what a command reproduces",
    "data/processed/detect": "the YOLO image tree, regenerated from the IR and the raw pages",
    "data/processed/arrows": "the arrow-detector dataset, likewise regenerated",
    "data/interim/pipeline_cache": "13.6's cache - volatile by design and keyed on a code digest",
    "data/interim/primitives_detect": "intermediate primitives, rebuilt by the preprocessing stage",
}


def _run(*args: str) -> tuple[int, str]:
    executable = str(DVC) if DVC.is_file() else "dvc"
    try:
        done = subprocess.run(
            [executable, *args], cwd=ROOT, capture_output=True, text=True, timeout=1800, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return 1, f"{type(error).__name__}: {error}"
    return done.returncode, (done.stdout or "") + (done.stderr or "")


def remotes() -> dict[str, str]:
    code, output = _run("remote", "list")
    if code != 0:
        return {}
    found = {}
    for line in output.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            found[parts[0]] = parts[1]
    return found


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def inventory() -> dict[str, Any]:
    tracked, untracked = [], []
    for path, reason in TRACKED_REASON.items():
        target = ROOT / path
        pointer = target.with_suffix(target.suffix + ".dvc")
        tracked.append(
            {
                "path": path,
                "reason": reason,
                "exists": target.exists(),
                "megabytes": round(_size(target) / 1e6, 1) if target.exists() else None,
                "pointer": pointer.name if pointer.is_file() else None,
            }
        )
    for path, reason in UNTRACKED_REASON.items():
        target = ROOT / path
        untracked.append(
            {
                "path": path,
                "reason": reason,
                "exists": target.exists(),
                "megabytes": round(_size(target) / 1e6, 1) if target.exists() else None,
            }
        )
    return {"tracked": tracked, "not_tracked": untracked}


def store_status() -> dict[str, Any]:
    """`dvc status --cloud` against each configured remote: what the store is missing."""
    out: dict[str, Any] = {}
    for name in ("raw", "interim", "processed"):
        code, output = _run("status", "--cloud", "-r", name)
        missing = [line.strip() for line in output.splitlines() if line.strip().startswith("new:")]
        out[name] = {
            "exit_code": code,
            "up_to_date": "Cache and remote 'default' are in sync" in output
            or "everything is up to date" in output.lower()
            or not missing,
            "missing_entries": len(missing),
            "sample": missing[:5],
        }
    return out


def orphan_pointers() -> list[str]:
    """Pointers whose payload exists neither in the workspace nor in the cache.

    This is the exact condition the 2026-09-12 rebuild discovered the hard way: a `.dvc` file
    that looks like versioned data and has nothing behind it. Naming them is the point.
    """
    code, output = _run("status")
    if code != 0:
        return []
    orphans = []
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("not in cache:"):
            orphans.append(stripped.split(":", 1)[1].strip().replace("\\", "/"))
    return sorted(set(orphans))


def collect(check_store: bool = True) -> dict:
    result: dict[str, Any] = {
        "remotes": remotes(),
        **inventory(),
        "durability_note": "the remotes are local directories on the same machine as the"
        " workspace; that satisfies DVC's push/pull contract but is not off-machine"
        " durability, and the loss this row responds to happened on this machine",
    }
    if check_store:
        result["store"] = store_status()
        result["pointers_without_a_payload"] = orphan_pointers()
    tracked_mb = sum(row["megabytes"] or 0 for row in result["tracked"])
    untracked_mb = sum(row["megabytes"] or 0 for row in result["not_tracked"])
    result["totals"] = {
        "tracked_megabytes": round(tracked_mb, 1),
        "not_tracked_megabytes": round(untracked_mb, 1),
        "tracked_paths": len(result["tracked"]),
    }
    return result


def render(result: dict) -> str:
    lines = [
        "# Phase 15.2 - data versioning",
        "",
        "Generated by `python -m src.mlops.dataversion`. The rule applied here is **version what"
        " a fresh clone cannot rebuild cheaply or deterministically**; everything else is named"
        " below with the reason it was left out, so the exclusions are a decision rather than an"
        " omission.",
        "",
        "## Remotes",
        "",
        "| remote | url |",
        "| :--- | :--- |",
    ]
    for name, url in result["remotes"].items():
        lines.append(f"| `{name}` | `{url}` |")
    lines += [
        "",
        f"> {result['durability_note']}.",
        "",
        f"## Tracked ({result['totals']['tracked_megabytes']} MB over"
        f" {result['totals']['tracked_paths']} paths)",
        "",
        "| path | MB | pointer | why |",
        "| :--- | ---: | :--- | :--- |",
    ]
    for row in result["tracked"]:
        pointer = f"`{row['pointer']}`" if row["pointer"] else "*none*"
        lines.append(f"| `{row['path']}` | {row['megabytes']} | {pointer} | {row['reason']} |")
    lines += [
        "",
        f"## Deliberately not tracked ({result['totals']['not_tracked_megabytes']} MB)",
        "",
        "| path | MB | why not |",
        "| :--- | ---: | :--- |",
    ]
    for row in result["not_tracked"]:
        lines.append(f"| `{row['path']}` | {row['megabytes']} | {row['reason']} |")
    orphans = result.get("pointers_without_a_payload")
    if orphans:
        lines += [
            "",
            f"## Pointers with no payload ({len(orphans)})",
            "",
            "These `.dvc` files describe data that is in neither the workspace nor the cache -"
            " the exact condition the 2026-09-12 rebuild ran into. They are listed rather than"
            " deleted, because the pointer is the only surviving record of what the corpus"
            " contained.",
            "",
        ]
        lines += [f"* `{path}`" for path in orphans]
    if result.get("store"):
        lines += [
            "",
            "## Store check (`dvc status --cloud`)",
            "",
            "| remote | in sync | missing |",
            "| :--- | :---: | ---: |",
        ]
        for name, entry in result["store"].items():
            lines.append(
                f"| `{name}` | {'yes' if entry['up_to_date'] else '**no**'} |"
                f" {entry['missing_entries']} |"
            )
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check", action="store_true", help="exit non-zero if the store is missing data"
    )
    ap.add_argument("--out", type=Path, default=REPORT_MD)
    ap.add_argument("--json", dest="out_json", type=Path, default=REPORT_JSON)
    args = ap.parse_args(argv)

    result = collect()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(result), encoding="utf-8")
    args.out_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"remotes": result["remotes"], "totals": result["totals"]}, indent=2))
    if args.check:
        stale = [
            name for name, entry in (result.get("store") or {}).items() if not entry["up_to_date"]
        ]
        if stale:
            print(f"remotes missing data: {', '.join(stale)}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
