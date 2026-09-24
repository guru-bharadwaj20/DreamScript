"""Phase 1.1.8 - which tracked payloads exist, and which are gone for good.

    python -m src.ingest losses           # rewrite docs/data_losses.md
    python -m src.ingest losses --check   # non-zero if a source's state changed

`dvc status -c` reports a payload with no content as "missing from remote and local", and it
prints the same line for a source you simply have not pulled yet. Those are different problems:

    not pulled   one `dvc pull` away
    lost         needs re-acquiring from its origin, which for some of these means an external
                 download that may or may not still resolve

Four of the eight raw sources are in the second state - `cghd_extracted`, `chaos`, `didi`,
`iam_line` - and nothing in this repository said so. `reports/license_audit.md` says "4 of 9
registered sources are present locally", which is true and does not distinguish the two.

This reads the `.dvc` pointers directly rather than shelling out to `dvc`, so it answers while a
`dvc repro` holds the lock, and on a machine with no DVC installed at all.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.utils.config import ROOT

RAW = ROOT / "data" / "raw"
CACHE = ROOT / ".dvc" / "cache" / "files" / "md5"
DOC = ROOT / "docs" / "data_losses.md"

#: The DVC store this project is configured against. A directory on one machine - see
#: docs/data_remote.md - which is why "on the remote" and "on this disk" are the same question.
STORE = Path("C:/Users/Temp/dreamscript-dvc-store")


def _config_store() -> Path:
    """The store path from `.dvc/config`, so this does not hard-code one machine's layout."""
    config = ROOT / ".dvc" / "config"
    if not config.is_file():
        return STORE
    for line in config.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("url =") and "remote" not in line:
            return Path(line.split("=", 1)[1].strip())
    return STORE


def _in_store(md5: str, store: Path) -> bool:
    key = Path(md5[:2]) / md5[2:]
    candidates = [store / "files" / "md5" / key]
    candidates += [store / sub / "files" / "md5" / key for sub in ("raw", "interim", "processed")]
    return any(path.exists() for path in candidates)


def survey() -> dict[str, dict]:
    """source -> where its payload can be found, if anywhere."""
    import yaml

    store = _config_store()
    out: dict[str, dict] = {}
    for pointer in sorted(RAW.glob("*.dvc")):
        spec = yaml.safe_load(pointer.read_text(encoding="utf-8"))["outs"][0]
        md5 = str(spec.get("md5", ""))
        key = Path(md5[:2]) / md5[2:]
        out[pointer.stem] = {
            "md5": md5,
            "files": spec.get("nfiles"),
            "worktree": (RAW / pointer.stem).exists(),
            "cache": (CACHE / key).exists(),
            "store": _in_store(md5, store) if md5 else False,
        }
    return out


def lost(state: dict[str, dict] | None = None) -> list[str]:
    """Sources whose content exists in none of the three places."""
    state = survey() if state is None else state
    return sorted(
        name for name, row in state.items() if not (row["worktree"] or row["cache"] or row["store"])
    )


def render(state: dict[str, dict]) -> str:
    mark = {True: "\u2713", False: "\u2717"}
    gone = lost(state)
    lines = [
        "# What is gone",
        "",
        f"{len(gone)} of the {len(state)} raw sources are tracked by a `.dvc` pointer whose "
        "content exists **nowhere** - not in the working tree, not in `.dvc/cache`, not in the "
        'store. They are not "not pulled yet". They are unrecoverable from this repository.',
        "",
        "| Source | files | `.dvc` md5 | worktree | local cache | store |",
        "| :--- | ---: | :--- | :---: | :---: | :---: |",
    ]
    for name, row in sorted(state.items(), key=lambda item: (item[0] in gone, item[0])):
        lines.append(
            f"| `{name}` | {row['files']} | `{row['md5'][:12]}...` | "
            f"{mark[row['worktree']]} | {mark[row['cache']]} | {mark[row['store']]} |"
        )
    lines += [
        "",
        "Regenerate with `python -m src.ingest losses`.",
        "",
        "## Why this needed its own page",
        "",
        '`dvc status -c` reports these as "missing from remote and local", which reads exactly '
        "like the line it prints for a source you simply have not pulled. The distinction is the "
        "whole point:",
        "",
        "* a **not pulled** source is one `dvc pull` away;",
        "* a **lost** source needs re-acquiring from its origin, and for three of these four that "
        "means an external download that may or may not still resolve.",
        "",
        '`reports/license_audit.md` already said "4 of 9 registered sources are present '
        'locally", which is true and does not say which of the two states the other four are in.',
        "",
        "## What each one cost",
        "",
        "| Source | Where it was used | What its absence means |",
        "| :--- | :--- | :--- |",
        "| `didi` | `configs/llm.yaml` training sources, `src/ir/convert/didi.py`, "
        "`pairs.REAL_SOURCES` | ~3,000 IR documents the corpus does not have. Removed from the "
        "training sources in this batch; the converter is kept because the payload is "
        "re-acquirable from the published DIDI release. |",
        "| `iam_line` | `src/ingest/manifest.from_iam` | handwriting lines for the Phase 9.3 "
        "recogniser, which was trained on label crops instead. |",
        "| `cghd_extracted` | `routing.SOURCE_TYPE` maps `cghd -> circuit` | why the router can "
        "never learn `circuit`; named in `routing.UNCOVERED_SOURCES`. |",
        "| `chaos` | `src/ingest/chaos_builder.py`, `reports/chaos_corpus.md` | the "
        "adverse-condition corpus. Its report survives; its images do not. |",
        "",
        "## What to do",
        "",
        "The `.dvc` pointers stay. They are the record of what the corpus was built from, and "
        "deleting them would make the manifest, the reports and `contributing.md` refer to "
        "sources that leave no trace. Re-acquiring any of them is a matter of running its "
        "downloader in `src/ingest/datasets/` and `dvc add`-ing the result; the md5 above is "
        "what a successful re-acquisition should reproduce.",
        "",
        "See also [data_remote.md](data_remote.md), which is why none of this is recoverable "
        "from a second machine either.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if the document is stale")
    args = parser.parse_args(argv)

    state = survey()
    if args.check:
        current = DOC.read_text(encoding="utf-8") if DOC.is_file() else ""
        if current != render(state):
            print(
                "docs/data_losses.md is out of date; run python -m src.ingest losses",
                file=sys.stderr,
            )
            return 1
        print(json.dumps({"sources": len(state), "lost": lost(state)}, indent=2))
        return 0

    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(render(state), encoding="utf-8")
    print(f"wrote {DOC.relative_to(ROOT)}: {len(lost(state))} lost of {len(state)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
