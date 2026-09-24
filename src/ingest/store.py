"""Phase 1.1.9 - what the store holds, and what exists only in one machine's cache.

    python -m src.ingest store           # the survey
    python -m src.ingest store --check   # non-zero if a cached output is not in the store

`dvc status -c` reported **1,507 objects as `new:`** - in `.dvc/cache`, absent from the store.
That is not a harmless backlog. `.dvc/cache` is a working directory: `dvc gc`, a reformatted
disk or a `rm -rf` on a tree somebody thought was scratch takes it, and the store is the only
copy that is supposed to survive that. Everything in that list was one such accident away from
joining the four sources in `docs/data_losses.md`, and it included the entire Phase 12.1 target
corpus (1,478 files), the arrow pose checkpoint and the TrOCR fine-tune.

Nothing said so. `dvc status -c` prints `new:` - which reads like "recently added", not "the
only copy is the one you can delete by accident" - and it prints one line per file, so a real
backlog of 1,507 scrolls past as noise. `dvc push` fixes the state; this is what notices.

## Why this reads the pointers instead of calling dvc

`dvc status -c` takes a minute here and needs the DVC lock, so it cannot run while a `dvc repro`
holds it, and it needs DVC installed. Every fact it reports for this question is on disk: the
`.dvc` pointers, `dvc.lock`, and the content-addressed layout `<store>/files/md5/ab/cdef...`.
This stats those paths, which is fast enough to be a `--check` and answers on a machine with no
DVC at all. `src.ingest.losses` reads the same layout for the raw sources and imports the two
location helpers from here, so there is one definition of "where would the store keep this".

## Three things that are easy to conflate

**`cache: false` outputs are not missing.** `dvc.yaml` declares the twelve report artefacts
uncached - they are small text files, they are in git, and `dvc push` correctly never sends them.
Counting them as absent from the store would make this check cry wolf on every run.

**A directory output is one hash and many objects.** `data/processed/targets` is a single
`.dir` entry in its pointer and 1,478 files underneath. The `.dir` object is a JSON manifest
listing each member's md5, so a directory is only genuinely stored when the manifest *and* every
member is - which is exactly the state an interrupted push leaves half-finished.

**Per-output remotes put objects in different subdirectories.** Four of the five pointers under
`data/processed/` declare `remote: processed` and land in `<store>/processed/files/md5`;
`data/processed/targets.dvc` declares none and lands in `<store>/files/md5`. All four remotes in
`.dvc/config` are subdirectories of one store on one disk, so for the durability question they
are the same place - but a check that looked only where the pointer says would report `targets`
as missing, and one that never looked would not be able to say where anything is. This looks in
the declared place first and records which subtree the object was actually found in.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.utils.config import ROOT

CACHE = ROOT / ".dvc" / "cache" / "files" / "md5"
CONFIG = ROOT / ".dvc" / "config"
LOCK = ROOT / "dvc.lock"
PIPELINE = ROOT / "dvc.yaml"

#: The store this project is configured against when `.dvc/config` cannot be read. A directory on
#: one machine - see docs/data_remote.md - which is why "on the remote" and "on this disk" are
#: the same question here.
STORE = Path("C:/Users/Temp/dreamscript-dvc-store")

#: The per-output remotes in `.dvc/config`, each a subdirectory of the store. An object may be
#: under any of them depending on what its pointer declared, and `""` is the default remote.
SUBTREES = ("", "raw", "interim", "processed")


def config_store() -> Path:
    """The default remote's path from `.dvc/config`, so this does not hard-code one machine."""
    if not CONFIG.is_file():
        return STORE
    for line in CONFIG.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("url =") and "remote" not in line:
            return Path(line.split("=", 1)[1].strip())
    return STORE


def object_path(md5: str, store: Path, subtree: str = "") -> Path:
    """Where the content-addressed layout puts `md5` under one subtree of the store."""
    base = store / subtree if subtree else store
    return base / "files" / "md5" / md5[:2] / md5[2:]


def _order(remote: str) -> list[str]:
    """The subtrees to look in, the one the pointer declared first."""
    if not remote:
        return list(SUBTREES)
    return [remote, *(name for name in SUBTREES if name != remote)]


def in_store(md5: str, store: Path, remote: str = "") -> str | None:
    """Which subtree of `store` holds `md5`, or None. The declared remote is tried first."""
    for subtree in _order(remote):
        if object_path(md5, store, subtree).exists():
            return subtree or "."
    return None


def in_cache(md5: str) -> bool:
    return (CACHE / md5[:2] / md5[2:]).exists()


def _members(md5: str, store: Path, remote: str) -> list[str]:
    """The md5s listed by a `.dir` manifest, read from wherever a copy of it can be found.

    A directory whose manifest is in neither place is one of the lost sources - there is nothing
    to expand and nothing to push, and `losses` is where that is reported.
    """
    candidates = [CACHE / md5[:2] / md5[2:]]
    candidates += [object_path(md5, store, subtree) for subtree in _order(remote)]
    for path in candidates:
        if path.is_file():
            entries = json.loads(path.read_text(encoding="utf-8"))
            return [str(entry["md5"]) for entry in entries]
    return []


def uncached() -> set[str]:
    """Output paths `dvc.yaml` declares `cache: false`. `dvc push` never sends these."""
    import yaml

    if not PIPELINE.is_file():
        return set()
    spec = yaml.safe_load(PIPELINE.read_text(encoding="utf-8")) or {}
    out: set[str] = set()
    for stage in (spec.get("stages") or {}).values():
        for entry in stage.get("outs") or []:
            if isinstance(entry, dict):
                for path, options in entry.items():
                    if isinstance(options, dict) and options.get("cache") is False:
                        out.add(str(path))
    return out


def tracked() -> list[dict]:
    """Every cached DVC output in the repo: `.dvc` pointers first, then `dvc.lock` stage outs."""
    import yaml

    skip = uncached()
    found: list[dict] = []
    for pointer in sorted((ROOT / "data").rglob("*.dvc")):
        spec = yaml.safe_load(pointer.read_text(encoding="utf-8"))["outs"][0]
        found.append(
            {
                "path": (pointer.parent / str(spec["path"])).relative_to(ROOT).as_posix(),
                "md5": str(spec.get("md5", "")),
                "nfiles": spec.get("nfiles"),
                "remote": str(spec.get("remote", "")),
                "from": pointer.relative_to(ROOT).as_posix(),
            }
        )
    if LOCK.is_file():
        lock = yaml.safe_load(LOCK.read_text(encoding="utf-8")) or {}
        for name, stage in (lock.get("stages") or {}).items():
            for spec in stage.get("outs") or []:
                path = str(spec.get("path", ""))
                if path in skip:
                    continue
                found.append(
                    {
                        "path": path,
                        "md5": str(spec.get("md5", "")),
                        "nfiles": spec.get("nfiles"),
                        "remote": "",
                        "from": f"dvc.lock:{name}",
                    }
                )
    return found


def survey(store: Path | None = None) -> list[dict]:
    """One row per tracked output, with its objects counted in the cache and in the store."""
    store = config_store() if store is None else store
    rows: list[dict] = []
    for out in tracked():
        md5, remote = out["md5"], out["remote"]
        if not md5:
            continue
        # A `.dir` with no manifest in either place is one of the lost sources: there is nothing
        # to expand, so the row is one object and `absent` is what reports it.
        objects = [md5, *_members(md5, store, remote)] if md5.endswith(".dir") else [md5]
        where = [in_store(one, store, remote) for one in objects]
        rows.append(
            {
                **out,
                "objects": len(objects),
                "cached": sum(in_cache(one) for one in objects),
                "stored": sum(place is not None for place in where),
                "subtree": next((place for place in where if place), None),
            }
        )
    return rows


def unpushed(rows: list[dict]) -> list[dict]:
    """Outputs with content in the cache that the store does not have. `dvc push` fixes these."""
    return [row for row in rows if row["cached"] and row["stored"] < row["cached"]]


def absent(rows: list[dict]) -> list[dict]:
    """Outputs with no content in either place - the `docs/data_losses.md` state."""
    return [row for row in rows if not row["cached"] and not row["stored"]]


def render(rows: list[dict]) -> str:
    width = max((len(row["path"]) for row in rows), default=0)
    lines = []
    for row in sorted(rows, key=lambda row: row["path"]):
        if not row["cached"] and not row["stored"]:
            state = "gone - in neither the cache nor the store"
        elif row["stored"] >= row["cached"]:
            state = f"stored ({row['stored']}/{row['objects']} in {row['subtree']})"
        else:
            state = f"NOT PUSHED - {row['cached'] - row['stored']} of {row['objects']} cache-only"
        lines.append(f"  {row['path']:<{width}}  {state}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if anything is cache-only")
    parser.add_argument("--json", action="store_true", help="print the survey as JSON")
    args = parser.parse_args(argv)

    store = config_store()
    if not store.is_dir():
        # Every machine but one. Failing here would make the check noise on the machines
        # docs/data_remote.md already says cannot reach the payload.
        print(f"no store at {store} - see docs/data_remote.md")
        return 0

    rows = survey(store)
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0

    behind, gone = unpushed(rows), absent(rows)
    print(render(rows))
    files = sum(row["cached"] - row["stored"] for row in behind)
    print(
        f"\n{len(rows)} cached outputs, {len(behind)} not fully in the store "
        f"({files} objects), {len(gone)} gone."
    )
    if behind:
        print("run `dvc push` (or `make push`) to put them in the store", file=sys.stderr)
        return 1 if args.check else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
