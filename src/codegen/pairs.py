"""Phase 12.1.1 - build, split, write and load the (diagram_type, IR, traversal) -> code pairs.

    python -m src.codegen.pairs build                  # real IR corpus -> per-source shards
    python -m src.codegen.pairs merge                  # shards -> train/validation/test.jsonl
    python -m src.codegen.pairs stats

    from src.codegen.pairs import load_pairs
    for record in load_pairs("train", sources=("hdbpmn", "synthetic")): ...

## Layout on disk (data/processed/codegen/pairs/, gitignored, rebuilt by the commands above)

    <source>.jsonl                 one shard per producer, every record already split
    train.jsonl / validation.jsonl / test.jsonl
                                   the merge of every shard present, sources in sorted order
    manifest.json                  per-split x per-source counts and each file's sha256

Shards are the unit of rebuilding: 12.1.4 writes `synthetic.jsonl`, 12.1.5 writes
`sketch2code_html.jsonl`, and `merge` never has to re-emit the real corpus to add them.

## How one pair is made (`make_pair`)

1. `diagram_type` is canonicalised to schemas/ir.schema.json's enum (`er` -> `er_diagram`).
2. The traversal is 7.3.3's `src.parse.sequences.traversal` - the reading-order DFS the rest of
   the pipeline already uses as gold order - unless the caller supplies one.
3. `ir_text = serialise(diagram, traversal)`; the stored `traversal` is `ir_text`'s `O` line,
   so the two agree by construction rather than by a second computation.
4. `target_code` is 12.1.6's emitter run on the diagram **with its id replaced by
   `CANONICAL_ID`**. The emitters name the function / class / screen / netlist title after the
   diagram id, which `ir_text` never shows, so an un-canonicalised target teaches the model to
   produce an identifier it cannot see - and for hdbpmn that identifier is the writer's id.
5. `splits.assign` (12.1.8) fills `split`, `split_basis` and `scribe`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from multiprocessing import Pool
from pathlib import Path
from typing import Any

from src.codegen import schema
from src.utils.config import ROOT

OUT_DIR = ROOT / "data" / "processed" / "codegen"
PAIRS_DIR = OUT_DIR / "pairs"
REJECT_DIR = OUT_DIR / "rejected"
IR_DIR = ROOT / "data" / "processed" / "ir"

#: The name every emitter is given in place of the diagram id (see module docstring, step 4).
CANONICAL_ID = "diagram"

#: Real IR sources that become pairs. flowchartseg is excluded, not forgotten: its 1,319 pages
#: carry node polygons and zero edges, so every target is an unbranched list of statements and
#: the pairs would teach "ignore connectivity" to a model whose whole job is connectivity.
REAL_SOURCES: tuple[str, ...] = ("didi", "fa_bresler", "hdbpmn", "sketch2code")
EXCLUDED_SOURCES: dict[str, str] = {"flowchartseg": "0 edges on every page"}

_TYPE_ALIASES = {"er": "er_diagram", "erd": "er_diagram", "bpmn": "flowchart"}


def canonical_type(diagram_type: str) -> str:
    """The IR schema's name for a diagram type."""
    value = str(diagram_type or "unknown")
    value = _TYPE_ALIASES.get(value, value)
    return value if value in schema.DIAGRAM_TYPES else "unknown"


def make_pair(
    diagram: dict,
    source: str,
    *,
    traversal: list[str] | None = None,
    target_code: str | None = None,
    meta: dict | None = None,
) -> dict:
    """One unsplit pair record (`split` is filled by `assign_splits`)."""
    from src.codegen import serialise, targets
    from src.parse.sequences import traversal as gold_traversal

    doc = dict(diagram, diagram_type=canonical_type(diagram.get("diagram_type")))
    order = list(traversal) if traversal is not None else gold_traversal(doc)[0]
    ir_text = serialise.serialise(doc, order)
    full_order = serialise.parse(ir_text)["traversal"]
    extra = dict(meta or {})
    if target_code is None:
        anonymous = dict(doc, id=CANONICAL_ID)
        if doc["diagram_type"] == "flowchart":
            target_code, extra["flowchart_mode"] = targets.flowchart_mode(anonymous, full_order)
        else:
            target_code = targets.for_type(doc["diagram_type"])(anonymous, full_order)
    return {
        "diagram_id": str(diagram.get("id")),
        "diagram_type": doc["diagram_type"],
        "ir_text": ir_text,
        "traversal": full_order,
        "target_code": target_code,
        "language": targets.language_for(doc["diagram_type"]),
        "source": source,
        "split": None,
        "split_basis": None,
        "scribe": None,
        "meta": extra,
    }


def assign_splits(records: list[dict]) -> list[dict]:
    """12.1.8's split, in schema field order. Never mutates the input."""
    from src.codegen import splits

    assigned = splits.assign(records)
    return [{key: record.get(key) for key in schema.FIELDS} for record in assigned]


def _real_pair(path_and_source: tuple[str, str]) -> dict:
    path, source = path_and_source
    diagram = json.loads(Path(path).read_text(encoding="utf-8"))
    rel = Path(path).relative_to(ROOT).as_posix()
    return make_pair(diagram, source, meta={"ir_path": rel})


def build_real(sources: Iterable[str] = REAL_SOURCES, workers: int | None = None) -> dict:
    """Emit, split, validate and write one shard per real source. Returns per-shard counts."""
    report: dict[str, Any] = {}
    for source in sources:
        paths = sorted((IR_DIR / source).glob("*.ir.json"))
        jobs = [(str(p), source) for p in paths]
        with Pool(workers or max(1, (os.cpu_count() or 2) - 2)) as pool:
            records = pool.map(_real_pair, jobs, chunksize=16)
        report[source] = write_shard(source, assign_splits(records))
    return report


def write_shard(source: str, records: list[dict]) -> dict:
    """12.1.7's filter, its postcondition, then the shard. Rejections go to `rejected/`.

    Nothing reaches `pairs/<source>.jsonl` without passing `quality.filter_pairs`, and the kept
    set is re-checked from scratch by `assert_all_compile` before a byte is written.
    """
    from src.codegen import quality

    kept, rejected = quality.filter_pairs(records)
    quality.assert_all_compile(kept)
    written = schema.write_jsonl(kept, PAIRS_DIR / f"{source}.jsonl")
    rejects = REJECT_DIR / f"{source}.jsonl"
    rejects.parent.mkdir(parents=True, exist_ok=True)
    with rejects.open("w", encoding="utf-8", newline="\n") as handle:
        for entry in rejected:
            row = {k: entry[k] for k in ("diagram_id", "kind", "detail", "language")}
            handle.write(json.dumps({**row, "record": entry["record"]}, ensure_ascii=False) + "\n")
    return {
        "candidates": len(records),
        "pairs": written,
        "rejected": len(rejected),
        "reject_kinds": dict(Counter(entry["kind"] for entry in rejected)),
        "splits": dict(Counter(r["split"] for r in kept)),
        "types": dict(Counter(r["diagram_type"] for r in kept)),
    }


# -- merge and load ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def shard_paths() -> list[Path]:
    return sorted(p for p in PAIRS_DIR.glob("*.jsonl") if p.stem not in schema.SPLITS)


def merge() -> dict:
    """Concatenate every shard into the three split files and write `manifest.json`."""
    counts: dict[str, Counter] = defaultdict(Counter)
    handles = {
        split: (PAIRS_DIR / f"{split}.jsonl").open("w", encoding="utf-8", newline="\n")
        for split in schema.SPLITS
    }
    try:
        for shard in shard_paths():
            for record in schema.read_jsonl(shard):
                handles[record["split"]].write(schema.to_line(record) + "\n")
                counts[record["split"]][record["source"]] += 1
    finally:
        for handle in handles.values():
            handle.close()
    manifest = {
        "schema_version": schema.SCHEMA_VERSION,
        "splits": {
            split: {
                "total": sum(counts[split].values()),
                "by_source": dict(sorted(counts[split].items())),
            }
            for split in schema.SPLITS
        },
        "sha256": {p.name: _sha256(p) for p in sorted(PAIRS_DIR.glob("*.jsonl"))},
    }
    (PAIRS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_pairs(
    split: str | None = None,
    sources: Iterable[str] | None = None,
    diagram_types: Iterable[str] | None = None,
    root: Path | None = None,
) -> Iterator[dict]:
    """Stream pair records from the merged split files.

    `split` is one of `schema.SPLITS` or None for all three; `sources` / `diagram_types` filter.
    Records are returned exactly as written - already validated by `schema.validate`.
    """
    base = root or PAIRS_DIR
    wanted_sources = set(sources) if sources is not None else None
    wanted_types = set(diagram_types) if diagram_types is not None else None
    for name in [split] if split else schema.SPLITS:
        if name not in schema.SPLITS:
            raise ValueError(f"split must be one of {schema.SPLITS}, got {name!r}")
        path = base / f"{name}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"{path} missing; run `python -m src.codegen.pairs merge`")
        for record in schema.read_jsonl(path):
            if wanted_sources is not None and record["source"] not in wanted_sources:
                continue
            if wanted_types is not None and record["diagram_type"] not in wanted_types:
                continue
            yield record


def stats(root: Path | None = None) -> dict:
    """Split x source x type counts and the invalid count, recomputed from the split files."""
    table: dict[str, Counter] = defaultdict(Counter)
    invalid = 0
    for record in load_pairs(root=root):
        table[record["split"]][f"{record['source']}/{record['diagram_type']}"] += 1
        invalid += bool(schema.validate(record))
    return {"invalid": invalid, "splits": {k: dict(sorted(v.items())) for k, v in table.items()}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 12.1.1 training pairs")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--sources", nargs="+", default=list(REAL_SOURCES))
    build.add_argument("--workers", type=int, default=None)
    sub.add_parser("merge")
    sub.add_parser("stats")
    args = parser.parse_args(argv)
    if args.command == "build":
        out = build_real(args.sources, args.workers)
    elif args.command == "merge":
        out = merge()
    else:
        out = stats()
    json.dump(out, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
