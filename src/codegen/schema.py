"""Phase 12.1.1 - the fixed training-pair schema, its validator, and the JSONL it lives in.

    python -m src.codegen.schema data/processed/codegen/pairs/train.jsonl   # validate a shard

## The pair, frozen at `SCHEMA_VERSION = 1`

Eleven keys, exactly, in this order, one JSON object per line:

    diagram_id    str        the IR document's `id` - the join key back to data/processed/ir
    diagram_type  str        one of `DIAGRAM_TYPES`, which is schemas/ir.schema.json's enum
    ir_text       str        `src.codegen.serialise.serialise(diagram, traversal)` - nothing else
    traversal     list[str]  node ids in emission order: exactly the ids of `ir_text`'s `O` line
    target_code   str        the reference program
    language      str        12.1.6's target for `diagram_type` (`LANGUAGE_BY_TYPE`)
    source        str        where the pair came from (`SOURCES`)
    split         str        train | validation | test, from 12.1.8 `src.codegen.splits.assign`
    split_basis   str        why the pair is in that split (12.1.8's basis names)
    scribe        str|None   namespaced writer id, or null when the source publishes none
    meta          dict       provenance only (structure family, flowchart mode, image path);
                             never read by the prompt, so nothing in it can leak into training

The *model* input is `(diagram_type, ir_text)` - `ir_text` already carries the traversal as its
`O` line - and the output is `target_code`. Everything else is bookkeeping a loader, a split
audit or a 12.3 evaluator needs without re-parsing prompt text. `src.codegen.prompt` (12.2.4)
turns a record into chat messages; `src.codegen.pairs.load_pairs` streams records back.

## Decisions that differ from the inherited draft, and why each was forced

The inherited draft was internally inconsistent with the three committed modules it sits
between, and three of its choices would have made every real pair invalid:

    split "val"            12.1.8 `splits.SPLITS` is `("train", "validation", "test")`, and
                           `splits.assign` writes "validation". The draft's enum rejected every
                           validation pair `assign` produced. Now "validation".
    language "tsx"         12.1.6 `targets.LANGUAGES` maps wireframe to "react", and 12.1.7's
                           `quality.CHECKS` has no "tsx" key, so a "tsx" pair would have been
                           rejected by the very filter meant to check it. Now "react".
    8 keys, extras refused `splits.assign` adds `split_basis` and `scribe` to every record, so
                           the draft refused the output of the split it was paired with. Both are
                           now schema fields - `scribe` is what a leak audit needs row by row.

The draft also cross-checked `traversal` against `ir_text` with a hand-rolled `split("|")`
that ignores 12.1.2's escaping (`\\p` is a literal pipe); `validate` now reads `ir_text` through
`serialise.parse`, the format's own reader, so the check cannot disagree with the format.

## Rules `validate` enforces beyond key presence

Every one is a failure a key-presence check misses and a pair builder can actually produce:

- `ir_text`'s `T|` line equals `diagram_type` - synthetic ER graphs say `er`, the IR enum says
  `er_diagram`, and a pair carrying both would train the model on a type name it is never shown.
- `traversal` equals the `O` line, has no repeats, and names exactly the `N|` rows.
- `language` is 12.1.6's language for `diagram_type`.
- `target_code` does not contain `diagram_id` (for ids of 6+ characters). The emitters name the
  function / class / screen after the diagram id, and an id like `ex00_writer0001` is invisible
  in `ir_text`: the model would be trained to produce an identifier it cannot see - a
  hallucination by construction, and a writer id leaking into code. Pair builders emit against
  the canonical id `diagram` instead (`src.codegen.pairs.CANONICAL_ID`).
- validation/test pairs name a scribe and synthetic pairs are train-only - 12.1.8's two
  invariants, re-checked per record so a hand-edited shard cannot break them silently.

`validate` returns a **list of every problem**, not the first and not a bool: a 10K-pair shard
that fails on one rule is fixed once, not once per rule.

## Rejected

    a JSON Schema file       the interesting rules are cross-field (traversal vs ir_text,
                             type vs language, id vs code), which JSON Schema expresses worst.
    Parquet / one JSON array JSONL appends, streams one record at a time, shards and diffs.
    storing the diagram dict that is the 27x token blowup 12.1.2 measured, kept on disk;
    in the record              `diagram_id` joins back to the IR corpus.
    rebuilding `ir_text` at   makes the training input a function of whatever `serialise` is on
    load time                 the day the loader runs; the adapter must be reproducible against
                              the bytes it was trained on.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

#: The record's keys, in canonical order. `write_jsonl` emits exactly these, in this order.
FIELDS: tuple[str, ...] = (
    "diagram_id",
    "diagram_type",
    "ir_text",
    "traversal",
    "target_code",
    "language",
    "source",
    "split",
    "split_basis",
    "scribe",
    "meta",
)

#: `schemas/ir.schema.json`'s enum, restated so a pair cannot carry a type the IR cannot.
DIAGRAM_TYPES: tuple[str, ...] = (
    "flowchart",
    "wireframe",
    "state_machine",
    "er_diagram",
    "circuit",
    "unknown",
)

#: 12.1.6's language tags - the keys of `src.codegen.quality.CHECKS` that a target may use.
LANGUAGES: tuple[str, ...] = ("python", "sql", "react", "spice")

#: 12.1.6's mapping, enforced. `unknown` may target anything: Phase 5 has not classified it.
LANGUAGE_BY_TYPE: dict[str, tuple[str, ...]] = {
    "flowchart": ("python",),
    "state_machine": ("python",),
    "er_diagram": ("sql",),
    "wireframe": ("react",),
    "circuit": ("spice",),
    "unknown": LANGUAGES,
}

#: IR corpus directories, 12.1.5's converted Sketch2Code HTML, 12.1.4 and 12.1.3.
SOURCES: tuple[str, ...] = (
    "didi",
    "fa_bresler",
    "flowchartseg",
    "hdbpmn",
    "sketch2code",
    "sketch2code_html",
    "synthetic",
    "handwritten",
)

#: 12.1.8's split names, verbatim from `src.codegen.splits.SPLITS`.
SPLITS: tuple[str, ...] = ("train", "validation", "test")

_STR_FIELDS = (
    "diagram_id",
    "diagram_type",
    "ir_text",
    "target_code",
    "language",
    "source",
    "split",
    "split_basis",
)

#: Ids shorter than this are too generic (`n0`, `s2c`) for a substring test to mean a leak.
_MIN_LEAK_ID = 6


def _parse(ir_text: str) -> dict:
    from src.codegen.serialise import parse

    return parse(ir_text)


def validate(record: Any) -> list[str]:
    """Every problem with `record`, as strings. An empty list means the record is valid."""
    if not isinstance(record, dict):
        return [f"record is {type(record).__name__}, not a dict"]

    problems = [f"missing key: {key}" for key in FIELDS if key not in record]
    problems += [f"unexpected key: {key}" for key in sorted(set(record) - set(FIELDS))]
    if problems:
        return problems

    for key in _STR_FIELDS:
        if not isinstance(record[key], str):
            problems.append(f"{key} is {type(record[key]).__name__}, not str")
    traversal = record["traversal"]
    if not isinstance(traversal, list) or not all(isinstance(x, str) for x in traversal):
        problems.append("traversal is not a list[str]")
    if record["scribe"] is not None and not isinstance(record["scribe"], str):
        problems.append("scribe is neither str nor null")
    if not isinstance(record["meta"], dict):
        problems.append("meta is not a dict")
    if problems:
        return problems

    if not record["diagram_id"]:
        problems.append("diagram_id is empty")
    if not record["target_code"].strip():
        problems.append("target_code is empty")
    if record["diagram_type"] not in DIAGRAM_TYPES:
        problems.append(f"diagram_type not in {DIAGRAM_TYPES}: {record['diagram_type']!r}")
    if record["language"] not in LANGUAGES:
        problems.append(f"language not in {LANGUAGES}: {record['language']!r}")
    if record["source"] not in SOURCES:
        problems.append(f"source not in {SOURCES}: {record['source']!r}")
    if record["split"] not in SPLITS:
        problems.append(f"split not in {SPLITS}: {record['split']!r}")
    if not record["split_basis"]:
        problems.append("split_basis is empty")

    allowed = LANGUAGE_BY_TYPE.get(record["diagram_type"])
    if allowed and record["language"] in LANGUAGES and record["language"] not in allowed:
        problems.append(
            f"language {record['language']!r} is not 12.1.6's target for "
            f"diagram_type {record['diagram_type']!r} (expected one of {allowed})"
        )

    if record["split"] in ("validation", "test") and not record["scribe"]:
        problems.append(f"{record['split']} pair has no scribe (12.1.8: held-out pairs need one)")
    if record["source"] == "synthetic" and record["split"] != "train":
        problems.append("synthetic pair outside train (12.1.8: synthetic is train-only)")

    diagram_id = record["diagram_id"]
    if len(diagram_id) >= _MIN_LEAK_ID and diagram_id in record["target_code"]:
        problems.append("target_code contains diagram_id, which ir_text never shows")

    ir_text = record["ir_text"]
    if not ir_text.startswith("T|"):
        problems.append("ir_text does not begin with a T| type line")
        return problems
    parsed = _parse(ir_text)
    if parsed["diagram_type"] != record["diagram_type"]:
        problems.append(
            f"ir_text type {parsed['diagram_type']!r} != diagram_type {record['diagram_type']!r}"
        )
    node_ids = [node["id"] for node in parsed["nodes"]]
    if len(set(traversal)) != len(traversal):
        problems.append("traversal repeats a node id")
    for nid in sorted(set(traversal) - set(node_ids)):
        problems.append(f"traversal id {nid!r} has no N| line in ir_text")
    for nid in sorted(set(node_ids) - set(traversal)):
        problems.append(f"ir_text node {nid!r} is missing from traversal")
    if traversal != parsed["traversal"]:
        problems.append("traversal does not match ir_text's O| line")
    return problems


# -- JSONL, the only storage this schema has ---------------------------------------------------


def to_line(record: dict) -> str:
    """One compact JSON line in `FIELDS` order - byte-stable for a given record."""
    return json.dumps({k: record[k] for k in FIELDS}, ensure_ascii=False, separators=(",", ":"))


def write_jsonl(records: Iterable[dict], path: Path | str, *, check: bool = True) -> int:
    """Write pairs to `path`, one per line. Returns the count. Raises on an invalid record."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            if check:
                problems = validate(record)
                if problems:
                    raise ValueError(f"{record.get('diagram_id')!r}: {'; '.join(problems)}")
            handle.write(to_line(record) + "\n")
            written += 1
    return written


def read_jsonl(path: Path | str) -> Iterator[dict[str, Any]]:
    """Stream records back, one line in memory at a time. A bad line is an error."""
    with Path(path).open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: {exc}") from exc


def validate_file(path: Path | str) -> dict[str, Any]:
    """Validate a whole shard. Returns counts plus the first few failing lines."""
    total = 0
    bad: list[dict[str, Any]] = []
    for lineno, record in enumerate(read_jsonl(path), 1):
        total += 1
        problems = validate(record)
        if problems:
            bad.append(
                {"line": lineno, "diagram_id": record.get("diagram_id"), "problems": problems}
            )
    return {"path": str(path), "records": total, "invalid": len(bad), "examples": bad[:10]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 12.1.1 - validate a training-pair shard")
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args(argv)
    bad = 0
    for path in args.paths:
        report = validate_file(path)
        bad += report["invalid"]
        json.dump(report, sys.stdout, indent=2)
        sys.stdout.write("\n")
    return 1 if bad else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
