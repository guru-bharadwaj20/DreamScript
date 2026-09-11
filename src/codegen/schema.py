"""Phase 12.1.1 - the fixed training-pair schema, its validator, and the JSONL it lives in.

    python -m src.codegen.schema data/processed/pairs/train.jsonl     # validate a shard

## The pair, frozen

Eight keys, exactly, in this order, one JSON object per line:

    diagram_id    str   the IR document's `id` - the join key back to data/processed/ir
    diagram_type  str   one of `DIAGRAM_TYPES` (schemas/ir.schema.json's enum)
    ir_text       str   `src.codegen.serialise.serialise(diagram, traversal)` - nothing else
    traversal     list  node ids in reading order, from `src.parse.sequences.traversal`
    target_code   str   the reference program
    language      str   one of `LANGUAGES`, and consistent with `diagram_type` (12.1.6)
    source        str   the corpus directory: didi, fa_bresler, flowchartseg, hdbpmn, sketch2code
                        or `synthetic` (12.1.4) or `handwritten` (12.1.3)
    split         str   train | val | test

`traversal` is stored **as well as** being embedded in `ir_text`'s `O` line, which is redundant
by construction, and that redundancy is the point: it is the one field 12.1.1 names that a
downstream consumer (the sampler, the 12.3 evaluator, a data audit) needs as a list without
re-parsing prompt text, and `validate` cross-checks the two so the redundancy cannot rot. The
alternative - store only `traversal` and rebuild `ir_text` at load time - was rejected because
it makes the training input a function of whatever `serialise` happens to be on the day the
loader runs, and 12.2.5's adapter must be reproducible against the bytes it was trained on.

## What `validate` catches, measured on 500 real diagrams x 12 deliberate corruptions

Every record is checked structurally, not just for key presence. The mutation test in
`tests/test_codegen_schema.py` builds a valid record from a real diagram and then breaks it one
way at a time; **12 of 12 corruptions are caught** - missing key, extra key, wrong type,
empty `ir_text`, empty `target_code`, unknown `diagram_type`, unknown `language`, unknown
`split`, unknown `source`, a `traversal` id absent from `ir_text`, an `ir_text` node absent from
`traversal`, and a `language` that is legal but wrong for the `diagram_type` (a `wireframe`
labelled `python`). The last three are the ones a key-presence check misses entirely and they
are the failure modes that actually happen: the pair builder is two functions, the traversal and
the serialiser, and a mismatch between them is silent everywhere else.

`validate` returns a **list of every problem**, not the first one and not a bool. A shard of
10K pairs (12.1.4) that fails on one rule is a one-line fix; discovering the rules one run at a
time is not, and 10.2.4 settled this house style already - four independent answers rather than
one `ok`.

## Rejected

    a JSON Schema file           REJECTED for this record. `schemas/ir.schema.json` exists
    (mirroring schemas/ir.*)     because IR is exchanged and versioned; a training pair is
                                 internal and its interesting rules - traversal/ir_text
                                 agreement, type/language agreement - are cross-field, which is
                                 exactly what a JSON Schema expresses worst.
    Parquet / a single JSON      REJECTED. 12.1.4 wants >=10K pairs written incrementally and
    array                        streamed back; JSONL appends, tails, shards and diffs in git,
                                 and `read_jsonl` never holds more than one line at a time.
    storing the diagram dict     REJECTED. That is the 27x token blowup 12.1.2 measured, kept
    alongside `ir_text`          on disk this time; `diagram_id` is the join key and the IR
                                 corpus is right there.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

#: The record's keys, in their canonical order. `write_jsonl` emits exactly these, in this
#: order, so a shard diffs line-for-line against a re-run.
FIELDS: tuple[str, ...] = (
    "diagram_id",
    "diagram_type",
    "ir_text",
    "traversal",
    "target_code",
    "language",
    "source",
    "split",
)

#: `schemas/ir.schema.json`'s enum, restated here so a pair cannot carry a type the IR cannot.
DIAGRAM_TYPES: tuple[str, ...] = (
    "flowchart",
    "wireframe",
    "state_machine",
    "er_diagram",
    "circuit",
    "unknown",
)

#: 12.1.6's five target generators, by the tag that goes in `language` and in the fenced block.
LANGUAGES: tuple[str, ...] = ("python", "sql", "tsx", "spice")

#: 12.1.6's mapping, enforced. `unknown` diagrams may target anything - the classifier of Phase
#: 5 has not spoken, and rejecting the pair would be inventing a fact.
LANGUAGE_BY_TYPE: dict[str, tuple[str, ...]] = {
    "flowchart": ("python",),
    "state_machine": ("python",),
    "er_diagram": ("sql",),
    "wireframe": ("tsx",),
    "circuit": ("spice",),
    "unknown": LANGUAGES,
}

#: The five corpus directories under data/processed/ir, plus the two Phase 12 producers.
SOURCES: tuple[str, ...] = (
    "didi",
    "fa_bresler",
    "flowchartseg",
    "hdbpmn",
    "sketch2code",
    "synthetic",
    "handwritten",
)

SPLITS: tuple[str, ...] = ("train", "val", "test")

_STR_FIELDS = (
    "diagram_id",
    "diagram_type",
    "ir_text",
    "target_code",
    "language",
    "source",
    "split",
)


@dataclass(frozen=True)
class Pair:
    """One training pair. Frozen because a record that mutates after validation is a lie."""

    diagram_id: str
    diagram_type: str
    ir_text: str
    traversal: list[str] = field(default_factory=list)
    target_code: str = ""
    language: str = "python"
    source: str = "synthetic"
    split: str = "train"

    def to_dict(self) -> dict[str, Any]:
        """Plain dict in `FIELDS` order - the thing that gets written and validated."""
        raw = asdict(self)
        return {k: raw[k] for k in FIELDS}

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> Pair:
        return cls(**{k: record[k] for k in FIELDS if k in record})


def ir_text_node_ids(ir_text: str) -> list[str]:
    """Node ids in an `ir_text`, in emission order. Kept here so `validate` needs no importer."""
    out = []
    for line in ir_text.split("\n"):
        if line.startswith("N|"):
            parts = line.split("|")
            if len(parts) > 1:
                out.append(parts[1])
    return out


def ir_text_order(ir_text: str) -> list[str]:
    """The `O` line's ids, or `[]` if there is no `O` line."""
    for line in ir_text.split("\n"):
        if line.startswith("O|"):
            return line[2:].split()
    return []


def validate(record: Any) -> list[str]:
    """Every problem with `record`, as strings. An empty list means the record is valid.

    Deliberately returns all problems rather than raising on the first - a 10K-pair shard is
    debugged once, not once per rule.
    """
    problems: list[str] = []
    if isinstance(record, Pair):
        record = record.to_dict()
    if not isinstance(record, dict):
        return [f"record is {type(record).__name__}, not a dict"]

    for key in FIELDS:
        if key not in record:
            problems.append(f"missing key: {key}")
    for key in sorted(set(record) - set(FIELDS)):
        problems.append(f"unexpected key: {key}")
    if problems:
        return problems

    for key in _STR_FIELDS:
        if not isinstance(record[key], str):
            problems.append(f"{key} is {type(record[key]).__name__}, not str")
    if not isinstance(record["traversal"], list) or not all(
        isinstance(x, str) for x in record["traversal"]
    ):
        problems.append("traversal is not a list[str]")
    if problems:
        return problems

    if not record["diagram_id"]:
        problems.append("diagram_id is empty")
    if not record["ir_text"]:
        problems.append("ir_text is empty")
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

    allowed = LANGUAGE_BY_TYPE.get(record["diagram_type"])
    if allowed and record["language"] in LANGUAGES and record["language"] not in allowed:
        problems.append(
            f"language {record['language']!r} is not 12.1.6's target for "
            f"diagram_type {record['diagram_type']!r} (expected one of {allowed})"
        )

    if record["ir_text"] and not record["ir_text"].startswith("T|"):
        problems.append("ir_text does not begin with a T| type line")
    nodes = set(ir_text_node_ids(record["ir_text"]))
    stated = set(record["traversal"])
    for nid in sorted(stated - nodes):
        problems.append(f"traversal id {nid!r} has no N| line in ir_text")
    for nid in sorted(nodes - stated):
        problems.append(f"ir_text node {nid!r} is missing from traversal")
    if record["ir_text"] and list(record["traversal"]) != ir_text_order(record["ir_text"]):
        problems.append("traversal does not match ir_text's O| line")
    return problems


# -- JSONL, which is the only storage this schema has ---------------------------------------


def write_jsonl(records: Iterable[Any], path: Path | str, *, check: bool = True) -> int:
    """Write pairs to `path`, one compact JSON object per line. Returns the count written.

    Raises on the first invalid record when `check` - a shard that silently contains a broken
    pair is worse than one that fails to build.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            data = record.to_dict() if isinstance(record, Pair) else {k: record[k] for k in FIELDS}
            if check:
                problems = validate(data)
                if problems:
                    raise ValueError(f"{data.get('diagram_id')!r}: {'; '.join(problems)}")
            handle.write(json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n")
            written += 1
    return written


def read_jsonl(path: Path | str) -> Iterator[dict[str, Any]]:
    """Stream records back. One line in memory at a time - 12.1.4 wants >=10K pairs."""
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
