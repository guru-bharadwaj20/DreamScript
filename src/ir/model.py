"""Phase 2.1.3 - the DreamScript IR as Python objects.

`schemas/ir.schema.json` is the contract; this is the ergonomic way to build something that
satisfies it. Every converter in `src/ir/convert/` produces a `Diagram`, and `Diagram.save`
refuses to write a file the schema would reject - so an invalid IR never reaches disk.

    python -m src.ir.model path/to/diagram.ir.json    # validate and summarise one file
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.ir import schema

IR_VERSION = "1.0"

#: Written next to every IR file so a reader can tell IR from anything else in the directory.
SUFFIX = ".ir.json"


class IRError(ValueError):
    """An IR object that does not satisfy the schema, raised with every problem listed."""


@dataclass
class Node:
    id: str
    shape: str
    bbox: list[float] | None
    text: str = ""
    semantic_role: str = "unknown"
    confidence: float = 1.0
    source_id: str | None = None
    attrs: dict[str, Any] | None = None

    @property
    def centre(self) -> tuple[float, float] | None:
        if self.bbox is None:
            return None
        x, y, w, h = self.bbox
        return (x + w / 2, y + h / 2)


@dataclass
class Edge:
    id: str
    src: str | None
    dst: str | None
    directed: bool = True
    label: str = ""
    polyline: list[list[float]] | None = None
    confidence: float = 1.0
    source_id: str | None = None
    attrs: dict[str, Any] | None = None

    @property
    def dangling(self) -> bool:
        return self.src is None or self.dst is None


@dataclass
class Diagram:
    id: str
    diagram_type: str
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    ir_version: str = IR_VERSION
    #: Phase 2.1.6. Three kinds of ambiguity a hand-drawn diagram leaves behind, recorded
    #: rather than resolved. A converter or detector that cannot decide writes the doubt down
    #: here; Phase 10 repairs what it can and Phase 13 asks the user about the rest.
    unresolved_edges: list[dict[str, Any]] = field(default_factory=list)
    crossed_out: list[dict[str, Any]] = field(default_factory=list)
    low_conf_text: list[dict[str, Any]] = field(default_factory=list)

    # -- conversion -------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Schema-shaped dict. Optional fields that were never set are dropped rather than
        serialised as null, because `"attrs": null` is not the same as no attrs and only one
        of the two is legal."""

        def clean(d: dict[str, Any], optional: tuple[str, ...]) -> dict[str, Any]:
            return {k: v for k, v in d.items() if k not in optional or v is not None}

        return {
            "ir_version": self.ir_version,
            "id": self.id,
            "diagram_type": self.diagram_type,
            "nodes": [clean(asdict(n), ("source_id", "attrs")) for n in self.nodes],
            "edges": [clean(asdict(e), ("source_id", "attrs")) for e in self.edges],
            "unresolved_edges": [dict(u) for u in self.unresolved_edges],
            "crossed_out": [dict(c) for c in self.crossed_out],
            "low_conf_text": [dict(t) for t in self.low_conf_text],
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Diagram:
        return cls(
            id=d["id"],
            diagram_type=d["diagram_type"],
            nodes=[Node(**n) for n in d.get("nodes", [])],
            edges=[Edge(**e) for e in d.get("edges", [])],
            meta=dict(d.get("meta", {})),
            ir_version=d.get("ir_version", IR_VERSION),
            unresolved_edges=[dict(u) for u in d.get("unresolved_edges", [])],
            crossed_out=[dict(c) for c in d.get("crossed_out", [])],
            low_conf_text=[dict(t) for t in d.get("low_conf_text", [])],
        )

    # -- ambiguity --------------------------------------------------------------------

    def record_unresolved(self, edge: Edge, candidates: list[str] | None = None, **extra) -> None:
        """Note that `edge` has an open end, deriving the reason from which end it is."""
        if edge.src is None and edge.dst is None:
            reason = "both-ends-open"
        elif edge.src is None:
            reason = "no-source"
        elif edge.dst is None:
            reason = "no-target"
        else:
            reason = extra.pop("reason", "ambiguous-endpoint")
        entry = {"edge": edge.id, "reason": reason}
        if candidates:
            entry["candidates"] = list(candidates)
        entry.update(extra)
        self.unresolved_edges.append(entry)

    def sync_unresolved(self) -> int:
        """Make sure every dangling edge is also listed as unresolved, and return how many
        entries had to be added.

        The two representations exist for different readers - `edges` for anything walking the
        graph, `unresolved_edges` for anything reporting on quality - and the failure mode is
        that a converter updates one and forgets the other. This is the cheap fix; the QA pass
        in Phase 2.2.5 is the check that it was called.
        """
        listed = {u["edge"] for u in self.unresolved_edges}
        added = 0
        for edge in self.edges:
            if edge.dangling and edge.id not in listed:
                self.record_unresolved(edge)
                added += 1
        return added

    # -- validation -------------------------------------------------------------------

    def problems(self) -> list[str]:
        return schema.problems("ir", self.to_dict())

    def require_valid(self) -> Diagram:
        probs = self.problems()
        if probs:
            raise IRError(f"{self.id}: " + "; ".join(probs))
        return self

    # -- io ---------------------------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        self.require_valid()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False, sort_keys=False)
            fh.write("\n")
        return path

    @classmethod
    def load(cls, path: str | Path) -> Diagram:
        with Path(path).open(encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    # -- graph helpers ----------------------------------------------------------------

    def node(self, node_id: str) -> Node | None:
        return next((n for n in self.nodes if n.id == node_id), None)

    @property
    def node_ids(self) -> set[str]:
        return {n.id for n in self.nodes}

    def out_edges(self, node_id: str) -> list[Edge]:
        return [e for e in self.edges if e.src == node_id]

    def in_edges(self, node_id: str) -> list[Edge]:
        return [e for e in self.edges if e.dst == node_id]

    def __repr__(self) -> str:  # a converter's output is read far more often than debugged
        return f"<Diagram {self.id} {self.diagram_type} {len(self.nodes)}n {len(self.edges)}e>"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+", type=Path)
    args = ap.parse_args(argv)

    ok = True
    for path in args.paths:
        d = Diagram.load(path)
        probs = d.problems()
        ok &= not probs
        print(f"  {'PASS' if not probs else 'FAIL'}  {path.name}  {d!r}")
        for p in probs:
            print(f"          {p}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
