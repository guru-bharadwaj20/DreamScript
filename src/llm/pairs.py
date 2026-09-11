"""Phase 12.2 - the code pairs the LLM rows train and score on, read through 12.1.1's loader.

Nothing here re-decides a pair: records come from `src.codegen.pairs.load_pairs(split)` (12.1.1's
frozen schema, 12.1.7's quality gate, 12.1.8's scribe-disjoint split), and this module only adds
the two lookups the LLM rows need - the IR document behind a pair, for scoring against the
drawing rather than against the reference program, and a train/test contamination check.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SPLITS = ("train", "validation", "test")


@lru_cache(maxsize=8)
def _load(split: str, sources: tuple[str, ...] | None) -> tuple[dict, ...]:
    from src.codegen.pairs import load_pairs

    return tuple(load_pairs(split, sources=list(sources) if sources else None))


def load(split: str, sources: Iterable[str] | None = None) -> list[dict]:
    """Validated pair records for one split (a fresh list; the records are shared)."""
    return list(_load(split, tuple(sorted(sources)) if sources else None))


def real_pairs() -> tuple[dict, ...]:
    """Every scribed validation and test pair plus all train pairs, for split-agnostic callers."""
    return tuple(p for split in SPLITS for p in _load(split, None))


def by_split(pairs: Iterable[dict], split: str) -> list[dict]:
    return [p for p in pairs if p["split"] == split]


def ir_path(pair: dict) -> Path:
    meta = pair.get("meta") or {}
    if isinstance(meta, str):
        meta = json.loads(meta)
    return ROOT / meta["ir_path"]


def diagram_for(pair: dict) -> dict:
    """The IR document behind a pair (for functional / hallucination scoring)."""
    return json.loads(ir_path(pair).read_text(encoding="utf-8"))


def contamination(train: Iterable[dict], held_out: Iterable[dict]) -> dict[str, int]:
    """Overlap between a training pool and a held-out set: diagram ids, scribes, exact IR text.

    Exact `ir_text` matters separately from ids: the same drawing ingested twice under two ids
    would pass an id check and still be a leak.
    """
    train = list(train)
    held_out = list(held_out)
    ids = {p["diagram_id"] for p in train} & {p["diagram_id"] for p in held_out}
    scribes = {p.get("scribe") for p in train if p.get("scribe")} & {
        p.get("scribe") for p in held_out if p.get("scribe")
    }
    texts = {p["ir_text"] for p in train} & {p["ir_text"] for p in held_out}
    return {"diagram_ids": len(ids), "scribes": len(scribes), "ir_texts": len(texts)}
