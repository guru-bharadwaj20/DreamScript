"""Phase 1.1 — the dataset registry.

One record per external dataset: where it comes from, what license governs it, how it is
fetched, and whether its raw files may be redistributed. Every downloader in
`src/ingest/datasets/` registers itself here, so `python -m src.ingest --config
configs/ingest.yaml` can report the acquisition state of the whole corpus in one place.

Raw data never enters git (see `.gitignore`); the registry records provenance so the corpus
can be rebuilt from scratch on another machine.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import date
from enum import Enum
from pathlib import Path

from src.utils.config import ROOT

RAW = ROOT / "data" / "raw"


class Availability(str, Enum):
    """How the dataset can actually be obtained, as verified — not as advertised."""

    OPEN = "open"  # direct public download, no account
    ACCOUNT = "account"  # free registration or acceptance of terms required
    REQUEST = "request"  # must email the authors
    UNAVAILABLE = "unavailable"  # original host is dead and no mirror was found


class Redistribution(str, Enum):
    YES = "yes"
    ATTRIBUTION = "attribution"
    NO = "no"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Dataset:
    slug: str
    name: str
    url: str
    license: str
    availability: Availability
    redistribution: Redistribution
    diagram_types: tuple[str, ...]
    phases: tuple[str, ...]
    data_card: str
    notes: str = ""
    checked_on: str = field(default_factory=lambda: date.today().isoformat())

    @property
    def path(self) -> Path:
        return RAW / self.slug

    @property
    def present(self) -> bool:
        return self.path.is_dir() and any(self.path.iterdir())

    def as_dict(self) -> dict:
        d = asdict(self)
        d["availability"] = self.availability.value
        d["redistribution"] = self.redistribution.value
        d["present"] = self.present
        d["path"] = str(self.path.relative_to(ROOT))
        return d


REGISTRY: dict[str, Dataset] = {}


def register(ds: Dataset) -> Dataset:
    REGISTRY[ds.slug] = ds
    return ds


def get(slug: str) -> Dataset:
    if slug not in REGISTRY:
        raise KeyError(f"unknown dataset {slug!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[slug]


def status() -> list[dict]:
    """Acquisition state of every registered dataset, for the ingest CLI and reports."""
    return [ds.as_dict() for ds in REGISTRY.values()]


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def write_provenance(ds: Dataset, extra: dict | None = None) -> Path:
    """Record what was fetched, when, and how much of it, next to the raw files."""
    out = ds.path / "_provenance.json"
    payload = ds.as_dict()
    files = [p for p in ds.path.rglob("*") if p.is_file() and p.name != "_provenance.json"]
    payload["n_files"] = len(files)
    payload["bytes"] = sum(p.stat().st_size for p in files)
    payload["fetched_on"] = date.today().isoformat()
    if extra:
        payload.update(extra)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out


# Registrations live in the individual downloader modules so importing one is enough to
# make it visible; importing the package registers all of them.
from src.ingest import datasets  # noqa: E402,F401  (import for side effects)
