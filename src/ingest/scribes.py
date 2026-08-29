"""Phase 1.2.6 — the scribe registry.

Style variation is the axis this project is most likely to fool itself on: a corpus drawn
mostly by one person produces excellent numbers that mean nothing (risk D3 in `docs/risks.md`).
The registry is what makes the ≥8-scribe requirement checkable rather than aspirational, and
it is the key the scribe-disjoint splits (Phase 1.3.3) and leave-one-scribe-out evaluation
(Phase 14.6) join on.

    python -m src.ingest.scribes --init          # create the registry with 8 placeholder rows
    python -m src.ingest.scribes                 # show the registry and its checks
    python -m src.ingest.scribes --validate      # fail if the corpus and registry disagree

Deliberately minimal on personal data: an opaque id, a self-declared handwriting style, the
media a person used, and whether they consented. No names, no contact details in the repo.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from src.ingest.collection import MEDIA, MIN_SCRIBES, collected
from src.utils.config import ROOT

REGISTRY_CSV = ROOT / "data" / "raw" / "chaos" / "scribes.csv"
TEMPLATE_CSV = ROOT / "docs" / "collection" / "scribes_template.csv"

# Self-declared, because the point is to span the range, not to grade anyone's handwriting.
STYLES = ("neat", "average", "messy")
HANDEDNESS = ("left", "right")


@dataclass
class Scribe:
    scribe_id: str
    style: str  # neat | average | messy - the axis grouped CV splits on
    handedness: str
    media_used: str  # semicolon-separated subset of MEDIA
    consent: str  # yes | no - no row without consent is used
    notes: str = ""

    def validate(self) -> list[str]:
        problems = []
        if not self.scribe_id.startswith("scribe"):
            problems.append(f"{self.scribe_id}: id must start with 'scribe'")
        if self.style not in STYLES:
            problems.append(f"{self.scribe_id}: style {self.style!r} not in {STYLES}")
        if self.handedness not in HANDEDNESS:
            problems.append(f"{self.scribe_id}: handedness {self.handedness!r} not in {HANDEDNESS}")
        for m in filter(None, self.media_used.split(";")):
            if m not in MEDIA:
                problems.append(f"{self.scribe_id}: unknown medium {m!r}")
        if self.consent != "yes":
            problems.append(f"{self.scribe_id}: consent is {self.consent!r}, must be 'yes'")
        return problems


def template_rows() -> list[Scribe]:
    """Eight placeholder rows spanning the style range, for the collector to fill in.

    The style mix is deliberate: two neat, four average, two messy. A corpus of only neat
    drafters is exactly the failure mode Phase 5.2.2's grouped CV exists to detect.
    """
    plan = [
        ("scribe01", "neat", "right", "pencil;ballpoint"),
        ("scribe02", "neat", "left", "ballpoint;stylus"),
        ("scribe03", "average", "right", "pencil;marker"),
        ("scribe04", "average", "right", "whiteboard;marker"),
        ("scribe05", "average", "left", "ballpoint;pencil"),
        ("scribe06", "average", "right", "stylus;pencil"),
        ("scribe07", "messy", "right", "marker;whiteboard"),
        ("scribe08", "messy", "right", "ballpoint;pencil"),
    ]
    return [
        Scribe(sid, style, hand, media, consent="", notes="placeholder - fill in on collection")
        for sid, style, hand, media in plan
    ]


def write_template() -> Path:
    TEMPLATE_CSV.parent.mkdir(parents=True, exist_ok=True)
    cols = [f.name for f in fields(Scribe)]
    with TEMPLATE_CSV.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for s in template_rows():
            w.writerow(asdict(s))
    return TEMPLATE_CSV


def load() -> list[Scribe]:
    if not REGISTRY_CSV.is_file():
        return []
    with REGISTRY_CSV.open(encoding="utf-8", newline="") as fh:
        return [Scribe(**{k: (v or "") for k, v in row.items()}) for row in csv.DictReader(fh)]


def report() -> dict:
    registry = load()
    consented = [s for s in registry if s.consent == "yes"]
    corpus_scribes = {r["scribe_id"] for r in collected()}
    registered = {s.scribe_id for s in registry}

    problems: list[str] = []
    for s in registry:
        problems.extend(s.validate())
    unregistered = sorted(corpus_scribes - registered)
    if unregistered:
        problems.append(f"drawings exist for unregistered scribes: {unregistered}")

    styles = {st: sum(1 for s in consented if s.style == st) for st in STYLES}

    return {
        "registry_exists": REGISTRY_CSV.is_file(),
        "n_registered": len(registry),
        "n_consented": len(consented),
        "style_mix": styles,
        "scribes_with_drawings": sorted(corpus_scribes),
        "unregistered_scribes": unregistered,
        "problems": problems,
        "checks": {
            "min_scribes_registered": len(consented) >= MIN_SCRIBES,
            "all_consented": bool(registry) and len(consented) == len(registry),
            "style_range_covered": (
                all(styles.get(st, 0) >= 1 for st in STYLES) if consented else False
            ),
            "no_unregistered_drawings": not unregistered,
            "no_validation_problems": not problems,
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--init", action="store_true", help="write the registry template")
    ap.add_argument("--validate", action="store_true", help="exit non-zero on any problem")
    args = ap.parse_args(argv)

    if args.init:
        path = write_template()
        print(f"wrote {path.relative_to(ROOT)}")
        print(f"copy it to {REGISTRY_CSV.relative_to(ROOT)} and fill it in as people draw")
        return 0

    r = report()
    print(
        f"registry: {REGISTRY_CSV.relative_to(ROOT)} "
        f"({'present' if r['registry_exists'] else 'NOT CREATED YET'})"
    )
    print(
        f"  registered: {r['n_registered']}  consented: {r['n_consented']} " f"(need {MIN_SCRIBES})"
    )
    print(f"  style mix : {r['style_mix']}")
    print(f"  drawings from: {r['scribes_with_drawings'] or 'nobody yet'}")
    for p in r["problems"]:
        print(f"  PROBLEM: {p}")
    print("-" * 60)
    for name, ok in r["checks"].items():
        print(f"  {'PASS' if ok else 'PENDING'}  {name}")
    if args.validate:
        return 0 if all(r["checks"].values()) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
