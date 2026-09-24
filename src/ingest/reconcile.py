"""Phase 1.1.7 - the three corpus counts, and why they differ.

    python -m src.ingest reconcile          # writes reports/corpus_reconciliation.md
    python -m src.ingest reconcile --check  # non-zero if a difference is not accounted for

Three artefacts describe "the corpus" and **none of them agrees with the others**:

    data/processed/manifest.parquet   3,054 rows over 4 sources
    data/processed/ir/                2,796 documents over 4 sources
    data/processed/detect/index.json  2,312 pages over 3 sources

Nothing reconciled them, so each of those numbers has been quoted somewhere as "the corpus" and
a reader had no way to tell a legitimate filter from data loss. Both differences turn out to be
legitimate, and that is exactly why they were worth resolving rather than assuming.

**sketch2code, 731 -> 484.** The manifest row is a *sketch* and the IR document is the *webpage*
it depicts. Sketch2Code pairs 731 human sketches with 484 real pages, several sketches per page,
and the manifest id carries both (`sketch2code/<page>_<n>`). 484 is the number of distinct page
ids in the manifest - not an approximation of it, exactly it.

**hdbpmn, 704 -> 693.** Eleven manifest entries have no IR document, and they are not scattered:
nine belong to two writers (0099, 0105). The IR is a strict subset - no document exists that the
manifest does not list - so this is conversion loss rather than divergence, and the eleven are
named in the report so that it is a list somebody can act on rather than a number.

**detect, 2,796 -> 2,312.** sketch2code has no detection annotations anywhere in this repo
(9.1.2's export is hdbpmn, flowchartseg and fa_bresler), so its 484 are absent by construction:
2,796 - 484 = 2,312.

`--check` asserts all three relationships. A count that moves for a fourth reason fails here
instead of being quoted.
"""

from __future__ import annotations

import argparse
import json
import re
import sys

from src.utils.config import ROOT

MANIFEST = ROOT / "data" / "processed" / "manifest.parquet"
IR_DIR = ROOT / "data" / "processed" / "ir"
DETECT_INDEX = ROOT / "data" / "processed" / "detect" / "index.json"
REPORT = ROOT / "reports" / "corpus_reconciliation.md"

#: Sources with no detection annotations anywhere in this repo. 9.1.2's export is hdbpmn,
#: flowchartseg and fa_bresler; sketch2code's wireframes were never annotated with boxes.
NO_DETECTION = ("sketch2code",)

#: Sources whose manifest row is not the IR document's unit. The row is a sketch and the document
#: is the page it depicts, so the two counts are *meant* to differ and the relationship is
#: "distinct page ids", not "rows".
PER_PAGE = ("sketch2code",)

#: How an IR filename is built from a manifest id, per source. The converters do not agree, and
#: pretending they do is what makes a reconciliation report wrong rather than useful:
#:
#:   fa_bresler     `fa_bresler/writer000_fa_001`     -> `writer000_fa_001`     the id's last part
#:   hdbpmn         `hdbpmn/ex00/ex00_writer0001`     -> `ex00_writer0001`      the id's last part
#:   sketch2code    `sketch2code/10018_0`             -> `s2c_10018`            prefix + page id
#:   flowchartseg   `flowchartseg/train/00000`        -> `fcseg_00000`          **renumbered**
#:
#: flowchartseg is the one that cannot be mapped: its manifest ids repeat across splits (1,319
#: rows, 1,187 distinct last parts) and the converter assigns a fresh sequence. So for that
#: source the reconciliation is by count only, and this table is what says which sources can be
#: reconciled *by name* - and therefore for which ones an "unconverted" list means anything.
NAME_PREFIX: dict[str, str] = {"sketch2code": "s2c_"}
COUNT_ONLY = ("flowchartseg",)


def _frame():
    import pandas as pd

    return pd.read_parquet(MANIFEST)


def manifest_counts() -> dict[str, int]:
    return {str(k): int(v) for k, v in _frame().groupby("source").size().items()}


def manifest_units() -> dict[str, set[str]]:
    """source -> the set of ids the IR converter would produce one document for."""
    out: dict[str, set[str]] = {}
    for source, rows in _frame().groupby("source"):
        ids = [str(value) for value in rows["id"]]
        if str(source) in PER_PAGE:
            # `sketch2code/<page>_<n>` - the page is the unit and the sketch index is not.
            out[str(source)] = {
                match.group(1) for value in ids if (match := re.search(r"/(\d+)_", value))
            }
        else:
            out[str(source)] = {value.rsplit("/", 1)[-1] for value in ids}
    return out


def ir_counts() -> dict[str, int]:
    if not IR_DIR.is_dir():
        return {}
    return {
        path.name: len(list(path.glob("*.ir.json")))
        for path in sorted(IR_DIR.iterdir())
        if path.is_dir()
    }


def detect_counts() -> dict[str, int]:
    if not DETECT_INDEX.is_file():
        return {}
    counts: dict[str, int] = {}
    for row in json.loads(DETECT_INDEX.read_text(encoding="utf-8")):
        counts[row["source"]] = counts.get(row["source"], 0) + 1
    return counts


def unconverted() -> dict[str, list[str]]:
    """source -> manifest units with no IR document. The gap as a list rather than a number.

    Only for sources in `NAME_PREFIX` or neither table - `COUNT_ONLY` sources are renumbered by
    their converter, so set arithmetic on their names produces a gap of everything and means
    nothing.
    """
    out: dict[str, list[str]] = {}
    for source, units in manifest_units().items():
        directory = IR_DIR / source
        if not directory.is_dir() or source in COUNT_ONLY:
            continue
        prefix = NAME_PREFIX.get(source, "")
        present = {
            path.name.removesuffix(".ir.json").removeprefix(prefix)
            for path in directory.glob("*.ir.json")
        }
        gap = sorted(units - present)
        if gap:
            out[source] = gap
    return out


def reconcile() -> dict:
    manifest, ir, detect = manifest_counts(), ir_counts(), detect_counts()
    units, gaps = manifest_units(), unconverted()
    return {
        "manifest_rows": sum(manifest.values()),
        "ir_documents": sum(ir.values()),
        "detect_pages": sum(detect.values()),
        "by_source": {
            source: {
                "manifest": manifest.get(source, 0),
                "units": len(units.get(source, ())),
                "ir": ir.get(source, 0),
                "detect": detect.get(source, 0),
                "unconverted": gaps.get(source, []),
            }
            for source in sorted(set(manifest) | set(ir) | set(detect))
        },
        "no_detection": list(NO_DETECTION),
        "per_page": list(PER_PAGE),
    }


def problems(result: dict) -> list[str]:
    """Every difference the three rules above do not account for."""
    found: list[str] = []
    for source, row in result["by_source"].items():
        if source in COUNT_ONLY:
            # Renumbered by the converter: the only claim available is that none was lost.
            if row["ir"] != row["manifest"]:
                found.append(
                    f"{source}: {row['manifest']} manifest rows against {row['ir']} IR documents"
                )
        elif row["units"] != row["ir"] + len(row["unconverted"]):
            found.append(
                f"{source}: {row['units']} manifest units, {row['ir']} IR documents and "
                f"{len(row['unconverted'])} unconverted do not add up"
            )
        expected = 0 if source in NO_DETECTION else row["ir"]
        if row["detect"] != expected:
            reason = "no detection annotations" if source in NO_DETECTION else "one per document"
            found.append(
                f"{source}: {row['detect']} detect pages against {expected} expected ({reason})"
            )
    return found


def render(result: dict) -> str:
    lines = [
        "# Corpus reconciliation",
        "",
        "Generated by `python -m src.ingest reconcile`. Three artefacts describe the corpus and "
        "none of them agrees with the others; this is where the differences are accounted for.",
        "",
        "| Artefact | Rows | One row is |",
        "| :--- | ---: | :--- |",
        f"| `manifest.parquet` | {result['manifest_rows']} | one acquired item |",
        f"| `data/processed/ir/` | {result['ir_documents']} | one diagram |",
        f"| `detect/index.json` | {result['detect_pages']} | one annotated page |",
        "",
        "| Source | manifest | units | IR | unconverted | detect |",
        "| :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for source, row in result["by_source"].items():
        # `units` is meaningless for a renumbered source - showing 1,187 next to 1,319 documents
        # invites exactly the confusion this report exists to remove.
        units = "n/a" if source in COUNT_ONLY else str(row["units"])
        gap = "n/a" if source in COUNT_ONLY else str(len(row["unconverted"]))
        lines.append(
            f"| {source} | {row['manifest']} | {units} | {row['ir']} | {gap} | {row['detect']} |"
        )
    lines += [
        "",
        "**units** is what the IR converter treats as one document. For every source but "
        f"{', '.join(PER_PAGE)} that is the manifest row; for those the row is a *sketch* and "
        "the document is the *webpage* it depicts, several sketches per page.",
        "",
        f"**detect** is zero for {', '.join(NO_DETECTION)} by construction: 9.1.2's export "
        "covers hdbpmn, flowchartseg and fa_bresler, and nothing in this repo carries box "
        "annotations for anything else.",
        "",
        f"**{', '.join(COUNT_ONLY)}** is reconciled by count alone. Its converter assigns a "
        "fresh sequence (`flowchartseg/train/00000` becomes `fcseg_00000`) and its manifest ids "
        "repeat across splits, so matching names would produce a gap of everything and mean "
        "nothing. Row count equals document count, which is the claim available.",
        "",
    ]
    for source, gap in sorted(result["by_source"].items()):
        names = gap["unconverted"]
        if not names:
            continue
        lines += [
            f"## {source}: {len(names)} manifest entries with no IR document",
            "",
            "Named rather than counted, because a number is not something anyone can act on.",
            "",
            *[f"- `{name}`" for name in names],
            "",
        ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail on an unaccounted difference")
    args = parser.parse_args(argv)

    if not MANIFEST.is_file():
        print(f"no manifest at {MANIFEST}; run `python -m src.ingest manifest`", file=sys.stderr)
        return 2

    result = reconcile()
    found = problems(result)
    if args.check:
        for line in found:
            print(line, file=sys.stderr)
        print(json.dumps({k: v for k, v in result.items() if k != "by_source"}, indent=2))
        return 1 if found else 0

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(render(result), encoding="utf-8")
    print(f"wrote {REPORT.relative_to(ROOT)}")
    for line in found:
        print("unaccounted:", line, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
