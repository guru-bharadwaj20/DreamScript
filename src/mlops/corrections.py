"""Phase 15.7 - the correction store: a user's fix, kept as labeled data.

    python -m src.mlops.corrections --value              # what each kind of correction is worth
    python -m src.mlops.corrections --record <json>      # append one correction
    python -m src.mlops.corrections --export <parquet>   # corrections as training rows
    python -m src.mlops.corrections --stats              # what the store holds

16.2.8 is the screen that produces these and it is Phase 16, not built. What this row owes is the
store behind it, and one thing worth more than the plumbing: **evidence about which correction is
worth collecting**, so the app screen is designed around the answer rather than guessing.

## Which correction matters, measured rather than assumed

From S5's own edit mass over the 162 held-out test pages (`reports/s5_test_large.json`):

    sub_text      1,275 edits   50.7%   <- half of every edit the assembler gets wrong
    edge_insert     604          24.0%
    edge_delete     395          15.7%
    node_insert     105           4.2%
    node_delete      70           2.8%
    sub_shape        12           0.5%

And the ladder in the same report says what fixing each would buy: median GED **13.0 as
assembled, 4.0 with perfect text, 9.0 with perfect edges, 0.0 with both**. So text corrections
alone take the median from 13 to 4 - a 69% reduction, and within sight of S5's `<= 3` bar - while
every other correction type put together buys 4.

**That is the design constraint for the app screen**: a correction UI that only lets someone
retype a label captures half the error mass, and one that makes them redraw edges captures a
quarter for far more effort per page. It is also why this store's schema treats `text` as the
first-class case and keeps the rest general.

## What the store is

Append-only JSONL at `data/interim/corrections.jsonl`. Append-only because a correction is an
observation with a timestamp, not a row to be updated - a user correcting their own correction is
two records, and which came second is exactly what training wants to know. JSONL because the file
stays readable and greppable at the sizes this will ever reach, and because a partial write costs
one line rather than the store.

Every record is validated against the IR it claims to correct: an unknown page, an unknown node,
or a correction that does not change anything is rejected at the door rather than exported into
training data later.

**No real user corrections exist yet**, because the screen that makes them is Phase 16. The store
is demonstrated round-trip and `--stats` reports an empty store honestly rather than seeding it
with invented ones.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.config import ROOT

STORE = ROOT / "data" / "interim" / "corrections.jsonl"
IR_ROOT = ROOT / "data" / "processed" / "ir"
REPORT_MD = ROOT / "reports" / "corrections.md"
REPORT_JSON = ROOT / "reports" / "corrections.json"
S5 = ROOT / "reports" / "s5_test_large.json"

#: What a correction may change. `text` first because it is half the edit mass; the rest are
#: general so the store does not have to change when the app grows a screen.
FIELDS = ("text", "shape", "semantic_role", "edge_src", "edge_dst", "edge_label")

#: A correction targets a node or an edge, and nothing else - a page-level "this is wrong" is a
#: complaint rather than a label, and cannot be trained on.
TARGETS = ("node", "edge")


class Rejected(Exception):
    """The correction does not describe a change that could be made to the IR it names."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ir_path(diagram_id: str) -> Path:
    """`<source>/<stem>` -> the IR document on disk."""
    if "/" in diagram_id:
        source, stem = diagram_id.split("/", 1)
        return IR_ROOT / source / f"{stem}.ir.json"
    return IR_ROOT / f"{diagram_id}.ir.json"


def load_ir(diagram_id: str) -> dict:
    path = ir_path(diagram_id)
    if not path.is_file():
        raise Rejected(f"no IR for {diagram_id!r} at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate(record: dict, diagram: dict | None = None) -> dict:
    """Reject anything that could not be applied, before it can reach training data."""
    for key in ("diagram_id", "target", "target_id", "field", "new_value"):
        if key not in record:
            raise Rejected(f"missing {key!r}")
    if record["target"] not in TARGETS:
        raise Rejected(f"target must be one of {TARGETS}, got {record['target']!r}")
    if record["field"] not in FIELDS:
        raise Rejected(f"field must be one of {FIELDS}, got {record['field']!r}")

    diagram = load_ir(record["diagram_id"]) if diagram is None else diagram
    collection = diagram.get("nodes" if record["target"] == "node" else "edges") or []
    found = next((item for item in collection if item.get("id") == record["target_id"]), None)
    if found is None:
        raise Rejected(
            f"{record['target']} {record['target_id']!r} is not in {record['diagram_id']!r}"
        )

    old = found.get(record["field"].replace("edge_", ""))
    if str(old) == str(record["new_value"]):
        # A "correction" that changes nothing is the user confirming, not correcting. It carries
        # no gradient and would dilute any set it was exported into.
        raise Rejected("new_value equals the current value; nothing was corrected")

    return {
        "diagram_id": record["diagram_id"],
        "target": record["target"],
        "target_id": record["target_id"],
        "field": record["field"],
        "old_value": old,
        "new_value": record["new_value"],
        # The IR the user was looking at. If the pipeline re-runs and the IR changes, a correction
        # made against the old one may no longer apply, and this is what lets `export` tell.
        "ir_sha256": hashlib.sha256(
            json.dumps(diagram, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16],
        "source": record.get("source") or record["diagram_id"].split("/")[0],
        "recorded_at": record.get("recorded_at") or _now(),
        "client": record.get("client", "unknown"),
    }


def record(entry: dict, store: Path = STORE) -> dict:
    """Validate and append. Returns the stored record."""
    stored = validate(entry)
    store.parent.mkdir(parents=True, exist_ok=True)
    with store.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(stored, ensure_ascii=False) + "\n")
    return stored


def read(store: Path = STORE) -> list[dict]:
    if not store.is_file():
        return []
    rows = []
    for line in store.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def latest(rows: list[dict]) -> list[dict]:
    """One record per (diagram, target, field): the last correction wins.

    Append-only keeps the history, but training wants the user's final answer. `recorded_at` is
    the ordering and ties fall back to file order, which is arrival order.
    """
    newest: dict[tuple, dict] = {}
    for index, row in enumerate(rows):
        key = (row["diagram_id"], row["target"], row["target_id"], row["field"])
        current = newest.get(key)
        if current is None or (row.get("recorded_at"), index) >= (
            current.get("recorded_at"),
            current["_index"],
        ):
            newest[key] = {**row, "_index": index}
    return [{k: v for k, v in row.items() if k != "_index"} for row in newest.values()]


def stale(rows: list[dict]) -> list[dict]:
    """Corrections made against an IR that has since changed; they may no longer apply."""
    out = []
    for row in rows:
        try:
            diagram = load_ir(row["diagram_id"])
        except Rejected:
            out.append({**row, "why": "the IR is gone"})
            continue
        digest = hashlib.sha256(json.dumps(diagram, sort_keys=True).encode("utf-8")).hexdigest()[
            :16
        ]
        if digest != row.get("ir_sha256"):
            out.append({**row, "why": "the IR changed after the correction was made"})
    return out


def export(rows: list[dict] | None = None, out: Path | None = None):
    """Corrections as training rows: one per final correction, stale ones excluded."""
    import pandas as pd

    rows = read() if rows is None else rows
    final = latest(rows)
    bad = {
        (r["diagram_id"], r["target"], r["target_id"], r["field"]) for r in stale(final)
    }
    usable = [
        r
        for r in final
        if (r["diagram_id"], r["target"], r["target_id"], r["field"]) not in bad
    ]
    frame = pd.DataFrame(
        usable,
        columns=[
            "diagram_id",
            "source",
            "target",
            "target_id",
            "field",
            "old_value",
            "new_value",
            "recorded_at",
            "client",
            "ir_sha256",
        ],
    )
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(out)
    return frame


def value() -> dict[str, Any]:
    """What each correction kind is worth, from S5's measured edit mass rather than intuition."""
    if not S5.is_file():
        return {"available": False, "why": f"no {S5.relative_to(ROOT)}"}
    payload = json.loads(S5.read_text(encoding="utf-8"))
    overall = payload.get("overall", {})
    ladder = overall.get("ladder", {})
    shares = overall.get("edit_share", {})
    mass = overall.get("edit_mass", {})
    base = ladder.get("as_assembled")
    return {
        "available": True,
        "pages": overall.get("pages"),
        "edit_mass": mass,
        "edit_share": shares,
        "ladder": ladder,
        "text_share": shares.get("sub_text"),
        "median_ged_as_assembled": base,
        "median_ged_with_perfect_text": ladder.get("perfect_text"),
        "median_ged_with_perfect_edges": ladder.get("perfect_edges"),
        "reading": (
            "text corrections are half the edit mass and take the median GED from"
            f" {base} to {ladder.get('perfect_text')}, while every other correction type together"
            f" takes it to {ladder.get('perfect_edges')} - so a screen that only lets a user"
            " retype a label captures the larger half for the smaller effort"
        ),
    }


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def stats(store: Path = STORE) -> dict[str, Any]:
    import collections

    rows = read(store)
    final = latest(rows)
    return {
        # A store outside the repo is legitimate - a test's tmp_path, or a deployment writing
        # somewhere else - so this must not assume it sits under ROOT.
        "store": _relative(store),
        "exists": store.is_file(),
        "records": len(rows),
        "after_last_wins": len(final),
        "stale": len(stale(final)),
        "by_field": dict(collections.Counter(r["field"] for r in final)),
        "by_source": dict(collections.Counter(r.get("source") for r in final)),
        "note": (
            "no real user corrections exist yet: 16.2.8, the screen that produces them, is Phase"
            " 16 and is not built. An empty store is reported as empty rather than seeded."
        )
        if not rows
        else None,
    }


def render(result: dict) -> str:
    worth, holdings = result["value"], result["stats"]
    lines = [
        "# Phase 15.7 - the correction store",
        "",
        "Generated by `python -m src.mlops.corrections`.",
        "",
        "Append-only JSONL at `data/interim/corrections.jsonl`. Append-only because a correction"
        " is an observation with a timestamp rather than a row to update - someone correcting"
        " their own correction is two records, and which came second is what training wants.",
        "",
        "Every record is validated against the IR it names: an unknown page, an unknown node, or"
        " a change that changes nothing is rejected at the door rather than discovered later"
        " inside a training set.",
        "",
        "## Which correction is worth collecting",
        "",
    ]
    if worth.get("available"):
        lines += [
            f"From S5's own edit mass over {worth['pages']} held-out pages:",
            "",
            "| edit | count | share |",
            "| :--- | ---: | ---: |",
        ]
        for kind, count in sorted(
            worth["edit_mass"].items(), key=lambda kv: -kv[1]
        ):
            lines.append(f"| `{kind}` | {count} | {worth['edit_share'].get(kind, 0):.1%} |")
        lines += [
            "",
            f"And what fixing each buys, from the same report's ladder: median GED"
            f" **{worth['median_ged_as_assembled']} as assembled**,"
            f" **{worth['median_ged_with_perfect_text']} with perfect text**,"
            f" **{worth['median_ged_with_perfect_edges']} with perfect edges**,"
            f" **{worth['ladder'].get('both_perfect')} with both**.",
            "",
            f"**{worth['reading'].capitalize()}.** That is the design constraint for 16.2.8"
            " rather than a preference: the highest-value screen is the one that lets someone"
            " retype a label, and it is worth building before any edge-editing affordance.",
            "",
        ]
    else:
        lines += [f"Not available: {worth.get('why')}.", ""]

    lines += [
        "## What the store holds",
        "",
        "| | |",
        "| :--- | ---: |",
        f"| records | {holdings['records']} |",
        f"| after last-wins | {holdings['after_last_wins']} |",
        f"| stale (IR changed since) | {holdings['stale']} |",
        "",
    ]
    if holdings.get("note"):
        lines += [f"**{holdings['note']}**", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 15.7 correction store")
    ap.add_argument("--record", help="a JSON file or literal holding one correction")
    ap.add_argument("--export", type=Path, help="write the usable corrections to this parquet")
    ap.add_argument("--stats", action="store_true", help="what the store holds")
    ap.add_argument("--value", action="store_true", help="what each correction kind is worth")
    args = ap.parse_args(argv)

    if args.record:
        payload = Path(args.record)
        entry = json.loads(payload.read_text(encoding="utf-8") if payload.is_file() else args.record)
        try:
            stored = record(entry)
        except Rejected as exc:
            print(f"rejected: {exc}", file=sys.stderr)
            return 1
        json.dump(stored, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    if args.export:
        frame = export(out=args.export)
        print(f"wrote {len(frame)} corrections to {args.export}")
        return 0

    if args.value:
        json.dump(value(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    result = {"value": value(), "stats": stats()}
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    REPORT_MD.write_text(render(result), encoding="utf-8")
    json.dump(result["stats"], sys.stdout, indent=2)
    sys.stdout.write("\n")
    print(f"-> {REPORT_MD}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
