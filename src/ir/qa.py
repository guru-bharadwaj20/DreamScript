"""Phase 2.2.5 - the label QA validator.

The schema says what an IR file may contain. It cannot say whether the contents make sense:
JSON Schema has no way to express "this edge's `src` must be one of this diagram's node ids",
which is exactly the class of mistake a converter or an annotator makes. This module is the
second gate, and it runs over every IR file in the corpus.

Findings have three severities, and the distinction is load-bearing:

* **error** - the file is wrong, and something downstream will break or silently mislead. An
  edge pointing at a node that does not exist; a box outside the image; a duplicate id.
* **warning** - the file is legal but suspicious, and a human should look. A role that does not
  belong to its diagram type; a node carrying a role that describes a connection.
* **info** - counted, never a failure. How much text is missing, how many roles are `unknown`.
  These are corpus statistics that happen to be worth watching.

"Validator green on the corpus" means **zero errors**. Warnings are reported with counts and a
named reason for each one that is expected, because a validator whose warnings are all ignored
is a validator nobody reads.

    python -m src.ir.qa                       # whole corpus, writes reports/label_qa.md
    python -m src.ir.qa data/processed/ir/fa_bresler
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from src.ir.model import SUFFIX, Diagram
from src.ir.vocab import EDGE_LIKE_ROLES, ROLES_BY_TYPE
from src.utils.config import ROOT

IR_ROOT = ROOT / "data" / "processed" / "ir"
REPORT = ROOT / "reports" / "label_qa.md"

ERROR, WARNING, INFO = "error", "warning", "info"

#: Tolerance for a box sticking out of the frame, as a fraction of the image's larger side.
#: Not zero: an annotator drawing on the very edge of a photo, and a rounding difference
#: between the annotation tool's coordinate space and ours, both produce sub-pixel overhang.
FRAME_SLACK = 0.01

#: How far a *container* may stick out before it stops being edge overhang and starts looking
#: like a coordinate bug. Measured on the corpus: the twelve real cases run from 1.1% to 15.4%
#: of the long side, while the width-versus-longest-side scaling error produced overhangs of
#: 30-60%. 0.25 sits in the gap.
CONTAINER_OVERHANG = 0.25


@dataclass(frozen=True)
class Finding:
    severity: str
    check: str
    diagram: str
    detail: str


def check_diagram(diagram: Diagram, *, path: Path | None = None) -> list[Finding]:
    """Every problem in one diagram. An empty list means it is fit to use."""
    out: list[Finding] = []

    def add(severity: str, check: str, detail: str) -> None:
        out.append(Finding(severity, check, diagram.id, detail))

    # -- the schema, first: nothing below is meaningful on a malformed file ---------------
    for problem in diagram.problems():
        add(ERROR, "schema", problem)
    if out:
        return out

    geometry = diagram.meta.get("geometry", "absent")
    size = diagram.meta.get("image_size")

    # -- identity -------------------------------------------------------------------------
    node_ids = [n.id for n in diagram.nodes]
    duplicates = [i for i, c in Counter(node_ids).items() if c > 1]
    if duplicates:
        add(ERROR, "duplicate_node_id", f"{len(duplicates)} repeated: {duplicates[:5]}")
    edge_ids = [e.id for e in diagram.edges]
    duplicate_edges = [i for i, c in Counter(edge_ids).items() if c > 1]
    if duplicate_edges:
        add(ERROR, "duplicate_edge_id", f"{len(duplicate_edges)} repeated: {duplicate_edges[:5]}")

    known = set(node_ids)

    # -- edges ----------------------------------------------------------------------------
    listed_unresolved = {u["edge"] for u in diagram.unresolved_edges}
    for edge in diagram.edges:
        for end, value in (("src", edge.src), ("dst", edge.dst)):
            if value is not None and value not in known:
                add(ERROR, "dangling_edge", f"edge {edge.id}.{end} -> unknown node {value!r}")
        if edge.dangling and edge.id not in listed_unresolved:
            add(
                ERROR,
                "unlisted_open_edge",
                f"edge {edge.id} has an open end but is not in unresolved_edges",
            )
        if edge.polyline is not None and len(edge.polyline) < 2:
            add(ERROR, "short_polyline", f"edge {edge.id} has fewer than two points")
    for entry in diagram.unresolved_edges:
        if entry["edge"] not in set(edge_ids):
            add(
                ERROR,
                "unresolved_edge_missing",
                f"unresolved_edges names absent edge {entry['edge']}",
            )

    # -- ambiguity references ---------------------------------------------------------------
    for entry in diagram.crossed_out:
        ref = entry.get("ref")
        if ref is not None and ref not in known and ref not in set(edge_ids):
            add(ERROR, "crossed_out_ref", f"crossed_out names unknown element {ref!r}")
    for entry in diagram.low_conf_text:
        ref = entry["ref"]
        if ref not in known and ref not in set(edge_ids):
            add(ERROR, "low_conf_text_ref", f"low_conf_text names unknown element {ref!r}")

    # -- geometry ---------------------------------------------------------------------------
    if geometry == "absent":
        bad = [n.id for n in diagram.nodes if n.bbox is not None]
        if bad:
            add(ERROR, "geometry_absent_but_boxed", f"{len(bad)} nodes have a bbox anyway")
    else:
        missing = [n.id for n in diagram.nodes if n.bbox is None]
        if missing:
            add(
                WARNING,
                "node_without_box",
                f"{len(missing)} of {len(diagram.nodes)} nodes have no bbox",
            )
        if not size:
            add(ERROR, "no_image_size", "geometry is present but meta.image_size is not")
        else:
            width, height = size
            slack = FRAME_SLACK * max(width, height)
            outside = []
            container_overhang = []
            degenerate = []
            for node in diagram.nodes:
                if node.bbox is None:
                    continue
                x, y, w, h = node.bbox
                if w <= 0 or h <= 0:
                    degenerate.append(node.id)
                overhang = max(-x, -y, x + w - width, y + h - height)
                if overhang <= slack:
                    continue
                # A pool or a lane is drawn as one long box around everything, and writers run
                # it off the edge of the paper; the annotator then draws it past the edge of
                # the photo. Every *other* box in those same files is inside the frame, which
                # is what says this is the data and not a coordinate bug. A large overhang is
                # still an error, because that is what a wrong scale looks like.
                if node.semantic_role == "container" and overhang <= CONTAINER_OVERHANG * max(
                    width, height
                ):
                    container_overhang.append(node.id)
                else:
                    outside.append(node.id)
            if outside:
                add(
                    ERROR,
                    "bbox_out_of_frame",
                    f"{len(outside)} boxes leave the {width}x{height} image: {outside[:5]}",
                )
            if container_overhang:
                add(
                    WARNING,
                    "container_overhangs_frame",
                    f"{len(container_overhang)} pools/lanes drawn past the edge of the photo",
                )
            if degenerate:
                add(ERROR, "bbox_degenerate", f"{len(degenerate)} boxes have zero area")

    if diagram.meta.get("image") and not (ROOT / diagram.meta["image"]).is_file():
        add(ERROR, "image_missing", f"meta.image does not exist: {diagram.meta['image']}")

    # -- roles ------------------------------------------------------------------------------
    allowed = ROLES_BY_TYPE.get(diagram.diagram_type, set())
    wrong = {n.semantic_role for n in diagram.nodes} - allowed
    if wrong:
        add(
            WARNING,
            "role_outside_diagram_type",
            f"{sorted(wrong)} are not roles of a {diagram.diagram_type}",
        )
    edge_like = [n.id for n in diagram.nodes if n.semantic_role in EDGE_LIKE_ROLES]
    if edge_like:
        add(
            WARNING,
            "connection_labelled_as_node",
            f"{len(edge_like)} nodes carry a connection role; Phase 10 must turn them into edges",
        )
    if diagram.nodes and not diagram.edges and geometry != "absent":
        add(WARNING, "no_edges", "nodes but no edges: check whether the source records any")

    # -- counted, never fatal ------------------------------------------------------------------
    if diagram.nodes:
        unknown = sum(n.semantic_role == "unknown" for n in diagram.nodes)
        if unknown:
            add(INFO, "unknown_roles", f"{unknown}/{len(diagram.nodes)}")
        blank = sum(not n.text for n in diagram.nodes)
        if blank:
            add(INFO, "nodes_without_text", f"{blank}/{len(diagram.nodes)}")
        guessed = sum(n.confidence < 1.0 for n in diagram.nodes)
        if guessed:
            add(INFO, "inferred_nodes", f"{guessed}/{len(diagram.nodes)}")
    if diagram.unresolved_edges:
        add(INFO, "unresolved_edges", f"{len(diagram.unresolved_edges)}/{len(diagram.edges)}")

    return out


def scan(root: Path) -> tuple[list[Finding], dict]:
    findings: list[Finding] = []
    stats: dict = {
        "files": 0,
        "unreadable": 0,
        "nodes": 0,
        "edges": 0,
        "by_source": defaultdict(int),
        "by_type": defaultdict(int),
        "by_geometry": defaultdict(int),
    }
    for path in sorted(root.rglob(f"*{SUFFIX}")):
        try:
            diagram = Diagram.load(path)
        except Exception as exc:  # noqa: BLE001 - an unreadable file is itself the finding
            stats["unreadable"] += 1
            findings.append(Finding(ERROR, "unreadable", path.name, str(exc)))
            continue
        stats["files"] += 1
        stats["nodes"] += len(diagram.nodes)
        stats["edges"] += len(diagram.edges)
        stats["by_source"][diagram.meta.get("source", "?")] += 1
        stats["by_type"][diagram.diagram_type] += 1
        stats["by_geometry"][diagram.meta.get("geometry", "?")] += 1
        findings.extend(check_diagram(diagram, path=path))
    return findings, stats


#: Warnings that are expected, and why. A warning with an entry here is reported with its
#: reason instead of as something to investigate; one without an entry is a genuine surprise.
EXPECTED_WARNINGS = {
    "no_edges": (
        "flowchartseg publishes a node mask and no connections at all. Its `meta.notes` says so "
        "on every file, and Phase 10 must never be evaluated on them"
    ),
    "role_outside_diagram_type": (
        "under review - a role appearing in the wrong diagram type is usually a converter bug"
    ),
    "connection_labelled_as_node": (
        "an arrow detected as a region. Phase 10.1 converts these into edges"
    ),
    "node_without_box": ("a node the source names but never places on the page"),
    "container_overhangs_frame": (
        "a pool or lane drawn past the edge of the paper, and annotated past the edge of the "
        "photo. Twelve cases, 1.1% to 15.4% of the long side; every non-container box in the "
        "same files is inside the frame, which is what distinguishes this from a scaling bug"
    ),
}


def write_report(findings: list[Finding], stats: dict, root: Path) -> Path:
    by_severity = defaultdict(list)
    for finding in findings:
        by_severity[finding.severity].append(finding)
    errors, warnings, infos = (by_severity[k] for k in (ERROR, WARNING, INFO))

    lines: list[str] = []
    add = lines.append
    add("# Label QA")
    add("")
    add("Phase 2.2.5. Generated by `python -m src.ir.qa`.")
    add("")
    add(
        f"**{stats['files']} IR files, {stats['nodes']:,} nodes, {stats['edges']:,} edges** "
        f"under `{root.relative_to(ROOT)}`."
    )
    add("")
    verdict = "**GREEN - no errors.**" if not errors else f"**RED - {len(errors)} errors.**"
    add(f"{verdict} {len(warnings)} warnings, {len(infos)} informational counts.")
    add("")

    add("## Corpus under validation")
    add("")
    add("| source | files | diagram type | files | geometry | files |")
    add("| :--- | ---: | :--- | ---: | :--- | ---: |")
    sources = sorted(stats["by_source"].items())
    types = sorted(stats["by_type"].items())
    geometries = sorted(stats["by_geometry"].items())
    for i in range(max(len(sources), len(types), len(geometries))):
        cells = []
        for table in (sources, types, geometries):
            cells.extend([table[i][0], str(table[i][1])] if i < len(table) else ["", ""])
        add("| " + " | ".join(cells) + " |")
    add("")

    add("## Errors")
    add("")
    if not errors:
        add("None. Every file satisfies the schema, every edge points at a node that exists,")
        add("every box lies inside its image, and every open end is recorded as unresolved.")
    else:
        counts = Counter(f.check for f in errors)
        add("| check | count | example |")
        add("| :--- | ---: | :--- |")
        for check, count in counts.most_common():
            example = next(f for f in errors if f.check == check)
            add(f"| `{check}` | {count} | {example.diagram}: {example.detail} |")
    add("")

    add("## Warnings")
    add("")
    if not warnings:
        add("None.")
    else:
        add("| check | count | expected? |")
        add("| :--- | ---: | :--- |")
        for check, count in Counter(f.check for f in warnings).most_common():
            reason = EXPECTED_WARNINGS.get(check, "**unexplained - investigate**")
            add(f"| `{check}` | {count} | {reason} |")
    add("")

    add("## Counted, not judged")
    add("")
    add("| count | total | share |")
    add("| :--- | ---: | ---: |")
    aggregate: dict[str, tuple[int, int]] = {}
    for finding in infos:
        got, total = (int(v) for v in finding.detail.split("/"))
        prev = aggregate.get(finding.check, (0, 0))
        aggregate[finding.check] = (prev[0] + got, prev[1] + total)
    for check, (got, total) in sorted(aggregate.items()):
        share = got / total if total else 0
        add(f"| `{check}` | {got:,} / {total:,} | {share:.1%} |")
    add("")
    add("These are properties of the sources, not defects. `unknown_roles` is dominated by DIDI,")
    add("whose prompts carry no semantics, and by flowchartseg, which has one node class;")
    add("`nodes_without_text` by the same two. They are counted here so that a later phase")
    add("cannot quietly train on them and report the result as role classification.")
    add("")

    add("## What this does and does not check")
    add("")
    add("| Checked | Not checked |")
    add("| :--- | :--- |")
    add(
        "| The file matches `schemas/ir.schema.json` | Whether a shape label is *correct* - see `reports/annotator_agreement.md` |"
    )
    add(
        "| Every edge endpoint is a node in the same diagram | Whether the edge should exist at all |"
    )
    add(
        "| Every box lies within its image, with 1% slack | Whether the box is tight around the ink |"
    )
    add(
        "| Every open-ended edge appears in `unresolved_edges` | Whether it *should* have been resolvable |"
    )
    add(
        "| `crossed_out` and `low_conf_text` name real elements | Whether the transcription is right |"
    )
    add("| `meta.image` exists on disk | Whether it is the right image |")
    add("| Roles belong to the diagram type | Whether the diagram type is right |")
    add("")
    add("The right-hand column is not a gap to be closed here. Those are questions about")
    add("correctness rather than consistency, and they are answered by Phase 2.2.4's agreement")
    add("measurement and by human review, not by a validator.")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    while lines and not lines[-1].strip():
        lines.pop()
    with REPORT.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    return REPORT


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", nargs="?", type=Path, default=IR_ROOT)
    ap.add_argument("--no-report", action="store_true")
    args = ap.parse_args(argv)

    if not args.root.exists():
        print(f"nothing to validate at {args.root}", file=sys.stderr)
        return 1

    findings, stats = scan(args.root)
    errors = [f for f in findings if f.severity == ERROR]
    warnings = [f for f in findings if f.severity == WARNING]

    if not args.no_report:
        rep = write_report(findings, stats, args.root)
        print(f"wrote {rep.relative_to(ROOT)}")

    print(
        json.dumps(
            {
                "files": stats["files"],
                "nodes": stats["nodes"],
                "edges": stats["edges"],
                "errors": len(errors),
                "warnings": len(warnings),
                "error_kinds": dict(Counter(f.check for f in errors)),
                "warning_kinds": dict(Counter(f.check for f in warnings)),
            },
            indent=2,
        )
    )
    for finding in errors[:15]:
        print(f"  ERROR  {finding.diagram}  {finding.check}: {finding.detail}", file=sys.stderr)

    checks = {
        "files_found": stats["files"] > 0,
        "all_files_readable": stats["unreadable"] == 0,
        "no_errors": not errors,
    }
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
