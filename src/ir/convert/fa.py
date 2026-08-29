"""Phase 2.2.2 - the FA (finite automata) InkML database to DreamScript IR.

Bresler's FA database is the only source in the corpus with complete, human-made structural
ground truth for a diagram type other than flowcharts: 300 automata whose InkML records which
strokes form each state, which form each arrow, which arrow connects which two states, and
what every label reads. That makes it the source of the project's state-machine IR and, in
Phase 2.2.6, of state-machine target code that is correct rather than plausible.

The ink is in tablet units, so the converter maps it into the frame of the rasterised image
that `src/ingest/chaos_builder.py` produced for the chaos corpus - the *same* fit, because IR
coordinates that index a different image than the one on disk are worse than no coordinates.
`tests/test_ir_convert.py` pins the two transforms together.

    python -m src.ir.convert.fa --limit 20
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from src.ir.model import SUFFIX, Diagram, Edge, Node
from src.utils.config import ROOT

SOURCE = "fa_bresler"
RAW = ROOT / "data" / "raw" / "fa_bresler" / "FA_1.0"
OUT = ROOT / "data" / "processed" / "ir" / SOURCE

#: Must match `src.ingest.chaos_builder.render_inkml`, which rendered the images these
#: coordinates index. Guarded by a test rather than by a comment alone.
CANVAS = 1024
MARGIN = 0.06


def _traces(root: ET.Element) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for tr in root.iter():
        if tr.tag.rsplit("}", 1)[-1] != "trace":
            continue
        pts = []
        for token in (tr.text or "").strip().split(","):
            nums = []
            for part in token.replace("'", " ").replace('"', " ").split():
                try:
                    nums.append(float(part))
                except ValueError:
                    continue
            if len(nums) >= 2:
                pts.append((nums[0], nums[1]))
        if len(pts) >= 2:
            out[tr.get("id", str(len(out)))] = np.asarray(pts, dtype=np.float64)
    return out


def fit(all_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (scale, offset) mapping tablet units into the rasterised canvas.

    Each axis is normalised independently - the aspect ratio is not preserved. That is a
    property of the renderer that produced the corpus images, not a choice made here, and
    changing it would silently invalidate every image already on disk.
    """
    lo = all_points.min(axis=0)
    hi = all_points.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    scale = (1 - 2 * MARGIN) * CANVAS / span
    offset = MARGIN * CANVAS - lo * scale
    return scale, offset


def _bbox(points: np.ndarray) -> list[float]:
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    return [float(lo[0]), float(lo[1]), float(hi[0] - lo[0]), float(hi[1] - lo[1])]


def convert(path: Path, *, image: Path | None = None) -> Diagram:
    root = ET.parse(path).getroot()
    traces = _traces(root)
    if not traces:
        raise ValueError(f"{path.name}: no usable traces")

    scale, offset = fit(np.vstack(list(traces.values())))
    traces = {k: v * scale + offset for k, v in traces.items()}

    # -- symbols: id -> (truth, text, points) -----------------------------------------
    symbols: dict[str, dict] = {}
    for group in root.iter():
        if group.tag.rsplit("}", 1)[-1] != "traceGroup":
            continue
        gid = group.get("id")
        truth, meaning = "", ""
        refs: list[str] = []
        for child in group:
            tag = child.tag.rsplit("}", 1)[-1]
            if tag == "annotation" and child.get("type") == "truth":
                truth = (child.text or "").strip()
            elif tag == "annotation" and child.get("type") == "textMeaning":
                meaning = (child.text or "").strip()
            elif tag == "traceView":
                refs.append(child.get("traceDataRef", ""))
        pts = [traces[r] for r in refs if r in traces]
        if gid is None or not pts:
            continue
        symbols[gid] = {"truth": truth, "text": meaning, "points": np.vstack(pts)}

    # -- relations ---------------------------------------------------------------------
    connections: list[tuple[str, str, str]] = []  # (arrow, from, to)
    arrow_in: list[tuple[str, str]] = []  # (arrow, state) - the initial-state marker
    arrow_labels: dict[str, list[str]] = {}
    text_inside: dict[str, list[str]] = {}
    for group in root.iter():
        if group.tag.rsplit("}", 1)[-1] != "symbolGroup":
            continue
        truth = ""
        refs = []
        for child in group:
            tag = child.tag.rsplit("}", 1)[-1]
            if tag == "annotation" and child.get("type") == "truth":
                truth = (child.text or "").strip()
            elif tag == "symbolView":
                refs.append(child.get("symbolDataRef", ""))
        refs = [r for r in refs if r in symbols]
        kind = lambda r: symbols[r]["truth"]  # noqa: E731 - a local lookup, not an API

        if truth == "arrow_connection" and len(refs) == 3:
            connections.append((refs[0], refs[1], refs[2]))
        elif truth == "arrow_in" and len(refs) == 2:
            arrow = next((r for r in refs if kind(r) == "arrow"), None)
            state = next((r for r in refs if kind(r) != "arrow"), None)
            if arrow and state:
                arrow_in.append((arrow, state))
        elif truth == "arrow_label" and len(refs) == 2:
            arrow = next((r for r in refs if kind(r) == "arrow"), None)
            label = next((r for r in refs if kind(r) == "label"), None)
            if arrow and label:
                arrow_labels.setdefault(arrow, []).append(label)
        elif truth == "text_inside" and len(refs) == 2:
            label = next((r for r in refs if kind(r) == "label"), None)
            state = next((r for r in refs if kind(r) != "label"), None)
            if label and state:
                text_inside.setdefault(state, []).append(label)

    initial = {state for _, state in arrow_in}

    # -- nodes -------------------------------------------------------------------------
    nodes: list[Node] = []
    for sid, sym in symbols.items():
        if sym["truth"] not in {"state", "final state"}:
            continue
        final = sym["truth"] == "final state"
        text = " ".join(symbols[t]["text"] for t in text_inside.get(sid, []) if symbols[t]["text"])
        role = "final-state" if final else ("initial-state" if sid in initial else "state")
        nodes.append(
            Node(
                id=f"s{sid}",
                shape="double-circle" if final else "circle",
                bbox=_bbox(sym["points"]),
                text=text.strip(),
                semantic_role=role,
                confidence=1.0,
                source_id=sid,
                attrs={"accepting": final, "initial": sid in initial, "shape_basis": "annotated"},
            )
        )
    present = {n.id for n in nodes}

    # -- edges -------------------------------------------------------------------------
    edges: list[Edge] = []
    for arrow, src, dst in connections:
        label = " ".join(
            symbols[t]["text"] for t in arrow_labels.get(arrow, []) if symbols[t]["text"]
        )
        pts = symbols[arrow]["points"]
        edges.append(
            Edge(
                id=f"e{arrow}",
                src=f"s{src}" if f"s{src}" in present else None,
                dst=f"s{dst}" if f"s{dst}" in present else None,
                directed=True,
                label=label.strip(),
                polyline=[[float(x), float(y)] for x, y in pts],
                confidence=1.0,
                source_id=arrow,
                attrs={"self_loop": src == dst},
            )
        )
    # The "arrow from nowhere" that marks the start state is a real drawn stroke, so it is a
    # real edge with no source - exactly the case `unresolved_edges` exists for.
    for arrow, state in arrow_in:
        pts = symbols[arrow]["points"]
        edges.append(
            Edge(
                id=f"e{arrow}",
                src=None,
                dst=f"s{state}" if f"s{state}" in present else None,
                directed=True,
                label="",
                polyline=[[float(x), float(y)] for x, y in pts],
                confidence=1.0,
                source_id=arrow,
                attrs={"initial_marker": True},
            )
        )

    writer = path.stem.split("_")[0]
    image = image or (ROOT / "data" / "raw" / "chaos" / "state_machine")
    diagram = Diagram(
        id=path.stem,
        diagram_type="state_machine",
        nodes=nodes,
        edges=edges,
        meta={
            "source": SOURCE,
            "geometry": "derived",
            "image": "",
            "image_size": [CANVAS, CANVAS],
            "scribe_id": f"fa-{writer}",
            "producer": "src.ir.convert.fa",
            "notes": (
                "Structure and text are ground truth from the InkML annotation. Coordinates "
                "are the ink mapped into the 1024px canvas used by chaos_builder.render_inkml, "
                "which normalises each axis independently and therefore does not preserve "
                "aspect ratio."
            ),
        },
    )
    for edge in edges:
        if edge.attrs and edge.attrs.get("initial_marker"):
            diagram.record_unresolved(edge, note="start marker: an arrow with no source state")
    diagram.sync_unresolved()
    return diagram


def files() -> list[Path]:
    return sorted(RAW.glob("*.inkml"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)

    paths = files()[: args.limit]
    if not paths:
        print(f"no InkML under {RAW}", file=sys.stderr)
        return 1

    stats = {"nodes": 0, "edges": 0, "accepting": 0, "initial": 0, "labelled_edges": 0}
    written, failed = 0, []
    for path in paths:
        try:
            d = convert(path)
        except Exception as exc:  # noqa: BLE001
            failed.append((path.name, str(exc)))
            continue
        d.save(args.out / f"{d.id}{SUFFIX}")
        written += 1
        stats["nodes"] += len(d.nodes)
        stats["edges"] += len(d.edges)
        stats["accepting"] += sum(n.semantic_role == "final-state" for n in d.nodes)
        stats["initial"] += sum(n.semantic_role == "initial-state" for n in d.nodes)
        stats["labelled_edges"] += sum(bool(e.label) for e in d.edges)

    print(f"wrote {written} IR files to {args.out.relative_to(ROOT)}")
    print(json.dumps(stats, indent=2, sort_keys=True))
    for name, err in failed[:10]:
        print(f"  FAIL  {name}: {err}", file=sys.stderr)
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
