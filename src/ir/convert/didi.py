"""Phase 2.2.2 - DIDI prompts to DreamScript IR.

    python -m src.ir convert.didi

**Kept deliberately against an absent payload.** `data/raw/didi.dvc` points at content that
`dvc status -c` calls "neither locally nor on remote", so this converter has nothing to convert
today and `configs/llm.yaml` no longer lists didi as a training source. The code is what would
read the payload if it came back, and `tests/test_assemble_serialise.py::ABSENT_SOURCES` fails
when it does, which is when its 3,000 files rejoin the corpus counts.

DIDI is 22,287 diagrams drawn by people who were shown a generated prompt and asked to copy
it. The prompt's graph is published as graphviz `.dot` and its laid-out form as `.xdot`, so the
structure behind every drawing is known exactly. What is *not* published is any mapping from
that layout to the pen strokes.

Recovering it is the interesting part of this converter. The drawing surface is given per
record as `writing_guide`, and the xdot layout has its own bounding box in points. Fitting the
second into the first with a single uniform scale, centred, reproduces where the ink actually
lies: on a sample of records the ink extent agrees with the fitted layout extent to within a
few percent on both axes. That is a *derived* geometry - `meta.geometry` says so - but it is a
derivation with evidence behind it, not a guess.

Two honest limits are recorded on every diagram this produces:

* the prompts carry **no diagram semantics**. Every node has a real text label - all 1,005 in
  a 300-record sample do, a mix of colour names and UI verbs ("dark brown", "Search", "Quit?")
  - and that text is genuine OCR ground truth. What the prompt never says is what a node
  *means*: a diamond labelled "Quit?" may or may not be a decision. Every role is therefore
  `unknown`, and a model trained on DIDI roles would be learning from shape alone.
* the ink is a *copy* of the prompt, so a stroke may disagree with the layout wherever the
  writer improvised. The IR describes the prompt, which is what the writer was aiming at.

    python -m src.ir.convert.didi --limit 200
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from src.ir.model import SUFFIX, Diagram, Edge, Node
from src.ir.vocab import canonical_shape
from src.utils.config import ROOT

SOURCE = "didi"
RAW = ROOT / "data" / "raw" / "didi"
NDJSON = RAW / "diagrams_20200131.ndjson"
OUT = ROOT / "data" / "processed" / "ir" / SOURCE

POINTS_PER_INCH = 72.0

BB_RE = re.compile(r'bb="([-0-9.,]+)"')
NODE_RE = re.compile(
    r"^\s*(?P<name>\"[^\"]+\"|\w+)\s*\[(?P<body>.*?)\];",
    re.MULTILINE | re.DOTALL,
)
EDGE_RE = re.compile(
    r"^\s*(?P<src>\"[^\"]+\"|\w+)\s*->\s*(?P<dst>\"[^\"]+\"|\w+)\s*\[(?P<body>.*?)\];",
    re.MULTILINE | re.DOTALL,
)
ATTR_RE = re.compile(r"(\w+)=(\"[^\"]*\"|[^,\s\]]+)")


def _attrs(body: str) -> dict[str, str]:
    return {k: v.strip('"') for k, v in ATTR_RE.findall(body)}


def _unquote(name: str) -> str:
    return name.strip().strip('"')


class Fit:
    """Uniform, centred fit of the graphviz page into the writing guide.

    graphviz measures from the bottom-left in points; the writing guide measures from the
    top-left in the same units the strokes use, so y is flipped as well as scaled.
    """

    def __init__(self, bb: tuple[float, float, float, float], guide: tuple[float, float]):
        x0, y0, x1, y1 = bb
        self.x0, self.y0 = x0, y0
        self.page_w = max(x1 - x0, 1e-6)
        self.page_h = max(y1 - y0, 1e-6)
        gw, gh = guide
        self.s = min(gw / self.page_w, gh / self.page_h)
        self.dx = (gw - self.page_w * self.s) / 2
        self.dy = (gh - self.page_h * self.s) / 2

    def __call__(self, x: float, y: float) -> tuple[float, float]:
        return (
            (x - self.x0) * self.s + self.dx,
            (self.page_h - (y - self.y0)) * self.s + self.dy,
        )

    def length(self, v: float) -> float:
        return v * self.s


def parse_xdot(text: str, guide: tuple[float, float]) -> tuple[list[Node], list[Edge]]:
    m = BB_RE.search(text)
    if not m:
        raise ValueError("xdot has no bounding box")
    x0, y0, x1, y1 = (float(v) for v in m.group(1).split(","))
    fit = Fit((x0, y0, x1, y1), guide)

    nodes: list[Node] = []
    seen: set[str] = set()
    for match in NODE_RE.finditer(text):
        name = _unquote(match.group("name"))
        attrs = _attrs(match.group("body"))
        if name in {"graph", "node", "edge"} or "pos" not in attrs or name in seen:
            continue
        seen.add(name)
        cx, cy = (float(v) for v in attrs["pos"].split(","))
        w = float(attrs.get("width", 0.75)) * POINTS_PER_INCH
        h = float(attrs.get("height", 0.5)) * POINTS_PER_INCH
        px, py = fit(cx, cy)
        pw, ph = fit.length(w), fit.length(h)
        nodes.append(
            Node(
                id=f"n{name}",
                shape=canonical_shape(attrs.get("shape", "box")),
                bbox=[px - pw / 2, py - ph / 2, pw, ph],
                text=attrs.get("label", "").strip(),
                semantic_role="unknown",  # DIDI prompts carry shape, never meaning
                confidence=1.0,
                source_id=name,
                attrs={"shape_basis": "prompt"},
            )
        )

    present = {n.id for n in nodes}
    edges: list[Edge] = []
    for i, match in enumerate(EDGE_RE.finditer(text)):
        src = f"n{_unquote(match.group('src'))}"
        dst = f"n{_unquote(match.group('dst'))}"
        attrs = _attrs(match.group("body"))
        polyline = None
        pos = attrs.get("pos", "")
        if pos:
            points = []
            for token in pos.replace("e,", " ").replace("s,", " ").split():
                parts = token.split(",")
                if len(parts) == 2:
                    try:
                        points.append(list(fit(float(parts[0]), float(parts[1]))))
                    except ValueError:
                        continue
            if len(points) >= 2:
                # graphviz writes the arrowhead point first and the spline after it.
                polyline = points[1:] + [points[0]]
        edges.append(
            Edge(
                id=f"e{i}",
                src=src if src in present else None,
                dst=dst if dst in present else None,
                directed=True,
                label=attrs.get("label", "").strip(),
                polyline=polyline,
                confidence=1.0,
                source_id=f"{src}->{dst}",
            )
        )
    return nodes, edges


def convert(record: dict) -> Diagram:
    label_id = record["label_id"]
    xdot = RAW / "prompts" / "xdot" / f"{label_id}.xdot"
    guide = (float(record["writing_guide"]["width"]), float(record["writing_guide"]["height"]))
    nodes, edges = parse_xdot(xdot.read_text(encoding="utf-8"), guide)

    diagram = Diagram(
        id=record["key"],
        diagram_type="flowchart",
        nodes=nodes,
        edges=edges,
        meta={
            "source": SOURCE,
            "geometry": "derived",
            "image": "",
            "image_size": [int(guide[0]), int(guide[1])],
            "split": record.get("split") if record.get("split") in {"train", "test"} else None,
            "producer": "src.ir.convert.didi",
            "didi_key": record["key"],
            "label_id": label_id,
            "notes": (
                "Structure is the generated prompt, exactly. Coordinates are the graphviz "
                "layout fitted uniformly and centred into the writing guide, which reproduces "
                "the ink extent to within a few percent. Text is the prompt's own label and "
                "is ground truth. Roles are 'unknown': the prompt never says what a node "
                "means, only what it is shaped like and what it reads."
            ),
        },
    )
    if diagram.meta["split"] is None:
        del diagram.meta["split"]
    diagram.sync_unresolved()
    return diagram


def records(limit: int | None = None):
    with NDJSON.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if limit is not None and i >= limit:
                return
            line = line.strip()
            if line:
                yield json.loads(line)


def ink_agreement(record: dict, diagram: Diagram) -> dict[str, float]:
    """How well the fitted layout predicts where the person actually drew.

    This is the evidence behind `geometry: derived`. It compares the bounding box of the ink
    with the bounding box of the fitted prompt, as a fraction of the writing guide.
    """
    xs = [c for stroke in record["drawing"] for c in stroke[0]]
    ys = [c for stroke in record["drawing"] for c in stroke[1]]
    if not xs or not ys:
        return {}
    boxes = [n.bbox for n in diagram.nodes if n.bbox]
    if not boxes:
        return {}
    lx = min(b[0] for b in boxes)
    ly = min(b[1] for b in boxes)
    hx = max(b[0] + b[2] for b in boxes)
    hy = max(b[1] + b[3] for b in boxes)
    gw, gh = diagram.meta["image_size"]
    return {
        "dx_lo": abs(min(xs) - lx) / gw,
        "dx_hi": abs(max(xs) - hx) / gw,
        "dy_lo": abs(min(ys) - ly) / gh,
        "dy_hi": abs(max(ys) - hy) / gh,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)

    written, failed = 0, []
    stats = {"nodes": 0, "edges": 0, "shapes": {}}
    agreement: list[float] = []
    for record in records(args.limit):
        try:
            d = convert(record)
        except Exception as exc:  # noqa: BLE001
            failed.append((record.get("key", "?"), str(exc)))
            continue
        d.save(args.out / f"{d.id}{SUFFIX}")
        written += 1
        stats["nodes"] += len(d.nodes)
        stats["edges"] += len(d.edges)
        for n in d.nodes:
            stats["shapes"][n.shape] = stats["shapes"].get(n.shape, 0) + 1
        agreement.extend(ink_agreement(record, d).values())

    if agreement:
        stats["ink_extent_error_mean"] = round(sum(agreement) / len(agreement), 4)
        stats["ink_extent_error_p90"] = round(sorted(agreement)[int(0.9 * len(agreement))], 4)
    print(f"wrote {written} IR files to {args.out.relative_to(ROOT)}")
    print(json.dumps(stats, indent=2, sort_keys=True))
    for key, err in failed[:10]:
        print(f"  FAIL  {key}: {err}", file=sys.stderr)
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
