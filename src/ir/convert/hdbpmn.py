"""Phase 2.2.2 - hdBPMN annotations to DreamScript IR.

hdBPMN is the richest source in the corpus: 704 photographed process diagrams, each paired
with a hand-made BPMN 2.0 XML that carries both the semantics (what each shape is, what
connects to what) and the geometry (where each shape sits on the photo). This converter is
therefore the one that produces ground-truth IR rather than an estimate of it.

Two things need care.

**Coordinates.** BPMN DI bounds are not in image pixels. Each file opens with a comment
`{"backgroundSize": 1000}`, meaning the annotator worked on the photo scaled so its **width**
was 1000 units; every bound and waypoint is multiplied by `width / backgroundSize` to land back
on the original photo.

That it is the width and not the longer side was found the hard way. The first version used
`max(width, height)`, which is identical on a landscape photo and wrong on a portrait one -
and the diagram checked by eye happened to be landscape. The Phase 2.2.5 validator then
reported 162 diagrams with boxes outside the frame, all portrait. Measured across 198
annotations: the width rule puts every box inside the image, the longest-side rule fails on
exactly the 38 portrait ones. `tests/test_ir_convert.py` now checks portrait and landscape
separately.

**Shape versus role.** The XML says an element is a `task`; it does not say the writer drew a
rounded rectangle. The role is ground truth, the shape is the BPMN drawing convention applied
to it, and every node records which of the two it is in `attrs.shape_basis`. Phase 2.2.4
measures how often that convention is wrong.

    python -m src.ir.convert.hdbpmn --limit 20
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from PIL import Image

from src.ir.model import SUFFIX, Diagram, Edge, Node
from src.ir.vocab import canonical_role, canonical_shape
from src.utils.config import ROOT

SOURCE = "hdbpmn"
RAW = ROOT / "data" / "raw" / "hdbpmn" / "data"
OUT = ROOT / "data" / "processed" / "ir" / SOURCE

#: BPMN elements that are things. Anything else with an id is either a connection (below) or
#: bookkeeping (process, collaboration, laneSet) that never gets drawn.
NODE_TAGS = {
    "task",
    "userTask",
    "serviceTask",
    "manualTask",
    "sendTask",
    "receiveTask",
    "scriptTask",
    "businessRuleTask",
    "subProcess",
    "callActivity",
    "startEvent",
    "endEvent",
    "intermediateCatchEvent",
    "intermediateThrowEvent",
    "boundaryEvent",
    "exclusiveGateway",
    "inclusiveGateway",
    "parallelGateway",
    "eventBasedGateway",
    "complexGateway",
    "dataObjectReference",
    "dataStoreReference",
    "participant",
    "lane",
    "textAnnotation",
    "group",
}

#: BPMN elements that are connections, with whether an arrowhead is part of the notation.
EDGE_TAGS = {
    "sequenceFlow": True,
    "messageFlow": True,
    "dataInputAssociation": True,
    "dataOutputAssociation": True,
    "association": False,
}

#: How BPMN says each element is drawn. This is convention, not observation - see the module
#: docstring. `freeform` is used where BPMN's symbol has no counterpart in our twelve shapes
#: (the folded-corner page of a data object, the cylinder of a data store); calling those
#: rectangles would assert something about the drawing that nobody checked.
SHAPE_BY_TAG = {
    "startEvent": "circle",
    "endEvent": "circle",
    "intermediateCatchEvent": "circle",
    "intermediateThrowEvent": "circle",
    "boundaryEvent": "circle",
    "exclusiveGateway": "diamond",
    "inclusiveGateway": "diamond",
    "parallelGateway": "diamond",
    "eventBasedGateway": "diamond",
    "complexGateway": "diamond",
    "dataObjectReference": "freeform",
    "dataStoreReference": "freeform",
    "textAnnotation": "text-block",
    "participant": "rectangle",
    "lane": "rectangle",
    "group": "rectangle",
}
DEFAULT_SHAPE = "rounded-rect"  # every task-like element

BACKGROUND_RE = re.compile(r'"backgroundSize"\s*:\s*([0-9.]+)')


def _tag(el: ET.Element) -> str:
    return el.tag.rsplit("}", 1)[-1]


def _text(el: ET.Element | None) -> str:
    return (el.text or "").strip() if el is not None else ""


def _background_size(raw: str) -> float:
    m = BACKGROUND_RE.search(raw)
    return float(m.group(1)) if m else 1000.0


def _endpoints(el: ET.Element) -> tuple[str | None, str | None]:
    """sourceRef/targetRef are attributes on a sequenceFlow and child elements on a data
    association - the BPMN spec allows both spellings and hdBPMN contains both."""
    src = el.get("sourceRef")
    dst = el.get("targetRef")
    for child in el:
        if _tag(child) == "sourceRef" and not src:
            src = _text(child)
        elif _tag(child) == "targetRef" and not dst:
            dst = _text(child)
    return (src or None), (dst or None)


def image_for(annotation: Path) -> Path:
    return RAW / "images" / annotation.parent.name / f"{annotation.stem}.jpg"


def convert(annotation: Path, *, image: Path | None = None) -> Diagram:
    raw = annotation.read_text(encoding="utf-8")
    root = ET.fromstring(raw)

    image = image or image_for(annotation)
    if image.is_file():
        with Image.open(image) as im:
            width, height = im.size
    else:  # geometry is meaningless without knowing what it indexes
        raise FileNotFoundError(f"no image beside {annotation.name}: {image}")
    scale = width / _background_size(raw)

    # -- semantics: every element that could be drawn, keyed by id --------------------
    kinds: dict[str, str] = {}
    names: dict[str, str] = {}
    edge_elements: dict[str, ET.Element] = {}
    for el in root.iter():
        eid = el.get("id")
        if not eid:
            continue
        tag = _tag(el)
        if tag in NODE_TAGS:
            kinds[eid] = tag
            names[eid] = (el.get("name") or "").replace("\n", " ").strip()
        elif tag in EDGE_TAGS:
            edge_elements[eid] = el
            names[eid] = (el.get("name") or "").replace("\n", " ").strip()

    # -- geometry: BPMNShape / BPMNEdge ------------------------------------------------
    bounds: dict[str, list[float]] = {}
    waypoints: dict[str, list[list[float]]] = {}
    for el in root.iter():
        tag = _tag(el)
        ref = el.get("bpmnElement")
        if not ref:
            continue
        if tag == "BPMNShape":
            b = next((c for c in el if _tag(c) == "Bounds"), None)
            if b is not None:
                bounds[ref] = [
                    float(b.get("x", 0)) * scale,
                    float(b.get("y", 0)) * scale,
                    float(b.get("width", 0)) * scale,
                    float(b.get("height", 0)) * scale,
                ]
        elif tag == "BPMNEdge":
            pts = [
                [float(c.get("x", 0)) * scale, float(c.get("y", 0)) * scale]
                for c in el
                if _tag(c) == "waypoint"
            ]
            if len(pts) >= 2:
                waypoints[ref] = pts

    # -- nodes -------------------------------------------------------------------------
    nodes: list[Node] = []
    for eid, tag in kinds.items():
        if eid not in bounds:
            # Annotated in the model but never placed on the plane: it was not drawn.
            continue
        nodes.append(
            Node(
                id=eid,
                shape=canonical_shape(SHAPE_BY_TAG.get(tag, DEFAULT_SHAPE)),
                bbox=bounds[eid],
                text=names.get(eid, ""),
                semantic_role=canonical_role(tag),
                confidence=1.0,
                source_id=eid,
                attrs={"bpmn_tag": tag, "shape_basis": "bpmn-convention"},
            )
        )
    present = {n.id for n in nodes}

    # -- edges -------------------------------------------------------------------------
    edges: list[Edge] = []
    for eid, el in edge_elements.items():
        src, dst = _endpoints(el)
        edges.append(
            Edge(
                id=eid,
                src=src if src in present else None,
                dst=dst if dst in present else None,
                directed=EDGE_TAGS[_tag(el)],
                label=names.get(eid, ""),
                polyline=waypoints.get(eid),
                confidence=1.0,
                source_id=eid,
                attrs={"bpmn_tag": _tag(el)},
            )
        )

    diagram = Diagram(
        id=annotation.stem,
        diagram_type="flowchart",
        nodes=nodes,
        edges=edges,
        meta={
            "source": SOURCE,
            "geometry": "annotated",
            "image": str(image.relative_to(ROOT)).replace("\\", "/"),
            "image_size": [width, height],
            "scribe_id": annotation.stem.split("_")[-1],
            "producer": "src.ir.convert.hdbpmn",
            "exercise": annotation.parent.name,
            "notes": (
                "Roles are ground truth from the BPMN model. Shapes are the BPMN drawing "
                "convention for those roles, not an observation of the photo."
            ),
        },
    )
    _resolve_parallel_gateways(diagram)
    diagram.sync_unresolved()
    return diagram


def _resolve_parallel_gateways(diagram: Diagram) -> None:
    """A BPMN parallel gateway is a fork or a join depending only on its degree; the XML tag
    is the same either way, so the vocabulary maps it to `fork` and this fixes up the joins."""
    for node in diagram.nodes:
        if (node.attrs or {}).get("bpmn_tag") != "parallelGateway":
            continue
        out_degree = len(diagram.out_edges(node.id))
        in_degree = len(diagram.in_edges(node.id))
        node.semantic_role = "join" if in_degree > out_degree else "fork"


def annotations() -> list[Path]:
    return sorted(RAW.glob("annotations/*/*.bpmn"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, help="convert only the first N files")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)

    files = annotations()[: args.limit]
    if not files:
        print(f"no annotations under {RAW}", file=sys.stderr)
        return 1

    written, failed = 0, []
    stats: dict[str, Any] = {"nodes": 0, "edges": 0, "dangling": 0, "roles": {}}
    for path in files:
        try:
            d = convert(path)
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the batch
            failed.append((path.name, str(exc)))
            continue
        d.save(args.out / f"{d.id}{SUFFIX}")
        written += 1
        stats["nodes"] += len(d.nodes)
        stats["edges"] += len(d.edges)
        stats["dangling"] += len(d.unresolved_edges)
        for n in d.nodes:
            stats["roles"][n.semantic_role] = stats["roles"].get(n.semantic_role, 0) + 1

    print(f"wrote {written} IR files to {args.out.relative_to(ROOT)}")
    print(json.dumps(stats, indent=2, sort_keys=True))
    for name, err in failed[:10]:
        print(f"  FAIL  {name}: {err}", file=sys.stderr)
    if failed:
        print(f"  {len(failed)} of {len(files)} files failed", file=sys.stderr)
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
