"""Phase 2.2.1 - Label Studio as the human labelling front end.

Label Studio was chosen over CVAT for one reason: CVAT is built around boxes and masks, and
this project's hardest annotation is not a box but a *relation* - which arrow leaves which
shape. Label Studio's `<Relations>` control makes that a first-class thing an annotator draws,
and its per-region `<Choices>` and `<TextArea>` cover role and transcription in the same pass,
so one image is labelled once rather than three times in three tools.

This module is the bridge in both directions:

* `tasks()` turns corpus images into Label Studio tasks;
* `prediction()` turns an IR diagram into a Label Studio *pre-annotation*, so an annotator
  corrects the converter's output instead of starting from a blank image - which is the
  difference between labelling 260 diagrams and labelling 260 diagrams twice;
* `to_ir()` turns a completed annotation back into IR.

The vocabularies in `labeling/label_studio/config.xml` are the same twelve shapes and
twenty-three roles as `src.ir.vocab`; `check()` fails if they ever drift apart.

    python -m src.ir.labelstudio --check
    python -m src.ir.labelstudio --tasks data/processed/ir/hdbpmn --limit 50
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from src.ir.model import SUFFIX, Diagram, Edge, Node
from src.ir.vocab import ROLES, SHAPES, canonical_role, canonical_shape
from src.utils.config import ROOT

CONFIG = ROOT / "labeling" / "label_studio" / "config.xml"
TASKS = ROOT / "labeling" / "tasks"

#: Relation names in the config, and what each means in the IR.
RELATIONS = {
    "flow": {"directed": True, "label": ""},
    "flow-yes": {"directed": True, "label": "yes"},
    "flow-no": {"directed": True, "label": "no"},
    "undirected": {"directed": False, "label": ""},
    "contains": {"directed": True, "label": "", "kind": "contains"},
    "uncertain": {"directed": True, "label": "", "uncertain": True},
}


# -- config -----------------------------------------------------------------------------


def config_vocabularies(path: Path = CONFIG) -> dict[str, list[str]]:
    """Read the shape, role, flag and relation lists straight out of the XML."""
    root = ET.fromstring(path.read_text(encoding="utf-8"))
    out: dict[str, list[str]] = {}
    for tag in root.iter("RectangleLabels"):
        out["shapes"] = [el.get("value", "") for el in tag.iter("Label")]
    for tag in root.iter("Choices"):
        name = tag.get("name", "")
        out[name] = [el.get("value", "") for el in tag.iter("Choice")]
    out["relations"] = [el.get("value", "") for el in root.iter("Relation")]
    return out


def check() -> list[str]:
    """Problems that would make an export unconvertible. Empty list means consistent."""
    problems = []
    vocab = config_vocabularies()
    if vocab.get("shapes", []) != list(SHAPES):
        missing = set(SHAPES) - set(vocab.get("shapes", []))
        extra = set(vocab.get("shapes", [])) - set(SHAPES)
        problems.append(
            f"shape labels differ from src.ir.vocab.SHAPES: missing {missing}, extra {extra}"
        )
    if vocab.get("role", []) != list(ROLES):
        missing = set(ROLES) - set(vocab.get("role", []))
        extra = set(vocab.get("role", [])) - set(ROLES)
        problems.append(
            f"role choices differ from src.ir.vocab.ROLES: missing {missing}, extra {extra}"
        )
    if set(vocab.get("relations", [])) != set(RELATIONS):
        problems.append("relation names differ from src.ir.labelstudio.RELATIONS")
    return problems


# -- IR out: tasks and pre-annotations ---------------------------------------------------


def image_url(path: str | Path) -> str:
    """Label Studio serves repo files through its local-files endpoint; the document root is
    set to the repository, so the query is the repo-relative path."""
    rel = str(path).replace("\\", "/")
    return f"/data/local-files/?d={rel}"


def _region_id(node_id: str) -> str:
    return f"r_{node_id}"


def prediction(diagram: Diagram) -> dict[str, Any]:
    """An IR diagram as a Label Studio pre-annotation.

    Percentages, not pixels: Label Studio stores region geometry as a percentage of the image,
    which is also why `image_size` is required on any diagram that has boxes.
    """
    size = diagram.meta.get("image_size")
    if not size:
        raise ValueError(f"{diagram.id}: cannot pre-annotate without meta.image_size")
    width, height = size
    results: list[dict[str, Any]] = []

    for node in diagram.nodes:
        if node.bbox is None:
            continue
        x, y, w, h = node.bbox
        value = {
            "x": 100.0 * x / width,
            "y": 100.0 * y / height,
            "width": 100.0 * w / width,
            "height": 100.0 * h / height,
            "rotation": 0,
        }
        rid = _region_id(node.id)
        results.append(
            {
                "id": rid,
                "type": "rectanglelabels",
                "from_name": "shape",
                "to_name": "image",
                "original_width": width,
                "original_height": height,
                "image_rotation": 0,
                "value": {**value, "rectanglelabels": [node.shape]},
            }
        )
        results.append(
            {
                "id": rid,
                "type": "choices",
                "from_name": "role",
                "to_name": "image",
                "original_width": width,
                "original_height": height,
                "image_rotation": 0,
                "value": {**value, "choices": [node.semantic_role]},
            }
        )
        if node.text:
            results.append(
                {
                    "id": rid,
                    "type": "textarea",
                    "from_name": "transcription",
                    "to_name": "image",
                    "original_width": width,
                    "original_height": height,
                    "image_rotation": 0,
                    "value": {**value, "text": [node.text]},
                }
            )

    uncertain = {u["edge"] for u in diagram.unresolved_edges}
    for edge in diagram.edges:
        if edge.src is None or edge.dst is None:
            continue  # a relation needs two ends; the dangling ones stay in the IR only
        name = "uncertain" if edge.id in uncertain else _relation_name(edge)
        results.append(
            {
                "type": "relation",
                "from_id": _region_id(edge.src),
                "to_id": _region_id(edge.dst),
                "direction": "right",
                "labels": [name],
            }
        )

    return {
        "model_version": diagram.meta.get("producer", "ir"),
        "result": results,
    }


def _relation_name(edge: Edge) -> str:
    if (edge.attrs or {}).get("kind") == "contains":
        return "contains"
    if not edge.directed:
        return "undirected"
    label = (edge.label or "").strip().lower()
    if label in {"yes", "y", "true"}:
        return "flow-yes"
    if label in {"no", "n", "false"}:
        return "flow-no"
    return "flow"


def task(diagram: Diagram, *, with_prediction: bool = True) -> dict[str, Any]:
    image = diagram.meta.get("image", "")
    if not image:
        raise ValueError(f"{diagram.id}: no image to label")
    payload: dict[str, Any] = {
        "data": {
            "image": image_url(image),
            "diagram_id": diagram.id,
            "source": diagram.meta.get("source", ""),
            "diagram_type": diagram.diagram_type,
        }
    }
    if with_prediction and diagram.meta.get("image_size"):
        payload["predictions"] = [prediction(diagram)]
    return payload


def tasks(ir_dir: Path, *, limit: int | None = None, with_predictions: bool = True) -> list[dict]:
    out = []
    for path in sorted(ir_dir.glob(f"*{SUFFIX}"))[:limit]:
        diagram = Diagram.load(path)
        if not diagram.meta.get("image"):
            continue  # structure-only sources have nothing to show an annotator
        out.append(task(diagram, with_prediction=with_predictions))
    return out


# -- IR in: a completed annotation --------------------------------------------------------


def to_ir(
    annotation: dict[str, Any],
    *,
    diagram_id: str,
    diagram_type: str = "unknown",
    image: str = "",
    meta: dict[str, Any] | None = None,
) -> Diagram:
    """Turn one Label Studio annotation (an export entry's `result` list) into IR."""
    results = annotation.get("result", annotation) if isinstance(annotation, dict) else annotation
    nodes: dict[str, Node] = {}
    order: list[str] = []
    flags: dict[str, set[str]] = {}
    size: list[int] | None = None
    relations = []

    for item in results:
        kind = item.get("type")
        if kind == "relation":
            relations.append(item)
            continue
        rid = item.get("id")
        if rid is None:
            continue
        value = item.get("value", {})
        width = item.get("original_width")
        height = item.get("original_height")
        if width and height:
            size = [int(width), int(height)]
        if rid not in nodes:
            nodes[rid] = Node(id=rid, shape="freeform", bbox=None, text="", semantic_role="unknown")
            order.append(rid)
        node = nodes[rid]
        if {"x", "y", "width", "height"} <= set(value) and width and height:
            node.bbox = [
                value["x"] * width / 100.0,
                value["y"] * height / 100.0,
                value["width"] * width / 100.0,
                value["height"] * height / 100.0,
            ]
        if kind == "rectanglelabels" and value.get("rectanglelabels"):
            node.shape = canonical_shape(value["rectanglelabels"][0])
        elif kind == "choices" and item.get("from_name") == "role" and value.get("choices"):
            node.semantic_role = canonical_role(value["choices"][0])
        elif kind == "choices" and item.get("from_name") == "flags":
            flags[rid] = set(value.get("choices", []))
        elif kind == "textarea" and value.get("text"):
            node.text = " ".join(value["text"]).strip()

    diagram = Diagram(
        id=diagram_id,
        diagram_type=diagram_type,
        nodes=[nodes[r] for r in order],
        edges=[],
        meta={
            "source": "label_studio",
            "geometry": "annotated",
            "image": image,
            "image_size": size,
            "producer": "src.ir.labelstudio",
            **(meta or {}),
        },
    )

    for i, rel in enumerate(relations):
        name = (rel.get("labels") or ["flow"])[0]
        spec = RELATIONS.get(name, RELATIONS["flow"])
        edge = Edge(
            id=f"e{i}",
            src=rel.get("from_id"),
            dst=rel.get("to_id"),
            directed=bool(spec["directed"]),
            label=str(spec["label"]),
            polyline=None,
            confidence=0.5 if spec.get("uncertain") else 1.0,
            attrs={"kind": spec["kind"]} if spec.get("kind") else None,
        )
        diagram.edges.append(edge)
        if spec.get("uncertain"):
            diagram.record_unresolved(edge, reason="ambiguous-endpoint", note="annotator flagged")

    for rid, marks in flags.items():
        node = nodes.get(rid)
        if node is None:
            continue
        if "crossed-out" in marks:
            diagram.crossed_out.append({"ref": rid, "bbox": node.bbox or [0, 0, 0, 0]})
        if "text-uncertain" in marks:
            diagram.low_conf_text.append({"ref": rid, "confidence": 0.3})
        if "shape-uncertain" in marks:
            node.confidence = 0.5

    diagram.sync_unresolved()
    return diagram


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="config and vocabularies agree")
    ap.add_argument("--tasks", type=Path, help="IR directory to build a task file from")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--out", type=Path, help="where to write the task file")
    args = ap.parse_args(argv)

    if args.check or not args.tasks:
        problems = check()
        for p in problems:
            print(f"  FAIL  {p}")
        if not problems:
            vocab = config_vocabularies()
            print(f"  PASS  {len(vocab['shapes'])} shapes, {len(vocab['role'])} roles agree")
            print(f"  PASS  {len(vocab['relations'])} relations agree")
        if not args.tasks:
            return 1 if problems else 0
        if problems:
            return 1

    built = tasks(args.tasks, limit=args.limit)
    out = args.out or TASKS / f"{args.tasks.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(built, fh, indent=2)
        fh.write("\n")
    predicted = sum(1 for t in built if t.get("predictions"))
    print(f"wrote {len(built)} tasks ({predicted} pre-annotated) to {out.relative_to(ROOT)}")
    return 0 if built else 1


if __name__ == "__main__":
    sys.exit(main())
