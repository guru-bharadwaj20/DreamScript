"""Phase 2.2.2 - Sketch2Code webpages to DreamScript IR.

Sketch2Code pairs 731 human sketches with the 484 real webpages they depict. The pairing is
what makes it valuable and also what makes it awkward: the ground truth is *HTML*, which says
what elements exist and how they nest, but nothing about where anything sits on the page
unless a browser renders it. There is no renderer in this pipeline, and inventing coordinates
would be worse than having none.

So this converter emits **structure-only IR**: `meta.geometry` is `absent`, every `bbox` and
`polyline` is null, and the edges are containment relations rather than drawn arrows. The QA
pass in Phase 2.2.5 knows to skip every geometric check on such a file, and Phase 9's detector
cannot train on them at all. What they are for is Phase 12: a wireframe IR paired with the
real HTML is a target-code pair, and that is what Phase 2.2.6 uses them as.

One more limit worth stating. A real webpage's DOM is far finer-grained than any sketch of it:
a person draws eight boxes where the markup has four hundred nodes. This converter keeps only
elements that map to a wireframe role, collapses single-child container chains, and caps the
result - so the IR is an *over-complete* reference for the sketch, not a stroke-level match.
`meta.dom_nodes` and `meta.truncated` record what that cost.

    python -m src.ir.convert.sketch2code --limit 25
"""

from __future__ import annotations

import argparse
import json
import sys
from html.parser import HTMLParser
from pathlib import Path

from src.ir.model import SUFFIX, Diagram, Edge, Node
from src.ir.vocab import ROLE_ALIASES, canonical_role
from src.utils.config import ROOT

SOURCE = "sketch2code"
RAW = ROOT / "data" / "raw" / "sketch2code"
OUT = ROOT / "data" / "processed" / "ir" / SOURCE

#: Elements whose whole subtree is invisible. Depth-counted, because they nest.
SKIP_SUBTREE = {"script", "style", "head", "title", "noscript", "template", "svg"}

#: Void elements that carry no content. These must NOT be depth-counted: HTML in the wild
#: writes `<meta ...>` with no closing tag, so counting them leaves the parser permanently
#: inside a skipped region and the entire page disappears.
SKIP_VOID = {"meta", "link", "br", "hr", "source", "track", "wbr", "base", "col", "param"}

#: Structural wrappers that exist in every document and mean nothing in a sketch.
TRANSPARENT = {"html"}

#: Shape drawn for each wireframe role. A sketched wireframe is boxes and squiggles: every
#: block is a rectangle and every run of text is a text-block. This is a convention, recorded
#: as `shape_basis: "wireframe-convention"`, not an observation of the sketch.
SHAPE_BY_ROLE = {
    "ui-label": "text-block",
    "ui-button": "rounded-rect",
    "ui-input": "rectangle",
    "ui-image": "rectangle",
    "container": "rectangle",
    "unknown": "rectangle",
}

MAX_NODES = 200


class _Tree(HTMLParser):
    """Minimal DOM builder: the stdlib parser plus a stack, which is all this needs."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root: dict = {"tag": "body", "children": [], "text": "", "parent": None}
        self.stack = [self.root]
        self.seen = 0
        self.skipping = 0

    def handle_starttag(self, tag, attrs):
        if tag in SKIP_SUBTREE:
            self.skipping += 1
            return
        if self.skipping or tag in SKIP_VOID or tag in TRANSPARENT:
            return
        self.seen += 1
        node = {"tag": tag, "children": [], "text": "", "parent": self.stack[-1]}
        self.stack[-1]["children"].append(node)
        self.stack.append(node)

    def handle_startendtag(self, tag, attrs):
        if tag in SKIP_SUBTREE or tag in SKIP_VOID or self.skipping:
            return
        self.seen += 1
        self.stack[-1]["children"].append(
            {"tag": tag, "children": [], "text": "", "parent": self.stack[-1]}
        )

    def handle_endtag(self, tag):
        if tag in SKIP_SUBTREE:
            self.skipping = max(0, self.skipping - 1)
            return
        if self.skipping or tag in SKIP_VOID or tag in TRANSPARENT:
            return
        for i in range(len(self.stack) - 1, 0, -1):
            if self.stack[i]["tag"] == tag:
                del self.stack[i:]
                break

    def handle_data(self, data):
        if self.skipping:
            return
        text = " ".join(data.split())
        if text:
            self.stack[-1]["text"] = (self.stack[-1]["text"] + " " + text).strip()[:200]


def _keep(node: dict) -> bool:
    """Keep elements that carry a wireframe role, and containers only when they branch."""
    role = canonical_role(node["tag"])
    if node["tag"] not in ROLE_ALIASES:
        return False
    if role == "container":
        return len(node["children"]) > 1  # a chain of single-child divs is invisible in a sketch
    return True


def _flatten(node: dict, parent_id: str | None, out: list[tuple[str, dict, str | None]]) -> None:
    my_id = parent_id
    if len(out) < MAX_NODES and (parent_id is None or _keep(node)):
        my_id = f"n{len(out)}"
        out.append((my_id, node, parent_id))
    for child in node["children"]:
        _flatten(child, my_id, out)


def convert(html_path: Path) -> Diagram:
    parser = _Tree()
    parser.feed(html_path.read_text(encoding="utf-8", errors="replace"))

    flat: list[tuple[str, dict, str | None]] = []
    _flatten(parser.root, None, flat)

    nodes: list[Node] = []
    edges: list[Edge] = []
    for node_id, element, parent_id in flat:
        role = canonical_role(element["tag"]) if element["tag"] != "body" else "container"
        nodes.append(
            Node(
                id=node_id,
                shape=SHAPE_BY_ROLE.get(role, "rectangle"),
                bbox=None,
                text=element["text"],
                semantic_role=role,
                confidence=1.0 if element["tag"] in ROLE_ALIASES else 0.5,
                source_id=element["tag"],
                attrs={"html_tag": element["tag"], "shape_basis": "wireframe-convention"},
            )
        )
        if parent_id is not None:
            edges.append(
                Edge(
                    id=f"c{len(edges)}",
                    src=parent_id,
                    dst=node_id,
                    directed=True,
                    label="",
                    polyline=None,
                    confidence=1.0,
                    attrs={"kind": "contains"},
                )
            )

    page_id = html_path.stem
    sketches = sorted(RAW.glob(f"sketches/{page_id}_*.png"))
    return Diagram(
        id=f"s2c_{page_id}",
        diagram_type="wireframe",
        nodes=nodes,
        edges=edges,
        meta={
            "source": SOURCE,
            "geometry": "absent",
            "image": "",
            "image_size": None,
            "producer": "src.ir.convert.sketch2code",
            "webpage": str(html_path.relative_to(ROOT)).replace("\\", "/"),
            "sketches": [str(p.relative_to(ROOT)).replace("\\", "/") for p in sketches],
            "dom_nodes": parser.seen,
            "truncated": len(flat) >= MAX_NODES,
            "notes": (
                "Structure and text come from the real HTML and are ground truth. There are no "
                "coordinates: nothing here was rendered, and edges are containment, not drawn "
                "arrows. The DOM is far finer-grained than any sketch of it, so this is an "
                "over-complete reference for the paired sketches, not a stroke-level match."
            ),
        },
    )


def pages() -> list[Path]:
    return sorted(RAW.glob("webpages/*.html"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)

    paths = pages()[: args.limit]
    if not paths:
        print(f"no webpages under {RAW}", file=sys.stderr)
        return 1

    written, failed = 0, []
    stats = {"nodes": 0, "edges": 0, "truncated": 0, "with_sketch": 0, "roles": {}}
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
        stats["truncated"] += bool(d.meta["truncated"])
        stats["with_sketch"] += bool(d.meta["sketches"])
        for n in d.nodes:
            stats["roles"][n.semantic_role] = stats["roles"].get(n.semantic_role, 0) + 1

    print(f"wrote {written} IR files to {args.out.relative_to(ROOT)}")
    print(json.dumps(stats, indent=2, sort_keys=True))
    for name, err in failed[:10]:
        print(f"  FAIL  {name}: {err}", file=sys.stderr)
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
