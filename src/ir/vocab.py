"""Phase 2.1.4 / 2.1.5 - the frozen vocabularies.

Two closed sets: what a thing looks like (`SHAPES`) and what it means (`ROLES`). They are
deliberately separate. A diamond in a flowchart is a decision and a diamond in an ER diagram
is a relationship; a state and a start event are both circles. Collapsing appearance into
meaning is the mistake that makes a model unable to generalise across diagram types, so the
IR records both and Phase 10 is where one is inferred from the other plus context.

`schemas/shape.schema.json` and `schemas/role.schema.json` are generated from these tuples by
`python -m src.ir.vocab --write`, which is how the enum in the schema and the enum in the code
are kept from drifting.
"""

from __future__ import annotations

import argparse
import json
import sys

from src.utils.config import ROOT

# ---------------------------------------------------------------------------------------
# Shapes - what was drawn
# ---------------------------------------------------------------------------------------

#: Frozen. Ten of these are the vocabulary named in plan.md 2.1.4. Two were added because the
#: corpus contains them in quantity and mapping them onto a neighbour would destroy the one
#: feature that distinguishes them:
#:
#:   double-circle   96 accepting states in the FA database. Calling them `circle` erases the
#:                   only mark that separates an accepting state from an ordinary one.
#:   octagon         2,413 DIDI prompts. It is one of that dataset's five prompt shapes;
#:                   folding it into `freeform` would throw away a fifth of DIDI's structure.
SHAPES: tuple[str, ...] = (
    "rectangle",
    "rounded-rect",
    "diamond",
    "ellipse",
    "circle",
    "double-circle",
    "parallelogram",
    "octagon",
    "arrow",
    "line",
    "text-block",
    "freeform",
)

#: Spellings other people's datasets use, mapped onto ours. A converter calls
#: `canonical_shape` rather than hardcoding one of these, so a new source only ever adds a row
#: here instead of widening the vocabulary.
SHAPE_ALIASES: dict[str, str] = {
    "oval": "ellipse",
    "box": "rectangle",
    "rect": "rectangle",
    "square": "rectangle",
    "roundedrect": "rounded-rect",
    "rounded_rectangle": "rounded-rect",
    "roundrect": "rounded-rect",
    "doublecircle": "double-circle",
    "double_circle": "double-circle",
    "final": "double-circle",
    "rhombus": "diamond",
    "polyline": "line",
    "segment": "line",
    "text": "text-block",
    "label": "text-block",
    "textblock": "text-block",
    "other": "freeform",
    "unknown": "freeform",
}

#: What a shape looks like, in the terms Phase 3 will actually measure. Used by the geometric
#: second annotator in Phase 2.2.4 and as a sanity reference for Phase 4's feature design.
SHAPE_GEOMETRY: dict[str, str] = {
    "rectangle": "4 corners, opposite sides parallel, aspect free",
    "rounded-rect": "4 corners with curvature at each, aspect free",
    "diamond": "4 corners, vertices near the bbox edge midpoints",
    "ellipse": "no corners, aspect away from 1",
    "circle": "no corners, aspect near 1",
    "double-circle": "two concentric closed curves, aspect near 1",
    "parallelogram": "4 corners, one pair of sides slanted",
    "octagon": "6-8 corners, convex, aspect near 1",
    "arrow": "open curve with a convergent head at one end",
    "line": "open curve, no head",
    "text-block": "dense small components, no enclosing outline",
    "freeform": "none of the above",
}


def canonical_shape(raw: str) -> str:
    """Map a source's spelling onto the vocabulary. Unrecognised shapes become `freeform`
    rather than raising: a converter meeting a new shape should degrade, not crash, and the
    QA pass in Phase 2.2.5 counts how often it happens."""
    key = (raw or "").strip().lower().replace(" ", "")
    if key in SHAPES:
        return key
    key2 = key.replace("-", "").replace("_", "")
    for shape in SHAPES:
        if shape.replace("-", "") == key2:
            return shape
    return SHAPE_ALIASES.get(key, SHAPE_ALIASES.get(key2, "freeform"))


# ---------------------------------------------------------------------------------------
# Roles - what it means. Filled in by Phase 2.1.5.
# ---------------------------------------------------------------------------------------

ROLES: tuple[str, ...] = ()


def _enum_schema(name: str, title: str, values: tuple[str, ...], description: str) -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"https://dreamscript.local/schemas/{name}.schema.json",
        "title": title,
        "description": description,
        "type": "string",
        "enum": list(values),
    }


def write_schemas() -> list[str]:
    written = []
    specs = [
        (
            "shape",
            "DreamScript shape vocabulary",
            SHAPES,
            "What was drawn, independent of what it means. Frozen at Phase 2.1.4; generated "
            "from src.ir.vocab.SHAPES.",
        )
    ]
    if ROLES:
        specs.append(
            (
                "role",
                "DreamScript semantic role vocabulary",
                ROLES,
                "What the shape means in its diagram type. Frozen at Phase 2.1.5; generated "
                "from src.ir.vocab.ROLES.",
            )
        )
    for name, title, values, desc in specs:
        path = ROOT / "schemas" / f"{name}.schema.json"
        with path.open("w", encoding="utf-8", newline="\n") as fh:
            json.dump(_enum_schema(name, title, values, desc), fh, indent=2)
            fh.write("\n")
        written.append(str(path.relative_to(ROOT)))
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true", help="regenerate the enum schemas")
    args = ap.parse_args(argv)
    if args.write:
        for path in write_schemas():
            print(f"wrote {path}")
        return 0
    print(f"shapes ({len(SHAPES)}): " + ", ".join(SHAPES))
    print(f"roles  ({len(ROLES)}): " + ", ".join(ROLES))
    return 0


if __name__ == "__main__":
    sys.exit(main())
