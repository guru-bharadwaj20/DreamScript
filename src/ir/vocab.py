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

#: Frozen. Ten of these are the vocabulary named in contributing.md 2.1.4. Two were added because the
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

#: Frozen. Seventeen of these are the vocabulary named in contributing.md 2.1.5. Six were added, each
#: because a real count in the corpus showed that folding it into a neighbour would make the
#: generated code wrong rather than merely coarse:
#:
#:   fork / join     661 parallel gateways in hdBPMN. A parallel gateway has no condition;
#:                   calling it `decision` would make Phase 12 emit an `if` where the drawing
#:                   means concurrency. Fork and join are told apart by degree, not by shape.
#:   event           796 intermediate catch and 298 intermediate throw events in hdBPMN -
#:                   events that are neither the start nor the end of the process.
#:   initial-state   the FA database marks these with an incoming arrow from nowhere. A state
#:   final-state     machine's target code is simply wrong without knowing which state it
#:                   starts in and which states accept; 96 of 199 states in a 60-file sample
#:                   are accepting.
#:   unknown         a converter that cannot tell must be able to say so. Every other value
#:                   here is a claim, and `unknown` is what keeps the claims honest. The QA
#:                   pass in 2.2.5 counts it rather than accepting it silently.
ROLES: tuple[str, ...] = (
    # flowchart / process
    "start",
    "end",
    "process",
    "decision",
    "io",
    "fork",
    "join",
    "event",
    # state machine
    "state",
    "initial-state",
    "final-state",
    "transition",
    # entity-relationship
    "entity",
    "attribute",
    "relationship",
    # shared
    "container",
    # user interface
    "ui-input",
    "ui-button",
    "ui-label",
    "ui-image",
    # circuit
    "component",
    "wire",
    # the honest default
    "unknown",
)

#: Which roles may appear in which diagram type. Not enforced by the schema - a mislabelled
#: diagram_type would then cascade into a validation failure that hides the real error - but
#: checked and counted by the QA pass in Phase 2.2.5, where it reads as a warning.
ROLES_BY_TYPE: dict[str, frozenset[str]] = {
    "flowchart": frozenset(
        {
            "start",
            "end",
            "process",
            "decision",
            "io",
            "fork",
            "join",
            "event",
            "container",
            "unknown",
        }
    ),
    "state_machine": frozenset(
        {"state", "initial-state", "final-state", "transition", "container", "unknown"}
    ),
    "er_diagram": frozenset({"entity", "attribute", "relationship", "container", "unknown"}),
    "wireframe": frozenset(
        {"ui-input", "ui-button", "ui-label", "ui-image", "container", "unknown"}
    ),
    "circuit": frozenset({"component", "wire", "ui-label", "container", "unknown"}),
    "unknown": frozenset(ROLES),
}

#: Roles that describe a connection rather than a thing. A node carrying one of these is a
#: detection-stage artifact - Phase 10.1 turns it into an edge - and the QA pass says so.
EDGE_LIKE_ROLES: frozenset[str] = frozenset({"transition", "wire"})

#: The shape a role is usually drawn as. Used by the geometric second annotator in Phase 2.2.4
#: and as the prior Phase 10 starts from; never as a hard rule, because the whole difficulty of
#: this project is that people draw a decision as a squashed circle when they are in a hurry.
SHAPE_ROLE_PRIOR: dict[str, tuple[str, ...]] = {
    "start": ("circle", "ellipse", "rounded-rect"),
    "end": ("circle", "double-circle", "ellipse", "rounded-rect"),
    "process": ("rectangle", "rounded-rect"),
    "decision": ("diamond",),
    "io": ("parallelogram", "rectangle"),
    "fork": ("diamond", "line", "rectangle"),
    "join": ("diamond", "line", "rectangle"),
    "event": ("circle", "ellipse"),
    "state": ("circle", "ellipse", "rounded-rect"),
    "initial-state": ("circle", "ellipse"),
    "final-state": ("double-circle", "circle"),
    "transition": ("arrow", "line"),
    "entity": ("rectangle",),
    "attribute": ("ellipse", "circle"),
    "relationship": ("diamond",),
    "container": ("rectangle", "rounded-rect"),
    "ui-input": ("rectangle",),
    "ui-button": ("rectangle", "rounded-rect"),
    "ui-label": ("text-block", "line"),
    "ui-image": ("rectangle",),
    "component": ("freeform", "rectangle", "circle"),
    "wire": ("line", "arrow"),
    "unknown": SHAPES,
}

#: Source spellings mapped onto our roles. BPMN tag names, graphviz node kinds and the FA
#: database's truth strings all land here rather than in each converter.
ROLE_ALIASES: dict[str, str] = {
    # BPMN
    "task": "process",
    "usertask": "process",
    "servicetask": "process",
    "manualtask": "process",
    "sendtask": "process",
    "receivetask": "process",
    "scripttask": "process",
    "businessruletask": "process",
    "subprocess": "process",
    "callactivity": "process",
    "startevent": "start",
    "endevent": "end",
    "intermediatecatchevent": "event",
    "intermediatethrowevent": "event",
    "boundaryevent": "event",
    "exclusivegateway": "decision",
    "inclusivegateway": "decision",
    "eventbasedgateway": "decision",
    "complexgateway": "decision",
    "parallelgateway": "fork",  # resolved to join by degree; see convert.hdbpmn
    "dataobjectreference": "io",
    "datastorereference": "io",
    "dataobject": "io",
    "participant": "container",
    "lane": "container",
    "pool": "container",
    "textannotation": "unknown",
    "group": "container",
    # finite automata
    "state": "state",
    "finalstate": "final-state",
    "final state": "final-state",
    "initialstate": "initial-state",
    # markup / wireframe
    "input": "ui-input",
    "textarea": "ui-input",
    "select": "ui-input",
    "button": "ui-button",
    "a": "ui-button",
    "img": "ui-image",
    "svg": "ui-image",
    "video": "ui-image",
    "figure": "ui-image",
    "h1": "ui-label",
    "h2": "ui-label",
    "h3": "ui-label",
    "h4": "ui-label",
    "h5": "ui-label",
    "h6": "ui-label",
    "p": "ui-label",
    "span": "ui-label",
    "li": "ui-label",
    "label": "ui-label",
    "div": "container",
    "section": "container",
    "nav": "container",
    "header": "container",
    "footer": "container",
    "main": "container",
    "aside": "container",
    "form": "container",
    "ul": "container",
    "ol": "container",
    "table": "container",
    "body": "container",
}


def canonical_role(raw: str) -> str:
    """Map a source's spelling onto the vocabulary; anything unrecognised becomes `unknown`."""
    key = (raw or "").strip().lower()
    if key in ROLES:
        return key
    flat = key.replace(" ", "").replace("_", "").replace("-", "")
    for role in ROLES:
        if role.replace("-", "") == flat:
            return role
    return ROLE_ALIASES.get(key, ROLE_ALIASES.get(flat, "unknown"))


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
