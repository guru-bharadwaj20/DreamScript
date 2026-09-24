"""Every document `src.synth.graphs` produces satisfies `schemas/ir.schema.json`.

This exists because the generator failed the project's own schema on **every** document it had
ever produced - 200 of 200 across the type x structure x seed grid - and nothing noticed, for
three reasons that are each worth a test rather than a comment:

    ir_version      declared here as the float `1.0`; the schema wants a string matching
                    `^[0-9]+[.][0-9]+$`. 160 of the 200.
    diagram_type    `"er"`, which is the key `codegen.targets` dispatches on and is *not* in the
                    schema's enum (`er_diagram` is). The other 40.
    semantic_role   `"source"`, `"resistor"`, `"capacitor"`, `"inductor"` on circuit nodes. The
                    role vocabulary is closed and allows a circuit only `component`, `wire`,
                    `ui-label`, `container`, `unknown`; the device kind belongs in
                    `attrs["component"]`, and always did.

None of the three is visible downstream: `codegen.pairs` canonicalises the type, the pair schema
is a different schema, and nothing else validated the IR at all. So the check has to be here, on
the generator's own output, against the IR schema itself.
"""

from __future__ import annotations

import pytest

from src.ir.model import IR_VERSION
from src.ir.schema import problems
from src.synth import graphs

GRID = [(t, s) for t in graphs.DIAGRAM_TYPES for s in graphs.STRUCTURES]


@pytest.mark.parametrize(("diagram_type", "structure"), GRID)
def test_every_generated_diagram_satisfies_the_ir_schema(diagram_type: str, structure: str):
    for seed in range(8):
        diagram = graphs.random_diagram(diagram_type, structure, seed)
        found = problems("ir", diagram)
        assert not found, f"{diagram_type}/{structure}/{seed}: " + "; ".join(found[:4])


def test_the_generator_does_not_keep_its_own_ir_version():
    """One definition. Two drifted, and the float lost."""
    assert graphs.IR_VERSION is IR_VERSION
    assert isinstance(graphs.IR_VERSION, str)


def test_diagram_types_are_exactly_the_schema_enum_minus_unknown():
    from src.ir.schema import load_schema

    enum = set(load_schema("ir")["properties"]["diagram_type"]["enum"])
    assert set(graphs.DIAGRAM_TYPES) == enum - {"unknown"}


@pytest.mark.parametrize("alias", sorted(graphs.TYPE_ALIASES))
def test_an_alias_is_accepted_and_canonicalised_never_emitted(alias: str):
    """`codegen.targets` keys its ER emitter on `"er"`, so the spelling stays callable - but what
    comes out is the schema's name, not the caller's."""
    diagram = graphs.random_diagram(alias, "linear", 1)
    assert diagram["diagram_type"] == graphs.TYPE_ALIASES[alias]
    assert not problems("ir", diagram)
