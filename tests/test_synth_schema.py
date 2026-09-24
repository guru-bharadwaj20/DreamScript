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


def test_the_note_that_called_this_generator_invalid_is_no_longer_true():
    """`codegen/synthetic.py` listed four reasons it does not use `src.synth.graphs`, and the
    third was "0 of 500 of its diagrams pass schemas/ir.schema.json". That one is fixed, and the
    sentence that followed it - "it is left unchanged because the 11.2 curriculum shares it" -
    was the reason it survived three phases: a generator shared by two consumers is a generator
    whose defects reach both, and 11.2's curriculum was being fed invalid IR throughout."""
    from src.utils.config import ROOT

    text = (ROOT / "src" / "codegen" / "synthetic.py").read_text(encoding="utf-8")
    assert "0 of 500 of its diagrams pass" not in text.replace('"0 of 500', "@@")
    assert "It is left unchanged because the 11.2 curriculum shares it" not in text


def test_the_curriculum_gets_schema_valid_diagrams():
    """11.2 is the other consumer, and the one the note said could not be disturbed."""
    import inspect

    from src.ir.schema import problems
    from src.rl import curriculum

    assert "random_diagram" in inspect.getsource(curriculum), "11.2 stopped using this generator"
    for structure in graphs.STRUCTURES:
        diagram = graphs.random_diagram("flowchart", structure, 5)
        assert not problems("ir", diagram)
