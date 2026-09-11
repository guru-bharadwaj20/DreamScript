"""Phase 12.1.7 - the quality filter, driven with targets whose defect is known.

The postcondition test builds a deliberately half-invalid corpus and asserts that what survives
passes every check again. The tool-stage tests are the ones that matter most: each case is
clean to the static checker and must still be rejected by the real tool behind it.
"""

from __future__ import annotations

import pytest

from src.codegen import pairs, quality
from src.eval import react, spice
from src.synth import graphs

NODE = pytest.mark.skipif(not react.toolchain_ready(), reason="needs node + esbuild + react")
NGSPICE = pytest.mark.skipif(not spice.available(), reason="needs ngspice")


def record(diagram_type: str = "flowchart", structure: str = "branching", seed: int = 1) -> dict:
    built = pairs.make_pair(graphs.random_diagram(diagram_type, structure, seed), "synthetic")
    return pairs.assign_splits([built])[0]


def test_compile_catches_what_ast_parse_accepts():
    import ast

    for code in ("return 1\n", "def f(a, a):\n    pass\n"):
        ast.parse(code)
        assert quality.check_python(code)[1] == "python_compile"


@pytest.mark.parametrize(
    "ddl",
    [
        "CREATE TABLE a (id INTEGER PRIMARY KEY, b_id INTEGER REFERENCES nosuch(id));",
        "CREATE TABLE t (id INTEGER PRIMARY KEY, x BANANA);",
        "SELECT 1;",
    ],
)
def test_sql_goes_through_the_fk_resolving_checker(ddl):
    import sqlite3

    sqlite3.connect(":memory:").executescript(ddl)  # the draft's check, which accepts all three
    assert not quality.check_sql(ddl)[0]


@pytest.mark.parametrize(
    "code",
    [
        "V1 in 0 5\nR1 in out 1k\nR2 out 0 1k\n",
        "* x\nZ1 a b 1k\nR1 a 0 1k\n.end\n",
        "* x\nV1 in 0 5\nR1 in out 1k\n.end\n",
    ],
)
def test_structural_spice_rules(code):
    assert not quality.check_spice(code)[0]


@NODE
@NGSPICE
def test_self_test_has_no_false_accepts_or_false_rejects():
    report = quality.self_test()
    assert report["errors"] == []
    assert report["by_language"]["react"]["caught_only_by_tool"] == 3
    assert report["by_language"]["spice"]["caught_only_by_tool"] == 3


@NGSPICE
def test_parallel_inductors_are_rejected_by_the_simulator_not_the_parser():
    deck = "* x\nV1 1 0 DC 5\nR1 1 2 1k\nL1 2 0 1m\nL2 2 0 1m\n.op\n.end\n"
    assert quality.check_spice(deck)[0]
    assert quality.check_code(deck, "spice")["kind"] == "spice.no_convergence"


@NODE
def test_undefined_identifier_is_rejected_at_render():
    code = "export default function P() {\n  return (\n    <div>{items.length}</div>\n  );\n}\n"
    assert quality.check_react(code)[0]
    assert quality.check_code(code, "react")["kind"] == "react.render_error"


def test_a_missing_tool_rejects_rather_than_passes(monkeypatch):
    monkeypatch.setattr(react, "toolchain_ready", lambda: False)
    monkeypatch.setattr(spice, "available", lambda: False)
    good_react = "export default function P() {\n  return (\n    <div />\n  );\n}\n"
    good_spice = "* d\nV1 in 0 5\nR1 in out 1k\nR2 out 0 1k\n.op\n.end\n"
    kinds = [
        r["kind"] for r in quality.check_many_code([(good_react, "react"), (good_spice, "spice")])
    ]
    assert kinds == ["react.unavailable", "spice.unavailable"]


def test_unknown_language_and_schema_failures_are_rejected():
    assert quality.check_code("x", "cobol")["kind"] == "unsupported_language"
    bad = record()
    bad["split"] = "val"
    assert quality.check(bad)["kind"] == "schema"


@NODE
@NGSPICE
def test_postcondition_holds_on_a_half_broken_corpus():
    good = [
        record(t, s, i) for t in graphs.DIAGRAM_TYPES for s in ("linear", "nested") for i in (1, 2)
    ]
    broken = []
    for rec in good:
        clone = dict(rec)
        clone["target_code"] = (
            rec["target_code"]
            .replace("return", "retrun (", 1)
            .replace("</", "<", 1)
            .replace(".end", "", 1)
            .replace("CREATE TABLE", "CREATE TABEL", 1)
        )
        broken.append(clone)
    kept, rejected = quality.filter_pairs(good + broken)
    assert quality.assert_all_compile(kept) == len(kept)
    assert all(r["record"] in broken or r["kind"] == "spice.no_convergence" for r in rejected)
    assert len(rejected) >= len(broken)


def test_assert_all_compile_raises_on_a_bad_survivor():
    bad = record()
    bad["target_code"] = "def f(:\n"
    with pytest.raises(quality.QualityError):
        quality.assert_all_compile([bad])
