"""SPICE decks through ngspice: the classifier on fixed transcripts, and the binary when present."""

from __future__ import annotations

import pytest

from src.eval import spice

NGSPICE = pytest.mark.skipif(not spice.available(), reason="needs ngspice")


def test_classify_reads_ngspice_output():
    assert spice.classify(0, "Total analysis time (seconds) = 0.001")[1] == "spice.ok"
    assert (
        spice.classify(1, "Error on line 4 or its substitute:\n  q1 a b")[1] == "spice.parse_error"
    )
    assert spice.classify(0, "Warning: singular matrix:  check node 2")[1] == "spice.no_convergence"
    assert spice.classify(0, "Error: no simulations run!")[1] == "spice.parse_error"


def test_unavailable_is_not_a_pass(monkeypatch):
    monkeypatch.setattr(spice, "BINARY", spice.TOOL_DIR / "does-not-exist")
    assert spice.check("* x\nV1 1 0 5\nR1 1 0 1k\n.op\n.end\n")["kind"] == "spice.unavailable"


@NGSPICE
@pytest.mark.parametrize(
    ("deck", "kind"),
    [
        ("* ok\nV1 1 0 DC 5\nR1 1 2 1k\nR2 2 0 1k\n.op\n.end\n", "spice.ok"),
        ("* model\nV1 1 0 5\nR1 1 2 1k\nQ1 a b\n.op\n.end\n", "spice.parse_error"),
        ("* float\nV1 1 0 5\nR1 2 3 1k\n.op\n.end\n", "spice.no_convergence"),
        ("* loop\nV1 1 0 5\nR1 1 2 1k\nL1 2 0 1m\nL2 2 0 1m\n.op\n.end\n", "spice.no_convergence"),
    ],
)
def test_real_simulator_verdicts(deck, kind):
    assert spice.check(deck)["kind"] == kind


@NGSPICE
def test_check_many_keeps_order():
    decks = ["* a\nV1 1 0 5\nR1 1 0 1k\n.op\n.end\n", "* b\nV1 1 0 5\nR1 2 3 1k\n.op\n.end\n"]
    assert [r["ok"] for r in spice.check_many(decks, workers=2)] == [True, False]
