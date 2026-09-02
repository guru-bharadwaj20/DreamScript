"""Phase 7.3.2 - the discrete emission alphabet and its three factors."""

from __future__ import annotations

import pytest

from src.parse import observations as obs

# -- the alphabet ---------------------------------------------------------------------------------


def test_the_alphabet_is_the_product_of_the_three_factors():
    assert len(obs.full_alphabet()) == (
        len(obs.SHAPE_CLASSES) * len(obs.KEYWORD_CLASSES) * len(obs.DEGREE_CLASSES)
    )


def test_the_alphabet_is_a_hundred_and_twenty_symbols():
    """The size is the whole design constraint; a change here changes 7.3.5's matrix."""
    assert len(obs.full_alphabet()) == 120


def test_the_alphabet_has_no_duplicates():
    assert len(set(obs.full_alphabet())) == len(obs.full_alphabet())


def test_every_ir_shape_has_a_class():
    from src.ir.vocab import SHAPES

    assert set(SHAPES) <= set(obs.SHAPE_TO_CLASS)


def test_every_shape_class_is_declared():
    assert set(obs.SHAPE_TO_CLASS.values()) <= set(obs.SHAPE_CLASSES)


# -- shape ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("shape", ["rectangle", "rounded-rect", "octagon"])
def test_the_box_family_collapses(shape):
    assert obs.shape_class(shape) == "box"


@pytest.mark.parametrize("shape", ["circle", "double-circle", "ellipse"])
def test_the_round_family_collapses(shape):
    assert obs.shape_class(shape) == "round"


def test_the_diamond_stays_alone_because_it_is_the_decision_evidence():
    assert obs.shape_class("diamond") == "diamond"


def test_an_unknown_shape_falls_back_rather_than_raising():
    assert obs.shape_class("hexagram") == "other"


# -- keyword --------------------------------------------------------------------------------------


def test_a_question_mark_decides_on_its_own():
    assert obs.keyword_class("total > 0?") == "question"


def test_terminal_words_beat_io_words():
    """'print report and end' is a terminal; the rarer fact wins."""
    assert obs.keyword_class("print report and end") == "terminal-word"


def test_io_words_are_found():
    assert obs.keyword_class("read the file") == "io-word"


def test_empty_text_is_its_own_class():
    assert obs.keyword_class("") == "empty"
    assert obs.keyword_class("   ") == "empty"


def test_a_label_with_no_letters_is_other_not_empty():
    """fa_bresler labels states '0' and '1'; those are not missing text."""
    assert obs.keyword_class("0") == "other"


# -- degree ---------------------------------------------------------------------------------------


def test_a_source_is_a_source_even_when_it_branches():
    assert obs.degree_class(0, 3) == "source"


def test_a_sink_has_no_way_out():
    assert obs.degree_class(2, 0) == "sink"


def test_two_or_more_outgoing_is_branching():
    assert obs.degree_class(1, 2) == "branching"


def test_one_in_one_out_is_linear():
    assert obs.degree_class(1, 1) == "linear"


# -- the symbol -----------------------------------------------------------------------------------


def test_the_symbol_is_the_readable_triple():
    assert obs.symbol("diamond", "ok?", 1, 2) == "diamond|question|branching"


def test_the_symbol_is_always_in_the_full_alphabet():
    assert obs.symbol("freeform", "", 0, 0) in obs.full_alphabet()


def test_observe_reads_the_node_fields():
    node = {"shape": "rectangle", "text": "start here"}
    assert obs.observe({}, node, 0, 1) == "box|terminal-word|source"


def test_observe_survives_a_node_with_no_text_or_shape():
    assert obs.observe({}, {}, 1, 1) == "other|empty|linear"
