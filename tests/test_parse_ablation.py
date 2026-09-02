"""Phase 7.3.11 - the four-rung ladder, and which rung the gain is actually on."""

from __future__ import annotations

import pytest

from src.parse import ablation


def sequence(observations, states):
    return {
        "observations": list(observations),
        "states": list(states),
        "diagram_type": "flowchart",
    }


# -- dropping a factor -------------------------------------------------------------------------------


def test_the_symbol_keeps_its_three_parts_when_a_factor_is_dropped():
    """A two-part string would be a change of representation, not a loss of information."""
    out = ablation.rebuild([sequence(["box|other|linear"], ["process"])], drop="keyword")
    assert out[0]["observations"] == ["box|-|linear"]


@pytest.mark.parametrize(
    "factor,expected",
    [
        ("shape", "-|other|linear"),
        ("keyword", "box|-|linear"),
        ("degree", "box|other|-"),
    ],
)
def test_each_factor_is_blanked_in_its_own_position(factor, expected):
    out = ablation.rebuild([sequence(["box|other|linear"], ["process"])], drop=factor)
    assert out[0]["observations"] == [expected]


def test_dropping_nothing_returns_the_symbols_unchanged():
    original = sequence(["box|other|linear", "diamond|question|branching"], ["process", "decision"])
    assert ablation.rebuild([original])[0]["observations"] == original["observations"]


def test_dropping_a_factor_can_only_merge_symbols_never_split_them():
    """That is what makes the ablation a loss: the alphabet shrinks or stays."""
    sequences = [sequence(["box|a|linear", "box|b|linear", "diamond|a|branching"], ["process"] * 3)]
    before = {s for q in sequences for s in q["observations"]}
    after = {s for q in ablation.rebuild(sequences, drop="keyword") for s in q["observations"]}
    assert len(after) < len(before)


def test_the_states_are_carried_through_untouched():
    out = ablation.rebuild([sequence(["box|other|linear"], ["decision"])], drop="shape")
    assert out[0]["states"] == ["decision"]


def test_the_original_sequences_are_not_mutated():
    original = sequence(["box|other|linear"], ["process"])
    ablation.rebuild([original], drop="shape")
    assert original["observations"] == ["box|other|linear"]


# -- the lookup table --------------------------------------------------------------------------------


def test_the_table_maps_each_symbol_to_its_commonest_state():
    sequences = [sequence(["a", "a", "a", "b"], ["process", "process", "decision", "terminal"])]
    table = ablation.majority_map(sequences, lambda s: s)
    assert table == {"a": "process", "b": "terminal"}


def test_the_key_function_is_what_makes_it_a_bag_of_shapes():
    """The baseline rung: the same table keyed on the shape factor alone."""
    sequences = [
        sequence(
            ["box|x|linear", "box|y|branching", "diamond|z|branching"],
            ["process", "process", "decision"],
        )
    ]
    table = ablation.majority_map(sequences, lambda s: s.split("|")[0])
    assert table == {"box": "process", "diamond": "decision"}


def test_a_tie_still_produces_a_single_answer():
    sequences = [sequence(["a", "a"], ["process", "decision"])]
    assert ablation.majority_map(sequences, lambda s: s)["a"] in {"process", "decision"}


# -- the memoryless decoder --------------------------------------------------------------------------


def test_a_perfectly_separable_corpus_is_decoded_perfectly():
    sequences = [sequence(["a", "b"], ["process", "decision"]) for _ in range(20)]
    assert ablation.memoryless(sequences, 2, lambda s: s)["accuracy"] == 1.0


def test_a_symbol_never_seen_in_training_falls_back_rather_than_raising():
    """Every fold holds out pages, so unseen symbols at test time are the normal case."""
    sequences = [sequence(["a"], ["process"]) for _ in range(9)]
    sequences.append(sequence(["zzz"], ["process"]))
    assert ablation.memoryless(sequences, 2, lambda s: s)["accuracy"] > 0


def test_the_fallback_is_the_commonest_state_overall():
    sequences = [sequence(["a"], ["process"]) for _ in range(9)]
    sequences.append(sequence(["unseen"], ["decision"]))
    # The held-out `unseen` row falls back to `process`, so it is scored wrong, not skipped.
    assert ablation.memoryless(sequences, 2, lambda s: s)["accuracy"] < 1.0


def test_every_node_is_scored_exactly_once():
    sequences = [sequence(["a", "b"], ["process", "decision"]) for _ in range(10)]
    result = ablation.memoryless(sequences, 5, lambda s: s)
    assert result["accuracy"] == pytest.approx(1.0)


# -- the rungs are ordered ---------------------------------------------------------------------------


def test_the_full_symbol_is_never_worse_than_the_shape_alone():
    """The ablation's premise: more of the symbol cannot carry less information."""
    sequences = [
        sequence(["box|x|linear", "box|y|branching"], ["process", "decision"]) for _ in range(20)
    ]
    shape_only = ablation.memoryless(sequences, 2, lambda s: s.split("|")[0])
    full = ablation.memoryless(sequences, 2, lambda s: s)
    assert full["accuracy"] >= shape_only["accuracy"]
    assert full["accuracy"] == 1.0
    assert shape_only["accuracy"] < 1.0
