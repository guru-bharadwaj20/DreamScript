"""Tests for 11.1.6 - `src.rl.abstraction`: the bound, and that it actually holds.

The row's definition of done is "table size bounded". A bound that is asserted in prose is worth
nothing, so `test_bound_holds_under_*` drives real episodes and checks every key that comes out,
and `test_key_index_is_injective` proves the factored encoding never collides - which is the
property the rejected feature-hashing alternative could not offer.
"""

from __future__ import annotations

import itertools
import random

import pytest

from src.rl import actions as A
from src.rl.abstraction import (
    BOUND,
    FACTOR_SIZES,
    AbstractState,
    abstract,
    assert_bounded,
    key_index,
    table_bytes,
)
from src.rl.episode import Episode
from src.rl.state import ROLE_VOCAB, DiagramGraph, TraversalState
from tests.test_rl_episode import DISCONNECTED, SELF_LOOP, SINGLE
from tests.test_rl_state import CYCLE, DIAMOND, LINE

FIXTURES = [LINE, DIAMOND, CYCLE, DISCONNECTED, SELF_LOOP, SINGLE]


# -- the bound itself -------------------------------------------------------------------


def test_bound_is_the_product_of_the_factors():
    product = 1
    for size in FACTOR_SIZES:
        product *= size
    assert product == BOUND


def test_bound_is_128000():
    """The inherited docstring said 140,800 with an 11-way role factor; the vocabulary is 10."""
    assert FACTOR_SIZES[0] == len(ROLE_VOCAB) == 10
    assert BOUND == 128_000


def test_factor_count_matches_the_key_width():
    assert len(FACTOR_SIZES) == len(AbstractState._fields)


def test_table_fits_in_memory_at_the_bound():
    """9.22 MB of float64 at the bound, 0.13 MB at the measured occupancy of 1,766 keys."""
    assert table_bytes() == BOUND * A.N_ACTIONS * 8
    assert table_bytes() < 16 * 1024 * 1024
    assert table_bytes(1766) < 1024 * 1024


# -- key_index --------------------------------------------------------------------------


def test_key_index_is_inside_the_bound():
    for doc in FIXTURES:
        graph = DiagramGraph.from_ir(doc)
        assert 0 <= key_index(abstract(graph, graph.initial_state())) < BOUND


def test_key_index_is_injective():
    """No two distinct keys share an index. This is what the rejected 2^16 hash could not do.

    Checked exhaustively over a slice of the factor space rather than all 128,000 keys, which
    would be slow for no extra confidence: the encoding is a mixed-radix number, so a collision
    anywhere would appear here.
    """
    ranges = [range(min(size, 3)) for size in FACTOR_SIZES]
    seen = {}
    for combination in itertools.product(*ranges):
        key = AbstractState(*combination)
        index = key_index(key)
        assert index not in seen, f"{key} collides with {seen.get(index)}"
        seen[index] = key


def test_key_index_spans_the_extremes():
    lowest = AbstractState(*[0] * len(FACTOR_SIZES))
    highest = AbstractState(*[size - 1 for size in FACTOR_SIZES])
    assert key_index(lowest) == 0
    assert key_index(highest) == BOUND - 1


@pytest.mark.parametrize("factor", range(len(FACTOR_SIZES)))
def test_key_index_rejects_an_out_of_range_factor(factor):
    values = [0] * len(FACTOR_SIZES)
    values[factor] = FACTOR_SIZES[factor]
    with pytest.raises(ValueError):
        key_index(AbstractState(*values))


def test_key_index_rejects_negative_factors():
    values = [0] * len(FACTOR_SIZES)
    values[0] = -1
    with pytest.raises(ValueError):
        key_index(AbstractState(*values))


def test_assert_bounded_returns_the_key():
    graph = DiagramGraph.from_ir(LINE)
    key = abstract(graph, graph.initial_state())
    assert assert_bounded(key) is key


# -- the abstraction --------------------------------------------------------------------


def test_abstract_returns_a_named_tuple_of_nine_ints():
    graph = DiagramGraph.from_ir(DIAMOND)
    key = abstract(graph, graph.initial_state())
    assert isinstance(key, AbstractState)
    assert len(key) == 9
    assert all(isinstance(v, int) for v in key)


def test_abstract_is_hashable_and_deterministic():
    graph = DiagramGraph.from_ir(DIAMOND)
    state = graph.initial_state()
    assert abstract(graph, state) == abstract(graph, state)
    assert hash(abstract(graph, state)) == hash(abstract(graph, state))


def test_abstract_on_an_empty_diagram_is_bounded():
    graph = DiagramGraph.from_ir({"nodes": [], "edges": []})
    assert_bounded(abstract(graph, graph.initial_state()))


def test_abstract_carries_no_node_identity():
    """Two structurally identical diagrams with different node ids must share a key.

    This is the property that makes the table transferable; without it a Q-value learned on one
    page is useless on the next, which is the whole point of the row.
    """
    from tests.test_rl_state import ir

    first = DiagramGraph.from_ir(ir([("a", "x"), ("b", "y")], [("a", "b")], diagram_id="one"))
    second = DiagramGraph.from_ir(ir([("p", "x"), ("q", "y")], [("p", "q")], diagram_id="two"))
    assert abstract(first, first.initial_state()) == abstract(second, second.initial_state())


def test_abstract_carries_no_diagram_size():
    """A 4-node and a 40-node line must share a key at the same *relative* progress.

    "Relative" is load-bearing and the first version of this test got it wrong: one node visited
    is 1/4 on the small diagram and 1/40 on the large one, which are different fractions and
    correctly land in different buckets. Compared at the same fraction - half visited and half
    emitted - the keys must agree, which is the property that lets one Q-value serve both.
    """
    from tests.test_rl_state import ir

    def line(n):
        nodes = [(f"n{i}",) for i in range(n)]
        edges = [(f"n{i}", f"n{i + 1}") for i in range(n - 1)]
        return DiagramGraph.from_ir(ir(nodes, edges))

    def halfway(graph):
        half = graph.n_nodes // 2
        mask = (1 << half) - 1
        return TraversalState(current=0, visited=mask, emitted=mask)

    small, large = line(4), line(40)
    assert abstract(small, halfway(small)) == abstract(large, halfway(large))


def test_out_degree_bucket_saturates_at_three():
    from tests.test_rl_state import ir

    wide = DiagramGraph.from_ir(
        ir([("a",)] + [(f"n{i}",) for i in range(5)], [("a", f"n{i}") for i in range(5)])
    )
    key = abstract(wide, wide.initial_state())
    assert key.out_degree == 3


def test_emitted_flag_tracks_the_current_node():
    graph = DiagramGraph.from_ir(LINE)
    state = graph.initial_state()
    assert abstract(graph, state).current_emitted == 0
    emitted = TraversalState(current=state.current, emitted=1 << state.current)
    assert abstract(graph, emitted).current_emitted == 1


def test_unresolved_flag_is_visible_in_the_key():
    """The flag distinguishes the 530 broken-edge diagrams; dropping it was tried and rejected."""
    from tests.test_rl_state import ir

    doc = ir([("a",), ("b",)], [("a", "b")])
    doc["unresolved_edges"] = [{"src": "a", "dst": None, "reason": "no-target"}]
    graph = DiagramGraph.from_ir(doc)
    assert abstract(graph, graph.initial_state()).unresolved == 1

    clean = DiagramGraph.from_ir(ir([("a",), ("b",)], [("a", "b")]))
    assert abstract(clean, clean.initial_state()).unresolved == 0


def test_fraction_buckets_hit_both_ends():
    graph = DiagramGraph.from_ir(DIAMOND)
    nothing = TraversalState(current=0, visited=0, emitted=0)
    everything = TraversalState(current=0, visited=graph.full_mask, emitted=graph.full_mask)
    assert abstract(graph, nothing).visited_frac == 0
    assert abstract(graph, everything).visited_frac == 4
    assert abstract(graph, everything).emitted_frac == 4


# -- the bound holds under real play -----------------------------------------------------


@pytest.mark.parametrize("seed", range(4))
def test_bound_holds_under_random_play(seed):
    rng = random.Random(seed)
    for doc in FIXTURES:
        graph = DiagramGraph.from_ir(doc)
        episode = Episode(graph)
        assert_bounded(abstract(graph, episode.state))
        while not episode.done():
            episode.apply(rng.choice(episode.legal_actions()))
            assert_bounded(abstract(graph, episode.state))


@pytest.mark.parametrize("seed", range(4))
def test_bound_holds_when_the_policy_avoids_terminate(seed):
    """The long episodes are where unusual stack depths and fractions appear."""
    rng = random.Random(seed)
    for doc in FIXTURES:
        graph = DiagramGraph.from_ir(doc)
        episode = Episode(graph)
        while not episode.done():
            legal = [a for a in episode.legal_actions() if a != A.TERMINATE] or [A.TERMINATE]
            episode.apply(rng.choice(legal))
            assert_bounded(abstract(graph, episode.state))


def test_bound_holds_under_gold_play():
    for doc in FIXTURES:
        graph = DiagramGraph.from_ir(doc)
        episode = Episode(graph)
        for action in episode.gold_actions():
            if episode.done():
                break
            episode.apply(action)
            assert_bounded(abstract(graph, episode.state))


def test_abstraction_compresses_against_the_raw_state():
    """Distinct abstract keys must be strictly fewer than distinct raw states over real play."""
    rng = random.Random(0)
    abstract_keys = set()
    raw_keys = set()
    for doc in FIXTURES:
        graph = DiagramGraph.from_ir(doc)
        for _ in range(40):
            episode = Episode(graph)
            while not episode.done():
                legal = [a for a in episode.legal_actions() if a != A.TERMINATE] or [A.TERMINATE]
                episode.apply(rng.choice(legal))
                state = episode.state
                abstract_keys.add(abstract(graph, state))
                raw_keys.add(
                    (graph.diagram_id, state.current, state.visited, state.emitted, state.stack)
                )
    assert len(abstract_keys) < len(raw_keys)
    assert len(abstract_keys) <= BOUND
