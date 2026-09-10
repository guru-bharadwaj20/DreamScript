"""Tests for 11.1.1 - `src.rl.state`: the encoder, the compiled graph, and the self-loop fix.

The corpus fixtures are small hand-built IR documents rather than real pages, so the tests state
their own facts instead of depending on a data directory. Three of them (`test_roots_ignores_*`,
`test_unresolved_*`, `test_garbled_text_*`) are regression tests for things the real corpus
contains and the inherited module got wrong or never exercised.
"""

from __future__ import annotations

import dataclasses

import pytest

from src.rl.state import (
    ROLE_VOCAB,
    UNKNOWN_ROLE,
    DiagramGraph,
    StateEncoder,
    TraversalState,
    graph_of,
    role_index,
)


def ir(nodes, edges, diagram_id="d", **extra):
    """A minimal IR document in the shape `src.parse.sequences.load_ir` yields."""
    return {
        "id": diagram_id,
        "nodes": [
            {
                "id": n[0],
                "text": n[1] if len(n) > 1 else "",
                "semantic_role": n[2] if len(n) > 2 else None,
            }
            for n in nodes
        ],
        "edges": [
            {"id": f"e{i}", "src": a, "dst": b, "directed": True} for i, (a, b) in enumerate(edges)
        ],
        **extra,
    }


LINE = ir(
    [("a", "start", "start"), ("b", "work", "process"), ("c", "end", "terminal")],
    [("a", "b"), ("b", "c")],
)
DIAMOND = ir(
    [
        ("a", "start", "start"),
        ("b", "yes", "process"),
        ("c", "no", "process"),
        ("d", "join", "terminal"),
    ],
    [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")],
)
CYCLE = ir([("a", "one"), ("b", "two"), ("c", "three")], [("a", "b"), ("b", "c"), ("c", "b")])


# -- role vocabulary --------------------------------------------------------------------


def test_role_vocab_is_states_plus_unknown():
    assert len(ROLE_VOCAB) == 10
    assert ROLE_VOCAB[-1] == "unknown"
    assert UNKNOWN_ROLE == 9


@pytest.mark.parametrize("role", [None, "", "nonsense-role"])
def test_unknown_roles_share_the_last_slot(role):
    assert role_index(role) == UNKNOWN_ROLE


def test_known_states_map_to_themselves():
    for i, name in enumerate(ROLE_VOCAB[:-1]):
        assert role_index(name) == i


def test_annotation_vocabulary_is_routed_not_dropped():
    """`container`/`event`/`fork` are 2.1.5 labels, not 7.3.1 states; they must still map."""
    for raw in ("container", "event", "fork"):
        assert role_index(raw) != UNKNOWN_ROLE


# -- compilation ------------------------------------------------------------------------


def test_from_ir_indexes_nodes_and_edges():
    graph = DiagramGraph.from_ir(LINE)
    assert graph.n_nodes == 3
    assert graph.n_edges == 2
    assert graph.node_ids == ("a", "b", "c")
    assert graph.successors[graph.index_of["a"]] == (graph.index_of["b"],)
    assert graph.predecessors[graph.index_of["c"]] == (graph.index_of["b"],)


def test_edges_to_missing_nodes_are_dropped_not_crashed():
    """Real IR carries unresolved edges with `src`/`dst` None. They must not index anything."""
    doc = ir([("a",), ("b",)], [("a", "b")])
    doc["edges"] += [
        {"id": "x", "src": None, "dst": "b"},
        {"id": "y", "src": "a", "dst": None},
        {"id": "z", "src": "ghost", "dst": "b"},
    ]
    graph = DiagramGraph.from_ir(doc)
    assert graph.n_edges == 1


def test_empty_diagram_compiles():
    graph = DiagramGraph.from_ir({"nodes": [], "edges": []})
    assert graph.n_nodes == 0
    assert graph.full_mask == 0
    assert graph.initial_state().current == 0


def test_full_mask_covers_every_node():
    graph = DiagramGraph.from_ir(DIAMOND)
    assert graph.full_mask == (1 << 4) - 1
    assert int(graph.full_mask).bit_count() == graph.n_nodes


def test_back_edges_come_from_traversal():
    graph = DiagramGraph.from_ir(CYCLE)
    assert graph.back_edges, "a 3-node cycle must expose at least one back edge"
    for a, b in graph.back_edges:
        assert 0 <= a < graph.n_nodes and 0 <= b < graph.n_nodes


def test_components_label_every_node():
    doc = ir([("a",), ("b",), ("c",), ("d",)], [("a", "b"), ("c", "d")])
    graph = DiagramGraph.from_ir(doc)
    assert len(graph.component_of) == 4
    assert graph.n_components == 2


# -- the self-loop bug in roots ---------------------------------------------------------


def test_roots_ignores_self_loops():
    """A node whose only predecessor is itself is still an entry point.

    The inherited `roots()` tested `not predecessors[i]`, so `a` here was never a root and the
    episode started at whichever node `gold_order` happened to put first - from which `a` is
    unreachable. Measured over the corpus this cost gold play full coverage on 811 diagrams.
    """
    doc = ir([("a",), ("b",)], [("a", "a"), ("a", "b")])
    graph = DiagramGraph.from_ir(doc)
    assert graph.index_of["a"] in graph.roots()
    assert graph.roots()[0] == graph.index_of["a"]


def test_roots_prefers_zero_in_degree():
    graph = DiagramGraph.from_ir(LINE)
    assert graph.roots()[0] == graph.index_of["a"]


def test_roots_falls_back_when_every_node_has_a_predecessor():
    """A pure cycle has no zero-in-degree node; `roots` must still return something."""
    doc = ir([("a",), ("b",)], [("a", "b"), ("b", "a")])
    graph = DiagramGraph.from_ir(doc)
    assert len(graph.roots()) > 0
    assert graph.initial_state().current in range(graph.n_nodes)


def test_initial_state_visits_only_its_start():
    graph = DiagramGraph.from_ir(LINE)
    state = graph.initial_state()
    assert state.n_visited() == 1
    assert state.has_visited(state.current)
    assert state.n_emitted() == 0


# -- unresolved-edge flags --------------------------------------------------------------


def test_unresolved_edges_set_flags_and_count():
    doc = ir([("a",), ("b",)], [("a", "b")])
    doc["unresolved_edges"] = [{"src": "a", "dst": None, "reason": "no-target"}]
    graph = DiagramGraph.from_ir(doc)
    assert graph.n_unresolved == 1
    assert graph.unresolved_out[graph.index_of["a"]]


def test_no_unresolved_edges_means_all_flags_false():
    graph = DiagramGraph.from_ir(LINE)
    assert graph.n_unresolved == 0
    assert not any(graph.unresolved_out)
    assert not any(graph.unresolved_in)


# -- TraversalState ---------------------------------------------------------------------


def test_state_is_frozen_and_hashable():
    state = TraversalState(current=0, visited=1)
    assert hash(state) == hash(TraversalState(current=0, visited=1))
    with pytest.raises(dataclasses.FrozenInstanceError):
        state.current = 3  # type: ignore[misc]


def test_bitmask_helpers_agree_with_counts():
    state = TraversalState(current=1, visited=0b1011, emitted=0b0010, loop_marked=0b1000)
    assert state.n_visited() == 3
    assert state.n_emitted() == 1
    assert state.has_visited(0) and state.has_visited(1) and not state.has_visited(2)
    assert state.has_emitted(1) and not state.has_emitted(0)
    assert state.is_loop_marked(3)


def test_emit_sequence_preserves_order_the_bitmask_loses():
    state = TraversalState(current=0, emitted=0b111, emit_sequence=(2, 0, 1))
    assert state.emitted_order() == [2, 0, 1]
    assert state.n_emitted() == 3


# -- StateEncoder -----------------------------------------------------------------------


def test_encoder_width_is_28_and_constant():
    widths = {
        len(StateEncoder(DiagramGraph.from_ir(doc)).feature_names)
        for doc in (LINE, DIAMOND, CYCLE, {"nodes": [], "edges": []})
    }
    assert widths == {28}


def test_encoder_feature_names_are_unique():
    names = StateEncoder(DiagramGraph.from_ir(DIAMOND)).feature_names
    assert len(names) == len(set(names))


def test_features_match_declared_width():
    graph = DiagramGraph.from_ir(DIAMOND)
    encoder = StateEncoder(graph)
    features = encoder.features(graph.initial_state())
    assert len(features) == len(encoder.feature_names)


def test_every_feature_is_in_unit_interval():
    """The observation space is Box(0, 1); a feature outside it would silently break 11.2.5."""
    for doc in (LINE, DIAMOND, CYCLE):
        graph = DiagramGraph.from_ir(doc)
        encoder = StateEncoder(graph)
        for node in range(graph.n_nodes):
            state = TraversalState(
                current=node,
                visited=graph.full_mask,
                emitted=graph.full_mask,
                loop_marked=graph.full_mask,
                stack=tuple(range(graph.n_nodes)),
                steps=999,
            )
            assert all(0.0 <= v <= 1.0 for v in encoder.features(state))


def test_features_on_empty_diagram_do_not_crash():
    graph = DiagramGraph.from_ir({"nodes": [], "edges": []})
    encoder = StateEncoder(graph)
    assert len(encoder.features(graph.initial_state())) == 28


def test_role_one_hot_has_exactly_one_hot():
    graph = DiagramGraph.from_ir(LINE)
    encoder = StateEncoder(graph)
    features = encoder.features(graph.initial_state())
    assert sum(features[: len(ROLE_VOCAB)]) == pytest.approx(1.0)


def test_features_are_deterministic():
    graph = DiagramGraph.from_ir(DIAMOND)
    encoder = StateEncoder(graph)
    state = graph.initial_state()
    assert encoder.features(state) == encoder.features(state)


def test_vector_is_a_mutable_copy():
    graph = DiagramGraph.from_ir(LINE)
    encoder = StateEncoder(graph)
    vector = encoder.vector(graph.initial_state())
    assert isinstance(vector, list)
    vector[0] = 99.0  # must not raise


def test_garbled_text_does_not_break_compilation():
    """OCR output is not clean. Blank, unicode and control-character labels must all compile."""
    doc = ir(
        [("a", ""), ("b", "  \t "), ("c", "→§±"), ("d", "x" * 400)],
        [("a", "b"), ("b", "c"), ("c", "d")],
    )
    graph = DiagramGraph.from_ir(doc)
    assert graph.n_nodes == 4
    assert len(StateEncoder(graph).features(graph.initial_state())) == 28


# -- caching ----------------------------------------------------------------------------


def test_graph_of_caches_equal_documents():
    first = graph_of(LINE)
    second = graph_of(dict(LINE))
    assert first is second


def test_graph_of_matches_direct_construction():
    assert graph_of(DIAMOND).node_ids == DiagramGraph.from_ir(DIAMOND).node_ids
