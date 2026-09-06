"""Phase 10.1.3 - the tracer's two named bugs, plus the invariants the corpus barely exercises.

Both regression tests below reconstruct the exact failure a synthetic skeleton makes visible
without needing a real page: a module constant bound as a function's default argument (so
monkeypatching the constant after import has no effect), and a junction walk that stops after
one elementary path instead of continuing through to the far end.
"""

from __future__ import annotations

from src.assemble import tracing
from src.assemble.tracing import (
    attach,
    box_distance,
    chain,
    elementary_paths,
    erase_nodes,
    node_boxes,
    to_diagram,
    trace,
)


def straight_line_with_spur(length=21, y=5, spur_x=10, spur_len=3):
    """A horizontal line with a short perpendicular spur - one real junction, one whisker."""
    pixels = {(y, x) for x in range(length)}
    pixels |= {(y + i, spur_x) for i in range(1, spur_len + 1)}
    return pixels


# ------------------------------------------------------------------------------------------
# regression: module constants bound as default args
# ------------------------------------------------------------------------------------------


def test_node_boxes_min_score_default_tracks_the_module_constant_when_patched(monkeypatch):
    boxes = [
        {"bbox": [0, 0, 10, 10], "cls": "rectangle", "score": 0.10},
        {"bbox": [20, 0, 10, 10], "cls": "rectangle", "score": 0.30},
    ]
    monkeypatch.setattr(tracing, "detections", lambda page: boxes)
    monkeypatch.setattr(tracing, "MIN_SCORE", 0.05)

    class FakePage:
        name = "p"

    # With the module constant patched down to 0.05, both boxes should pass. If `min_score`
    # had been bound at def-time to the old MIN_SCORE, this would silently keep filtering
    # with the value in effect when the module was imported and drop the 0.10-score box.
    kept = node_boxes(FakePage())
    assert len(kept) == 2


def test_node_boxes_explicit_min_score_argument_still_overrides():
    boxes = [
        {"bbox": [0, 0, 10, 10], "cls": "rectangle", "score": 0.10},
        {"bbox": [20, 0, 10, 10], "cls": "rectangle", "score": 0.30},
    ]

    class FakePage:
        name = "p"

    import src.assemble.tracing as tracing_mod

    old = tracing_mod.detections
    tracing_mod.detections = lambda page: boxes
    try:
        kept = node_boxes(FakePage(), min_score=0.5)
    finally:
        tracing_mod.detections = old
    assert kept == []


# ------------------------------------------------------------------------------------------
# regression: chaining through a junction must not stop after the first elementary path
# ------------------------------------------------------------------------------------------


def test_chain_joins_both_collinear_arms_across_a_junction_into_one_trace():
    pixels = straight_line_with_spur()
    paths = elementary_paths(pixels)
    traces = chain(paths)

    lengths = sorted(len(t) for t in traces)
    # The two collinear halves of the line (10 px either side of the junction, sharing the
    # junction pixel) must be walked into a single 19-pixel trace spanning the full line - if
    # the backward search that locates a chain's free end had marked paths as `done` before
    # the forward emission loop ran, the forward walk would see its own start already `done`
    # and stop immediately, leaving two 10-pixel stubs instead of one 19-pixel line.
    assert 19 in lengths, f"expected the two arms merged into one full-length trace, got {lengths}"
    full = max(traces, key=len)
    assert full[0] == (5, 0) or full[-1] == (5, 0)
    assert full[0] == (5, 20) or full[-1] == (5, 20)


def test_chain_leaves_the_spur_as_its_own_short_trace():
    pixels = straight_line_with_spur()
    traces = chain(elementary_paths(pixels))
    lengths = sorted(len(t) for t in traces)
    assert 3 in lengths or 4 in lengths  # spur of 3 pixels plus the shared junction pixel


# ------------------------------------------------------------------------------------------
# regression: elementary path dedup must not depend on which end the walk started from
# ------------------------------------------------------------------------------------------


def test_elementary_paths_does_not_double_count_a_path_walked_from_either_end():
    pixels = straight_line_with_spur()
    paths = elementary_paths(pixels)
    # Every physical arm is discovered twice (once from each endpoint cluster); a token that
    # disagreed with itself on reversal (e.g. sampling the walk's middle pixel, which differs
    # between a walk and its reverse when the walk has even length) would let the reversed
    # copy through and double the arm count.
    seen_pairs = [tuple(sorted(p["ends"])) for p in paths]
    assert len(seen_pairs) == len(set(seen_pairs))


# ------------------------------------------------------------------------------------------
# other behaviour
# ------------------------------------------------------------------------------------------


def test_erase_nodes_zeroes_an_inflated_box_region():
    import numpy as np

    mask = np.ones((20, 20), dtype=bool)
    out = erase_nodes(mask, [{"bbox": [5, 5, 4, 4]}], pad=0.0)
    assert not out[5:9, 5:9].any()
    assert out[0, 0]


def test_box_distance_is_zero_inside_and_positive_outside():
    bbox = [0.0, 0.0, 10.0, 10.0]
    assert box_distance((5.0, 5.0), bbox) == 0.0
    assert box_distance((20.0, 5.0), bbox) == 10.0


def test_attach_picks_the_nearest_box_within_tolerance():
    boxes = [{"id": "a", "bbox": [0, 0, 10, 10]}, {"id": "b", "bbox": [100, 0, 10, 10]}]
    assert attach((11.0, 5.0), boxes, tolerance=5.0) == "a"
    assert attach((50.0, 5.0), boxes, tolerance=5.0) is None


def test_trace_returns_no_edges_when_nothing_survives_erasure():
    import numpy as np

    class FakePage:
        name = "p"
        diagonal = 100.0

    mask = np.zeros((20, 20), dtype=bool)
    edges = trace(FakePage(), [], mask=mask)
    assert edges == []


def test_to_diagram_records_dangling_ends_as_unresolved():
    from src.ir.model import Edge

    class FakePage:
        name = "p"

    boxes = [{"id": "n0", "bbox": [0, 0, 10, 10], "cls": "rectangle"}]
    edges = [Edge(id="e0", src="n0", dst=None, polyline=[[0.0, 0.0], [5.0, 5.0]])]
    diagram = to_diagram(FakePage(), boxes, edges)
    assert diagram.edges == edges
    assert diagram.unresolved_edges, "a dangling end should be recorded as unresolved"
