"""`src.detect.arrows` - the geometry, without a detector.

The detector is trained and measured by ultralytics; what this file pins is everything around
it, because that is where this approach can silently stop working:

* which nodes an endpoint is allowed to snap to - the container rule, which is the single
  constant that moved the inference rule's measured ceiling from 0.2128 to 0.9512 on hdbpmn;
* which end is the source - the whole reason to regress two keypoints rather than detect a box;
* the label export, whose boxes and keypoints have to stay inside the image and normalised.

`detect` is never called. These tests load no weights.
"""

from __future__ import annotations

from types import SimpleNamespace

from src.detect import arrows


def _diagram(nodes):
    return {"nodes": nodes}


def _node(node_id, bbox):
    return {"id": node_id, "bbox": bbox}


# -- who an endpoint may attach to ---------------------------------------------------------


def test_a_container_is_not_a_snap_target():
    """A BPMN pool spans the page, and distance-to-box is zero anywhere inside it.

    Left in, it is the nearest node to every point on the page and wins every arrow - measured,
    that is the difference between recovering 21.3% of hdbpmn's edges and 95.1%.
    """
    diagram = _diagram([_node("task", [10, 10, 40, 40]), _node("pool", [0, 0, 1000, 900])])
    targets = arrows.snap_targets(diagram, (1000, 1000))
    assert set(targets) == {"task"}


def test_a_node_just_under_the_threshold_is_kept():
    side = ((arrows.CONTAINER_AREA - 0.01) * 1000 * 1000) ** 0.5
    diagram = _diagram([_node("big", [0, 0, side, side])])
    assert set(arrows.snap_targets(diagram, (1000, 1000))) == {"big"}


def test_a_node_just_over_the_threshold_is_dropped():
    side = ((arrows.CONTAINER_AREA + 0.01) * 1000 * 1000) ** 0.5
    diagram = _diagram([_node("bigger", [0, 0, side, side])])
    assert arrows.snap_targets(diagram, (1000, 1000)) == {}


def test_snap_targets_applies_the_scale_it_is_given():
    diagram = _diagram([_node("n", [10, 10, 20, 20])])
    targets = arrows.snap_targets(diagram, (1000, 1000), scale=2.0)
    assert targets["n"] == [20.0, 20.0, 40.0, 40.0]


def test_a_node_without_a_box_is_not_a_target():
    diagram = _diagram([{"id": "ghost"}, _node("real", [1, 1, 5, 5])])
    assert set(arrows.snap_targets(diagram, (100, 100))) == {"real"}


# -- which end is the source ---------------------------------------------------------------


def test_the_head_decides_direction():
    targets = {"a": [0, 0, 20, 20], "b": [100, 0, 20, 20]}
    assert arrows.link((10, 10), (110, 10), targets) == ("a", "b")
    assert arrows.link((110, 10), (10, 10), targets) == ("b", "a")


def test_an_endpoint_inside_a_box_snaps_to_that_box():
    targets = {"a": [0, 0, 50, 50], "b": [200, 200, 50, 50]}
    assert arrows.link((25, 25), (225, 225), targets) == ("a", "b")


def test_a_self_loop_is_returned_rather_than_dropped():
    """Real in an automaton; the caller decides, not the geometry."""
    targets = {"a": [0, 0, 20, 20]}
    assert arrows.link((5, 5), (10, 10), targets) == ("a", "a")


def test_link_with_no_targets_is_none_rather_than_a_crash():
    assert arrows.link((0, 0), (1, 1), {}) is None


def test_the_nearer_of_two_candidates_wins():
    targets = {"near": [0, 0, 10, 10], "far": [500, 500, 10, 10]}
    assert arrows.link((12, 12), (14, 14), targets) == ("near", "near")


# -- the exported labels -------------------------------------------------------------------


def _page(width=1000, height=800, scale=1.0):
    return SimpleNamespace(
        size=(float(width), float(height)), scale=scale, id="p", name="p", source="hdbpmn"
    )


def test_a_label_line_is_normalised_and_has_both_keypoints():
    page = _page()
    diagram = {"edges": [{"polyline": [[100, 100], [400, 300]], "src": "a", "dst": "b"}]}
    lines = arrows.pose_lines(page, diagram)
    assert len(lines) == 1
    parts = lines[0].split()
    assert parts[0] == "0"
    assert len(parts) == 11  # class + box(4) + two keypoints of (x, y, v)
    assert all(0.0 <= float(v) <= 1.0 for v in parts[1:5])
    assert parts[7] == "2" and parts[10] == "2"


def test_a_horizontal_arrow_still_gets_a_box_with_area():
    """Zero-height boxes are dropped by the trainer, so the padding has to rescue them."""
    page = _page()
    diagram = {"edges": [{"polyline": [[100, 400], [700, 400]], "src": "a", "dst": "b"}]}
    parts = arrows.pose_lines(page, diagram)[0].split()
    assert float(parts[4]) > 0.0


def test_an_edge_without_a_polyline_is_not_exported():
    page = _page()
    diagram = {"edges": [{"src": "a", "dst": "b"}, {"polyline": [[1, 1]], "src": "a", "dst": "b"}]}
    assert arrows.pose_lines(page, diagram) == []


def test_the_keypoints_are_the_polyline_ends_in_order():
    page = _page(1000, 1000)
    diagram = {
        "edges": [{"polyline": [[100, 200], [500, 500], [900, 800]], "src": "a", "dst": "b"}]
    }
    parts = arrows.pose_lines(page, diagram)[0].split()
    assert abs(float(parts[5]) - 0.1) < 1e-6
    assert abs(float(parts[6]) - 0.2) < 1e-6
    assert abs(float(parts[8]) - 0.9) < 1e-6
    assert abs(float(parts[9]) - 0.8) < 1e-6


def test_the_page_scale_is_applied_to_the_polyline():
    """Annotations are in native pixels; the exported image is smaller."""
    page = _page(1000, 1000, scale=0.5)
    diagram = {"edges": [{"polyline": [[200, 400], [800, 1000]], "src": "a", "dst": "b"}]}
    parts = arrows.pose_lines(page, diagram)[0].split()
    assert abs(float(parts[5]) - 0.1) < 1e-6  # 200 * 0.5 / 1000
    assert abs(float(parts[6]) - 0.2) < 1e-6


def test_the_confidence_floor_favours_recall():
    """A missed arrow is an edge_delete and a spurious one an edge_insert, 1,251 against 325."""
    assert arrows.CONF < 0.25
