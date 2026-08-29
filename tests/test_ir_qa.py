"""Phase 2.2.5 - the QA validator.

A validator is only worth having if it fails on bad input, so most of these tests build a
broken diagram and check that the right finding comes back.
"""

from __future__ import annotations

import pytest

from src.ir import qa
from src.ir.model import Diagram, Edge, Node


def good() -> Diagram:
    return Diagram(
        id="ok",
        diagram_type="flowchart",
        nodes=[
            Node("n1", "circle", [10, 10, 40, 40], "start", "start"),
            Node("n2", "rectangle", [10, 100, 80, 40], "do it", "process"),
        ],
        edges=[Edge("e1", "n1", "n2", True, "", [[30, 50], [30, 100]])],
        meta={
            "source": "test",
            "geometry": "annotated",
            "image": "",
            "image_size": [400, 300],
        },
    )


def checks(diagram: Diagram) -> dict[str, str]:
    return {f.check: f.severity for f in qa.check_diagram(diagram)}


def test_a_good_diagram_has_no_findings():
    assert qa.check_diagram(good()) == []


def test_edge_pointing_at_a_missing_node_is_an_error():
    d = good()
    d.edges[0].dst = "nope"
    assert checks(d)["dangling_edge"] == qa.ERROR


def test_open_edge_not_listed_as_unresolved_is_an_error():
    """The failure `Diagram.sync_unresolved` exists to prevent."""
    d = good()
    d.edges[0].dst = None
    assert checks(d)["unlisted_open_edge"] == qa.ERROR
    d.sync_unresolved()
    assert "unlisted_open_edge" not in checks(d)


def test_duplicate_ids_are_errors():
    d = good()
    d.nodes[1].id = "n1"
    assert checks(d)["duplicate_node_id"] == qa.ERROR


def test_box_outside_the_frame_is_an_error():
    d = good()
    d.nodes[1].bbox = [380, 100, 80, 40]
    assert checks(d)["bbox_out_of_frame"] == qa.ERROR


def test_a_container_may_overhang_a_little_but_not_a_lot():
    """Pools are drawn off the edge of the paper; a wrong coordinate scale is not."""
    d = good()
    d.nodes[1].semantic_role = "container"
    d.nodes[1].bbox = [10, 100, 430, 40]  # 30px past a 400px frame - real overhang
    assert checks(d)["container_overhangs_frame"] == qa.WARNING
    d.nodes[1].bbox = [10, 100, 800, 40]  # twice the frame - a scaling bug
    assert checks(d)["bbox_out_of_frame"] == qa.ERROR


def test_geometry_absent_means_no_boxes_at_all():
    d = good()
    d.meta["geometry"] = "absent"
    assert checks(d)["geometry_absent_but_boxed"] == qa.ERROR


def test_ambiguity_entries_must_name_real_elements():
    d = good()
    d.low_conf_text.append({"ref": "ghost", "confidence": 0.2})
    assert checks(d)["low_conf_text_ref"] == qa.ERROR


def test_missing_image_file_is_an_error():
    d = good()
    d.meta["image"] = "data/raw/does-not-exist.jpg"
    assert checks(d)["image_missing"] == qa.ERROR


def test_role_from_the_wrong_diagram_type_is_a_warning_not_an_error():
    """A mislabelled diagram_type would otherwise cascade into a wall of false errors."""
    d = good()
    d.nodes[1].semantic_role = "wire"
    result = checks(d)
    assert result["role_outside_diagram_type"] == qa.WARNING
    assert qa.ERROR not in result.values()


def test_schema_failure_short_circuits_the_rest():
    """No point reporting a dangling edge in a file that is not even valid IR."""
    d = good()
    d.diagram_type = "sudoku"
    d.edges[0].dst = "nope"
    result = qa.check_diagram(d)
    assert {f.check for f in result} == {"schema"}


def test_every_expected_warning_names_a_real_check():
    """Stops the explanations drifting away from the checks they explain."""
    d = good()
    d.nodes[1].bbox = None
    d.edges.clear()
    produced = {f.check for f in qa.check_diagram(d)}
    assert "no_edges" in produced
    assert "node_without_box" in produced
    assert set(qa.EXPECTED_WARNINGS) >= {"no_edges", "node_without_box"}


@pytest.mark.parametrize("check", sorted(qa.EXPECTED_WARNINGS))
def test_expected_warnings_have_a_stated_reason(check):
    assert len(qa.EXPECTED_WARNINGS[check]) > 30
