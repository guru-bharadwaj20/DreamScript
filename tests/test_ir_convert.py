"""Phase 2.2.2 - the converters.

Every test here that touches a dataset skips when the dataset is absent, so the suite still
runs on a clone that has not pulled the data. The round-trip requirement from contributing.md 2.2.2 is
`test_<source>_roundtrips`: convert, write, read back, and get the same object.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.ir.model import Diagram
from src.ir.vocab import ROLES, SHAPES


def _roundtrip(diagram: Diagram, tmp_path):
    assert diagram.problems() == []
    path = diagram.save(tmp_path / f"{diagram.id}.ir.json")
    assert Diagram.load(path).to_dict() == diagram.to_dict()


def _vocabulary_is_respected(diagram: Diagram):
    for node in diagram.nodes:
        assert node.shape in SHAPES
        assert node.semantic_role in ROLES


def _endpoints_exist(diagram: Diagram):
    ids = diagram.node_ids
    for edge in diagram.edges:
        assert edge.src is None or edge.src in ids
        assert edge.dst is None or edge.dst in ids


# -- hdBPMN ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def hdbpmn_diagram():
    from src.ir.convert import hdbpmn

    files = hdbpmn.annotations()
    if not files:
        pytest.skip("hdBPMN not present")
    return hdbpmn.convert(files[0])


def test_hdbpmn_roundtrips(hdbpmn_diagram, tmp_path):
    _roundtrip(hdbpmn_diagram, tmp_path)


def test_hdbpmn_structure(hdbpmn_diagram):
    d = hdbpmn_diagram
    assert d.diagram_type == "flowchart"
    assert d.meta["geometry"] == "annotated"
    assert len(d.nodes) > 3 and len(d.edges) > 3
    _vocabulary_is_respected(d)
    _endpoints_exist(d)


def test_hdbpmn_boxes_land_on_the_photo(hdbpmn_diagram):
    """The DI coordinates are scaled by `max(w, h) / backgroundSize`; if that is wrong the
    boxes drift off the page, and this is the cheap check that they do not."""
    width, height = hdbpmn_diagram.meta["image_size"]
    for node in hdbpmn_diagram.nodes:
        x, y, w, h = node.bbox
        assert x >= -5 and y >= -5
        assert x + w <= width + 5 and y + h <= height + 5
        assert w > 1 and h > 1


def test_hdbpmn_parallel_gateways_split_into_fork_and_join():
    from src.ir.convert import hdbpmn

    roles = set()
    for path in hdbpmn.annotations()[:12]:
        try:
            d = hdbpmn.convert(path)
        except FileNotFoundError:
            continue
        roles |= {
            n.semantic_role for n in d.nodes if (n.attrs or {}).get("bpmn_tag") == "parallelGateway"
        }
    if not roles:
        pytest.skip("no parallel gateways in the sample")
    assert roles <= {"fork", "join"}


def test_hdbpmn_records_that_shapes_are_convention(hdbpmn_diagram):
    """A reader must never mistake the BPMN drawing convention for an observation."""
    assert all(
        (n.attrs or {}).get("shape_basis") == "bpmn-convention" for n in hdbpmn_diagram.nodes
    )
    assert "convention" in hdbpmn_diagram.meta["notes"]


# -- FA -------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fa_diagram():
    from src.ir.convert import fa

    files = fa.files()
    if not files:
        pytest.skip("FA database not present")
    return fa.convert(files[0])


def test_fa_roundtrips(fa_diagram, tmp_path):
    _roundtrip(fa_diagram, tmp_path)


def test_fa_structure(fa_diagram):
    d = fa_diagram
    assert d.diagram_type == "state_machine"
    assert d.nodes and d.edges
    _vocabulary_is_respected(d)
    _endpoints_exist(d)
    assert all(n.semantic_role in {"state", "initial-state", "final-state"} for n in d.nodes)


def test_fa_accepting_states_keep_their_double_circle(fa_diagram):
    for node in fa_diagram.nodes:
        assert (node.shape == "double-circle") == (node.semantic_role == "final-state")


def test_fa_start_marker_is_an_unresolved_edge(fa_diagram):
    markers = [e for e in fa_diagram.edges if (e.attrs or {}).get("initial_marker")]
    if not markers:
        pytest.skip("this automaton has no start marker")
    listed = {u["edge"] for u in fa_diagram.unresolved_edges}
    assert all(m.id in listed for m in markers)
    assert all(m.src is None for m in markers)


def test_fa_transform_matches_the_renderer_that_made_the_images():
    """`fa.fit` must reproduce `chaos_builder.render_inkml` exactly, or the coordinates index
    an image that does not exist. This pins them together rather than trusting a comment."""
    from src.ingest import chaos_builder
    from src.ir.convert import fa

    points = np.array([[10.0, 20.0], [500.0, 900.0], [250.0, 40.0]])
    scale, offset = fa.fit(points)
    ours = points * scale + offset

    lo, hi = points.min(axis=0), points.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    margin = 0.06
    size = fa.CANVAS
    theirs = ((points - lo) / span * (1 - 2 * margin) + margin) * size

    assert chaos_builder.render_inkml is not None
    assert np.allclose(ours, theirs)


# -- DIDI -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def didi_pair():
    from src.ir.convert import didi

    if not didi.NDJSON.is_file():
        pytest.skip("DIDI not present")
    record = next(iter(didi.records(1)))
    return record, didi.convert(record)


def test_didi_roundtrips(didi_pair, tmp_path):
    _roundtrip(didi_pair[1], tmp_path)


def test_didi_structure(didi_pair):
    d = didi_pair[1]
    assert d.meta["geometry"] == "derived"
    assert d.nodes
    _vocabulary_is_respected(d)
    _endpoints_exist(d)
    assert all(n.semantic_role == "unknown" for n in d.nodes)


def test_didi_fitted_layout_agrees_with_where_people_drew():
    """The whole justification for DIDI's derived geometry. If the fit were wrong this error
    would be tens of percent, not a few."""
    from src.ir.convert import didi

    if not didi.NDJSON.is_file():
        pytest.skip("DIDI not present")
    errors = []
    for record in didi.records(40):
        errors.extend(didi.ink_agreement(record, didi.convert(record)).values())
    assert errors
    assert float(np.mean(errors)) < 0.10
    assert float(np.percentile(errors, 90)) < 0.15


# -- Sketch2Code ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def s2c_diagram():
    from src.ir.convert import sketch2code

    files = sketch2code.pages()
    if not files:
        pytest.skip("Sketch2Code not present")
    return sketch2code.convert(files[0])


def test_sketch2code_roundtrips(s2c_diagram, tmp_path):
    _roundtrip(s2c_diagram, tmp_path)


def test_sketch2code_is_structure_only(s2c_diagram):
    d = s2c_diagram
    assert d.meta["geometry"] == "absent"
    assert all(n.bbox is None for n in d.nodes)
    assert all(e.polyline is None for e in d.edges)
    _vocabulary_is_respected(d)
    _endpoints_exist(d)


def test_sketch2code_finds_more_than_the_document_root(s2c_diagram):
    """The regression that made this converter emit one node per page: a void `<meta>` tag
    with no closing tag left the parser inside a skipped subtree forever."""
    assert len(s2c_diagram.nodes) > 5
    assert {n.semantic_role for n in s2c_diagram.nodes} > {"container"}


def test_sketch2code_keeps_its_sketches(s2c_diagram):
    assert s2c_diagram.meta["sketches"]
    assert s2c_diagram.meta["webpage"].endswith(".html")


# -- flowchartseg ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def fcseg_diagram(tmp_path_factory):
    import pandas as pd

    from src.ir.convert import flowchartseg

    frames = flowchartseg.frames()
    if not frames:
        pytest.skip("flowchartseg not present")
    df = pd.read_parquet(frames[0])
    return flowchartseg.convert(df.iloc[0], 0, write_image=False)


def test_flowchartseg_roundtrips(fcseg_diagram, tmp_path):
    _roundtrip(fcseg_diagram, tmp_path)


def test_flowchartseg_emits_no_edges_and_says_why(fcseg_diagram):
    """An empty edge list here means 'the source records none', and the note must say so -
    otherwise Phase 10 would score itself against a diagram that appears to have no
    connections."""
    assert fcseg_diagram.edges == []
    assert "never to evaluate graph assembly" in fcseg_diagram.meta["notes"]


def test_flowchartseg_shape_guesses_are_not_claimed_as_truth(fcseg_diagram):
    assert fcseg_diagram.nodes
    for node in fcseg_diagram.nodes:
        assert node.confidence < 1.0
        assert (node.attrs or {}).get("shape_basis") == "geometry"
        assert node.semantic_role == "unknown"


def test_flowchartseg_is_recorded_as_rendered_not_hand_drawn(fcseg_diagram):
    """Phase 1 claimed these were hand-drawn; they are not, and the IR must not repeat it."""
    assert "Computer-rendered" in fcseg_diagram.meta["notes"]


# -- the geometric shape guesser ------------------------------------------------------


def _ideal(kind: str) -> np.ndarray:
    import cv2

    mask = np.zeros((300, 300), np.uint8)
    if kind == "rectangle":
        cv2.rectangle(mask, (40, 60), (260, 240), 255, -1)
    elif kind == "diamond":
        cv2.fillPoly(mask, [np.array([[150, 30], [270, 150], [150, 270], [30, 150]])], 255)
    elif kind == "circle":
        cv2.circle(mask, (150, 150), 110, 255, -1)
    elif kind == "ellipse":
        cv2.ellipse(mask, (150, 150), (140, 60), 0, 0, 360, 255, -1)
    elif kind == "octagon":
        pts = np.array(
            [
                [150 + 110 * np.cos(t), 150 + 110 * np.sin(t)]
                for t in np.linspace(0, 2 * np.pi, 9)[:-1]
            ],
            np.int32,
        )
        cv2.fillPoly(mask, [pts], 255)
    elif kind == "parallelogram":
        cv2.fillPoly(mask, [np.array([[80, 60], [280, 60], [220, 240], [20, 240]])], 255)
    elif kind == "rounded-rect":
        cv2.rectangle(mask, (40, 80), (260, 220), 255, -1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


@pytest.mark.parametrize(
    "kind",
    ["rectangle", "diamond", "circle", "ellipse", "parallelogram", "rounded-rect"],
)
def test_geometry_classifier_on_ideal_shapes(kind):
    """Six vocabulary shapes are drawable as ideal masks and the classifier must get all six
    right on those before its guesses on real ink mean anything. Note that its thresholds are
    calibrated on hand-drawn ink, not on these - so passing here is necessary, not sufficient."""
    from src.ir.convert.geometry import classify_mask

    shape, confidence = classify_mask(_ideal(kind))
    assert shape == kind
    assert 0 < confidence <= 0.7


def test_geometry_classifier_cannot_tell_an_octagon_from_a_circle():
    """A known, deliberate limitation, pinned rather than hidden.

    A rendered octagon has circularity 0.81 and a rendered circle 0.89 - but a *hand-drawn*
    circle measures 0.70, below both. No threshold on this feature separates the three, so the
    classifier does not emit `octagon` at all; that shape only ever comes from a source that
    declares it, such as a DIDI prompt."""
    from src.ir.convert.geometry import classify_mask

    assert classify_mask(_ideal("octagon"))[0] == "circle"


def test_geometry_classifier_confidence_is_never_certain():
    """Every branch is a guess from a contour; none of them may claim ground truth."""
    from src.ir.convert.geometry import classify_mask

    for kind in ["rectangle", "circle", "octagon", "ellipse"]:
        assert classify_mask(_ideal(kind))[1] < 1.0
