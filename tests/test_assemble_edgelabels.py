"""Phase 10.1.6 - reading the trigger written beside a connector.

**This file was rewritten.** It previously tested `classify`, `bind` and `match_read`, an API
`467d7e3` replaced wholesale when the module stopped guessing labels from geometry and started
reading pixels beside the polyline. Those 23 tests had been failing with `AttributeError` ever
since, which is worse than no tests: they cost a red suite and protected nothing, because the code
they exercised no longer existed.

What is tested here is the module as it is, and the parts that can go wrong silently. The decode
itself needs a 1.3 GB checkpoint and a GPU, so it is not exercised; everything around it - the ink
gate that decides what is decoded at all, and the cache that decides whether a decode happens -
is, because both have already caused real defects this project had to find the hard way.
"""

from __future__ import annotations

import json
import types

import numpy as np
import pytest

from src.assemble import edgelabels as E


class FakePage:
    def __init__(self, tmp_path, name="p1", source="fa_bresler"):
        self.name = name
        self.id = name
        self.source = source
        self._image = tmp_path / f"{name}.png"

    @property
    def image(self):
        return self._image


@pytest.fixture()
def fingerprint(monkeypatch):
    """Pin the checkpoint fingerprint so the cache tests are about the cache."""
    calls = {"value": "checkpoint-A"}
    monkeypatch.setattr(
        "src.assemble.statelabels.checkpoint_fingerprint", lambda *_: calls["value"]
    )
    return calls


@pytest.fixture()
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "CACHE", tmp_path / "edge_text")
    return tmp_path / "edge_text"


def test_the_ink_floor_is_the_lowest_value_that_was_swept():
    """The docstring's finding: precision falls as the floor rises, so it must not be tuned up."""
    assert E.MIN_INK == 0.001


def test_only_the_corpora_whose_edges_carry_labels_are_listed():
    assert set(E.SOURCES) == {"fa_bresler", "hdbpmn"}


def test_unit_falls_back_to_the_page_when_no_node_has_a_box():
    """A predicted diagram can arrive with no usable boxes; the crop rule still needs a scale."""
    unit = E._unit({"nodes": []}, (1200.0, 900.0))
    assert unit == pytest.approx(max(8.0, 900.0 / 60.0))


def test_unit_is_page_relative_when_nodes_have_boxes():
    small = E._unit({"nodes": [{"bbox": [0, 0, 60, 60]}]}, (1200.0, 900.0))
    large = E._unit({"nodes": [{"bbox": [0, 0, 600, 600]}]}, (1200.0, 900.0))
    assert large > small


def test_candidates_is_empty_when_the_page_image_is_unreadable(tmp_path):
    """A missing image must be nothing to decode, not a crash mid-assembly."""
    page = FakePage(tmp_path)  # the file is never written
    ids, patches = E.candidates(page, {"edges": [{"id": "e1", "polyline": [[0, 0], [10, 10]]}]})
    assert ids == []
    assert patches == []


def test_candidates_skips_an_edge_with_no_polyline(tmp_path, monkeypatch):
    import cv2

    page = FakePage(tmp_path)
    cv2.imwrite(str(page.image), np.full((100, 100), 255, dtype=np.uint8))
    ids, _ = E.candidates(page, {"edges": [{"id": "e1"}, {"id": "e2", "points": None}]})
    assert ids == []


def test_candidates_applies_the_ink_gate(tmp_path, monkeypatch):
    """The gate is what stops a decoder being asked to read blank paper and answering anyway."""
    import cv2

    page = FakePage(tmp_path)
    cv2.imwrite(str(page.image), np.full((100, 100), 255, dtype=np.uint8))
    diagram = {
        "nodes": [{"bbox": [0, 0, 20, 20]}],
        "edges": [
            {"id": "inked", "polyline": [[10, 10], [80, 80]]},
            {"id": "blank", "polyline": [[10, 80], [80, 10]]},
        ],
    }
    monkeypatch.setattr(
        "src.ocr.textcrops.edge_label_box", lambda *a, **k: (10, 10, 30, 30)
    )
    monkeypatch.setattr("src.ocr.textcrops.cut", lambda image, box: np.zeros((8, 8), np.uint8))
    seen = []

    def gate(image, box, polyline, unit, floor):
        seen.append(floor)
        return polyline[0] == [10, 10]  # only the first edge has ink beside it

    monkeypatch.setattr("src.ocr.textcrops.off_line_ink", gate)

    ids, patches = E.candidates(page, diagram)
    assert ids == ["inked"]
    assert len(patches) == 1
    # And the gate is asked with the swept floor, not some other number.
    assert set(seen) == {E.MIN_INK}


def test_candidates_skips_an_edge_whose_box_falls_off_the_page(tmp_path, monkeypatch):
    import cv2

    page = FakePage(tmp_path)
    cv2.imwrite(str(page.image), np.full((100, 100), 255, dtype=np.uint8))
    monkeypatch.setattr("src.ocr.textcrops.edge_label_box", lambda *a, **k: None)
    ids, _ = E.candidates(page, {"edges": [{"id": "e1", "polyline": [[0, 0], [9, 9]]}]})
    assert ids == []


def test_store_and_cached_round_trip(tmp_path, cache, fingerprint):
    page = FakePage(tmp_path)
    E.store(page, {"e1": "a", "e2": "b"})
    assert E.cached(page) == {"e1": "a", "e2": "b"}


def test_an_empty_read_is_stored_so_the_page_is_not_decoded_again(tmp_path, cache, fingerprint):
    """`cached` has to distinguish 'no file' from 'read it, found nothing'."""
    page = FakePage(tmp_path)
    assert E.cached(page) is None  # no file
    E.store(page, {})
    assert E.cached(page) == {}  # read, found none - and not None


def test_a_retrain_invalidates_the_cache(tmp_path, cache, fingerprint):
    """The defect this fingerprint exists for: 338 stale files survived two S3 retrains.

    Without it the cache outlives the model, S5 re-scores text the previous checkpoint produced,
    and the report comes back byte-identical - which reads as "the change did nothing" rather
    than "the change was never applied".
    """
    page = FakePage(tmp_path)
    E.store(page, {"e1": "a"})
    assert E.cached(page) == {"e1": "a"}

    fingerprint["value"] = "checkpoint-B"  # a retrain rewrote the weights
    assert E.cached(page) is None


def test_a_corrupt_cache_file_reads_as_absent_rather_than_raising(tmp_path, cache, fingerprint):
    page = FakePage(tmp_path)
    E.store(page, {"e1": "a"})
    path = cache / f"{page.source}__{page.id}.json"
    path.write_text("{not json", encoding="utf-8")
    assert E.cached(page) is None


def test_a_cache_file_without_labels_reads_as_absent(tmp_path, cache, fingerprint):
    page = FakePage(tmp_path)
    path = cache / f"{page.source}__{page.id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"checkpoint": "checkpoint-A"}), encoding="utf-8")
    assert E.cached(page) is None


def test_two_sources_do_not_share_a_cache_entry(tmp_path, cache, fingerprint):
    a = FakePage(tmp_path, name="same", source="fa_bresler")
    b = FakePage(tmp_path, name="same", source="hdbpmn")
    E.store(a, {"e1": "from_fa"})
    assert E.cached(b) is None


def test_apply_writes_labels_onto_dict_edges(tmp_path, cache, fingerprint):
    page = FakePage(tmp_path)
    E.store(page, {"e1": "a", "e2": "b"})
    diagram = {"edges": [{"id": "e1"}, {"id": "e2"}, {"id": "e3"}]}
    written = E.apply(page, diagram)
    assert written == 2
    assert diagram["edges"][0]["label"] == "a"
    assert diagram["edges"][1]["label"] == "b"
    assert "label" not in diagram["edges"][2]


def test_apply_writes_labels_onto_object_edges(tmp_path, cache, fingerprint):
    page = FakePage(tmp_path)
    E.store(page, {"e1": "a"})
    edge = types.SimpleNamespace(id="e1", label="")
    diagram = types.SimpleNamespace(edges=[edge], to_dict=lambda: {"edges": [{"id": "e1"}]})
    assert E.apply(page, diagram) == 1
    assert edge.label == "a"


def test_apply_reads_nothing_when_the_cache_says_the_page_is_blank(tmp_path, cache, fingerprint):
    """A stored empty result must short-circuit the decode, not fall through to it."""
    page = FakePage(tmp_path)
    E.store(page, {})

    def explode(*_a, **_k):
        raise AssertionError("read_page must not be called when the cache has an answer")

    original = E.read_page
    E.read_page = explode
    try:
        assert E.apply(page, {"edges": [{"id": "e1"}]}) == 0
    finally:
        E.read_page = original


def test_apply_returns_zero_rather_than_raising_on_an_edge_it_has_no_label_for(
    tmp_path, cache, fingerprint
):
    page = FakePage(tmp_path)
    E.store(page, {"unrelated": "x"})
    assert E.apply(page, {"edges": [{"id": "e1"}]}) == 0
