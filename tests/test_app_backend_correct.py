"""Phase 16.2.8 - the correction round trip: log it, apply it, re-emit the code.

The row calls this the highest-value screen in the app and the arithmetic is 14's: `sub_text` is 11
of hdbpmn's median 21 edits, so more than half the distance between a read page and a correct one is
a word the recogniser got wrong.

What is tested here is the part that has to be exactly right for that to be true:

  * the tap is **always logged**, whether or not it changed anything, because it is training data
    either way and `feedback.jsonl` outlives the prediction it refers to,
  * `was` is taken from **the record** and not from the request, because a client that sent the
    wrong one would poison the log with a correction of something that was never said,
  * the IR is edited in a **copy**, so a failed write cannot leave a mutated object behind,
  * the record keeps its **id**, because a correction that minted a new one would break every link
    already shared and the corrected reading is the same page,
  * and a kind that is not applied says so, rather than letting someone believe the code moved.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from app.backend.correct import APPLIED_KINDS, apply_correction, regenerate  # noqa: E402
from app.backend.main import build_app  # noqa: E402
from app.backend.store import Store  # noqa: E402
from app.backend.upstream import Upstream  # noqa: E402

IR = {
    "ir_version": "1.0",
    "id": "p",
    "diagram_type": "flowchart",
    "nodes": [
        {
            "id": "d000",
            "shape": "rectangle",
            "text": "insurer",
            "confidence": 0.41,
            "bbox": [0, 0, 10, 5],
        },
        {
            "id": "d001",
            "shape": "rectangle",
            "text": "insurer",
            "confidence": 0.9,
            "bbox": [0, 20, 10, 5],
        },
    ],
    "edges": [{"id": "a0", "src": "d000", "dst": "d001", "directed": True, "confidence": 0.8}],
}

RESULT = {
    "ok": True,
    "degraded": False,
    "stopped_at": None,
    "needs_confirmation": False,
    "diagram_type": "flowchart",
    "language": "python",
    "code": "def run(ctx):\n    ctx = insurer(ctx)\n    return ctx\n",
    "ir": IR,
    "traversal": ["d000", "d001"],
    "stages": [],
    "seconds": 1.0,
}


@pytest.fixture()
def wired(tmp_path):
    store = Store(tmp_path)
    client = TestClient(build_app(store=store, upstream=Upstream("http://model:8000")))
    record = store.put({**RESULT, "ir": {**IR, "nodes": [dict(n) for n in IR["nodes"]]}})
    return client, store, record["id"]


# == applying one, in isolation ================================================================


def test_a_text_correction_rewrites_only_the_node_it_names():
    corrected, changed = apply_correction(
        IR, {"kind": "sub_text", "node": "d000", "now": "Receive"}
    )
    assert changed is True
    texts = {n["id"]: n["text"] for n in corrected["nodes"]}
    assert texts == {"d000": "Receive", "d001": "insurer"}


def test_the_correction_is_made_on_a_copy():
    """The caller still holds the record it read. A half-failed write must not leave a mutated
    object behind claiming otherwise."""
    before = IR["nodes"][0]["text"]
    apply_correction(IR, {"kind": "sub_text", "node": "d000", "now": "something else"})
    assert IR["nodes"][0]["text"] == before


def test_a_corrected_label_stops_being_low_confidence():
    """The label is a person's now, not the recogniser's. 16.2.9 flags low-confidence text, and a
    corrected one must stop being flagged."""
    corrected, _ = apply_correction(IR, {"kind": "sub_text", "node": "d000", "now": "Receive"})
    node = next(n for n in corrected["nodes"] if n["id"] == "d000")
    assert node["confidence"] == 1.0
    assert node["corrected"] is True


def test_retyping_the_same_words_is_not_a_change():
    """A double tap, or two phones. Not an error, and not worth a regeneration."""
    _, changed = apply_correction(IR, {"kind": "sub_text", "node": "d000", "now": "insurer"})
    assert changed is False


@pytest.mark.parametrize("kind", ["sub_type", "del_node", "add_edge", "del_edge", "other"])
def test_a_kind_that_is_not_applied_changes_nothing(kind: str):
    """Only `sub_text` rewrites the IR today, and the restraint is deliberate: deleting a node or
    adding an edge changes the topology, which changes the traversal, which changes which branch
    the emitter writes - and a person retyping a word has not asked for that."""
    assert kind not in APPLIED_KINDS
    _, changed = apply_correction(IR, {"kind": kind, "node": "d000", "now": "x"})
    assert changed is False


def test_a_correction_naming_a_node_that_is_not_there_changes_nothing():
    _, changed = apply_correction(IR, {"kind": "sub_text", "node": "ghost", "now": "x"})
    assert changed is False


def test_a_correction_with_no_replacement_changes_nothing():
    _, changed = apply_correction(IR, {"kind": "sub_text", "node": "d000"})
    assert changed is False


def test_correcting_a_missing_ir_is_not_a_crash():
    assert apply_correction(None, {"kind": "sub_text", "node": "d000", "now": "x"}) == (None, False)


# == re-emitting ===============================================================================


def test_regenerate_produces_code_from_a_corrected_diagram():
    corrected, _ = apply_correction(
        IR, {"kind": "sub_text", "node": "d000", "now": "Receive order"}
    )
    code, language, _degraded, _reason = regenerate(corrected, ["d000", "d001"], "flowchart")
    assert language == "python"
    assert code and "receive_order" in code
    assert "insurer" in code, "the other node was not corrected and must be unchanged"


def test_regenerate_reports_a_type_it_has_no_language_for():
    code, language, _degraded, reason = regenerate(IR, ["d000"], "not_a_diagram_type")
    assert code is None and language is None
    assert "not_a_diagram_type" in reason


def test_the_emitter_pulls_in_no_model():
    """The app's promise about itself: this backend holds no model. The emitter is a template, and
    that is why a correction can be re-emitted in this process in milliseconds."""
    import sys

    before = set(sys.modules)
    import src.pipeline.generate  # noqa: F401

    pulled = {m.split(".")[0] for m in set(sys.modules) - before}
    assert not pulled & {"torch", "transformers", "ultralytics"}


# == the route =================================================================================


def test_the_round_trip_logs_applies_and_regenerates(wired):
    client, store, rid = wired
    body = client.post(
        f"/correct/{rid}", json={"kind": "sub_text", "node": "d000", "now": "Receive order"}
    ).json()

    assert body["correction"]["applied"] is True
    assert body["correction"]["regenerated"] is True
    assert body["id"] == rid, "a correction must not mint a new id"
    assert "receive_order" in body["code"]
    assert len(store.corrections(rid)) == 1


def test_it_is_persisted_and_survives_a_reread(wired):
    client, _, rid = wired
    client.post(
        f"/correct/{rid}", json={"kind": "sub_text", "node": "d000", "now": "Receive order"}
    )
    again = client.get(f"/predict/{rid}").json()
    texts = {n["id"]: n["text"] for n in again["ir"]["nodes"]}
    assert texts["d000"] == "Receive order"
    assert "receive_order" in again["code"]
    assert len(again["corrections"]) == 1


def test_was_is_taken_from_the_record_not_from_the_request(wired):
    """The half of a correction that makes it training data is what the recogniser actually said.
    A client that sent the wrong `was` would poison the log."""
    client, store, rid = wired
    client.post(
        f"/correct/{rid}",
        json={"kind": "sub_text", "node": "d000", "now": "Receive order", "was": "a lie"},
    )
    logged = store.corrections(rid)[0]
    assert logged["was"] == "insurer"


def test_an_unapplied_kind_is_still_logged_and_says_it_was_not_applied(wired):
    client, store, rid = wired
    body = client.post(f"/correct/{rid}", json={"kind": "del_edge", "edge": "a0"}).json()
    assert body["correction"]["applied"] is False
    assert body["correction"]["regenerated"] is False
    assert body["correction"]["applies_kinds"] == list(APPLIED_KINDS)
    assert len(store.corrections(rid)) == 1, "an unapplied correction is still training data"
    assert body["code"] == RESULT["code"], "nothing was regenerated, so nothing changed"


def test_two_corrections_to_one_node_keep_both_events(wired):
    """The second must not erase the evidence that the first was needed."""
    client, store, rid = wired
    client.post(f"/correct/{rid}", json={"kind": "sub_text", "node": "d000", "now": "Recieve"})
    client.post(f"/correct/{rid}", json={"kind": "sub_text", "node": "d000", "now": "Receive"})
    events = store.corrections(rid)
    assert [e["now"] for e in events] == ["Recieve", "Receive"]
    # And the second correction's `was` is the first correction's result, not the original.
    assert events[1]["was"] == "Recieve"


def test_an_unknown_kind_is_refused_rather_than_logged(wired):
    client, store, rid = wired
    assert client.post(f"/correct/{rid}", json={"kind": "typo", "node": "d000"}).status_code == 422
    assert store.corrections(rid) == []


def test_correcting_an_unknown_id_is_404(wired):
    client, _, _ = wired
    assert client.post("/correct/deadbeefdeadbeef", json={"kind": "sub_text"}).status_code == 404


def test_the_response_carries_the_whole_prediction_not_a_patch(wired):
    """The corrected IR, the regenerated code and the trust flags move together. A client merging
    a patch would be one refresh from showing a new label beside old code."""
    client, _, rid = wired
    body = client.post(
        f"/correct/{rid}", json={"kind": "sub_text", "node": "d000", "now": "Receive order"}
    ).json()
    for field in ("ok", "degraded", "stopped_at", "needs_confirmation", "ir", "code", "stages"):
        assert field in body, field
