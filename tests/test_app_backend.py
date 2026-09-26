"""Phase 16.1.1 - the app backend.

The pipeline is exercised by Phase 13's tests and the model server by 15.11's. What matters at
*this* boundary is what the backend adds and what it must not lose:

  * the degradation reporting 13.4 established survives the extra hop unflattened,
  * one upload becomes an id, and the reads against that id carry no image,
  * an upstream that is down, slow, wrong or not the model server at all are four different
    answers, because a client can act on the difference,
  * an id from a URL path never becomes a filesystem path,
  * a correction is appended and cannot be overwritten.

No model is loaded here. `FakeUpstream` stands in for 15.11, which is the point of `build_app`
taking its collaborators.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from app.backend import store as store_module  # noqa: E402
from app.backend.main import MAX_UPLOAD_BYTES, build_app  # noqa: E402
from app.backend.store import Store  # noqa: E402
from app.backend.upstream import Reply, Upstream  # noqa: E402

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64

#: What 15.11 answers with on a good page, in the shape `result_payload` builds. Degraded and
#: stopped_at are deliberately non-trivial in places below: a proxy that loses them is the failure
#: this file is mostly about.
GOOD = {
    "ok": True,
    "degraded": False,
    "stopped_at": None,
    "needs_confirmation": False,
    "diagram_type": "flowchart",
    "language": "python",
    "code": "def run(ctx):\n    return ctx\n",
    "ir": {"nodes": [{"id": "n1", "text": "Start"}], "edges": []},
    "traversal": ["n1"],
    "stages": [{"stage": "detect", "ok": True, "seconds": 0.4, "degraded": False}],
    "seconds": 1.23,
}


class FakeUpstream(Upstream):
    """15.11's contract without 15.11. Records what it was asked, answers what it was told to."""

    def __init__(self, reply: Reply | None = None, *, url: str = "http://model:8000") -> None:
        super().__init__(url)
        self.reply = reply if reply is not None else Reply(200, dict(GOOD), 0.5)
        self.calls: list[tuple[int, str, str]] = []

    def predict(self, body: bytes, filename: str, content_type: str) -> Reply:
        self.calls.append((len(body), filename, content_type))
        return self.reply

    def health(self) -> dict:
        return {"reachable": True, "url": self.url, "status": 200}


@pytest.fixture()
def wired(tmp_path):
    """A client, its store and its fake upstream, all three addressable by the test."""
    up = FakeUpstream()
    store = Store(tmp_path)
    return TestClient(build_app(store=store, upstream=up)), store, up


# -- health -----------------------------------------------------------------------------------


def test_health_does_not_probe_the_model_server(wired):
    """A health check that fails because *another* container is not ready restarts this one."""
    client, _, _ = wired
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert "model_server" not in body


def test_health_probes_the_model_server_when_asked(wired):
    """Opt-in, for a client that wants to show "server offline" rather than for an orchestrator."""
    client, _, _ = wired
    body = client.get("/health?upstream=1").json()
    assert body["model_server"]["reachable"] is True


def test_openapi_publishes_the_contract(wired):
    """The contract is generated from the code, so it cannot drift from it."""
    client, _, _ = wired
    paths = set(client.get("/openapi.json").json()["paths"])
    assert {"/health", "/predict", "/ir/{record_id}", "/code/{record_id}", "/feedback"} <= paths


# -- what /predict refuses before it costs an uplink ------------------------------------------


def test_a_non_image_suffix_is_refused_before_the_upload_is_read(wired):
    client, _, up = wired
    response = client.post("/predict", files={"image": ("notes.txt", b"x" * 100, "text/plain")})
    assert response.status_code == 415
    assert up.calls == [], "the model server must not be asked about a .txt file"


def test_an_empty_upload_is_400(wired):
    client, _, _ = wired
    assert client.post("/predict", files={"image": ("p.png", b"", "image/png")}).status_code == 400


def test_an_oversize_upload_is_413(wired):
    """The cap is self-defence, not abuse protection (16.1.4) - but it is a real cap."""
    client, _, up = wired
    body = b"\x89PNG\r\n\x1a\n" + b"\x00" * MAX_UPLOAD_BYTES
    response = client.post("/predict", files={"image": ("p.png", body, "image/png")})
    assert response.status_code == 413
    assert up.calls == []


def test_an_octet_stream_photograph_is_still_sent_as_an_image(wired):
    """A Blob built in JavaScript declares application/octet-stream for a perfectly good JPEG."""
    client, _, up = wired
    client.post("/predict", files={"image": ("page.jpg", PNG, "application/octet-stream")})
    assert up.calls[0][2] == "image/jpeg"


# -- the pass-through, which is the whole reason this is a proxy and not a rewrite -------------


def test_predict_returns_an_id_and_stores_the_answer(wired):
    client, store, _ = wired
    body = client.post("/predict", files={"image": ("page.png", PNG, "image/png")}).json()
    assert body["id"] in store.ids()
    assert body["bytes_in"] == len(PNG)
    assert body["code"] == GOOD["code"]


def test_degradation_survives_the_extra_hop(tmp_path):
    """13.4's rule at the second boundary. The phone is the caller with the least context."""
    degraded = dict(GOOD, degraded=True, ok=False, stopped_at="generate", needs_confirmation=True)
    client = TestClient(
        build_app(store=Store(tmp_path), upstream=FakeUpstream(Reply(200, degraded, 0.5)))
    )
    body = client.post("/predict", files={"image": ("p.png", PNG, "image/png")}).json()
    assert body["degraded"] is True
    assert body["ok"] is False
    assert body["stopped_at"] == "generate"
    assert body["needs_confirmation"] is True
    assert body["stages"] == degraded["stages"]


def test_the_backends_own_fields_sit_beside_the_upstreams_and_never_over_them(wired):
    client, _, _ = wired
    body = client.post("/predict", files={"image": ("page.png", PNG, "image/png")}).json()
    for key, value in GOOD.items():
        assert body[key] == value, f"the proxy changed {key!r}"
    assert {"id", "created", "filename", "bytes_in", "upstream_seconds"} <= set(body)


def test_a_page_that_never_reached_code_is_200_not_an_error(tmp_path):
    """15.11's rule, kept: an honest "I could not read this" is a successful answer."""
    stopped = dict(GOOD, ok=False, code=None, language=None, stopped_at="assemble")
    client = TestClient(
        build_app(store=Store(tmp_path), upstream=FakeUpstream(Reply(200, stopped, 0.2)))
    )
    response = client.post("/predict", files={"image": ("p.png", PNG, "image/png")})
    assert response.status_code == 200
    assert response.json()["stopped_at"] == "assemble"


# -- four ways for the upstream to fail, and four different answers ----------------------------


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        (Reply(503, {}, 0.0, "the model server at http://model:8000 is unreachable: x"), 503),
        (Reply(504, {}, 180.0, "did not answer within 180s"), 504),
        (Reply(415, {"detail": "unsupported file type .pdf"}, 0.1, "unsupported file type"), 415),
        (Reply(502, {}, 0.1, "answered 200 with text/html, not JSON"), 502),
    ],
)
def test_an_upstream_failure_keeps_its_classification(tmp_path, reply, expected):
    client = TestClient(build_app(store=Store(tmp_path), upstream=FakeUpstream(reply)))
    response = client.post("/predict", files={"image": ("p.png", PNG, "image/png")})
    assert response.status_code == expected
    assert reply.error.split(":")[0] in response.json()["detail"]


def test_an_unreachable_model_server_names_the_url_it_tried(tmp_path):
    """ "upstream error" is not actionable; the URL that is wrong is."""
    reply = Reply(
        503, {}, 0.0, "the model server at http://model:8000 is unreachable: ConnectError"
    )
    client = TestClient(build_app(store=Store(tmp_path), upstream=FakeUpstream(reply)))
    detail = client.post("/predict", files={"image": ("p.png", PNG, "image/png")}).json()["detail"]
    assert "http://model:8000" in detail


def test_a_failed_upstream_stores_nothing(tmp_path):
    store = Store(tmp_path)
    client = TestClient(build_app(store=store, upstream=FakeUpstream(Reply(503, {}, 0.0, "down"))))
    client.post("/predict", files={"image": ("p.png", PNG, "image/png")})
    assert store.ids() == []


def test_a_real_upstream_that_is_not_listening_is_503_not_a_traceback():
    """`Upstream` itself, against a port with nothing on it. Never raises.

    The port is bound and released rather than hardcoded, and that is not fussiness. The first
    version of this test used `127.0.0.1:1` and got **504, not 503**: Windows does not answer a
    connect to port 1 with an RST, it drops it, so the client waits out the whole timeout. The code
    was right and the test's premise was wrong - a refusal and a silence are genuinely different
    failures, which is why they are separate statuses. A port the OS has just handed back is one
    that refuses immediately.
    """
    import socket

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    reply = Upstream(f"http://127.0.0.1:{port}", timeout_s=5.0).predict(PNG, "p.png", "image/png")
    assert reply.status == 503, f"got {reply.status}: {reply.error}"
    assert reply.ok is False
    assert f"127.0.0.1:{port}" in reply.error


def test_a_model_server_that_is_up_but_silent_is_504_not_503():
    """The other half of that distinction, against a socket that accepts and never answers."""
    import socket
    import threading

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    held: list = []
    accepting = threading.Thread(target=lambda: held.append(listener.accept()), daemon=True)
    accepting.start()
    try:
        reply = Upstream(f"http://127.0.0.1:{port}", timeout_s=1.0).predict(
            PNG, "p.png", "image/png"
        )
        assert reply.status == 504, f"got {reply.status}: {reply.error}"
    finally:
        for sock, _ in held:
            sock.close()
        listener.close()


# -- the cheap reads --------------------------------------------------------------------------


def test_the_ir_read_carries_no_image_and_still_says_whether_to_trust_it(wired):
    client, _, _ = wired
    rid = client.post("/predict", files={"image": ("p.png", PNG, "image/png")}).json()["id"]
    body = client.get(f"/ir/{rid}").json()
    assert body["ir"] == GOOD["ir"]
    assert body["traversal"] == GOOD["traversal"]
    assert set(body) >= {"needs_confirmation", "degraded", "stopped_at"}
    assert "code" not in body


def test_code_as_text_is_what_a_clipboard_wants(wired):
    client, _, _ = wired
    rid = client.post("/predict", files={"image": ("p.png", PNG, "image/png")}).json()["id"]
    response = client.get(f"/code/{rid}?format=text")
    assert response.headers["content-type"].startswith("text/plain")
    assert response.text == GOOD["code"]


def test_code_as_text_on_a_page_with_no_code_is_404_not_an_empty_200(tmp_path):
    """Pasting the empty string into an editor is worse than being told there is nothing."""
    stopped = dict(GOOD, ok=False, code=None, stopped_at="classify")
    store = Store(tmp_path)
    client = TestClient(build_app(store=store, upstream=FakeUpstream(Reply(200, stopped, 0.1))))
    rid = client.post("/predict", files={"image": ("p.png", PNG, "image/png")}).json()["id"]
    response = client.get(f"/code/{rid}?format=text")
    assert response.status_code == 404
    assert "classify" in response.json()["detail"]


def test_an_unknown_format_is_refused_by_the_schema(wired):
    client, _, _ = wired
    rid = client.post("/predict", files={"image": ("p.png", PNG, "image/png")}).json()["id"]
    assert client.get(f"/code/{rid}?format=yaml").status_code == 422


def test_an_unknown_id_is_404_on_every_read(wired):
    client, _, _ = wired
    for route in ("/predict/deadbeef", "/ir/deadbeef", "/code/deadbeef", "/feedback/deadbeef"):
        assert client.get(route).status_code == 404, route


# -- an id from a URL path is not a filesystem path --------------------------------------------


@pytest.mark.parametrize("bad", ["..", "../../etc/passwd", "..%2f..%2fsecrets", "zzzz", "a" * 80])
def test_an_id_that_is_not_the_shape_ids_are_minted_in_is_refused(wired, bad):
    client, _, _ = wired
    assert client.get(f"/ir/{bad}").status_code in (404, 422)


def test_the_store_refuses_a_traversal_id_at_the_store_rather_than_the_route(tmp_path):
    """Filtered once, where every route gets it, not once per route."""
    store = Store(tmp_path)
    with pytest.raises(KeyError):
        store.path_for("../../secrets")
    assert store.get("../../secrets") is None


# -- feedback ---------------------------------------------------------------------------------


def test_a_correction_is_appended_and_readable(wired):
    client, store, _ = wired
    rid = client.post("/predict", files={"image": ("p.png", PNG, "image/png")}).json()["id"]
    first = client.post(
        "/feedback",
        json={"id": rid, "kind": "sub_text", "node": "n1", "was": "Stant", "now": "Start"},
    )
    assert first.status_code == 200
    assert first.json()["corrections"] == 1
    logged = client.get(f"/feedback/{rid}").json()["corrections"]
    assert logged[0]["now"] == "Start"
    assert logged[0]["diagram_type"] == "flowchart", "what was corrected must be recoverable"


def test_the_same_node_corrected_twice_keeps_both_events(wired):
    """A second correction must not erase the evidence that the first was needed."""
    client, _, _ = wired
    rid = client.post("/predict", files={"image": ("p.png", PNG, "image/png")}).json()["id"]
    client.post("/feedback", json={"id": rid, "kind": "sub_text", "node": "n1", "now": "Star"})
    client.post("/feedback", json={"id": rid, "kind": "sub_text", "node": "n1", "now": "Start"})
    events = client.get(f"/feedback/{rid}").json()["corrections"]
    assert [event["now"] for event in events] == ["Star", "Start"]


def test_the_correction_kinds_are_closed(wired):
    """A log that also contains "wrong" and "typo" cannot be counted against 14's taxonomy."""
    client, _, _ = wired
    rid = client.post("/predict", files={"image": ("p.png", PNG, "image/png")}).json()["id"]
    assert client.post("/feedback", json={"id": rid, "kind": "typo"}).status_code == 422


def test_feedback_against_an_unknown_id_is_404(wired):
    client, _, _ = wired
    assert client.post("/feedback", json={"id": "deadbeef", "kind": "other"}).status_code == 404


# -- the store's own two properties -----------------------------------------------------------


def test_a_record_survives_a_new_store_over_the_same_root(tmp_path):
    """The whiteboard has been wiped; an in-memory dict would have lost the page."""
    first = Store(tmp_path)
    rid = first.put(dict(GOOD))["id"]
    assert Store(tmp_path).get(rid)["result"]["code"] == GOOD["code"]


def test_the_store_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(store_module, "MAX_RECORDS", 3)
    store = Store(tmp_path)
    for _ in range(6):
        store.put(dict(GOOD))
    assert len(store.ids()) == 3


def test_eviction_never_touches_the_correction_log(tmp_path, monkeypatch):
    """The predictions are derived. The corrections are training data."""
    monkeypatch.setattr(store_module, "MAX_RECORDS", 1)
    store = Store(tmp_path)
    old = store.put(dict(GOOD))["id"]
    store.feedback(old, {"kind": "sub_text", "now": "Start"})
    for _ in range(3):
        store.put(dict(GOOD))
    assert store.get(old) is None
    assert len(store.corrections(old)) == 1


def test_a_truncated_last_line_does_not_invalidate_the_log(tmp_path):
    """An append-only file written by a process that was killed can end mid-line."""
    store = Store(tmp_path)
    rid = store.put(dict(GOOD))["id"]
    store.feedback(rid, {"kind": "sub_text", "now": "Start"})
    path = tmp_path / store_module.FEEDBACK_FILE
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"id": "' + rid + '", "kin')
    assert len(store.corrections(rid)) == 1


def test_a_truncated_record_reads_as_absent_rather_than_raising(tmp_path):
    store = Store(tmp_path)
    rid = store.put(dict(GOOD))["id"]
    store.path_for(rid).write_text("{not json", encoding="utf-8")
    assert store.get(rid) is None


def test_two_uploads_of_the_same_bytes_are_two_records(tmp_path):
    """Content addressing would silently return the first attempt's corrections for the second."""
    store = Store(tmp_path)
    assert store.put(dict(GOOD))["id"] != store.put(dict(GOOD))["id"]


def test_the_written_record_is_json_a_second_reader_can_parse(tmp_path):
    store = Store(tmp_path)
    rid = store.put(dict(GOOD), filename="page.jpg")["id"]
    on_disk = json.loads(store.path_for(rid).read_text(encoding="utf-8"))
    assert on_disk["filename"] == "page.jpg"
    assert not list(store.predictions.glob("*.tmp")), "the atomic write left its temp file behind"


# -- the module's own self-test ----------------------------------------------------------------


def test_self_test_passes_without_a_model_server():
    """`python -m app.backend.main --check` is what CI and a deployment smoke test run."""
    from app.backend.main import self_test

    result = self_test(FakeUpstream())
    assert result["ok"], json.dumps(result, indent=2, default=str)


def test_importing_the_module_does_not_build_the_app():
    """PEP 562, as in 15.11: `uvicorn app.backend.main:app` still resolves, the import is free."""
    from app.backend import main

    assert main._APP is None or isinstance(main._APP, object)
    with pytest.raises(AttributeError):
        main.__getattr__("nope")
