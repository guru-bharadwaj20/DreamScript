"""Phase 16.1.2 - streaming stage-by-stage progress, all three hops of it.

One row spanning three modules, and its tests are together because the guarantee is end to end:

  * `src.pipeline.core` observes its own stages, and cannot be broken by the observer,
  * `src.serve.api` turns that into server-sent events off a worker thread,
  * `app.backend` relays them, adds only what is its own, and stores the result.

The claim worth defending is the one in the docstrings: **this backend invents no progress**. A
relay that advanced a bar on a timer would pass any test that only checked "frames arrive", so the
tests here check that the stage frames are the upstream's own, that the two frames the backend adds
are the only two it adds, and that the id in the `result` frame is readable the instant a client
sees it.

No model is loaded. The pipeline is driven with stubs, and the relay is driven against a fake
upstream that emits a recorded frame sequence.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from app.backend.main import build_app  # noqa: E402
from app.backend.store import Store  # noqa: E402
from app.backend.stream import Reader, comment, frame, parse  # noqa: E402
from app.backend.upstream import Upstream, _error_lines  # noqa: E402
from src.pipeline.cache import StageCache  # noqa: E402
from src.pipeline.contracts import UNKNOWN, Outcome  # noqa: E402
from src.pipeline.core import STAGES, DreamScriptPipeline  # noqa: E402

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64

RESULT = {
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


def sse(events: list[tuple[str, dict]], *, pings: int = 0, final_blank: bool = True) -> list[str]:
    """A recorded model-server stream, as `httpx.aiter_lines()` would deliver it: no newlines."""
    lines: list[str] = []
    for index, (name, payload) in enumerate(events):
        if pings and index == 1:
            lines.append(": ping")
            lines.append("")
        lines.append(f"event: {name}")
        lines.append(f"data: {json.dumps(payload)}")
        lines.append("")
    if not final_blank and lines and lines[-1] == "":
        lines.pop()
    return lines


class StreamingUpstream(Upstream):
    """A model server that emits a script. Records what it was asked, so the relay can be checked
    against it rather than against a guess."""

    def __init__(self, lines: list[str], *, reachable: bool = True) -> None:
        super().__init__("http://model:8000")
        self.lines = lines
        self.reachable = reachable
        self.calls: list[tuple[int, str, str]] = []

    def health(self) -> dict:
        return {
            "reachable": self.reachable,
            "url": self.url,
            "status": 200 if self.reachable else 0,
        }

    async def stream(self, body: bytes, filename: str, content_type: str):
        self.calls.append((len(body), filename, content_type))
        for line in self.lines:
            yield line


def read(response) -> list[tuple[str, dict]]:
    """Every frame in a streamed response, in order."""
    return parse(response.iter_lines())


def wired(tmp_path, lines: list[str], **kw):
    up = StreamingUpstream(lines, **kw)
    store = Store(tmp_path)
    return TestClient(build_app(store=store, upstream=up)), store, up


# == the pipeline observes its own stages ======================================================


@pytest.fixture()
def page(tmp_path):
    """A real file on disk.

    Not a bare path: `run` takes `image_key(path)` before stage one, so a path that does not exist
    stops at `load` and never reaches the stages these tests are about. The first version of this
    file passed `tmp_path / "x.png"` and every observer test saw one `load` frame - the code was
    right and the fixture was not.
    """
    path = tmp_path / "page.png"
    path.write_bytes(PNG)
    return path


def _stubbed(**over) -> DreamScriptPipeline:
    """A pipeline whose stages answer instantly, so the observer can be tested with no model."""
    over.setdefault("cache", StageCache(enabled=False))
    pipeline = DreamScriptPipeline(**over)
    pipeline._detect = lambda path: Outcome(value=[{"cls": "circle"}])  # type: ignore[method-assign]
    pipeline._classify = lambda boxes: Outcome(  # type: ignore[method-assign]
        value=("flowchart", 0.99), confidence=0.99
    )
    pipeline._assemble = lambda path, boxes, kind="": Outcome(  # type: ignore[method-assign]
        value={"nodes": [{"id": "n1"}], "edges": []}
    )
    pipeline._traverse = lambda diagram: Outcome(value=["n1"])  # type: ignore[method-assign]
    pipeline._serialise = lambda diagram, order: Outcome(value="ir")  # type: ignore[method-assign]
    pipeline._generate = lambda d, o, k: Outcome(value=("print(1)", "python"))  # type: ignore[method-assign]
    pipeline._verify = lambda code, kind: Outcome(value=True)  # type: ignore[method-assign]
    return pipeline


def test_the_observer_declares_the_stage_list_before_any_stage_runs(page):
    """A phone must be able to draw seven greyed-out rows, not guess them."""
    seen: list[dict] = []
    _stubbed().run(page, observer=seen.append)
    assert seen[0]["event"] == "run_started"
    assert seen[0]["stages"] == list(STAGES)


def test_every_stage_is_announced_before_it_runs_and_reported_after(page):
    seen: list[dict] = []
    result = _stubbed().run(page, observer=seen.append)
    started = [e["stage"] for e in seen if e["event"] == "stage_started"]
    finished = [e["stage"] for e in seen if e["event"] == "stage_finished"]
    assert started == list(STAGES)
    assert finished == list(STAGES)
    assert [s.name for s in result.stages] == list(STAGES)
    assert seen[-1]["event"] == "run_finished"


def test_a_stage_finished_frame_carries_the_whole_table_row(page):
    """The stream and `timing_table()` must not be able to tell a client two different stories."""
    seen: list[dict] = []
    result = _stubbed().run(page, observer=seen.append)
    streamed = [
        {k: v for k, v in e.items() if k not in ("event", "index")}
        for e in seen
        if e["event"] == "stage_finished"
    ]
    assert streamed == result.timing_table()


def test_the_index_lets_a_client_show_three_of_seven(page):
    seen: list[dict] = []
    _stubbed().run(page, observer=seen.append)
    for event in seen:
        if event["event"] == "stage_started":
            assert event["index"] == STAGES.index(event["stage"]) + 1


def test_the_run_ends_on_the_confidence_gate_with_its_reason(page):
    """13.5's early return must still close the stream, or a client waits for nothing."""
    pipeline = _stubbed(confidence_floor=0.99)
    pipeline._classify = lambda boxes: Outcome(  # type: ignore[method-assign]
        value=("flowchart", 0.5), confidence=0.5
    )
    seen: list[dict] = []
    pipeline.run(page, observer=seen.append)
    assert seen[-1]["event"] == "run_finished"
    assert seen[-1]["stopped_at"] == "classify"
    assert seen[-1]["needs_confirmation"] is True
    assert "below the" in seen[-1]["reason"]


def test_an_unknown_type_also_closes_the_stream(page):
    pipeline = _stubbed()
    pipeline._classify = lambda boxes: Outcome(  # type: ignore[method-assign]
        value=(UNKNOWN, 0.0), confidence=0.0
    )
    seen: list[dict] = []
    pipeline.run(page, observer=seen.append)
    assert seen[-1]["event"] == "run_finished"


def test_a_failing_stage_closes_the_stream_after_reporting_itself(page):
    pipeline = _stubbed()
    pipeline._assemble = lambda path, boxes, kind="": Outcome.failed(  # type: ignore[method-assign]
        "no nodes survived assembly"
    )
    seen: list[dict] = []
    pipeline.run(page, observer=seen.append)
    failed = [e for e in seen if e["event"] == "stage_finished" and not e["ok"]]
    assert failed and failed[-1]["stage"] == "assemble"
    assert seen[-1] == {"event": "run_finished", "stopped_at": "assemble"}


def test_an_unreadable_page_is_still_reported_to_the_observer(tmp_path):
    """The one case that legitimately never reaches stage one."""
    seen: list[dict] = []
    DreamScriptPipeline().run(tmp_path / "does-not-exist.png", observer=seen.append)
    assert seen[0]["event"] == "run_started"
    assert seen[-1]["stage"] == "load"
    assert "unreadable" in seen[-1]["reason"]


def test_an_observer_that_raises_cannot_lose_the_page(page):
    """The observer is a socket belonging to a phone that can walk into a lift."""

    def hostile(event):
        raise RuntimeError("broken pipe")

    result = _stubbed().run(page, observer=hostile)
    assert result.ok
    assert result.code == "print(1)"
    assert [s.name for s in result.stages] == list(STAGES)


def test_a_run_with_no_observer_takes_the_same_path(page):
    with_observer = _stubbed().run(page, observer=lambda event: None)
    without = _stubbed().run(page)
    assert without.code == with_observer.code
    assert [s.name for s in without.stages] == [s.name for s in with_observer.stages]


def test_a_cached_stage_is_reported_as_cached_rather_than_silently_skipped(tmp_path, page):
    """A second run looking instant is the cache working, and the client should be able to say so."""
    cache = StageCache(tmp_path / "cache")
    _stubbed(cache=cache).run(page)
    seen: list[dict] = []
    _stubbed(cache=cache).run(page, observer=seen.append)
    cached = [e for e in seen if e["event"] == "stage_finished" and e.get("cached")]
    assert cached, "a warm cache reported nothing as cached"
    assert all(e["ok"] for e in cached)


# == the framing =============================================================================


def test_a_frame_is_terminated_by_a_blank_line():
    text = frame("stage_started", {"stage": "assemble"})
    assert text == 'event: stage_started\ndata: {"stage": "assemble"}\n\n'


def test_a_comment_is_a_colon_and_is_not_a_frame():
    assert comment() == ": ping\n\n"
    assert parse([": ping", ""]) == []


def test_parse_reads_back_what_frame_wrote():
    wire = frame("a", {"n": 1}) + frame("b", {"n": 2})
    assert parse(wire.splitlines()) == [("a", {"n": 1}), ("b", {"n": 2})]


def test_a_stream_that_ends_without_its_final_blank_line_keeps_its_last_frame():
    """Dropping the `result` frame because the socket closed a byte early would lose the answer."""
    lines = ["event: result", 'data: {"ok": true}']
    assert parse(lines) == [("result", {"ok": True})]


def test_a_malformed_data_line_is_dropped_rather_than_raised_on():
    lines = ["event: a", "data: {not json", "", "event: b", "data: {}", ""]
    assert parse(lines) == [("b", {})]


def test_the_reader_yields_a_frame_the_moment_its_blank_line_lands():
    """`parse` takes a whole stream, which is what a test has and a relay never does."""
    reader = Reader()
    assert reader.feed("event: stage_started") is None
    assert reader.feed('data: {"stage": "detect"}') is None
    assert reader.feed("") == ("stage_started", {"stage": "detect"})
    assert reader.feed("") is None


def test_the_reader_skips_comments_without_breaking_the_frame_in_progress():
    reader = Reader()
    reader.feed("event: x")
    assert reader.feed(": ping") is None
    reader.feed("data: {}")
    assert reader.feed("") == ("x", {})


def test_an_upstream_error_is_three_lines_not_one_blob():
    """Everything the relay yields is fed to `Reader` one line at a time, as httpx delivers it."""
    lines = _error_lines(503, "unreachable")
    assert all("\n" not in line for line in lines)
    assert parse(list(lines)) == [
        ("error", {"event": "error", "status": 503, "detail": "unreachable"})
    ]


# == the relay ================================================================================

SCRIPT = [
    ("run_started", {"event": "run_started", "stages": list(STAGES)}),
    ("stage_started", {"event": "stage_started", "stage": "detect", "index": 1}),
    ("stage_finished", {"event": "stage_finished", "stage": "detect", "ok": True, "seconds": 0.4}),
    ("run_finished", {"event": "run_finished", "stopped_at": None}),
    ("result", {"event": "result", **RESULT}),
]


def test_the_relay_adds_an_upload_frame_first_and_it_is_its_own_fact(tmp_path):
    client, _, _ = wired(tmp_path, sse(SCRIPT))
    response = client.post("/predict/stream", files={"image": ("page.jpg", PNG, "image/jpeg")})
    frames = read(response)
    assert frames[0][0] == "upload"
    assert frames[0][1] == {"event": "upload", "filename": "page.jpg", "bytes_in": len(PNG)}


def test_the_relay_invents_no_stage(tmp_path):
    """The whole argument of the row. A bar advanced on a timer would pass "frames arrive"."""
    client, _, _ = wired(tmp_path, sse(SCRIPT))
    response = client.post("/predict/stream", files={"image": ("p.png", PNG, "image/png")})
    frames = read(response)
    relayed = [(name, payload) for name, payload in frames if name.startswith(("stage_", "run_"))]
    assert relayed == [(name, payload) for name, payload in SCRIPT if name != "result"]


def test_the_only_two_frames_the_relay_owns_are_upload_and_the_id(tmp_path):
    client, _, _ = wired(tmp_path, sse(SCRIPT))
    frames = read(client.post("/predict/stream", files={"image": ("p.png", PNG, "image/png")}))
    names = [name for name, _ in frames]
    assert names == ["upload"] + [name for name, _ in SCRIPT]
    result = dict(frames[-1][1])
    for key, value in RESULT.items():
        assert result[key] == value, f"the relay changed {key!r}"
    assert set(result) - set(RESULT) == {
        "id",
        "created",
        "filename",
        "bytes_in",
        "upstream_seconds",
        "corrections",
        "event",
    }


def test_the_id_in_the_result_frame_is_readable_the_instant_a_client_sees_it(tmp_path):
    """An id a client can see before the file exists is one whose GET races the write."""
    client, store, _ = wired(tmp_path, sse(SCRIPT))
    frames = read(client.post("/predict/stream", files={"image": ("p.png", PNG, "image/png")}))
    record_id = frames[-1][1]["id"]
    assert record_id in store.ids()
    assert client.get(f"/ir/{record_id}").json()["ir"] == RESULT["ir"]
    assert client.get(f"/code/{record_id}?format=text").text == RESULT["code"]


def test_a_keep_alive_comment_is_relayed_rather_than_swallowed(tmp_path):
    """The proxy that would time this out sits between the client and here as often as beyond."""
    client, _, _ = wired(tmp_path, sse(SCRIPT, pings=1))
    response = client.post("/predict/stream", files={"image": ("p.png", PNG, "image/png")})
    assert any(line.startswith(":") for line in response.iter_lines())


def test_a_stream_that_ends_a_byte_early_still_stores_and_reports_the_result(tmp_path):
    client, store, _ = wired(tmp_path, sse(SCRIPT, final_blank=False))
    frames = read(client.post("/predict/stream", files={"image": ("p.png", PNG, "image/png")}))
    assert frames[-1][0] == "result"
    assert len(store.ids()) == 1


def test_the_response_is_an_event_stream_that_proxies_must_not_buffer(tmp_path):
    client, _, _ = wired(tmp_path, sse(SCRIPT))
    response = client.post("/predict/stream", files={"image": ("p.png", PNG, "image/png")})
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["x-accel-buffering"] == "no"
    assert response.headers["cache-control"] == "no-cache"


def test_an_unreachable_model_server_is_a_real_503_not_a_frame_inside_a_200(tmp_path):
    """The commonest failure here is nothing listening on MODEL_URL. Probed before the stream
    opens, because once the first byte is sent that option is gone."""
    client, _, _ = wired(tmp_path, sse(SCRIPT), reachable=False)
    response = client.post("/predict/stream", files={"image": ("p.png", PNG, "image/png")})
    assert response.status_code == 503
    assert "http://model:8000" in response.json()["detail"]


def test_a_failure_after_the_stream_opened_is_a_frame_carrying_the_status_it_would_have_been(
    tmp_path,
):
    client, store, _ = wired(
        tmp_path,
        sse(SCRIPT[:2]) + list(_error_lines(502, "pipeline failed: RuntimeError: card fell out")),
    )
    response = client.post("/predict/stream", files={"image": ("p.png", PNG, "image/png")})
    assert response.status_code == 200, "the headers left before the failure; it cannot be a status"
    frames = read(response)
    assert frames[-1][0] == "error"
    assert frames[-1][1]["status"] == 502
    assert "card fell out" in frames[-1][1]["detail"]
    assert store.ids() == [], "a failed run stored nothing"


def test_the_stream_route_refuses_what_predict_refuses(tmp_path):
    """One `_accept` for both routes: a streaming route that forgot the size cap would be a hole
    in one endpoint whose symptom points nowhere near the omission."""
    client, _, up = wired(tmp_path, sse(SCRIPT))
    bad = client.post("/predict/stream", files={"image": ("notes.txt", b"x" * 99, "text/plain")})
    assert bad.status_code == 415
    empty = client.post("/predict/stream", files={"image": ("p.png", b"", "image/png")})
    assert empty.status_code == 400
    assert up.calls == [], "the model server was asked about an upload that never should have gone"


def test_the_stream_route_guesses_the_content_type_the_same_way(tmp_path):
    client, _, up = wired(tmp_path, sse(SCRIPT))
    client.post("/predict/stream", files={"image": ("page.jpg", PNG, "application/octet-stream")})
    assert up.calls[0][2] == "image/jpeg"


def test_a_get_on_the_stream_path_is_not_mistaken_for_a_record_id(tmp_path):
    """`/predict/stream` and `/predict/{id}` share a prefix and differ only by method and shape."""
    client, _, _ = wired(tmp_path, sse(SCRIPT))
    assert client.get("/predict/stream").status_code == 404


# == the model server's own route =============================================================


def test_the_model_server_publishes_the_stream_route_as_an_event_stream():
    """The contract comes from the code, so a client can discover the media type."""
    from src.serve import api

    schema = api.__getattr__("app").openapi()
    assert "/predict/stream" in schema["paths"]
    content = schema["paths"]["/predict/stream"]["post"]["responses"]["200"]["content"]
    assert "text/event-stream" in content


def test_the_model_servers_frame_helper_matches_the_backends():
    """Two framers, one wire format. A mismatch here is a stream the relay cannot parse."""
    from src.serve.api import frame as server_frame

    assert server_frame("x", {"a": 1}) == frame("x", {"a": 1})


def test_the_heartbeat_is_under_the_usual_proxy_idle_timeout():
    from src.serve.api import HEARTBEAT_S

    assert 0 < HEARTBEAT_S < 30
