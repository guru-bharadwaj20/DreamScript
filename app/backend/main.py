"""Phase 16.1.1 - the app backend: one upload, four cheap reads, and a memory.

    uvicorn app.backend.main:app --host 0.0.0.0 --port 3000
    python -m app.backend.main --check          # wire and self-test without binding a port
    python -m app.backend.main --openapi        # the published contract

    curl -F "image=@page.jpg" http://localhost:3000/predict

OpenAPI is served at `/docs` and `/openapi.json` by FastAPI from the code, so the contract cannot
drift from the implementation - the same reason 15.11 does it.

## What this is, and why it is not the model server

15.11 is the **model server**: page in, IR and code out, nothing kept. This is the **app backend**:
it keeps what a phone needs kept and adds what a phone needs added, and it holds no model at all.
It never imports torch. The only thing it asks of the model server is `POST /predict`.

    GET  /health          this process. Add ?upstream=1 to probe the model server too
    POST /predict         one page -> the model server's answer, stored under a new id
    POST /predict/stream  the same, streamed stage by stage as server-sent events (16.1.2)
    GET  /predict/{id}    the whole stored record again
    GET  /ir/{id}         just the IR, and the boxes the overlay draws
    GET  /code/{id}       just the code (?format=text for text/plain, which is what a share sheet
                          and a clipboard actually want)
    POST /feedback        a correction against an id. Appended, never overwritten

## Why `/ir` and `/code` are reads of an id, not second uploads

A page photograph is several megabytes over a mobile uplink. The app needs the boxes to draw, the
IR to lay out, the code to show, and later a regeneration after a label is fixed. Charging the user
a second 3 MB upload for each of those is not an API design, it is a phone bill. `POST /predict`
carries the image exactly once; everything after it is an id.

## `/health` does not probe the model server unless asked

15.11's `/health` deliberately does not load the weights, because a liveness probe that pulls
1.3 GB gets the container killed during its own start period. The same argument applies one hop
out and is easier to get wrong: if *this* `/health` failed because the model container was not
ready, Docker would restart this process - which was never the thing that was unhealthy - in a
loop. So the default answer is about this process, and the upstream probe is opt-in for a client
that wants to show "server offline" rather than for an orchestrator that wants to know whether to
kill something.

## Degradation is passed through, never flattened

Every response that carries a result carries the model server's own `ok`, `degraded`, `stopped_at`,
`needs_confirmation` and per-stage table, verbatim. A phone showing a person their own whiteboard
is the caller with the least context in the system and the most need for that distinction; a proxy
that helpfully reduces it to `{"code": ...}` is where an honest pipeline turns into a confident
wrong answer.

## Streaming, and what this backend does not invent

`POST /predict/stream` makes the wait legible: a cold GPU run of a fixture page is 12.08 s, of
which `assemble` alone is 9.15 s, and a spinner held for nine seconds is indistinguishable from a
stalled request.

The stages in that stream are **not this backend's**. It holds no pipeline, so the only progress it
could report on its own authority would be the expected stage names advanced on a timer - a bar that
moves because seconds passed rather than because work finished. So 16.1.2 put the observer in
`src.pipeline.core` and the SSE route on the model server, and this relays it: it adds an `upload`
frame (its own fact - the photograph arrived), adds the `id` to the `result` frame (its own, from the
store), and relays everything else, keep-alive comments included, unchanged.

## Scope

The sandbox runner (16.1.3) and rate and size limits (16.1.4) are the next two rows and are not here
yet. The one cap present is `MAX_UPLOAD_BYTES`, which is self-defence against a malformed multipart
body rather than abuse protection - the same exception 15.11 makes, for the same reason.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import AsyncIterator
from typing import Any

# At module level rather than inside the factory, for the reason 15.11 records: `from __future__
# import annotations` turns every annotation into a string and FastAPI resolves `UploadFile`
# against this module's globals, so an import inside `build_app` leaves an unresolvable ForwardRef
# and route registration fails outright.
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from app.backend.store import Store
from app.backend.stream import Reader, comment, frame
from app.backend.upstream import Upstream

#: A phone photograph is a few megabytes and 25 is generous for one. Not abuse protection - that is
#: 16.1.4 - but self-defence: without a ceiling a malformed multipart body is read into memory in
#: full before anything can reject it. Matches 15.11's cap deliberately, so a payload this backend
#: accepts is never one the model server will refuse for size.
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

#: What a client may upload, by name. The cheap half of the check; the model server does the
#: content sniff and its 415 passes straight through, so this is not a second implementation of it
#: but a rejection that happens before 25 MB have crossed a mobile uplink.
ALLOWED_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")

#: The correction kinds a client may report. Closed rather than free text because 14's error
#: taxonomy is what these become: `sub_text` is 11 of hdbpmn's median 21 edits, and a log that
#: also contains "wrong", "label bad" and "typo" cannot be counted against that.
FEEDBACK_KINDS = ("sub_text", "sub_type", "add_node", "del_node", "add_edge", "del_edge", "other")

#: Version of the backend contract, reported by `/health` and as the OpenAPI version.
VERSION = "16.1"


def _content_type(filename: str, declared: str | None) -> str:
    """What to tell the model server this upload is.

    A browser that uses the File System Access API or a `Blob` built in JavaScript can send
    `application/octet-stream` for a perfectly good JPEG, and 15.11 checks the magic bytes rather
    than this header, so guessing from the suffix is safe and refusing on the header would reject
    real photographs from real phones.
    """
    if declared and declared.startswith("image/"):
        return declared
    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return {"jpg": "image/jpeg", "tif": "image/tiff"}.get(suffix, f"image/{suffix or 'png'}")


def build_app(*, store: Store | None = None, upstream: Upstream | None = None) -> FastAPI:
    """The FastAPI application.

    A factory taking both collaborators, so a test drives a temporary state root and a fake model
    server without monkeypatching module globals - and so `--check` can exercise every route that
    does not need a model.
    """
    app = FastAPI(
        title="DreamScript app backend",
        version=VERSION,
        description=(
            "Phase 16.1. One upload, then reads against its id. Holds no model: the pipeline "
            "answers on 15.11's model server and this passes its degradation reporting through "
            "unchanged. Keeps the request identity a phone needs and the corrections a tap "
            "produces."
        ),
    )
    state = store if store is not None else Store()
    model = upstream if upstream is not None else Upstream()
    app.state.store = state
    app.state.upstream = model

    # -- health -------------------------------------------------------------------------

    @app.get("/health", summary="This process. Probes the model server only if asked.")
    def health(
        upstream_probe: bool = Query(
            False,
            alias="upstream",
            description="Also probe the model server. Off by default: a container health check "
            "that fails because another container is not ready restarts the wrong process.",
        )
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "status": "ok",
            "service": "app-backend",
            "version": VERSION,
            "model_url": model.url,
            "stored": len(state.ids()),
        }
        if upstream_probe:
            body["model_server"] = model.health()
        return body

    # -- predict ------------------------------------------------------------------------

    @app.post("/predict", summary="One page photograph. Stored under a new id.")
    async def predict(image: UploadFile = File(...)) -> JSONResponse:
        filename, body = await _accept(image)
        reply = model.predict(body, filename, _content_type(filename, image.content_type))
        if not reply.ok:
            # The upstream's own status, already classified by `Upstream`. Its detail is returned
            # rather than a generic one because "the model server at http://model:8000 is
            # unreachable" is actionable and "upstream error" is not.
            raise HTTPException(
                status_code=reply.status,
                detail=reply.error or f"the model server answered {reply.status}",
            )

        record = state.put(
            reply.payload,
            filename=filename,
            bytes_in=len(body),
            upstream_seconds=round(reply.seconds, 3),
        )
        return JSONResponse(_public(record))

    @app.post(
        "/predict/stream",
        summary="The same answer, streamed stage by stage (16.1.2).",
        response_class=StreamingResponse,
        responses={
            200: {
                "content": {"text/event-stream": {}},
                "description": "`upload`, then the model server's own `run_started`, "
                "`stage_started`/`stage_finished` and `run_finished` frames relayed unchanged, "
                "then one `result` frame carrying what `/predict` returns plus its new `id`.",
            }
        },
    )
    async def predict_stream(image: UploadFile = File(...)) -> StreamingResponse:
        filename, body = await _accept(image)
        content_type = _content_type(filename, image.content_type)

        # Probed before the stream opens, so the commonest failure in this architecture - nothing
        # listening on MODEL_URL - is still a real 503 with a status line, not a frame the client
        # has to parse out of a 200. Once the first byte is sent that option is gone, which is why
        # it is taken here.
        probe = model.health()
        if not probe.get("reachable"):
            raise HTTPException(
                status_code=503,
                detail=f"the model server at {model.url} is not reachable: "
                f"{probe.get('detail') or probe.get('status')}",
            )

        return StreamingResponse(
            _relay(state, model, body, filename, content_type),
            media_type="text/event-stream",
            headers={
                # Without this an nginx or a CDN in front of this service buffers the response and
                # delivers every frame at once at the end - a working implementation of exactly the
                # thing this row exists to prevent.
                "X-Accel-Buffering": "no",
                "Cache-Control": "no-cache",
            },
        )

    @app.get("/predict/{record_id}", summary="A stored prediction, in full.")
    def stored(record_id: str) -> dict[str, Any]:
        return _public(_require(state, record_id))

    # -- the cheap reads ----------------------------------------------------------------

    @app.get("/ir/{record_id}", summary="Just the IR, and what the overlay needs to draw it.")
    def ir(record_id: str) -> dict[str, Any]:
        record = _require(state, record_id)
        result = record["result"]
        return {
            "id": record["id"],
            "diagram_type": result.get("diagram_type"),
            "ir": result.get("ir"),
            "traversal": result.get("traversal") or [],
            # Carried here too, and not only on /predict. A client that fetched the IR alone must
            # still be able to tell whether to trust it without a second request.
            "needs_confirmation": bool(result.get("needs_confirmation")),
            "degraded": bool(result.get("degraded")),
            "stopped_at": result.get("stopped_at"),
        }

    @app.get("/code/{record_id}", summary="Just the code. ?format=text for a clipboard.")
    def code(
        record_id: str,
        fmt: str = Query(
            "json", alias="format", pattern="^(json|text)$", description="json or text"
        ),
    ):
        record = _require(state, record_id)
        result = record["result"]
        body = result.get("code")
        if fmt == "text":
            # A share sheet, a clipboard and `curl | less` all want the program and nothing else.
            # 404 rather than an empty 200: a page that never reached code has no code to copy, and
            # pasting the empty string into an editor is worse than being told.
            if not body:
                raise HTTPException(
                    status_code=404,
                    detail=f"{record_id} produced no code; stopped at "
                    f"{result.get('stopped_at') or 'an unreported stage'}",
                )
            return PlainTextResponse(body)
        return {
            "id": record["id"],
            "language": result.get("language"),
            "code": body,
            "ok": bool(result.get("ok")),
            "degraded": bool(result.get("degraded")),
            "stopped_at": result.get("stopped_at"),
        }

    # -- feedback -----------------------------------------------------------------------

    @app.post("/feedback", summary="One correction against a stored prediction. Append-only.")
    def feedback(payload: dict[str, Any]) -> dict[str, Any]:
        record_id = str(payload.get("id") or "")
        record = _require(state, record_id)
        kind = str(payload.get("kind") or "other")
        if kind not in FEEDBACK_KINDS:
            raise HTTPException(
                status_code=422,
                detail=f"unknown correction kind {kind!r}; expected one of "
                f"{', '.join(FEEDBACK_KINDS)}",
            )
        event = state.feedback(
            record_id,
            {
                "kind": kind,
                "node": payload.get("node"),
                "edge": payload.get("edge"),
                "was": payload.get("was"),
                "now": payload.get("now"),
                "note": payload.get("note"),
                # What the pipeline itself said about this page, stored beside the correction. A
                # correction is only training data if what it corrected is recoverable, and the
                # prediction record is evictable while the log is not.
                "diagram_type": record["result"].get("diagram_type"),
                "degraded": bool(record["result"].get("degraded")),
            },
        )
        record["corrections"] = state.corrections(record_id)
        state.update(record)
        return {"stored": event, "corrections": len(record["corrections"])}

    @app.get("/feedback/{record_id}", summary="Every correction against one id.")
    def feedback_for(record_id: str) -> dict[str, Any]:
        _require(state, record_id)
        events = state.corrections(record_id)
        return {"id": record_id, "count": len(events), "corrections": events}

    return app


async def _accept(image: UploadFile) -> tuple[str, bytes]:
    """One upload, checked. Shared by `/predict` and `/predict/stream`.

    Factored out when the second route arrived rather than copied into it: three checks that must
    give the same answer on both routes is the code that drifts, and a streaming route that forgot
    the size cap would be a hole in one endpoint whose symptom points nowhere near the omission.

    The content *sniff* is deliberately not here - the model server does it, and its 415 passes
    straight through. Two implementations of "are these bytes really a PNG" is one more than this
    architecture needs, and the one that matters is the one next to the decoder.
    """
    filename = image.filename or "upload.png"
    suffix = f".{filename.rsplit('.', 1)[-1].lower()}" if "." in filename else ""
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail=f"unsupported file type {suffix or '(none)'}; expected one of "
            f"{', '.join(ALLOWED_SUFFIXES)}",
        )
    body = await image.read(MAX_UPLOAD_BYTES + 1)
    if len(body) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"image exceeds {MAX_UPLOAD_BYTES} bytes")
    if not body:
        raise HTTPException(status_code=400, detail="empty upload")
    return filename, body


async def _relay(
    state: Store, model: Upstream, body: bytes, filename: str, content_type: str
) -> AsyncIterator[str]:
    """Relay the model server's progress, add the two frames that are this backend's, store the
    result.

    The order of the last two things matters. The `result` frame is emitted **after** the record is
    written, because it carries the id and an id a client can see before the file exists is an id
    whose `GET /ir/{id}` races the write. The store's write is atomic, so once the frame is out the
    read cannot see half a record.

    A page is still stored when the client has gone. The generator keeps running to completion on a
    disconnect in every server that closes the response lazily, and a person whose train entered a
    tunnel after the detector finished should be able to reload and find their page by its id rather
    than photograph the whiteboard again - which by then has been wiped.
    """
    reader = Reader()
    yield frame("upload", {"event": "upload", "filename": filename, "bytes_in": len(body)})

    async for line in model.stream(body, filename, content_type):
        if line.startswith(":"):
            # Relayed, not swallowed. The proxy that would time this connection out sits between
            # the client and this process as often as between this process and the model server.
            yield comment(line[1:].strip() or "ping")
            continue
        event = reader.feed(line)
        if event is None:
            continue
        name, payload = event
        if name != "result":
            # Relayed unchanged. These are the model server's observations of its own stages and
            # this backend has nothing true to add to them.
            yield frame(name, payload)
            continue
        record = state.put(
            payload,
            filename=filename,
            bytes_in=len(body),
            upstream_seconds=payload.get("seconds"),
        )
        yield frame("result", _public(record))

    trailing = reader.flush()
    if trailing is not None:
        # A stream that ended without its final blank line still delivered its last frame, and
        # dropping it because the socket closed a byte early would lose the answer.
        name, payload = trailing
        if name == "result":
            yield frame(
                "result", _public(state.put(payload, filename=filename, bytes_in=len(body)))
            )
        else:
            yield frame(name, payload)


def _require(state: Store, record_id: str) -> dict[str, Any]:
    """The record, or a 404 naming what is gone.

    The store evicts oldest-first, so a 404 here is a real and expected outcome rather than only a
    typo - a demo that ran long enough can outlive its own first page, and the message says so.
    """
    record = state.get(record_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"no prediction {record_id!r}")
    return record


def _public(record: dict[str, Any]) -> dict[str, Any]:
    """A record as a response: the upstream payload, flat, with this backend's fields beside it.

    Flat rather than nested under `result` because the client reads `code` and `stages` on every
    screen, and `result.result.code` is the shape of a proxy that could not decide what it was.
    `id`, `created` and the upload facts are added *beside* those keys, never over them.
    """
    result = dict(record["result"])
    result.update(
        {
            "id": record["id"],
            "created": record["created"],
            "filename": record.get("filename"),
            "bytes_in": record.get("bytes_in"),
            "upstream_seconds": record.get("upstream_seconds"),
            "corrections": record.get("corrections") or [],
        }
    )
    return result


_APP: Any = None


def __getattr__(name: str) -> Any:
    """`app`, built on first access rather than at import. PEP 562, as in 15.11.

    `uvicorn app.backend.main:app` needs a module attribute, not a factory, and providing one by
    calling `build_app()` at import makes every test that imports this module register the routes
    and construct a FastAPI application. A module-level `__getattr__` runs on the first unresolved
    lookup, so the import is free and `app.backend.main:app` still resolves.
    """
    global _APP
    if name == "app":
        if _APP is None:
            _APP = build_app()
        return _APP
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def self_test(base: Any = None) -> dict[str, Any]:
    """Wire the app and exercise every route that does not need a model server."""
    import tempfile

    from fastapi.testclient import TestClient

    checks: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as tmp:
        store = Store(tmp)
        client = TestClient(build_app(store=store, upstream=base or Upstream()))

        health = client.get("/health")
        checks["health"] = {"status": health.status_code, "ok": health.status_code == 200}
        checks["health_skips_upstream"] = {
            "keys": sorted(health.json()),
            "ok": "model_server" not in health.json(),
        }

        schema = client.get("/openapi.json")
        paths = list(schema.json().get("paths", {})) if schema.status_code == 200 else []
        checks["openapi"] = {
            "paths": paths,
            "ok": {"/predict", "/ir/{record_id}", "/code/{record_id}", "/feedback", "/health"}
            <= set(paths),
        }

        rejected = client.post(
            "/predict", files={"image": ("notes.txt", b"not an image", "text/plain")}
        )
        checks["rejects_wrong_type"] = {
            "status": rejected.status_code,
            "ok": rejected.status_code == 415,
        }

        empty = client.post("/predict", files={"image": ("page.png", b"", "image/png")})
        checks["rejects_empty"] = {"status": empty.status_code, "ok": empty.status_code == 400}

        missing = client.get("/ir/deadbeefdeadbeef")
        checks["unknown_id_is_404"] = {
            "status": missing.status_code,
            "ok": missing.status_code == 404,
        }

        traversal = client.get("/code/..%2f..%2fetc%2fpasswd")
        checks["path_traversal_is_404"] = {
            "status": traversal.status_code,
            "ok": traversal.status_code == 404,
        }

        # The store, round tripped without a model server: this is the half of `/predict` that is
        # this backend's own and not a proxy of anything.
        record = store.put({"ok": True, "code": "print(1)", "language": "python", "ir": {}})
        held = client.get(f"/code/{record['id']}?format=text")
        checks["stored_code_reads_back"] = {
            "status": held.status_code,
            "ok": held.status_code == 200 and held.text == "print(1)",
        }
        logged = client.post(
            "/feedback", json={"id": record["id"], "kind": "sub_text", "now": "Start"}
        )
        checks["feedback_is_appended"] = {
            "status": logged.status_code,
            "ok": logged.status_code == 200 and logged.json()["corrections"] == 1,
        }
        bad_kind = client.post("/feedback", json={"id": record["id"], "kind": "typo"})
        checks["feedback_kind_is_closed"] = {
            "status": bad_kind.status_code,
            "ok": bad_kind.status_code == 422,
        }

    return {"checks": checks, "ok": all(check["ok"] for check in checks.values())}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 16.1 app backend")
    ap.add_argument("--check", action="store_true", help="self-test without binding a port")
    ap.add_argument("--openapi", action="store_true", help="print the OpenAPI document")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=3000)
    args = ap.parse_args(argv)

    if args.openapi:
        json.dump(__getattr__("app").openapi(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    if args.check:
        started = time.perf_counter()
        result = self_test()
        result["seconds"] = round(time.perf_counter() - started, 3)
        json.dump(result, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0 if result["ok"] else 1

    import uvicorn

    uvicorn.run(__getattr__("app"), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
