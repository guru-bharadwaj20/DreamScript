"""Phase 15.11 - the inference service: one page in, IR and code out.

    uvicorn src.serve.api:app --host 0.0.0.0 --port 8000
    python -m src.serve.api --check          # import, wire and self-test without binding a port

    curl -F "image=@page.png" http://localhost:8000/predict

OpenAPI is served at `/docs` and `/openapi.json` by FastAPI itself, so the endpoint contract is
published rather than described in prose that can drift from the code.

## What this is, and what it is not

This is the **model server**: it holds the pipeline, and it answers one question - given a page,
what is the IR and the code. It is deliberately separate from the UI, which is the point of the
row: a phone client, a notebook and a batch job all reach the same service.

It is **not** Phase 16's app backend. The hardened sandbox runner (16.1.3) and rate and size limits
(16.1.4) belong there and are not here. The one exception is `MAX_UPLOAD_BYTES`, which is not abuse
protection but self-defence: without it a malformed multipart body is read into memory in full
before anything can reject it.

## `/predict/stream`, and why the streaming half of 16.1.2 is here

The scope line above used to say streaming progress belonged in the app backend too, and that was
wrong in a way worth recording rather than quietly fixing. **The app backend cannot see a stage it
does not run.** It is a proxy: it holds no pipeline, and the only progress it could report on its
own would be invented from the *expected* stage names on a timer - a progress bar that advances
because time passed rather than because work finished. That is precisely the confident-wrong-answer
failure this repository spends 13.4, 13.7 and 13.8 refusing to ship, moved into the UI.

Progress can only be observed where the stages are, so `POST /predict/stream` is here, feeding off
16.1.2's `run(..., observer=...)`, and the app backend relays it. The relay adds its own frames and
the request id; it does not fabricate any.

    curl -N -F "image=@page.png" http://localhost:8000/predict/stream

    event: stage_started
    data: {"event":"stage_started","stage":"assemble","index":3}

The last frame is `event: result` with exactly the body `/predict` would have returned, so a client
that streams and a client that waits get the same answer, and the one thing that differs between
them is when they learn it. Measured on a cold GPU run of `tests/fixtures/flowchart.png`: 12.08 s
end to end, of which `assemble` alone is **9.15 s**. That is the row's argument in one number - a
person watching a spinner for nine seconds cannot tell a working request from a stalled one.

## Degradation is reported, never hidden

13.4 established that an emitter answer and a model answer must never be mixed in a measurement,
and the same rule holds at the API boundary. Every response carries:

    ok                  did this page reach code
    degraded            did any stage fall back to a rung below its first choice
    stopped_at          the stage that ended it, empty when nothing did
    stages              13.7's per-stage table, with each stage's own confidence and reason
    needs_confirmation  the router was below its floor; a caller showing this to a person
                        should say so

A caller that wants only the code can read `code`. A caller that wants to know whether to trust it
has everything it needs without a second request, which is the difference between a service that
can be integrated honestly and one that cannot.

## The model is loaded once, lazily

Constructing the pipeline imports torch and can load checkpoints, which must not happen at import
time - that would make `--check`, the OpenAPI dump and any test that imports this module pay for a
GPU load. It happens on the first request and is then reused.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

# Imported at module level rather than inside `build_app`: `from __future__ import annotations`
# turns every annotation into a string, and FastAPI resolves `UploadFile` against this module's
# globals. Imported inside the factory it is an unresolvable ForwardRef and route registration
# fails outright. fastapi is a declared dependency of this row, so the import costs nothing that
# is not already required.
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

#: Seconds of silence on `/predict/stream` before a `: ping` comment goes out. Below the 30 s that
#: is the usual proxy idle timeout and well below the 9.15 s `assemble` takes, so a long stage
#: produces several pings rather than one gamble.
HEARTBEAT_S = 5.0

#: A page photograph is a few megabytes; 25 is generous for one and small enough that a malformed
#: body cannot exhaust the process. Not abuse protection - that is 16.1.4 - but self-defence.
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

#: What a caller may upload, by name. The cheap half of the check, and the half that was here:
#: it rejects `notes.txt` before 25 MB have been read off the wire.
ALLOWED_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")

#: ...and the half that was documented and missing. The comment above this tuple used to read
#: "Checked by content, not only by the declared type: a client that labels a PDF as image/png
#: should get a clear rejection rather than a stack trace from OpenCV" - and the code looked at
#: `Path(filename).suffix` and nothing else. So a PDF named `x.png` reached `cv2.imread`, which
#: returns None, which `page_for` raises on, which arrives at the caller as a 502 naming the
#: pipeline: the documented failure, produced by the documented input.
#:
#: Signatures rather than a library. These are fixed byte prefixes, and taking on a dependency
#: that decodes untrusted uploads in order to tell you their type is a larger thing than a table.
MAGIC: tuple[tuple[str, int, bytes], ...] = (
    ("png", 0, b"\x89PNG\r\n\x1a\n"),
    ("jpeg", 0, b"\xff\xd8\xff"),
    ("bmp", 0, b"BM"),
    ("tiff", 0, b"II*\x00"),
    ("tiff", 0, b"MM\x00*"),
    # RIFF is a container, so this mark alone is also WAV and AVI; `sniff` requires both.
    ("webp", 0, b"RIFF"),
    ("webp", 8, b"WEBP"),
)


def sniff(body: bytes) -> str:
    """The image family these bytes actually are, or `""` for anything else.

    WebP needs both of its marks: `RIFF` at 0 on its own is a WAV file, and accepting one because
    it starts like the other is the same mistake as trusting the extension, one layer down.
    """
    hits = {name for name, offset, mark in MAGIC if body[offset : offset + len(mark)] == mark}
    if "webp" in hits and not (body[:4] == b"RIFF" and body[8:12] == b"WEBP"):
        hits.discard("webp")
    return sorted(hits)[0] if hits else ""


_PIPELINE: Any = None


def pipeline():
    """The pipeline, built once on first use rather than at import."""
    global _PIPELINE
    if _PIPELINE is None:
        from src.pipeline.core import DreamScriptPipeline

        _PIPELINE = DreamScriptPipeline()
    return _PIPELINE


def registry_versions() -> list[dict[str, Any]]:
    """What 15.3 says is deployed, so `/version` reports the registry rather than a constant."""
    try:
        from src.mlops.registry import listing

        held = listing()
    except Exception:  # noqa: BLE001 - a service must start without a registry
        return []
    return [row for row in held if row.get("stage") in ("Production", "Staging")]


def result_payload(result: Any, seconds: float) -> dict[str, Any]:
    """One page's `Result` as the response body. Degradation is a field, not a footnote."""
    stages = result.timing_table()
    return {
        "ok": bool(result.ok),
        "degraded": any(stage.get("degraded") for stage in stages),
        "stopped_at": result.stopped_at or None,
        "needs_confirmation": bool(result.needs_confirmation),
        "diagram_type": result.diagram_type,
        "language": result.language or None,
        "code": result.code or None,
        "ir": result.ir,
        "traversal": list(result.traversal or []),
        "stages": stages,
        "seconds": round(seconds, 3),
    }


def build_app():
    """The FastAPI application. A function so importing this module costs nothing - see
    `__getattr__`, which is what makes that true of the module attribute as well."""
    app = FastAPI(
        title="DreamScript inference",
        version="15.11",
        description=(
            "One page in, IR and code out. Separate from the UI by design: a phone client, a "
            "notebook and a batch job all reach this same service. Every response reports "
            "whether any stage degraded, so a caller can tell an emitter answer from a model "
            "answer without a second request."
        ),
    )

    @app.get("/health", summary="Liveness. Does not load the model.")
    def health() -> dict[str, Any]:
        # Deliberately does not touch `pipeline()`: a liveness probe that loads 1.3 GB of weights
        # is a liveness probe that times out and gets the container killed on startup.
        return {"status": "ok", "model_loaded": _PIPELINE is not None}

    @app.get("/version", summary="Which model versions this service would answer with.")
    def version() -> dict[str, Any]:
        return {"service": "15.11", "registry": registry_versions()}

    @app.post("/predict", summary="One page photograph to IR and code.")
    async def predict(image: UploadFile = File(...)) -> JSONResponse:
        body, suffix = await accept(image)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"upload{suffix}"
            path.write_bytes(body)
            started = time.perf_counter()
            try:
                result = pipeline().run(path)
            except Exception as exc:
                # 502 rather than 500: the pipeline is the upstream here, and the distinction
                # tells a caller whether to retry the page or the request.
                raise HTTPException(status_code=502, detail=f"pipeline failed: {exc}") from exc
            payload = result_payload(result, time.perf_counter() - started)

        # 200 even when the page did not reach code. A page the pipeline honestly could not read
        # is a successful answer to the question asked - `ok: false` with the stage that stopped
        # it is more useful to a client than a 4xx that says only "no".
        return JSONResponse(payload)

    @app.post(
        "/predict/stream",
        summary="The same answer, streamed stage by stage as server-sent events (16.1.2).",
        response_class=StreamingResponse,
        responses={
            200: {
                "content": {"text/event-stream": {}},
                "description": "`run_started`, then `stage_started`/`stage_finished` per stage, "
                "then one `result` frame carrying exactly what `/predict` returns.",
            }
        },
    )
    async def predict_stream(image: UploadFile = File(...)) -> StreamingResponse:
        # The upload is validated *before* the response starts, so a bad request is still a real
        # 415 or 413. Everything after the first byte of the stream can only be a frame, because
        # the status line has already gone out - which is why the checks happen out here.
        body, suffix = await accept(image)
        return StreamingResponse(
            stage_events(body, suffix),
            media_type="text/event-stream",
            headers={
                # Without this an nginx or a CDN in front of the service buffers the whole
                # response and delivers seven frames at once at the end, which is a working
                # implementation of the thing this row exists to avoid.
                "X-Accel-Buffering": "no",
                "Cache-Control": "no-cache",
            },
        )

    return app


async def accept(image: UploadFile) -> tuple[bytes, str]:
    """One upload, checked. Shared by `/predict` and `/predict/stream`.

    Factored out when the second route arrived rather than copied into it. Four checks that must
    give the same answer on both routes is exactly the code that drifts: the streaming route
    forgetting the content sniff would be a hole in one endpoint and not the other, and nothing
    about the symptom would point at the omission.
    """
    suffix = Path(image.filename or "").suffix.lower()
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
    if not sniff(body):
        # 415, like the suffix rejection, and for the same reason: the request is well formed
        # and the payload is of a type this service does not read.
        raise HTTPException(
            status_code=415,
            detail=f"{image.filename or 'upload'} is named {suffix} but its contents are "
            f"not a {'/'.join(sorted({name for name, _, _ in MAGIC}))} image",
        )
    return body, suffix


def frame(name: str, payload: dict[str, Any]) -> str:
    """One server-sent event. `\\n\\n` terminated, which is the entire framing rule."""
    return f"event: {name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def stage_events(body: bytes, suffix: str) -> AsyncIterator[str]:
    """Run one page and yield its progress as it happens.

    ## Why a thread and a queue

    `pipeline().run` is synchronous and holds the GIL only in the gaps between torch calls; it
    cannot be awaited and must not run on the event loop, because a 9-second `assemble` on the loop
    stops this process answering `/health`. So the run goes to a worker thread, its observer drops
    events on a `queue.Queue`, and this generator drains the queue.

    `to_thread.run_sync(q.get)` rather than polling: a poll loop either adds latency to every frame
    or burns a core waiting, and a blocking `get` in a worker thread does neither. The `timeout` is
    what makes the heartbeat possible.

    ## The heartbeat is not decoration

    `assemble` is 9.15 s on a cold GPU run and it emits nothing while it works. Nine seconds of
    silence on an HTTP response is long enough for an idle-timeout proxy to close the connection
    and for a mobile radio to drop to a low-power state, and the client cannot distinguish either
    from a slow page. A `: ping` comment every few seconds is bytes on the wire that mean nothing to
    the client and everything to the things between it and here.
    """
    import queue
    import threading

    from anyio import to_thread

    events: queue.Queue[dict[str, Any] | None] = queue.Queue()

    def work() -> None:
        started = time.perf_counter()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / f"upload{suffix}"
                path.write_bytes(body)
                result = pipeline().run(path, observer=events.put)
                payload = result_payload(result, time.perf_counter() - started)
            events.put({"event": "result", **payload})
        except Exception as error:  # noqa: BLE001 - 502's reason, as a frame rather than a status
            # The status line left before the first stage started, so a pipeline crash cannot be
            # the 502 that `/predict` answers with. It is a frame carrying the status it would have
            # been, which is the most honest thing available once the headers are gone.
            events.put(
                {
                    "event": "error",
                    "status": 502,
                    "detail": f"pipeline failed: {type(error).__name__}: {error}",
                }
            )
        finally:
            events.put(None)

    threading.Thread(target=work, name="predict-stream", daemon=True).start()

    while True:
        try:
            event = await to_thread.run_sync(lambda: events.get(timeout=HEARTBEAT_S))
        except queue.Empty:
            yield ": ping\n\n"
            continue
        if event is None:
            return
        yield frame(str(event.get("event", "stage")), event)


_APP: Any = None


def __getattr__(name: str) -> Any:
    """`app`, built on first access rather than at import. PEP 562.

    `uvicorn src.serve.api:app` needs a module *attribute*, not a factory, and this module used
    to provide one by calling `build_app()` at import - directly under a docstring saying "A
    function so importing this module costs nothing". Importing it registered four routes and
    constructed a FastAPI application, which is what `--check`, the OpenAPI dump and every test
    that imports this module paid for.

    A module-level `__getattr__` runs on the first *unresolved* attribute lookup, so
    `import src.serve.api` is free and `src.serve.api:app` still resolves - uvicorn does
    `getattr(module, "app")` and cannot tell the difference. The claim is true now.
    """
    global _APP
    if name == "app":
        if _APP is None:
            _APP = build_app()
        return _APP
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def self_test() -> dict[str, Any]:
    """Wire the app and exercise every route that does not need a model."""
    from fastapi.testclient import TestClient

    # `app` inside this module is not resolved by `__getattr__` - a global lookup that fails
    # goes to the module namespace, and a *successful* one never reaches it. Asked for explicitly.
    client = TestClient(__getattr__("app"))
    checks: dict[str, Any] = {}

    health = client.get("/health")
    checks["health"] = {"status": health.status_code, "ok": health.status_code == 200}

    version = client.get("/version")
    checks["version"] = {"status": version.status_code, "ok": version.status_code == 200}

    schema = client.get("/openapi.json")
    paths = list(schema.json().get("paths", {})) if schema.status_code == 200 else []
    checks["openapi"] = {
        "status": schema.status_code,
        "paths": paths,
        "ok": {"/predict", "/health", "/version"} <= set(paths),
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

    # The claim this module makes about itself, exercised rather than asserted: a PDF that says
    # it is a PNG in both its filename and its declared content type.
    disguised = client.post(
        "/predict",
        files={"image": ("page.png", b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n", "image/png")},
    )
    checks["rejects_disguised_content"] = {
        "status": disguised.status_code,
        "ok": disguised.status_code == 415,
    }

    return {"checks": checks, "ok": all(check["ok"] for check in checks.values())}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 15.11 inference service")
    ap.add_argument("--check", action="store_true", help="self-test without binding a port")
    ap.add_argument("--openapi", action="store_true", help="print the OpenAPI document")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args(argv)

    if args.openapi:
        json.dump(__getattr__("app").openapi(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    if args.check:
        result = self_test()
        json.dump(result, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0 if result["ok"] else 1

    import uvicorn

    uvicorn.run(__getattr__("app"), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
