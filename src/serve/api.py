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

It is **not** Phase 16's app backend. Streaming stage-by-stage progress (16.1.2), the hardened
sandbox runner (16.1.3), and rate and size limits (16.1.4) belong there and are not here. The one
exception is `MAX_UPLOAD_BYTES`, which is not abuse protection but self-defence: without it a
malformed multipart body is read into memory in full before anything can reject it.

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
from pathlib import Path
from typing import Any

# Imported at module level rather than inside `build_app`: `from __future__ import annotations`
# turns every annotation into a string, and FastAPI resolves `UploadFile` against this module's
# globals. Imported inside the factory it is an unresolvable ForwardRef and route registration
# fails outright. fastapi is a declared dependency of this row, so the import costs nothing that
# is not already required.
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

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
    async def predict(image: UploadFile = File(...)) -> JSONResponse:  # noqa: B008 - FastAPI's DI
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

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"upload{suffix}"
            path.write_bytes(body)
            started = time.perf_counter()
            try:
                result = pipeline().run(path)
            except Exception as exc:  # noqa: BLE001 - one bad page must not take the service down
                # 502 rather than 500: the pipeline is the upstream here, and the distinction
                # tells a caller whether to retry the page or the request.
                raise HTTPException(status_code=502, detail=f"pipeline failed: {exc}") from exc
            payload = result_payload(result, time.perf_counter() - started)

        # 200 even when the page did not reach code. A page the pipeline honestly could not read
        # is a successful answer to the question asked - `ok: false` with the stage that stopped
        # it is more useful to a client than a 4xx that says only "no".
        return JSONResponse(payload)

    return app


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
