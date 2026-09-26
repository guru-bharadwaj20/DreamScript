"""Phase 16.1.1 - the one thing this backend asks of 15.11's model server.

    from app.backend.upstream import Upstream
    up = Upstream()                      # MODEL_URL, or http://localhost:8000
    reply = up.predict(body, "page.jpg", "image/jpeg")
    reply.status, reply.payload

## The contract is one POST, and it is deliberately thin

`docker-compose.yml` has said `MODEL_URL: http://model:8000` since 15.10, before either side of
this existed. That is the whole coupling: this backend holds no model, imports no torch, and knows
exactly one upstream route - `POST /predict`, multipart, one image. Everything else it serves is
derived from what came back, or is its own (the store, the sandbox, the limits).

## The payload is passed through, not summarised

13.4's rule - never mix an emitter answer with a model answer in a measurement - holds at *two*
boundaries now, and the second one is easier to break than the first. The model server's response
already carries `ok`, `degraded`, `stopped_at`, `needs_confirmation` and the per-stage table
precisely so a caller can tell what it is looking at. A proxy that helpfully flattens that into
`{"code": ...}` destroys the distinction at the last hop, on the client that most needs it, because
a phone showing a person their own whiteboard is the caller with the least context of all.

So `payload` is the upstream JSON object verbatim. This module adds fields beside it, never over
it.

## Failure is classified, because the client can act on the classification

    upstream unreachable    503   the model server is down or MODEL_URL is wrong.
                                  Retrying this request later can work. Nothing else can.
    upstream timed out      504   it is up and the page is still being read.
    upstream 4xx            same  the client sent something wrong; the status and the detail are
                                  its own answer and re-labelling them would hide it
    upstream 5xx            502   the upstream failed. 15.11 already uses 502 for "the pipeline
                                  failed", so this is a consistent reading one hop out
    upstream non-JSON       502   something is listening on that port and it is not the model
                                  server - the single most common misconfiguration there is

A page the pipeline honestly could not read is **not** in that table: 15.11 answers `200` with
`ok: false`, and so does this.
"""

from __future__ import annotations

import contextlib
import json
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

#: The model server, by service name inside compose and by localhost outside it. Read from the
#: environment rather than taken as a constructor default only, because which model server answers
#: is a deployment decision - the same reason `src.pipeline.core` reads `DREAMSCRIPT_MODEL_URL`.
MODEL_URL_ENV = "MODEL_URL"
DEFAULT_MODEL_URL = "http://localhost:8000"

#: A page on CPU is seconds, not milliseconds. 16.1.5 measures the real number; this is the ceiling
#: past which something is wrong rather than slow, and it is generous on purpose: a cold first
#: request loads the detector and the recogniser before it reads a single pixel.
PREDICT_TIMEOUT_S = 180.0

#: `/health` upstream probes are a different question and get a different budget. A probe that
#: blocks for three minutes is not a probe.
PROBE_TIMEOUT_S = 2.0


def _error_lines(status: int, detail: str) -> tuple[str, str, str]:
    """A failure of the relay, in the wire format of the thing it was relaying.

    Three *lines* rather than one frame string, because everything this generator yields is fed to
    `stream.Reader` one line at a time - exactly as `httpx.aiter_lines()` delivers them, with no
    trailing newlines - and a single blob containing `\\n` would arrive as one unparseable line.
    The `status` field carries what the response code would have been had the failure happened
    before the headers went out.
    """
    payload = {"event": "error", "status": status, "detail": detail}
    return ("event: error", f"data: {json.dumps(payload, ensure_ascii=False)}", "")


@dataclass(frozen=True)
class Reply:
    """What came back, already classified. `payload` is upstream's JSON object, untouched."""

    status: int
    payload: dict[str, Any]
    seconds: float
    #: Empty when the upstream answered with JSON. Otherwise the local reason, for the detail line.
    error: str = ""

    @property
    def ok(self) -> bool:
        """Did the upstream answer at all. Not whether the page reached code - that is
        `payload["ok"]`, and conflating the two is the mistake this whole module is written
        against."""
        return self.status == 200 and not self.error


class Upstream:
    """A client for exactly one route on 15.11's model server."""

    def __init__(self, url: str | None = None, *, timeout_s: float = PREDICT_TIMEOUT_S) -> None:
        raw = url if url is not None else os.environ.get(MODEL_URL_ENV, "") or DEFAULT_MODEL_URL
        self.url = raw.rstrip("/")
        self.timeout_s = timeout_s

    # -- requests -----------------------------------------------------------------------

    def predict(self, body: bytes, filename: str, content_type: str) -> Reply:
        """One page to the model server. Never raises for anything the upstream does."""
        return self._post_image("/predict", body, filename, content_type)

    async def stream(self, body: bytes, filename: str, content_type: str) -> AsyncIterator[str]:
        """`POST /predict/stream` upstream, one line at a time (16.1.2).

        Async, unlike `predict`, and not for symmetry's sake: a relay that blocks the event loop
        while it waits on the model server serves one phone at a time. `predict` is synchronous
        because FastAPI already runs a `def` route in a threadpool; a streaming response's generator
        runs *on the loop*, so every await here has to be a real one.

        Yields raw lines, including the blank separators and the `:` keep-alive comments. Framing is
        `stream.Reader`'s job - this method's only responsibility is to not lose a byte and not
        raise: a failure arrives as an `error` line in the same wire format, because by the time
        this generator runs the status line has already been sent.
        """
        import httpx

        files = {"image": (filename, body, content_type)}
        try:
            async with (
                httpx.AsyncClient(timeout=self.timeout_s) as client,
                client.stream("POST", f"{self.url}/predict/stream", files=files) as response,
            ):
                if response.status_code != 200:
                    # The upstream refused before streaming anything - a 415 for a bad type, say.
                    # Its body must be read explicitly on a streamed response.
                    await response.aread()
                    detail = ""
                    with contextlib.suppress(ValueError):
                        decoded = response.json()
                        if isinstance(decoded, dict):
                            detail = str(decoded.get("detail", ""))
                    for line in _error_lines(
                        response.status_code,
                        detail or f"the model server answered {response.status_code}",
                    ):
                        yield line
                    return
                async for line in response.aiter_lines():
                    yield line
        except httpx.TimeoutException as error:
            for line in _error_lines(
                504, f"the model server at {self.url} stopped answering: {error}"
            ):
                yield line
        except Exception as error:  # noqa: BLE001 - a frame, because the headers are already gone
            for line in _error_lines(
                503,
                f"the model server at {self.url} is unreachable: {type(error).__name__}: {error}",
            ):
                yield line

    def health(self) -> dict[str, Any]:
        """Is the model server reachable, on a probe budget rather than a page budget.

        Deliberately not called by this backend's own `/health` unless asked for: see `main`. A
        container health check that fails because a *different* container is not ready restarts the
        wrong process forever, which is 15.11's own argument about `/health` and the weights, one
        level out.
        """
        import httpx

        try:
            with httpx.Client(timeout=PROBE_TIMEOUT_S) as client:
                response = client.get(f"{self.url}/health")
        except Exception as error:  # noqa: BLE001 - a probe reports, it does not raise
            return {
                "reachable": False,
                "url": self.url,
                "detail": f"{type(error).__name__}: {error}",
            }
        body: Any = None
        # Suppressed rather than handled: a probe reports reachability, and a model server that is
        # up and answering `/health` with something other than JSON is still up. What it said is a
        # bonus field, not the answer.
        with contextlib.suppress(ValueError):
            body = response.json()
        return {
            "reachable": response.status_code == 200,
            "url": self.url,
            "status": response.status_code,
            "upstream": body if isinstance(body, dict) else None,
        }

    # -- internals ----------------------------------------------------------------------

    def _post_image(self, route: str, body: bytes, filename: str, content_type: str) -> Reply:
        import time

        import httpx

        started = time.perf_counter()
        files = {"image": (filename, body, content_type)}
        try:
            with httpx.Client(timeout=self.timeout_s) as client:
                response = client.post(f"{self.url}{route}", files=files)
        except httpx.TimeoutException as error:
            return Reply(
                504,
                {},
                time.perf_counter() - started,
                f"the model server at {self.url} did not answer within {self.timeout_s:g}s: {error}",
            )
        except Exception as error:  # noqa: BLE001 - classified into a status, not propagated
            return Reply(
                503,
                {},
                time.perf_counter() - started,
                f"the model server at {self.url} is unreachable: {type(error).__name__}: {error}",
            )
        seconds = time.perf_counter() - started

        try:
            payload = response.json()
        except ValueError:
            # Something is listening and it is not the model server. The most common single
            # misconfiguration in this architecture, and it deserves to say so rather than to
            # arrive as a JSONDecodeError traceback.
            return Reply(
                502,
                {},
                seconds,
                f"the model server at {self.url} answered {response.status_code} with "
                f"{response.headers.get('content-type', 'no content type')}, not JSON",
            )
        if not isinstance(payload, dict):
            return Reply(502, {}, seconds, f"the model server returned {type(payload).__name__}")

        if response.status_code == 200:
            return Reply(200, payload, seconds)
        if 400 <= response.status_code < 500:
            # Passed through. A 415 for an unsupported type is the client's answer, and relabelling
            # it 502 would tell the phone to retry something that can never work.
            return Reply(response.status_code, payload, seconds, str(payload.get("detail", "")))
        return Reply(
            502,
            payload,
            seconds,
            f"the model server failed with {response.status_code}: {payload.get('detail', '')}",
        )
