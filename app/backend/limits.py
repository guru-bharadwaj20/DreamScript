"""Phase 16.1.4 - rate and size limits, and what they are honestly worth.

    from app.backend.limits import Limits, install
    install(app)                      # the middleware, with the defaults below

## Two different problems wearing one row number

**Size** is the easy half and it is not really abuse protection. A phone photograph is a few
megabytes; a multipart body claiming to be 4 GB is not a photograph. The important part is *when* the
refusal happens: `MAX_UPLOAD_BYTES` in `main` is checked after `await image.read(...)`, which means
the bytes have already crossed the socket and been buffered. A `Content-Length` far above the cap can
be refused before a single byte of body is read, and that is the difference between a bounded process
and one that merely reports being over the bound afterwards.

A client may of course lie about `Content-Length`, or omit it and use `Transfer-Encoding: chunked`.
So the early check is an optimisation and **never the only one**: `main._accept` reads at most
`MAX_UPLOAD_BYTES + 1` and refuses on the overflow byte, which is the check that actually holds. This
middleware does not wrap the ASGI receive channel to count a chunked body itself - that would be a
third place enforcing one number, and the route already enforces it against the bytes rather than
against the client's claim about them.

**Rate** is the half that is genuinely about abuse, and it is honest about its ceiling. See below.

## Not all routes cost the same, so not all routes share a budget

    predict   6 per minute      a page is seconds of GPU and megabytes of upload
    run       10 per minute     a program is a process, and the runner already caps four at once
    write     30 per minute     corrections. Cheap, and the thing we most want people to do
    read      120 per minute    an id lookup is a file read; a client polls these

One bucket for everything would be set by the most expensive route and would then throttle the
cheapest. A phone drawing an overlay, fetching the IR and reading the code makes three reads per
capture, and rate-limiting *that* to protect the GPU would be limiting the wrong thing.

## What this limiter is not

**It is per process, and it says so.** The buckets are a dict in memory. Two uvicorn workers give a
client twice the budget; three give three times. That is not a bug to be hidden behind a comment - it
is the reason the numbers above are conservative, and the reason `X-RateLimit-Limit` reports the
per-process number rather than a fiction about the deployment. A limit that must hold across
processes needs shared state, which needs Redis, which is a dependency this row does not justify:
`docker-compose.yml` runs **one** app container with **one** worker.

**It does not trust `X-Forwarded-For` unless told to, and that is deliberate.** A limiter keyed on a
header the client sets is a limiter with a bypass: `X-Forwarded-For: <random>` on every request is
one bucket per request. So the key is the socket peer by default. Behind a reverse proxy that makes
every client share one bucket, which is wrong in the other direction - so `TRUSTED_PROXY_HOPS`
(`DREAMSCRIPT_TRUSTED_PROXY_HOPS`) names how many proxies are in front of this process, and the key
is taken that many entries from the right of the chain. Zero means "no proxy, use the socket", and
zero is the default because the safe failure is a shared bucket rather than no bucket.

**It is a fixed window, not a token bucket.** A client can send its whole minute's allowance in the
last second of one window and again in the first second of the next - the classic 2x burst at a
boundary. A sliding window costs a deque per client and a token bucket costs two floats; either would
be defensible. The fixed window is chosen because the failure it permits is *twice the intended rate
for one second*, and the numbers above are set low enough that twice them is still a rate this
service can serve. That is a decision with a cost, recorded rather than presented as free.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

#: One counting window. Fixed, and 60 s so the numbers read as "per minute" without arithmetic.
WINDOW_S = 60.0

#: Requests per window, per client, per class. See the docstring for why these are not one number.
BUDGETS: dict[str, int] = {
    "predict": 6,
    "run": 10,
    "write": 30,
    "read": 120,
}

#: Hard ceiling on a request body, checked here against `Content-Length` before the body is read and
#: again in `main._accept` against the bytes themselves. 25 MB, matching `main.MAX_UPLOAD_BYTES` and
#: 15.11's cap: a payload this accepts must never be one the model server refuses for size.
MAX_BODY_BYTES = 25 * 1024 * 1024

#: How many reverse proxies sit in front of this process. 0 means none, so the rate-limit key is the
#: socket peer. Read from the environment because it is a property of the deployment and of nothing
#: else - and defaulting to 0 means a misconfiguration gives every client one shared bucket rather
#: than giving every client an unlimited one.
TRUSTED_PROXY_HOPS_ENV = "DREAMSCRIPT_TRUSTED_PROXY_HOPS"

#: Routes exempt from rate limiting entirely. `/health` is polled by an orchestrator on a schedule
#: the orchestrator chooses, and a limiter that answers 429 to a liveness probe gets the container
#: restarted - which is 15.11's lesson about health checks, arriving a third time.
EXEMPT_PATHS = ("/health", "/openapi.json", "/docs", "/redoc", "/docs/oauth2-redirect")

#: 16.3.2's client assets, also exempt.
#:
#: They are file reads off local disk and cost nothing this limiter exists to protect. A cold load
#: of the gallery is thirteen requests, so six reloads would spend most of a 120-per-minute read
#: budget on the app's own stylesheet - and a 429 for a stylesheet is an app that renders without
#: CSS, which looks like a catastrophe and is a rate limit.
EXEMPT_PREFIXES = ("/assets/", "/examples/", "/icons/")


def trusted_proxy_hops() -> int:
    """Read per call rather than bound at import: a module-level default is evaluated once, and a
    process that sets the environment afterwards would get the old value."""
    try:
        return max(0, int(os.environ.get(TRUSTED_PROXY_HOPS_ENV, "0") or 0))
    except ValueError:
        return 0


def classify(method: str, path: str) -> str:
    """Which budget a request draws on.

    By what it costs, which is a function of **both** the path and the method - and the first version
    of this used the path alone for `/predict`, which was wrong in a way only a real client found.

    `POST /predict` is seconds of GPU and megabytes of upload. `GET /predict/{id}` is a file read of
    a record that already exists, and it is the request the result screen makes *every time it
    opens*. Bucketing it with the upload gave it a budget of six per minute, and six page loads in a
    row - which is what a screenshot pass across three devices and two themes is - answered 429.
    Nothing was being protected: the expensive thing had already happened.

    So the rule is the honest one. A `GET` under `/predict` is a read; only a body-carrying request
    to it costs what the `predict` budget exists to limit.
    """
    if path.startswith("/predict"):
        return "predict" if method in ("POST", "PUT", "PATCH") else "read"
    if path.startswith("/run"):
        return "run"
    if method in ("POST", "PUT", "PATCH", "DELETE"):
        return "write"
    return "read"


@dataclass
class Window:
    """One client's count in one class, in the current window."""

    started: float
    count: int = 0


@dataclass
class Limits:
    """Fixed-window counters, per client per class. In memory, per process - see the docstring."""

    budgets: dict[str, int] = field(default_factory=lambda: dict(BUDGETS))
    window_s: float = WINDOW_S
    max_body_bytes: int = MAX_BODY_BYTES
    _windows: dict[tuple[str, str], Window] = field(default_factory=dict, repr=False)

    def check(self, client: str, kind: str, *, now: float | None = None) -> dict[str, Any]:
        """Count one request and say whether it is allowed.

        Returns the headers a client needs to behave well - `X-RateLimit-Limit`, `-Remaining`,
        `-Reset` - because a limiter that only says no teaches a client nothing, and a client that
        has been told its budget can pace itself instead of discovering the limit by hitting it.
        """
        budget = self.budgets.get(kind, self.budgets.get("read", 120))
        clock = time.monotonic() if now is None else now
        key = (client, kind)
        window = self._windows.get(key)
        if window is None or clock - window.started >= self.window_s:
            window = Window(started=clock)
            self._windows[key] = window
        window.count += 1
        resets_in = max(0.0, self.window_s - (clock - window.started))
        allowed = window.count <= budget
        return {
            "allowed": allowed,
            "kind": kind,
            "limit": budget,
            "remaining": max(0, budget - window.count),
            "reset_s": round(resets_in, 1),
            "retry_after": max(1, int(resets_in + 0.999)),
        }

    def prune(self, *, now: float | None = None) -> int:
        """Drop expired windows and return how many went.

        Without this the dict grows one entry per (client, class) forever, which on a public service
        is an unbounded allocation keyed by whatever the internet sends. Called opportunistically
        rather than on a timer: there is no background task here to own one.
        """
        clock = time.monotonic() if now is None else now
        stale = [
            key for key, window in self._windows.items() if clock - window.started >= self.window_s
        ]
        for key in stale:
            del self._windows[key]
        return len(stale)

    @property
    def tracked(self) -> int:
        """How many (client, class) windows are live. Reported by `/health`."""
        return len(self._windows)


def client_key(scope: dict[str, Any], hops: int | None = None) -> str:
    """Who to count this request against.

    The socket peer, unless `TRUSTED_PROXY_HOPS` says there are proxies in front. The arithmetic,
    spelled out, because getting it backwards is the usual mistake:

    A proxy appends the address it received *from* - nginx's `$proxy_add_x_forwarded_for` appends
    `$remote_addr`. So with one trusted hop (`C -> P -> us`) P appended C, and the client is the
    **last** entry; anything left of it is whatever C chose to send. With two hops
    (`C -> P1 -> P2 -> us`) P1 set `C` and P2 appended P1, so the client is `chain[-2]`. In general
    it is `chain[len - hops]`, and the entries to the left of that are unverified.

    Taking `chain[0]`, which is the common mistake, takes exactly the entry the client wrote.
    """
    depth = trusted_proxy_hops() if hops is None else hops
    if depth > 0:
        for name, value in scope.get("headers") or []:
            if name.lower() != b"x-forwarded-for":
                continue
            chain = [part.strip() for part in value.decode("latin-1").split(",") if part.strip()]
            if chain:
                # `hops` proxies appended `hops` entries; the client's own address is the one just
                # left of them. Clamped, because a chain shorter than declared means the request did
                # not come through the expected path and the leftmost entry is the best available.
                index = max(0, len(chain) - depth)
                return chain[min(index, len(chain) - 1)]
    peer = scope.get("client")
    return peer[0] if peer else "unknown"


def install(app: Any, limits: Limits | None = None) -> Limits:
    """Add the middleware to `app` and return the counters, so a caller can inspect them.

    Returned rather than hidden: `/health` reports how many windows are live, and a test needs to
    drive the clock. A limiter nothing can see is a limiter nobody can diagnose.
    """
    # From `fastapi.responses`, not `starlette.responses`. It is the same class - fastapi
    # re-exports it - but importing it from starlette makes starlette a *declared* dependency of
    # this module, and it is declared in no requirements file: it arrives transitively through
    # fastapi. `tests/test_requirements_declare_every_import.py` caught it, which is the second
    # time that test has found exactly this on this phase (the first was anyio, in 16.1.2).
    from fastapi.responses import JSONResponse

    state = limits if limits is not None else Limits()
    app.state.limits = state

    @app.middleware("http")
    async def guard(request, call_next):
        path = request.url.path
        if path in EXEMPT_PATHS or path.startswith(EXEMPT_PREFIXES):
            # A 429 to a liveness probe gets the container restarted. 15.11's lesson, third outing.
            return await call_next(request)

        declared = request.headers.get("content-length")
        if declared is not None:
            try:
                length = int(declared)
            except ValueError:
                return JSONResponse({"detail": "malformed Content-Length"}, status_code=400)
            if length > state.max_body_bytes:
                # Before a byte of body is read. A client may lie about this, which is why the
                # streaming count below and `main`'s per-read cap both stay.
                return JSONResponse(
                    {
                        "detail": f"body of {length} bytes exceeds the {state.max_body_bytes}-byte "
                        f"limit; a page photograph is a few megabytes"
                    },
                    status_code=413,
                )

        verdict = state.check(client_key(request.scope), classify(request.method, path))
        if not verdict["allowed"]:
            state.prune()
            return JSONResponse(
                {
                    "detail": f"{verdict['limit']} {verdict['kind']} requests per "
                    f"{int(state.window_s)}s; {verdict['reset_s']}s until the window resets"
                },
                status_code=429,
                headers=_headers(verdict, state) | {"Retry-After": str(verdict["retry_after"])},
            )

        response = await call_next(request)
        for name, value in _headers(verdict, state).items():
            # Set on every answer, not only on a refusal: a client told its budget can pace itself
            # instead of discovering the limit by hitting it.
            response.headers[name] = value
        return response

    return state


def _headers(verdict: dict[str, Any], state: Limits) -> dict[str, str]:
    return {
        "X-RateLimit-Limit": str(verdict["limit"]),
        "X-RateLimit-Remaining": str(verdict["remaining"]),
        "X-RateLimit-Reset": str(verdict["reset_s"]),
        # Named honestly. The limit holds within this process and the deployment may run more than
        # one; a client reading the header above should know what it is a limit on.
        "X-RateLimit-Scope": f"process:{int(state.window_s)}s",
    }
