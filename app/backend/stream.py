"""Phase 16.1.2 - server-sent events, and the relay that does not invent any.

    from app.backend.stream import frame, parse

    frame("stage_started", {"stage": "assemble"})   # -> "event: stage_started\\ndata: {...}\\n\\n"
    parse(["event: stage_started", "data: {}", ""])  # -> [("stage_started", {})]

## Why the relay is a relay

The app backend holds no pipeline, so it can observe no stage. The only progress it could report on
its own authority would be the *expected* stage names advanced on a timer - a bar that moves because
seconds passed rather than because work finished, indistinguishable to the person holding the phone
from one that means something. That is the confident-wrong-answer failure 13.4, 13.7 and 13.8 exist
to refuse, relocated into the UI, and it would be the first place this project shipped one.

So the stages come from `src/serve/api.py`'s `/predict/stream`, which feeds off 16.1.2's pipeline
observer and therefore reports work that actually happened. This module adds exactly two frames of
its own and rewrites exactly one:

    upload    added. The backend's own fact: the photograph arrived, all N bytes of it. A phone on
              a slow uplink spends real seconds here and it is the one part of the wait the client
              can attribute without help
    error     added on a failure of the relay itself, carrying the status it would have been
    result    rewritten - and only by *adding* the `id` the store minted. Every other key is the
              model server's

## Framing

The whole of the SSE wire format that matters here: `event:` names it, `data:` carries it, a blank
line ends it, and a line starting `:` is a comment that exists to keep the connection warm. Comments
are relayed rather than swallowed, because the proxy that would time this connection out sits
between the *client* and this process as often as between this process and the model server.

`data:` is one line per frame by construction - every producer here is `json.dumps`, which never
emits a newline - so the multi-line-data branch of the specification is deliberately not
implemented. `parse` asserts nothing about it; it simply keeps the last `data:` it saw, which is the
correct behaviour for single-line frames and a documented non-goal for the rest.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any


def frame(name: str, payload: dict[str, Any]) -> str:
    """One server-sent event, terminated. The `\\n\\n` is the entire framing rule."""
    return f"event: {name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def comment(text: str = "ping") -> str:
    """A keep-alive. Bytes on the wire that mean nothing to a client and everything to a proxy."""
    return f": {text}\n\n"


def parse(lines: Iterable[str]) -> list[tuple[str, dict[str, Any]]]:
    """`(event, payload)` for every complete frame in `lines`. Comments and blanks are skipped.

    A frame whose `data:` is not JSON is dropped rather than raised on. The producer is the model
    server and it always sends JSON, so this is not a path that should ever run - but the
    alternative is a relay that dies on one malformed frame and takes the finished page with it,
    and the page is the thing the user cares about.
    """
    out: list[tuple[str, dict[str, Any]]] = []
    name = ""
    data = ""
    for raw in lines:
        line = raw.rstrip("\r\n")
        if line.startswith(":"):
            continue
        if not line:
            if name or data:
                try:
                    payload = json.loads(data) if data else {}
                except json.JSONDecodeError:
                    payload = None  # type: ignore[assignment]
                if isinstance(payload, dict):
                    out.append((name or "message", payload))
            name = ""
            data = ""
            continue
        if line.startswith("event:"):
            name = line[6:].strip()
        elif line.startswith("data:"):
            data = line[5:].strip()
    # A stream that ends without its final blank line still delivered its last frame. httpx's
    # `aiter_lines` does not synthesise one, and dropping the `result` frame because the server
    # closed the socket a byte early would lose the answer.
    if name or data:
        try:
            payload = json.loads(data) if data else {}
        except json.JSONDecodeError:
            payload = None  # type: ignore[assignment]
        if isinstance(payload, dict):
            out.append((name or "message", payload))
    return out


class Reader:
    """Incremental `parse`, for a relay that must forward each frame as it arrives.

    `parse` takes a whole stream, which is what a test has and a relay never does. Feed this one
    line at a time and it yields complete frames the moment their blank line lands - which is the
    difference between relaying progress and collecting it.
    """

    def __init__(self) -> None:
        self.name = ""
        self.data = ""

    def feed(self, raw: str) -> tuple[str, dict[str, Any]] | None:
        """One line in; one complete frame out, or `None` while the frame is still arriving."""
        line = raw.rstrip("\r\n")
        if line.startswith(":"):
            return None
        if not line:
            return self.flush()
        if line.startswith("event:"):
            self.name = line[6:].strip()
        elif line.startswith("data:"):
            self.data = line[5:].strip()
        return None

    def flush(self) -> tuple[str, dict[str, Any]] | None:
        """The frame in hand, if there is one. Called on a blank line and at end of stream."""
        if not (self.name or self.data):
            return None
        name, data = self.name or "message", self.data
        self.name = ""
        self.data = ""
        try:
            payload = json.loads(data) if data else {}
        except json.JSONDecodeError:
            return None
        return (name, payload) if isinstance(payload, dict) else None
