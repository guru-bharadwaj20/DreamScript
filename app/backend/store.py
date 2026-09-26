"""Phase 16.1.1 - what the model server has no business remembering.

    from app.backend.store import Store
    store = Store()
    record = store.put(payload, filename="page.jpg", bytes_in=2_412_889)
    store.get(record["id"])["result"]["code"]
    store.feedback(record["id"], {"kind": "sub_text", "node": "n3", "now": "Start"})

## Why a store at all

15.11 answers one question per request and keeps nothing, which is right for a model server and
wrong for a phone. A page photograph is **several megabytes over a mobile uplink**, and the app
needs four things off one upload: the boxes to draw, the IR to lay out, the code to show, and a
regeneration after the user fixes a label. Four uploads of the same 3 MB photo to get them is not
a design decision, it is a bill the user pays.

So `/predict` uploads once and returns an `id`; `/ir/{id}`, `/code/{id}` and the correction round
trip are reads and writes against that id, and none of them carries an image.

## On disk, not in a dict

A demo is the whole point of Phase 16, and a demo is exactly when a process restarts. An
in-memory dict loses the page that was just photographed - the one irreplaceable thing in the
request, because the whiteboard has been wiped and the person is holding the phone. One JSON file
per prediction under the state root survives `uvicorn --reload`, a container restart and a crash.

It is **not a database and does not pretend to be one**. Two properties are wanted and both are
cheap: a write that is atomic (temp file plus `os.replace`, so a reader never sees half a record)
and a bound on the size (`MAX_RECORDS` oldest-first eviction, so a long demo cannot fill a disk).
Concurrent writers to the *same* id would race, and nothing in this design creates two - an id is
minted by one request and the corrections against it arrive from one phone.

## The corrections are append-only, and they are the deliverable

`feedback.jsonl` is not a log. 16.2.8's argument is that `sub_text` is 11 of hdbpmn's median 21
edits, so a tap that fixes a label is worth more than a recogniser retrain - and it is only worth
anything if it is kept. Appended as JSON lines rather than written into the prediction record,
because a correction is an event with a time, the same node can be corrected twice, and the second
correction must not erase the evidence that the first was needed.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

#: Where predictions and the correction log live. `runs/` is gitignored and derived, which is what
#: this is: every record here can be reproduced by re-uploading the page, and none of it is input.
DEFAULT_ROOT = Path("runs") / "app"

#: Oldest-first eviction ceiling. A page record is a few kilobytes of JSON with the IR in it, so
#: 500 is well under a megabyte of state and far more than any demo or test session needs.
MAX_RECORDS = 500

#: The correction log is never evicted. It is training data; the predictions are not.
FEEDBACK_FILE = "feedback.jsonl"

#: Ids are minted as lowercase hex and are checked against that alphabet before they touch a path.
_HEX = frozenset("0123456789abcdef")


def new_id() -> str:
    """A short opaque handle. Not content-addressed, on purpose.

    Two photographs of the same whiteboard a second apart are two attempts and the user may want
    to compare them, so the same bytes hashing to the same id - silently returning the first
    attempt's corrections against the second attempt's upload - is a bug waiting for a demo.
    """
    return uuid.uuid4().hex[:16]


class Store:
    """One directory of prediction records, plus one append-only correction log."""

    def __init__(self, root: str | os.PathLike[str] | None = None) -> None:
        if root is not None:
            self.root = Path(root)
        else:
            self.root = Path(os.environ.get("DREAMSCRIPT_APP_STATE", "") or DEFAULT_ROOT)
        self.predictions = self.root / "predictions"
        self.predictions.mkdir(parents=True, exist_ok=True)

    # -- predictions --------------------------------------------------------------------

    def path_for(self, record_id: str) -> Path:
        """The file for an id, with the id checked rather than trusted.

        `record_id` arrives in a URL path, and `Path(dir) / "../../secrets"` resolves perfectly
        well. An id that is not the shape ids are minted in is refused here rather than filtered in
        the route, because there are three routes and the fourth one to be written would have had
        to remember.
        """
        if not record_id or len(record_id) > 64 or not set(record_id) <= _HEX:
            raise KeyError(record_id)
        return self.predictions / f"{record_id}.json"

    def put(self, result: dict[str, Any], **meta: Any) -> dict[str, Any]:
        """Store one prediction and return its record."""
        record: dict[str, Any] = {
            "id": new_id(),
            "created": time.time(),
            "result": result,
            "corrections": [],
            **meta,
        }
        self._write(record)
        self._evict()
        return record

    def get(self, record_id: str) -> dict[str, Any] | None:
        """One record, or `None`. A missing id and a malformed id are the same answer here."""
        try:
            path = self.path_for(record_id)
        except KeyError:
            return None
        if not path.exists():
            return None
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # A record that cannot be read is a record that is gone. Raising would turn a truncated
            # file - the one thing the atomic write exists to prevent, and so the one thing worth
            # being robust about anyway - into a 500 on every later read of that id.
            return None
        return loaded if isinstance(loaded, dict) else None

    def update(self, record: dict[str, Any]) -> dict[str, Any]:
        """Rewrite a record in place. Used by the correction round trip, not by `/predict`."""
        self._write(record)
        return record

    def ids(self) -> list[str]:
        """Every stored id, newest first."""
        files = sorted(
            self.predictions.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True
        )
        return [path.stem for path in files]

    # -- corrections --------------------------------------------------------------------

    def feedback(self, record_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Append one correction event and return it as stored."""
        event = {"id": record_id, "at": time.time(), **payload}
        line = json.dumps(event, ensure_ascii=False, sort_keys=True)
        with (self.root / FEEDBACK_FILE).open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        return event

    def corrections(self, record_id: str | None = None) -> list[dict[str, Any]]:
        """Every correction event, or those against one id, oldest first."""
        path = self.root / FEEDBACK_FILE
        if not path.exists():
            return []
        events: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                # One unparseable line does not invalidate the file. An append-only log written by
                # a process that was killed mid-write can end in a partial line, and the good lines
                # before it are still training data.
                continue
            if record_id is None or event.get("id") == record_id:
                events.append(event)
        return events

    # -- internals ----------------------------------------------------------------------

    def _write(self, record: dict[str, Any]) -> None:
        """Atomically. A reader must never see a half-written record."""
        path = self.path_for(record["id"])
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)

    def _evict(self) -> None:
        """Keep the newest `MAX_RECORDS`. Corrections are not touched."""
        files = sorted(self.predictions.glob("*.json"), key=lambda p: p.stat().st_mtime)
        for stale in files[: max(0, len(files) - MAX_RECORDS)]:
            # Suppressed: a record another process already removed is a record that is evicted.
            # Raising here would make a successful `/predict` fail on the cleanup after it.
            with contextlib.suppress(OSError):
                stale.unlink()
