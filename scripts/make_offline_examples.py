"""Phase 16.2.10 — capture real predictions for the five bundled examples, for offline use.

The offline fallback in ``app/frontend/src/lib/offline.ts`` serves a *stored answer* when the
network or the server is gone. This script is where those answers come from, and the rule it exists
to enforce is that **nothing in that file is written by hand**: every byte is what
``app/backend/main.py`` actually answered for a fixture this repository commits, on a run that
happened, through the whole pipeline.

A hand-written "example result" would be the app's most visible screen showing a payload no version
of this pipeline ever produced — the one lie a demo mode is most tempted into and the hardest to
notice, because it would look perfect.

So the five answers are captured, warts included. Three of the five come back read as the wrong
diagram type - ``er_diagram``, ``wireframe`` and ``circuit`` all answer ``flowchart``, which is
4.1.5's aspect-ratio leak, since every fixture here is 320x240 - and the cache keeps that rather
than fixing it up, because the offline screen has to show what the reader does.

That count is worth noticing: the gallery asserted **two** by hand for as long as it existed, and
running this script is what measured the third. Nothing in the client says a number about these
pages any more - it reads them out of the payloads this writes.

Regenerate with the app backend running on port 3000::

    .venv/Scripts/python.exe scripts/make_offline_examples.py

The backend's upload budget is six per minute (16.1.4), which is exactly enough for five fixtures
and one retry; the script paces itself rather than earning a 429 on the fifth.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
OUT = ROOT / "app" / "frontend" / "src" / "lib" / "offline.data.json"

#: The gallery's five, in the gallery's order, with the words a person reads on the card.
EXAMPLES: tuple[tuple[str, str], ...] = (
    ("flowchart.png", "Flowchart"),
    ("state_machine.png", "State machine"),
    ("er_diagram.png", "ER diagram"),
    ("wireframe.png", "Wireframe"),
    ("circuit.png", "Circuit"),
)

#: Between uploads. 16.1.4 allows six per minute; five at ten seconds apart never approaches it.
PACE_S = 10.0

#: A cold first page is ~12 s (16.2.2's measurement) and `assemble` alone can be 9 s.
TIMEOUT_S = 300.0


def multipart(field: str, filename: str, payload: bytes) -> tuple[bytes, str]:
    """One file, as multipart/form-data.

    By hand rather than with ``requests``, which is not a declared dependency of this repository and
    would fail ``tests/test_requirements_declare_every_import.py`` — a 15-line encoder is cheaper
    than a dependency added for one script.
    """
    boundary = f"----dreamscript{uuid.uuid4().hex}"
    body = b"".join(
        (
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'.encode(),
            b"Content-Type: image/png\r\n\r\n",
            payload,
            f"\r\n--{boundary}--\r\n".encode(),
        )
    )
    return body, f"multipart/form-data; boundary={boundary}"


def predict(base: str, path: Path) -> dict[str, Any]:
    body, content_type = multipart("image", path.name, path.read_bytes())
    request = urllib.request.Request(
        f"{base}/predict", data=body, headers={"Content-Type": content_type}, method="POST"
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
        return json.loads(response.read().decode("utf-8"))


def health(base: str) -> dict[str, Any]:
    with urllib.request.urlopen(f"{base}/health?upstream=1", timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:3000", help="the app backend")
    args = parser.parse_args(argv)

    try:
        state = health(args.base)
    except (urllib.error.URLError, OSError) as failure:
        print(f"the app backend at {args.base} did not answer: {failure}", file=sys.stderr)
        print("start it first - the cached answers must be real ones.", file=sys.stderr)
        return 1

    upstream = state.get("model_server") or {}
    if not upstream.get("reachable"):
        # Without the model server every stage falls back and the cache would be five degraded
        # answers - technically real, and a demo of the wrong thing.
        print(f"the model server at {upstream.get('url')} is unreachable.", file=sys.stderr)
        return 1

    records: list[dict[str, Any]] = []
    for index, (file, name) in enumerate(EXAMPLES):
        source = FIXTURES / file
        if not source.exists():
            print(f"missing fixture {source}", file=sys.stderr)
            return 1
        if index:
            time.sleep(PACE_S)
        print(f"  {file} ...", end="", flush=True)
        started = time.perf_counter()
        prediction = predict(args.base, source)
        print(
            f" {time.perf_counter() - started:.2f}s"
            f" -> {prediction.get('diagram_type')} / {prediction.get('language')}"
            f" ({len((prediction.get('ir') or {}).get('nodes') or [])} nodes)"
        )
        records.append(
            {
                "file": file,
                "name": name,
                "id": prediction["id"],
                "bytes": source.stat().st_size,
                "prediction": prediction,
            }
        )

    document = {
        # Absolute, not "3 days ago": the app prints this date to a person and a relative one would
        # be wrong the moment the bundle is cached (16.3.1) rather than re-rendered.
        "generated": time.strftime("%Y-%m-%d", time.localtime()),
        "backend": state.get("version"),
        "model_url": state.get("model_url"),
        "note": (
            "Generated by scripts/make_offline_examples.py against a running backend. "
            "Every field is what the pipeline answered; nothing is hand-written."
        ),
        "examples": records,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(document, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1024:.1f} kB)")
    return 0


if __name__ == "__main__":  # pragma: no cover - a script
    raise SystemExit(main())
