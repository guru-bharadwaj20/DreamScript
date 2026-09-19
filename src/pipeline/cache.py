"""Phase 13.6 - a per-stage cache keyed on the image and the configuration that read it.

    cache = StageCache()
    cache.get("detect", key) or cache.put("detect", key, value)

The key is `sha256(image bytes) + sha256(the stage's own config)`, so a stage is re-run when
either the page or the thing that would change its answer changes, and not otherwise. Hashing
the *bytes* rather than the path matters on this corpus: 1.3.2 found nine duplicate groups
across 25 images, and two paths holding the same photograph should not be read twice.

Only stages whose output is JSON-shaped are cached - the IR, the traversal, the code. The
detector's boxes are cached as their dict form; model weights and images are not, because the
cost of serialising a page of pixels is the cost the cache exists to avoid.

A cache entry records the config hash it was written under, so a stale entry from a different
configuration is a miss rather than a wrong answer.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.utils.config import ROOT

#: Where cached stage outputs live. Under `data/interim` because they are derived and disposable.
CACHE_DIR = ROOT / "data" / "interim" / "pipeline_cache"


def digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()[:32]


def digest_obj(obj: Any) -> str:
    """A stable hash of any JSON-shaped configuration."""
    text = json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def image_key(path: str | Path) -> str:
    """The page's identity: its bytes, not its name."""
    data = Path(path).read_bytes()
    return digest_bytes(data)


class StageCache:
    """Read-through cache for stage outputs, one JSON file per (stage, key)."""

    def __init__(self, directory: Path = CACHE_DIR, enabled: bool = True) -> None:
        self.directory = Path(directory)
        self.enabled = enabled
        self.hits = 0
        self.misses = 0

    def _path(self, stage: str, key: str) -> Path:
        return self.directory / stage / f"{key}.json"

    def get(self, stage: str, key: str) -> Any | None:
        if not self.enabled:
            return None
        path = self._path(stage, key)
        if not path.is_file():
            self.misses += 1
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.misses += 1
            return None
        self.hits += 1
        return payload.get("value")

    def put(self, stage: str, key: str, value: Any) -> Any:
        if not self.enabled:
            return value
        path = self._path(stage, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.write_text(
                json.dumps({"key": key, "value": value}, default=str), encoding="utf-8"
            )
        except (OSError, TypeError, ValueError):
            # A stage whose output will not serialise is simply not cached; it is not an error.
            pass
        return value

    def stats(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses}

    def clear(self) -> int:
        """Remove every entry. Returns how many files went."""
        if not self.directory.is_dir():
            return 0
        gone = 0
        for path in self.directory.rglob("*.json"):
            path.unlink(missing_ok=True)
            gone += 1
        return gone
