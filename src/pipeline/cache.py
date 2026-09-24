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

import contextlib
import hashlib
import json
import os
from functools import lru_cache
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


#: The tree `code_key` fingerprints: all of `src/`.
#:
#: It was five packages - `pipeline`, `assemble`, `codegen`, `ocr`, `parse` - and a stage's answer
#: is decided by more than five. `detect` runs the detector, `ir` defines the document, `preprocess`
#: produces the ink mask every traced edge is derived from, `llm` answers `generate`'s first rung,
#: `utils` holds the seeding. Editing `preprocess/binarize.py` changed the mask, changed the trace,
#: changed the IR - and left the key untouched, so the cache went on serving IR the current code
#: would not produce. That is the same failure `code_key` was written to stop, one directory over.
#:
#: Whole-tree rather than a wider hand-maintained tuple, because the tuple is the defect: any list
#: of "the packages that matter" is a list that goes stale the next time a stage reaches one more.
#: The cost is ~325 files of a few hundred kilobytes hashed once per process, against stage work
#: measured in hundreds of milliseconds, and it errs towards an extra miss rather than a wrong hit.
CODE_ROOT = ROOT / "src"


@lru_cache(maxsize=1)
def code_key() -> str:
    """A digest of the code that produces stage outputs.

    **Without this the cache answers with results the current code would not produce.** The key
    was the image plus a config hash of `confidence_floor`, `generate` and `read_text` - three
    constructor arguments - and nothing about the pipeline itself. Wiring the S5 assembly stages
    in changed what `assemble` returns for every page, and a warm cache went on serving the old
    IR: a page whose machine had been rebuilt with real states and transitions still came back
    with one merged state and none. That is the failure this module's own docstring promises
    does not happen - "a stale entry from a different configuration is a miss rather than a
    wrong answer" - and it was true only of configuration, not of code.

    Hashing source text rather than mtimes, because a checkout, a rebase or a copy all move
    mtimes without changing behaviour, and the point is to invalidate on behaviour. Computed
    once per process: a few hundred small files, a few milliseconds, against stage work measured
    in hundreds of milliseconds. See `CODE_ROOT` on why it is the whole tree.
    """
    digest = hashlib.sha256()
    if not CODE_ROOT.is_dir():
        return digest.hexdigest()[:16]
    for file in sorted(CODE_ROOT.rglob("*.py")):
        # Relative path as well as content: moving a module without editing it changes which
        # import resolves where, and that changes answers.
        digest.update(str(file.relative_to(CODE_ROOT)).replace("\\", "/").encode("utf-8"))
        try:
            digest.update(file.read_bytes())
        except OSError:
            # An unreadable file is a reason to invalidate, not to crash.
            digest.update(str(file).encode("utf-8"))
    return digest.hexdigest()[:16]


#: Environment variables that change what a stage returns. Not configuration the pipeline holds -
#: configuration the *process* holds, which is exactly why it was missing from the key.
#:
#: `assemble.s5` reads three switches from the environment rather than from arguments, for a
#: reason it documents: `run` fans pages out to joblib workers, which are separate processes, and
#: a module constant set in the parent is re-imported at its default in every child. Three more
#: name a *checkpoint*: `S3_CHECKPOINT` and `HDBPMN_RECOGNISER` pick the text recogniser,
#: `ARROW_WEIGHTS` picks the pose model whose keypoints become the edges.
#:
#: Every one of them decides an answer and none of them was in the key, so flipping one and
#: re-running returned the previous configuration's result from a warm cache - the failure this
#: module's docstring promises does not happen.
ENV_KEYS = (
    "DREAMSCRIPT_ARROW_EDGES",
    # 13.4's first rung: set it and `generate` asks a served adapter, leave it and the emitter
    # answers. Nothing decides more about what comes back.
    "DREAMSCRIPT_MODEL_URL",
    "DREAMSCRIPT_EDGE_TEXT",
    "DREAMSCRIPT_PAGE_TEXT",
    "ARROW_WEIGHTS",
    "HDBPMN_RECOGNISER",
    "S3_CHECKPOINT",
)


def env_key(names: tuple[str, ...] = ENV_KEYS) -> str:
    """A digest of the environment that decides a stage's output.

    Unset and set-to-the-default are deliberately *different* keys. The alternative is to bake
    each variable's default in here, which is a fourth place for the defaults to drift from the
    modules that own them; an extra miss the first time a variable is set explicitly is cheaper
    than a wrong hit.
    """
    return digest_obj({name: os.environ.get(name) for name in names})


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
        if payload.get("code") != code_key():
            # Written by different code. A miss, not a wrong answer.
            self.misses += 1
            return None
        self.hits += 1
        return payload.get("value")

    def put(self, stage: str, key: str, value: Any) -> Any:
        if not self.enabled:
            return value
        path = self._path(stage, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # A stage whose output will not serialise is simply not cached; it is not an error.
        with contextlib.suppress(OSError, TypeError, ValueError):
            path.write_text(
                json.dumps({"key": key, "code": code_key(), "value": value}, default=str) + "\n",
                encoding="utf-8",
            )
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
