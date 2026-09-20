"""Phase 10.1.2 for state machines - reading the label inside a state, with a recogniser that
has actually seen one.

    from src.assemble.statelabels import read_page, cached, build_cache
    read_page(page, boxes)      # {detected box id: "q0"}

## Why this exists and `nodetext` was not enough

`nodetext` composes CRAFT + ownership + S3's recogniser, and on fa_bresler it labels **nothing**:
CRAFT finds 0.69 text regions per page against 4.9 labelled states, and what S3 returns for the
regions it does find is BPMN vocabulary - `'receive order typ choosen'` for a page whose states
are `q0 q1 q2`. Two separate causes, both fatal:

1. **A state's label is inside its circle**, so there is no text-detection problem to solve. The
   node box already is the crop. CRAFT is being asked a question that does not need asking, and
   answering it badly.
2. **S3 had never seen an automaton label.** `labelcrops.build` is hardcoded to
   `load_ir(["hdbpmn"])`, so the label-crop corpus it trains and evaluates on contains no
   fa_bresler in any split. S3 scores **1.3% exact** on fa_bresler node labels (3 of 233) -
   not badly, but *blindly*: it answers with the only vocabulary it was ever shown.

So this module crops the node box directly and reads it with a recogniser fine-tuned on the
text-crop corpus's fa_bresler rows, which do exist (2,646 train crops, 987 of them node labels).
On val node crops that model scores **94.4% exact against S3's 1.3%**.

## What that number is and is not

94.4% is measured on *annotated* boxes, because `textcrops.add_detected` is deliberately
hdbpmn-only - fa_bresler pages are renders, and detecting on a render measures the renderer. The
pipeline reads *detected* boxes instead, so the number that counts is the one S5 reports after
this is wired in, not the crop-level one.

Crops are cut the same way the training corpus cuts them - `textcrops.cut` with `INSET`, height
normalised - because a recogniser fed a differently-framed crop is being asked a different
question. Reading a whole circle interior at 0.66 scale instead scored 0/28.
"""

from __future__ import annotations

import json
import os
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.utils.config import ROOT

#: The state-machine recogniser, trained on fa_bresler's own crops.
CHECKPOINT = ROOT / "experiments" / "ocr" / "trocr_fa"

#: S3's checkpoint, which *is* trained on hdbpmn and reads its task labels. `HDBPMN_RECOGNISER`
#: names a directory under `experiments/ocr`, so a better recogniser reaches S5 as a run rather
#: than an edit. The default is `trocr_large` because it measures better on S3's own protocol -
#: **CER 0.2133 / exact 0.6526 against `trocr_s3`'s 0.2364 / 0.6198** over the same 2,835 crops
#: and 23 writer-disjoint evaluation writers - and because `sub_text` is more than half of
#: hdbpmn's median GED, so this is the only remaining lever on S5.
HDBPMN_CHECKPOINT = ROOT / "experiments" / "ocr" / os.environ.get(
    "HDBPMN_RECOGNISER", "trocr_large"
)

#: Which recogniser reads which corpus. The split is not a preference - each model is blind to
#: the other's labels: S3 scores 1.3% on automaton labels, and the fa model never saw a sentence.
RECOGNISER = {"fa_bresler": CHECKPOINT}

#: Read labels once in the parent and look them up in the workers - see `build_cache`.
CACHE = ROOT / "data" / "interim" / "state_text"

#: Beam width for the decode. 4 is what S3's own evaluation uses.
BEAMS = 4


def checkpoint_for(source: str) -> Path:
    """The recogniser that has actually seen this corpus's labels."""
    return RECOGNISER.get(source, HDBPMN_CHECKPOINT)


#: `s5.run` fans pages out with `prefer="threads"`, and transformers resolves its submodules
#: lazily on first attribute access. Six threads reaching that at once raced and one lost with
#: `ImportError: cannot import name 'VisionEncoderDecoderModel'` - a real traceback from an
#: import that succeeds when serialised. The lock covers the import, not just the construction,
#: because the import is what raced.
_IMPORT_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def processor():
    """One resident `TrOCRProcessor`, imported under the same lock as the model.

    **The lock has to cover every lazily-resolved transformers symbol, not one of them.** The
    first version locked `VisionEncoderDecoderModel` only, and the next parallel S5 run died the
    same way one symbol along: `ImportError: cannot import name 'TrOCRProcessor'`. transformers
    resolves submodules on first attribute access and `s5.run` uses `prefer="threads"`, so any
    first touch from six threads at once is the same race.

    Cached as well as locked, because `from_pretrained` on the processor was being called once
    per page per stage for the same reason the model was.
    """
    with _IMPORT_LOCK:
        from transformers import TrOCRProcessor

        from src.ocr.s3 import MODEL

    return TrOCRProcessor.from_pretrained(MODEL)


@lru_cache(maxsize=2)
def _load(checkpoint: str, device_key: str):
    """One resident recogniser per checkpoint. **Not caching this cost 5.4577 s per page.**

    `from_pretrained` reads ~1.3 GB from disk and rebuilds the graph, and it was being called
    once per page per stage - so a 25-page run loaded the same two checkpoints fifty times, and
    13.7's assemble stage measured 5.4577 s median against a detector at 0.0717 s. The weights
    do not change between pages; there was never a reason to re-read them.

    Keyed on the device as well as the path because a CPU and a CUDA copy are different objects,
    and `maxsize=2` because that is how many checkpoints exist - fa_bresler's and hdbpmn's.
    """
    with _IMPORT_LOCK:
        from transformers import VisionEncoderDecoderModel

    import torch

    return VisionEncoderDecoderModel.from_pretrained(checkpoint).to(torch.device(device_key))


def _model(device, source: str = "fa_bresler"):
    path = checkpoint_for(source)
    if not path.is_dir():
        raise FileNotFoundError(f"no recogniser at {path}; train it before reading labels")
    return _load(str(path), str(device))


def read_page(page, boxes: list[dict], batch: int = 32) -> dict[str, str]:
    """`{box id: label}` for every detected node box on one page."""
    from src.utils import gpu

    gpu.cap()
    import cv2
    import torch

    from src.ocr import s3, textcrops

    image = cv2.imread(str(page.image), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return {}

    ids, patches = [], []
    for box in boxes:
        patch = textcrops.cut(image, box["bbox"], textcrops.INSET)
        if patch is None:
            continue
        ids.append(box["id"])
        patches.append(patch)
    if not ids:
        return {}

    # `s3.predict` reads files, so the crops go through a scratch directory rather than being
    # re-implemented here: one cropping rule, shared with the corpus the model trained on.
    import tempfile

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _model(device, page.source)
    with tempfile.TemporaryDirectory() as tmp:
        files = []
        for name, patch in zip(ids, patches, strict=True):
            path = Path(tmp) / f"{name}.png"
            cv2.imwrite(str(path), patch)
            files.append(path)
        texts = s3.predict(model, processor(), files, device, batch=batch, beams=BEAMS)
    return {name: text.strip() for name, text in zip(ids, texts, strict=True)}


@lru_cache(maxsize=4)
def checkpoint_fingerprint(source: str = "fa_bresler") -> str:
    """Which recogniser a cached label was read with.

    **Without this the cache outlives the model and silently hides a retrain.** The key was the
    page name alone, so when S3 was retrained the 338 stored files stayed valid and S5 went on
    scoring text produced by the previous checkpoint - two retrains in one night reached the
    criterion not at all, and the report was byte-identical each time, which reads like "the
    change did nothing" rather than "the change was never applied".

    The weights file's size and mtime rather than its contents: it is 1.3 GB, this is called per
    page, and a retrain always rewrites it. `code_key` in `src.pipeline.cache` hashes source text
    because source files are small and a checkout can move mtimes without changing behaviour;
    neither applies to a 1.3 GB checkpoint that only ever changes when it is rewritten.
    """
    path = checkpoint_for(source) / "model.safetensors"
    try:
        stat = path.stat()
    except OSError:
        return "none"
    return f"{stat.st_size:x}-{int(stat.st_mtime):x}"


def cached(page) -> dict[str, str] | None:
    """The stored labels for a page, or None when it has not been read by *this* recogniser."""
    path = CACHE / f"{page.name}.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or "labels" not in payload:
        # Written before the fingerprint existed, so which model produced it is unknowable.
        return None
    if payload.get("checkpoint") != checkpoint_fingerprint(getattr(page, "source", "fa_bresler")):
        return None
    return dict(payload["labels"])


def _payload(page, labels: dict[str, str]) -> str:
    return json.dumps(
        {
            "checkpoint": checkpoint_fingerprint(getattr(page, "source", "fa_bresler")),
            "labels": labels,
        }
    )


def build_cache(pages: list[Any], boxes_for, batch: int = 32) -> int:
    """Read every page in the parent process and store the labels. Returns pages read.

    S5 scores pages across workers, and a worker that loaded its own recogniser would put one
    copy of the model on the card per worker. Reading here means the workers only look up.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    done = 0
    for page in pages:
        if cached(page) is not None:
            continue
        labels = read_page(page, boxes_for(page), batch=batch)
        (CACHE / f"{page.name}.json").write_text(_payload(page, labels), encoding="utf-8")
        done += 1
        if done % 25 == 0:
            print(f"[statelabels] read {done} pages", flush=True)
    return done
