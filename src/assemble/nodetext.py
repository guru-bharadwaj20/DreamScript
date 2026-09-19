"""Phase 10.1.2 composed with 9.3 - the step that actually puts a label on an assembled node.

    from src.assemble.nodetext import read_nodes
    read_nodes(page, boxes)      # {detected box id: "the words in it"}

## The gap this closes

10.1.1 built node boxes and left `Node.text` empty, because "10.1.2 owns binding". 10.1.2 then
scored the binding - which text region belongs to which node - **without ever reading a
character**, since its question was geometric. Nothing composed the two, so the assembled IR
that S5 diffs against the truth carried no text at all, and every matched node was charged a
`node_substitute`. That single omission was **36.7% of all S5 edits**, more than node detection,
shape and insertion put together.

So this module is deliberately thin: it is the join, not a new idea. CRAFT finds the writing
(9.3's finding that a geometry-derived crop cannot contain a fifth of its own labels), 10.1.2's
cascade decides which node owns each region, and the 9.3 recogniser reads it.

## The measurement, which is a negative result

**Wiring this in does not improve S5, and the looser ownership stages make it worse.** Over the
308 validation pages, on S5's own metric:

    ownership stages accepted   median GED   sub_text   share <= 3   nodes labelled
    none (text off)                    9.0       1947        0.416                0
    containment only                   9.0       1949        0.416              141
    containment + nearest              9.5       2155        0.373              629
    all three                         10.0       2248        0.351              896

So the default is `RULES = ("containment",)` and `s5.assemble(text=False)`: the break-even policy,
off, because compute spent for no gain is not worth spending.

Three things compose to produce that, and only the last is about recognition:

1. `irdiff` charges a substitution unless the label matches the truth **exactly** after
   normalisation, so a near-miss is worth the same as a blank.
2. Just over half the ground-truth nodes carry **no** text at all (`empty_text_share` 0.5048),
   and an empty prediction against an empty truth is free. Writing a label onto one of those
   *creates* an edit that did not exist - 316 of them under the loosest policy.
3. The label that is read is often correct and attached to the **wrong node**. Sampled pairs read
   `got 'notify customer by email' / want 'application rejected'` on a page where another node's
   truth is exactly `notify customer by email`. 9.3's own conclusion was that the remaining error
   is "selection, not recognition", and this is that finding arriving in the assembler.

S3 reads its own crops at **0.6336 exact match**; this path composes detector box, CRAFT region,
ownership and recogniser, and almost nothing survives all four exactly. **That is why the 36.7%
text edit mass is not recoverable by connecting the existing parts** - the gap is ownership, and
it has to be closed before reading the labels can pay for itself.

Multi-line labels are joined in reading order, top to bottom then left to right, because a task
box with two lines of writing is one label in the IR and two CRAFT regions here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.assemble.corpus import Page
from src.assemble.textbind import bind
from src.utils.config import ROOT

#: Read at most this many regions per page. A page of dense writing is a page whose labels are
#: mostly edge labels, and the recogniser is the expensive part of the loop.
MAX_REGIONS = 400

#: Which stages of 10.1.2's cascade a label is accepted from. `containment` is the writing
#: actually inside the box; `nearest` and `hungarian` are the fallbacks that claim the ink no
#: box contains - which on a BPMN page is largely edge labels, and attaching one of those to a
#: task is how a correctly read phrase lands on the wrong node.
RULES = ("containment",)

#: Where a page's read labels are kept between runs. The recogniser is the slow part of S5 by a
#: wide margin and the labels do not change unless the checkpoint does, so scoring a split twice
#: should not read the same handwriting twice.
CACHE = ROOT / "data" / "interim" / "node_text"

_MODEL: Any = None


def reader():
    """The 9.3 recogniser, loaded once. `None` when no checkpoint has been trained yet."""
    global _MODEL
    if _MODEL is None:
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        from src.ocr.s3 import CHECKPOINT

        if not (Path(CHECKPOINT) / "config.json").is_file():
            _MODEL = (None, None, None)
            return _MODEL
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        proc = TrOCRProcessor.from_pretrained(CHECKPOINT)
        model = VisionEncoderDecoderModel.from_pretrained(CHECKPOINT).to(device).eval()
        _MODEL = (model, proc, device)
    return _MODEL


def regions(image: np.ndarray) -> list[list[int]]:
    """Every block of writing on the page, as `[x, y, w, h]` in page pixels."""
    from src.ocr.labelcrops import text_boxes

    return text_boxes(image)[:MAX_REGIONS]


def read_regions(page: Page) -> list[dict] | None:
    """Every region read on this page: its box, its transcript and how it was bound.

    The transcripts are the expensive half and the ownership rule is the half still being
    argued about, so they are stored apart. Re-deciding which node owns which writing is then
    a dictionary lookup rather than another pass over the handwriting.
    """
    path = CACHE / f"{page.name}.json"
    if not path.is_file():
        return None
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return stored.get("regions") if isinstance(stored, dict) else None


def cached(page: Page, boxes: list[dict] | None = None, rules=RULES) -> dict[str, str] | None:
    """A page's labels from an earlier run under `rules`, or None if it has not been read."""
    regions_read = read_regions(page)
    if regions_read is None:
        return None
    return assign(regions_read, boxes, rules)


def assign(regions_read, boxes, rules=RULES) -> dict[str, str]:
    """`{box id: label}` from stored regions, keeping only bindings made by `rules`."""
    lines: dict[str, list[tuple[int, int, str]]] = {}
    for region in regions_read:
        if region.get("rule") not in rules or not region.get("text", "").strip():
            continue
        owner = region.get("owner")
        if owner is None:
            continue
        if boxes is not None:
            if not isinstance(owner, int) or owner >= len(boxes):
                continue
            owner = boxes[owner]["id"]
        lines.setdefault(str(owner), []).append(
            (int(region["box"][1]), int(region["box"][0]), region["text"].strip())
        )
    out: dict[str, str] = {}
    for node_id, found in lines.items():
        found.sort()
        out[node_id] = " ".join(word for _, _, word in found)
    return out


def build_cache(pages: list[Page], boxes_for, batch: int = 32) -> int:
    """Read every page in the parent process and store the labels. Returns pages read.

    S5 scores pages across worker processes, and a worker that loads its own recogniser would put
    one copy of the model on the card per worker - six of them, for a model that is wanted once.
    Reading here and caching means the workers only ever look the answer up.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    done = 0
    for page in pages:
        if read_regions(page) is not None:
            continue
        found = read_page(page, boxes_for(page), batch=batch)
        (CACHE / f"{page.name}.json").write_text(json.dumps({"regions": found}), encoding="utf-8")
        done += 1
        if done % 25 == 0:
            print(f"[nodetext] read {done} pages", flush=True)
    return done


def read_page(page: Page, boxes: list[dict], batch: int = 32) -> list[dict]:
    """Every region of writing on `page`: which box it was bound to, by which rule, and what
    it says. Ownership is *recorded*, not yet applied - `assign` decides what to keep."""
    import cv2

    from src.ocr.labelcrops import crop

    model, proc, device = reader()
    if model is None or not boxes:
        return []

    image = cv2.imread(str(page.image), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return []
    found = regions(image)
    if not found:
        return []

    # The binder works in detector pixels, which is the frame the boxes are already in.
    texts = [[float(x), float(y), float(x + w), float(y + h)] for x, y, w, h in found]
    node_boxes = [[float(v) for v in b["bbox"]] for b in boxes]
    bindings = bind(texts, node_boxes, page.diagonal)

    patches, kept = [], []
    for binding, block in zip(bindings, found, strict=True):
        if binding.node is None:
            continue
        patch = crop(image, list(block))
        if patch is None:
            continue
        patches.append(patch)
        kept.append((binding.node, binding.stage, list(block)))
    if not patches:
        return []

    words = _transcribe(model, proc, device, patches, batch)
    return [
        {"owner": node_index, "rule": rule, "box": block, "text": word}
        for (node_index, rule, block), word in zip(kept, words, strict=True)
    ]


def _transcribe(model, proc, device, patches, batch: int) -> list[str]:
    import torch

    from src.ocr.s3 import BEAMS, MAX_LENGTH

    out: list[str] = []
    with torch.no_grad():
        for start in range(0, len(patches), batch):
            chunk = patches[start : start + batch]
            rgb = [np.stack([p, p, p], axis=-1) for p in chunk]
            pixels = proc(images=rgb, return_tensors="pt").pixel_values.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                generated = model.generate(pixels, max_length=MAX_LENGTH, num_beams=BEAMS)
            out.extend(proc.batch_decode(generated, skip_special_tokens=True))
    return out
