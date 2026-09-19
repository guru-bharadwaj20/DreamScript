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

## What a label is worth, and why that is not the CER

`irdiff` charges a text substitution unless the prediction matches the truth **exactly** after
normalisation. A recogniser at 0.15 CER does not remove 85% of those edits - it removes the
share of labels it gets exactly right, which is a different and smaller number. `s5.ceiling_floor`
already models it that way (`(1 - exact) * sub_text`), and it is the honest way to read any gain
here: this module converts *exact* reads into removed edits and nothing else.

Multi-line labels are joined in reading order, top to bottom then left to right, because a task
box with two lines of writing is one label in the IR and two CRAFT regions here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.assemble.corpus import Page
from src.assemble.textbind import bind

#: Read at most this many regions per page. A page of dense writing is a page whose labels are
#: mostly edge labels, and the recogniser is the expensive part of the loop.
MAX_REGIONS = 400

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


def read_nodes(page: Page, boxes: list[dict], batch: int = 32) -> dict[str, str]:
    """`{box id: label}` for the detected boxes on `page`, empty where nothing was bound."""
    import cv2

    from src.ocr.labelcrops import crop

    model, proc, device = reader()
    if model is None or not boxes:
        return {}

    image = cv2.imread(str(page.image), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return {}
    found = regions(image)
    if not found:
        return {}

    # The binder works in detector pixels, which is the frame the boxes are already in.
    texts = [[float(x), float(y), float(x + w), float(y + h)] for x, y, w, h in found]
    node_boxes = [[float(v) for v in b["bbox"]] for b in boxes]
    bindings = bind(texts, node_boxes, page.diagonal)

    patches, owners = [], []
    for binding, block in zip(bindings, found, strict=True):
        if binding.node is None:
            continue
        patch = crop(image, list(block))
        if patch is None:
            continue
        patches.append(patch)
        owners.append((binding.node, block[1], block[0]))
    if not patches:
        return {}

    words = _transcribe(model, proc, device, patches, batch)

    # One node, several regions: reading order is top to bottom, then left to right.
    lines: dict[int, list[tuple[int, int, str]]] = {}
    for (node_index, top, left), word in zip(owners, words, strict=True):
        if word.strip():
            lines.setdefault(node_index, []).append((top, left, word.strip()))
    out: dict[str, str] = {}
    for node_index, found_lines in lines.items():
        found_lines.sort()
        out[boxes[node_index]["id"]] = " ".join(word for _, _, word in found_lines)
    return out


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
