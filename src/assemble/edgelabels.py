"""Phase 10.1.2 for edges - read the trigger written beside a connector.

    from src.assemble import edgelabels
    edgelabels.read_page(page, diagram)      # -> {edge id: label}

## Why this module had to exist

`statelabels` reads the text *inside* a shape. Nothing read the text *beside a line*, and on a
state machine that is the half that carries the meaning: the states are just names, but the
triggers are the alphabet. Measured on the 25 golden pages before this module, assembly produced
**0 of 113 annotated edge labels**, and the consequence was not a slightly worse score - with no
triggers every transition is an epsilon transition, epsilon-closure merges every state into one,
and the emitted machine had **one state and zero transitions on 9 of 9 state machines**. It
parsed. It could not accept or reject a single string.

## It is assembled from parts that were already here

Nothing below is a new idea; the pieces were written for the S3 *training* corpus, which crops
edge labels from ground-truth polylines, and were never pointed at *predicted* edges at
inference:

    textcrops.glyph_height   a page-relative unit, median node height / 6
    textcrops.edge_label_box the rectangle beside a polyline's arc-length midpoint
    textcrops.off_line_ink   is there ink here that is not the connector itself
    s3.predict               the recogniser that trained on exactly those crops

So the training distribution and the inference distribution are the same crop rule, which is the
whole reason to reuse them rather than write a fresh cropper here.

## The ink gate is what makes this safe to run

Most predicted edges have no label - the annotation labels 113 of 397 edges on the golden set,
and assembly over-segments, so the majority of predicted edges are fragments with nothing written
beside them. Handing every one of them to a decoder would invite the hallucination 9.3 measured:
an autoregressive decoder asked to read blank paper returns a confident short string. So a crop
is only decoded when `off_line_ink` finds ink that is not the connector.

The threshold was then swept rather than argued. The argument would have been that a false
trigger is worse than a missing one - a wrong trigger produces a machine that runs and is wrong,
where a missing trigger produces an epsilon edge, which is what the code already does today - so
the floor should sit above `off_line_ink`'s own. **The measurement says otherwise**: precision
falls as the floor rises, because what a higher floor rejects is the handwriting rather than the
connector. `MIN_INK` is therefore the lowest value swept, and the numbers are on the constant.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from src.utils.config import ROOT

#: Where read labels are stored, one JSON per page, so a re-score does not re-decode.
CACHE = ROOT / "data" / "interim" / "edge_text"

#: Beam width for the decode. 4, the same as `statelabels` and S3's own evaluation.
BEAMS = 4

#: Ink fraction a crop must carry, over and above the connector, before it is decoded at all.
#: **0.001, the lowest floor swept, because the expected trade-off does not exist here.** Over 12
#: fa_bresler pages the floor was swept 0.001 -> 0.016 and precision does not pay for recall:
#:
#:     0.001  recall 0.778  precision 0.856        0.006  recall 0.697  precision 0.841
#:     0.002  recall 0.768  precision 0.854        0.010  recall 0.616  precision 0.824
#:     0.004  recall 0.747  precision 0.851        0.016  recall 0.556  precision 0.821
#:
#: Precision *falls* as the floor rises, so a higher floor was removing labelled edges faster
#: than unlabelled ones - the ink it was rejecting was the handwriting, not the connector. The
#: gate still earns its place by dropping the majority of over-segmented fragments outright;
#: it just should not be tuned above the point where it starts eating real labels.
MIN_INK = 0.001

#: Corpora whose edges carry labels worth reading. fa_bresler's triggers are its alphabet;
#: hdbpmn writes sequence-flow conditions the same way.
SOURCES = ("fa_bresler", "hdbpmn")


def _unit(diagram: dict[str, Any], page_size: tuple[float, float]) -> float:
    """`textcrops.glyph_height` for a *predicted* diagram, which has no `meta`."""
    from src.ocr import textcrops

    nodes = [n for n in diagram.get("nodes", []) if n.get("bbox")]
    if not nodes:
        return max(8.0, min(page_size) / 60.0)
    return textcrops.glyph_height({"nodes": nodes, "meta": {"image_size": page_size}})


def candidates(page, diagram: dict[str, Any]) -> tuple[list[str], list[Any]]:
    """`(edge ids, crops)` for every predicted edge with ink written beside it."""
    import cv2

    from src.ocr import textcrops

    image = cv2.imread(str(page.image), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return [], []
    height, width = image.shape[:2]
    size = (float(width), float(height))
    unit = _unit(diagram, size)

    ids, patches = [], []
    for edge in diagram.get("edges", []):
        polyline = edge.get("polyline") or edge.get("points")
        if not polyline:
            continue
        box = textcrops.edge_label_box(polyline, unit, size)
        if box is None:
            continue
        # The gate: ink that is not the connector. Without it every fragment of an
        # over-segmented edge is handed to a decoder that will answer anyway.
        if not textcrops.off_line_ink(image, box, polyline, unit, floor=MIN_INK):
            continue
        patch = textcrops.cut(image, box)
        if patch is None:
            continue
        ids.append(str(edge.get("id")))
        patches.append(patch)
    return ids, patches


def read_page(page, diagram: dict[str, Any], batch: int = 32) -> dict[str, str]:
    """`{edge id: trigger}` for one page. Edges with no ink beside them are simply absent."""
    from src.utils import gpu

    gpu.cap()
    import cv2
    import torch

    from src.assemble import statelabels
    from src.ocr import s3

    ids, patches = candidates(page, diagram)
    if not ids:
        return {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = statelabels._model(device, getattr(page, "source", "fa_bresler"))
    # `s3.predict` reads files, so the crops go through a scratch directory rather than being
    # re-implemented here - one cropping rule, shared with the corpus the model trained on.
    with tempfile.TemporaryDirectory() as tmp:
        files = []
        for name, patch in zip(ids, patches, strict=True):
            path = Path(tmp) / f"{name}.png"
            cv2.imwrite(str(path), patch)
            files.append(path)
        texts = s3.predict(model, statelabels.processor(), files, device, batch=batch, beams=BEAMS)

    # The model is cached by `statelabels._load` and deliberately stays resident: freeing it here
    # would force a 1.3 GB re-read on the next page, which is the cost this caching removed.
    return {name: text.strip() for name, text in zip(ids, texts, strict=True) if text.strip()}


def apply(page, diagram: Any) -> int:
    """Read this page's edge labels onto `diagram` in place. Returns how many were set."""
    plain = diagram.to_dict() if hasattr(diagram, "to_dict") else diagram
    labels = cached(page)
    if labels is None:
        labels = read_page(page, plain)
        # Stored even when empty, so a page with nothing written beside any connector is not
        # re-decoded on every re-score. `cached` distinguishes "no file" from "read, found none".
        store(page, labels)
    if not labels:
        return 0
    written = 0
    for edge in getattr(diagram, "edges", plain.get("edges", [])):
        key = str(edge.id if hasattr(edge, "id") else edge.get("id"))
        text = labels.get(key)
        if not text:
            continue
        if hasattr(edge, "label"):
            edge.label = text
        else:
            edge["label"] = text
        written += 1
    return written


def cached(page) -> dict[str, str] | None:
    """Stored labels for a page, or None when they were not read by *this* recogniser.

    The checkpoint fingerprint is not decoration: without it a retrain leaves every stored file
    valid, and the criterion is re-scored with text the previous model produced. That happened -
    `statelabels` had 338 files keyed on the page name alone, and two S3 retrains reached S5 not
    at all while each report came back byte-identical.
    """
    from src.assemble.statelabels import checkpoint_fingerprint

    path = CACHE / f"{getattr(page, 'source', 'loose')}__{page.id or page.name}.json"
    if not path.is_file():
        return None
    try:
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or "labels" not in payload:
        return None
    if payload.get("checkpoint") != checkpoint_fingerprint("fa_bresler"):
        return None
    return dict(payload["labels"])


def store(page, labels: dict[str, str]) -> Path:

    from src.assemble.statelabels import checkpoint_fingerprint

    path = CACHE / f"{getattr(page, 'source', 'loose')}__{page.id or page.name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"checkpoint": checkpoint_fingerprint("fa_bresler"), "labels": labels}) + "\n",
        encoding="utf-8",
    )
    return path
