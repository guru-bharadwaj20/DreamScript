"""Phase 10.1.2 for BPMN - read a node's label from where BPMN actually writes it.

    from src.assemble import pagetext
    pagetext.apply(page, diagram)      # sets node.text on a predicted Diagram

## What this is worth, and a correction

S5 reads hdbpmn node labels with `statelabels`, which crops the **node box** - right for an
automaton, where the state name is written inside the circle, and wrong in principle for BPMN,
where the label is often written outside the glyph.

**How wrong was first measured as "completely", and that was a bug in the measurement.** The
first comparison fed ground-truth boxes, which live in the IR's native coordinate space
(2480x1612 on the page tested), to an image stored at 1280x832, so every crop fell outside the
image and the reader scored 0 of 51. Scaling by `page.scale` first:

    statelabels (node box)   19/51 = 0.373
    pagetext (ownership)     23/51 = 0.451

So the honest gain is **+7.8 points**, not a rescue. `sub_text` is 1,474 edits and 44.8% of
hdbpmn's S5 edit mass, which remains the criterion's largest single component, but it is a
two-thirds-wrong reader rather than a dead one. The misses are mostly near: `evaluate
application` read as `evaluation application`, `create new bank account` as `create relevant
bank accou`.

## Why this is an adapter and not a new idea

`src.ocr.ownership` already solves exactly this: CRAFT finds the writing, a learned ranker
decides which element owns each line, and Hungarian assignment resolves the competition. It
reads hdbpmn at **0.5823 exact** through S3. It was built for `labelcrops`, over *annotated*
diagrams, and nothing pointed it at a *predicted* one.

The one thing that does not transfer is how it decides which rule to apply. `ownership.kind_of`
reads the BPMN element id - `Participant_`, `Task_`, `startEvent_` - and a predicted node is
called `d003`. What a predicted node does carry is the **detector's shape class**, which is the
same information in a different vocabulary: a BPMN event is drawn as a circle and writes its
label underneath, a task is a rounded rectangle and writes it inside, a pool is a very large
rectangle and writes it in a rotated header strip.

So this module translates shape to the vocabulary `ownership` already understands, by giving
each predicted node a synthetic id with the right prefix, and translates the answer back. That
keeps one implementation of the ownership rules rather than a second copy that will drift.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from src.utils.config import ROOT

#: Where read labels are cached, one JSON per page.
CACHE = ROOT / "data" / "interim" / "page_text"

#: Beam width for the decode, matching `statelabels` and S3's own evaluation.
BEAMS = 4

#: Corpora whose labels are written outside the glyph and need this reader rather than
#: `statelabels`. fa_bresler is deliberately absent: its state names are inside the circle and
#: `statelabels` reads them at 94.4%.
SOURCES = ("hdbpmn",)

#: A node covering at least this share of the page is a pool or a lane, whose title is written
#: in a rotated header strip rather than beside the shape.
CONTAINER_AREA = 0.12

#: Detector classes whose BPMN counterpart writes its label outside the drawn glyph.
EXTERNAL_SHAPES = ("circle", "double-circle", "diamond")


def _synthetic_id(node: dict, page_area: float) -> str:
    """A `src.ocr.ownership` id whose prefix selects the rule this node's shape implies."""
    box = node.get("bbox") or [0, 0, 0, 0]
    area = float(box[2]) * float(box[3])
    if page_area > 0 and area / page_area >= CONTAINER_AREA:
        return f"Participant_{node['id']}"
    if str(node.get("shape")) in EXTERNAL_SHAPES:
        return f"Event_{node['id']}"
    return f"Task_{node['id']}"


def elements_for(diagram: dict[str, Any], page_shape) -> tuple[list[dict], dict[str, str]]:
    """`ownership`-shaped elements for a predicted diagram, plus the map back to node ids."""
    height, width = page_shape[:2]
    page_area = float(height) * float(width)
    elements, back = [], {}
    for node in diagram.get("nodes", []):
        if not node.get("bbox"):
            continue
        name = _synthetic_id(node, page_area)
        back[name] = str(node["id"])
        elements.append({"id": name, "bbox": node["bbox"], "kind": "node", "text": ""})
    return elements, back


def read_page(page, diagram: dict[str, Any], batch: int = 32) -> dict[str, str]:
    """`{node id: label}` for one predicted page, through the ownership rules."""
    import cv2
    import torch

    from src.assemble import statelabels
    from src.ocr import s3
    from src.ocr.labelcrops import cached_lines, crop, page_labels, text_boxes
    from src.utils import gpu

    gpu.cap()
    image = cv2.imread(str(page.image), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return {}

    elements, back = elements_for(diagram, image.shape)
    if not elements:
        return {}

    lines = cached_lines(getattr(page, "id", "") or page.name) or text_boxes(image)
    if not lines:
        return {}
    # **`page_labels`, not `ownership.blocks` directly.** The first version called the geometric
    # rules straight, which quietly skipped the fitted ranker: `page_labels` uses
    # `ownlearn.assign` whenever a ranker exists and only falls back to pure geometry when it
    # does not. The ranker is the part that was refitted to a learned-pick CER of 0.1441, so
    # bypassing it left most of that work on the floor.
    blocks = page_labels(image, elements, [list(map(int, b[:4])) for b in lines])

    ids, patches = [], []
    for element in elements:
        block = blocks.get(element["id"])
        if block is None:
            continue
        patch = crop(image, block)
        if patch is None:
            continue
        ids.append(back[element["id"]])
        patches.append(patch)
    if not ids:
        return {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = statelabels._model(device, "hdbpmn")
    with tempfile.TemporaryDirectory() as tmp:
        files = []
        for name, patch in zip(ids, patches, strict=True):
            path = Path(tmp) / f"{name}.png"
            cv2.imwrite(str(path), patch)
            files.append(path)
        texts = s3.predict(model, s3.processor(), files, device, batch=batch, beams=BEAMS)
    return {name: text.strip() for name, text in zip(ids, texts, strict=True) if text.strip()}


def apply(page, diagram: Any) -> int:
    """Read this page's node labels onto `diagram` in place. Returns how many were set."""
    plain = diagram.to_dict() if hasattr(diagram, "to_dict") else diagram
    labels = cached(page)
    if labels is None:
        labels = read_page(page, plain)
        store(page, labels)
    if not labels:
        return 0
    written = 0
    for node in getattr(diagram, "nodes", plain.get("nodes", [])):
        key = str(node.id if hasattr(node, "id") else node.get("id"))
        text = labels.get(key)
        if not text:
            continue
        if hasattr(node, "text"):
            node.text = text
        else:
            node["text"] = text
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
    if payload.get("checkpoint") != checkpoint_fingerprint("hdbpmn"):
        return None
    return dict(payload["labels"])


def store(page, labels: dict[str, str]) -> Path:
    import json

    from src.assemble.statelabels import checkpoint_fingerprint

    path = CACHE / f"{getattr(page, 'source', 'loose')}__{page.id or page.name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"checkpoint": checkpoint_fingerprint("hdbpmn"), "labels": labels}),
        encoding="utf-8",
    )
    return path
