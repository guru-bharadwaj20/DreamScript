"""Phase 10.1.9 - putting text on the assembled nodes, because S5 was paying for its absence.

    python -m src.assemble.labels --split val        # build the cache and score it
    python -m src.assemble.labels --split test

S5's error decomposition (`src.assemble.s5`) says the single largest block of edit mass -
**36.7% of every edit on the 308 val pages** - is `node_substitute` on *text*, and that all of it
is one omission rather than one inaccuracy: `tracing.to_diagram` emits `Node.text = None` for
every node it builds. 10.1.1 left text out on purpose ("10.1.2 owns binding") and 10.1.2 scored
the binding without ever reading a character, so nothing in Phase 10 ever composed the two. This
module is that composition, and nothing more:

    1  3.2.7's text proposals for the page (`textbind.text_boxes`)
    2  10.1.2's cascade binds each proposal to a detected node (`textbind.bind`)
    3  the proposals bound to one node are **merged into a single block** and cropped once
    4  9.3.1's `labelcrops.crop` normalises that block exactly as S3's training crops were
    5  S3's fine-tuned TrOCR checkpoint reads it

Step 3 is the one choice here worth arguing. Reading each proposal separately and joining the
strings would put this stage in a domain S3 never saw - it was fine-tuned on one crop per label,
not one per fragment - and would add a word-ordering problem on top of a recognition one. Merging
first keeps the crop in the distribution the checkpoint was trained on.

## What it is worth, on val (308 pages, 1,947 text substitutions to remove)

    condition                            median GED   hdbpmn median   text substitutions
    no text at all (before)                    10.0            37.0                1,947
    read (this module)                          8.0            31.0                1,431
    perfect text (oracle, unreachable)          5.0            23.5                    0

So the stage removes **26.5% of the text edit mass** and takes two edits off the pooled median -
a real gain, and well short of the oracle, for two reasons that are both measured upstream rather
than guessed at here: S3 reads a node label exactly right **0.654** of the time on *ground-truth*
crops, and 10.1.2's binding is **0.8616** accurate, so the compound ceiling on this stage is
about 0.56 even before the crops get worse than S3's. An exactly-right label is the only kind
that removes an edit - GED charges one edit for a wrong string and one for a missing one - which
is why a CER of 0.1964 buys so much less here than it looks like it should.

**It is not enough to move S5 and that is the point of measuring it.** See the S5 row.

## The cache

Reading 470 pages is a few minutes of GPU and the answer never changes, so transcripts are cached
to `experiments/assemble/labels.json`, keyed by page name. `--rebuild` forces a re-read.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from src.assemble import s5, textbind
from src.assemble.corpus import RUNS, Page, pages, truth
from src.ir.model import Diagram

#: Transcripts, keyed by page name then node id. Deleting it costs a GPU pass, nothing else.
CACHE = RUNS / "labels.json"

#: Beam width. S3 reports its 0.1964 CER at 4 beams, so this stage reads at the same setting or
#: the number it inherits is not the number it was given.
BEAMS = 4

#: Crops per generate() call.
BATCH = 24


# ------------------------------------------------------------------------------------------
# crops
# ------------------------------------------------------------------------------------------


def blocks(page: Page) -> dict[str, list[int]]:
    """Node id -> the one label block bound to it, as `[x, y, w, h]` in detector pixels.

    The node ids are `tracing.node_boxes`' `d{index}`, which index the same filtered detection
    list `textbind.node_boxes` does - same source, same `arrowhead` exclusion, same 0.25 score
    floor - so the binder's node index *is* the tracer's node id and no geometric re-matching is
    needed between the two stages.
    """
    texts = textbind.text_boxes(page)
    nodes = textbind.node_boxes(page)
    if not texts or not nodes:
        return {}

    owned: dict[int, list[list[float]]] = {}
    for binding in textbind.bind(texts, nodes, page.diagonal):
        if binding.node is not None:
            owned.setdefault(binding.node, []).append(texts[binding.text])

    out: dict[str, list[int]] = {}
    for index, boxes in owned.items():
        x1 = min(b[0] for b in boxes)
        y1 = min(b[1] for b in boxes)
        x2 = max(b[0] + b[2] for b in boxes)
        y2 = max(b[1] + b[3] for b in boxes)
        if x2 > x1 and y2 > y1:
            out[f"d{index:03d}"] = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
    return out


def read_pages(held: list[Page], *, beams: int = BEAMS, batch: int = BATCH) -> dict[str, dict]:
    """Every page's label blocks, cropped and transcribed in one batched GPU pass."""
    import cv2
    import torch
    from PIL import Image
    from transformers import VisionEncoderDecoderModel

    from src.ocr import labelcrops
    from src.ocr.s3 import CHECKPOINT, MAX_LENGTH, processor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VisionEncoderDecoderModel.from_pretrained(CHECKPOINT).to(device).eval()
    proc = processor()

    crops: list[Any] = []
    keys: list[tuple[str, str]] = []
    for page in held:
        image = cv2.imread(str(page.image), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        for node_id, block in blocks(page).items():
            patch = labelcrops.crop(image, block)
            if patch is None:
                continue
            crops.append(Image.fromarray(patch).convert("RGB"))
            keys.append((page.name, node_id))

    texts: list[str] = []
    with torch.no_grad():
        for start in range(0, len(crops), batch):
            chunk = crops[start : start + batch]
            values = proc(images=chunk, return_tensors="pt").pixel_values.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                generated = model.generate(values, max_length=MAX_LENGTH, num_beams=beams)
            texts.extend(proc.batch_decode(generated, skip_special_tokens=True))

    out: dict[str, dict] = {page.name: {} for page in held}
    for (name, node_id), text in zip(keys, texts, strict=True):
        out[name][node_id] = text.strip()
    return out


# ------------------------------------------------------------------------------------------
# the cache, and applying it
# ------------------------------------------------------------------------------------------


def cached(held: list[Page], *, rebuild: bool = False) -> dict[str, dict]:
    """Transcripts for `held`, reading only the pages the cache is missing."""
    store: dict[str, dict] = {}
    if CACHE.is_file() and not rebuild:
        store = json.loads(CACHE.read_text(encoding="utf-8"))
    missing = [p for p in held if p.name not in store]
    if missing:
        store.update(read_pages(missing))
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(store, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return store


def apply(diagram: Diagram, transcripts: dict[str, str]) -> Diagram:
    """Write `transcripts` onto the diagram's nodes, in place. Unread nodes keep `None`."""
    for node in diagram.nodes:
        text = transcripts.get(node.id)
        if text:
            node.text = text
    return diagram


def assemble(page: Page, transcripts: dict[str, dict]) -> Diagram:
    """`s5.assemble` with this stage's labels on it."""
    return apply(s5.assemble(page), transcripts.get(page.name, {}))


# ------------------------------------------------------------------------------------------
# scoring: what the stage is worth in S5's own units
# ------------------------------------------------------------------------------------------


def run(split: str = "val", limit: int | None = None, rebuild: bool = False) -> dict[str, Any]:
    """Score the same pages twice - without text and with it - and report the difference."""
    held = [p for p in pages() if p.split == split]
    if limit:
        held = held[:limit]
    store = cached(held, rebuild=rebuild)

    before: list[dict] = []
    after: list[dict] = []
    for page in held:
        actual = truth(page)
        plain = s5.assemble(page)
        before.append({"page": page.name, "source": page.source, **s5.decompose(plain, actual)})
        read = apply(plain, store.get(page.name, {}))
        after.append({"page": page.name, "source": page.source, **s5.decompose(read, actual)})

    sources = sorted({r["source"] for r in before})
    return {
        "stage": "10.1.9",
        "split": split,
        "beams": BEAMS,
        "checkpoint": "S3's fine-tuned trocr-base-handwritten",
        "before": s5.summarise(before),
        "after": s5.summarise(after),
        "by_source": {
            s: {
                "before": s5.summarise([r for r in before if r["source"] == s]),
                "after": s5.summarise([r for r in after if r["source"] == s]),
            }
            for s in sources
        },
        "text_edits_removed": sum(r["sub_text"] for r in before)
        - sum(r["sub_text"] for r in after),
        "text_edits_before": sum(r["sub_text"] for r in before),
        "nodes_read": sum(len(store.get(p.name, {})) for p in held),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="val", choices=("val", "test"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args(argv)

    result = run(args.split, args.limit, args.rebuild)
    out = RUNS / f"labels_{args.split}.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
