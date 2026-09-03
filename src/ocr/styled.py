"""Phase 8.7 - routing text crops to a per-style OCR head, and the control that prices the routing.

    python -m src.ocr.styled --build    # cache the text crops
    python -m src.ocr.styled            # fine-tune every arm and score them

8.6 produced three style clusters over 105 hdbpmn writers, established that they are about the
hand rather than the exercise (AMI against exercise -0.0080), and measured how often a page routes
to its own writer's cluster (0.7927 against a 0.3333 chance rate). This task spends those clusters
on the thing they were built for and measures whether they pay.

## The arms

    zero_shot   `microsoft/trocr-small-handwritten` untouched. The domain gap, priced.
    global      one head fine-tuned on every training crop.
    style       three heads, one per 8.6 cluster, each fine-tuned on its own writers' crops. A
                test crop is routed by the style cluster of the *page* it came from.
    random      three heads on three random writer groups **of the same three sizes**, routed the
                same way.

## The `random` arm is the whole experiment

A per-cluster head is trained on roughly a third of the data. So `style` beating `global` would be
remarkable and `style` losing to `global` would prove nothing - it would be the obvious
consequence of dividing the training set by three. **The question that can actually be answered is
whether the clusters are better than an arbitrary partition of the same shape**, and that is
`random`: identical group sizes, identical per-group data volume, identical training budget, and
the only difference is whether writers were grouped by how they write.

    style - random    what 8.6's clustering contributed
    style - global    what splitting the training set cost, net of that

Reporting only the second, which is what the plan's "CER improves >= 15%" invites, would credit or
blame the clustering for an effect that is mostly arithmetic.

## Splitting by writer

Held out by scribe, so no test writer's hand is in any training set. That is the deployment case -
a new page from a new person - and it is also the case 7.4.7's `unseen_scribe` control showed to
be the one that matters: a scheme that helps on a known writer and cannot run on a new one is a
different product. Routing still works here because 8.6 routes on *page* statistics, which a new
writer has.

## The crops

hdbpmn node boxes, inset 10% on each side to drop the drawn border, from the grayscale original
rather than the binarised layer - TrOCR was trained on photographs and 3.1's binarisation is a
domain shift it never saw. Ground truth is the node's `text` from the BPMN annotation, so this is
real handwriting with a real transcription, not a rendering.

## What it measured

9,807 hdbpmn text crops, 105 writers, held out by writer: **26 test writers, 79 training writers,
7,599 training crops, 2,208 test crops.** `trocr-small-handwritten`, three epochs per head.

    arm          CER      WER     exact match
    zero_shot   1.0573   1.4245     0.0023
    global      0.6628   0.8625     0.1245
    random      0.8210   1.0645     0.0086
    style       0.8704   1.0733     0.0154

    global vs zero_shot     -37.3% CER      (fine-tuning works)
    style  vs global        +31.3% CER      (routing is much worse)
    random vs global        +23.9% CER      (splitting the data is most of that)
    style  vs random         +6.0% CER      (the clustering costs a further six percent)

**The plan's target is a 15% relative CER improvement. The measured effect is 31% in the wrong
direction, and it fails against the random control too.** Style-conditioned routing should not be
built on this corpus.

## The zero-shot row is above 1.0, and that is not a bug

A CER of 1.0573 means the edit distance exceeds the length of the truth, which happens when the
model inserts more than it gets right. TrOCR-small was trained on IAM: single lines of cursive
English prose, one line per image. An hdbpmn crop is two or three words in block capitals wrapped
over two or three lines inside a hand-drawn box. Asked for a line of prose it produces one -
`evaluate the Application` came back as `adoption .` and `send a welcome pack in a letter` as
`a b c dwynson was a former election`. **Fine-tuning removes 37.3% of that error**, which is the
domain gap being closed and the only unambiguously positive number in this task.

## The `random` control is why this task has an answer instead of an excuse

Without it, `style` at 0.8704 against `global` at 0.6628 has two explanations and no way to choose
between them: the clustering is bad, or each head simply had a third of the data. The control
separates them, and it separates them cleanly.

**Dividing the training set into three costs 23.9% relative CER on its own** - that is the
`random` arm, which groups writers arbitrarily and is otherwise identical in group sizes, per-group
data volume and training budget. It is the price of the architecture, not of 8.6's clusters.

**8.6's clusters then cost a further 6.0% on top of that.** Grouping writers by *how they write*
is measurably worse than grouping them at random, which is the opposite of the design's
hypothesis and is the finding here.

## Why grouping similar writers together makes it worse

The mechanism is visible in the per-group rows and it is not subtle.

    style groups     train crops   CER          random groups   train crops   CER
       0                3,076     0.8078            0             2,452     0.7809
       1                  794     0.9288            1             1,064     0.8411
       2                3,729     0.8834            2             4,083     0.8572

**Every style head is worse than the random head of comparable size.** A style cluster is by
construction a *narrow* slice of handwriting: cluster 2 is 52 writers who all write small, dense
and round. A head trained only on those sees less variation than a head trained on 52 arbitrary
writers, and at test time it faces a writer it has never seen - because the split is by writer, as
the deployment case requires. Narrow training and unseen test writers is the combination that
punishes specialisation. A random group of the same size carries the full spread of hands and
generalises further.

The style split also delivers the most lopsided groups - 794 crops in cluster 1 against 1,064 in
the smallest random group - because 8.6's twelve-writer cluster is small in writers *and* those
writers wrote less. Its head is the worst in the table at 0.9288.

**Exact match separates the arms even more sharply than CER**: 0.1245 for `global` against 0.0154
and 0.0086. The global head gets one crop in eight exactly right; the split heads get one in
sixty-five and one in a hundred and sixteen. Whatever the split arms are doing, they are almost
never producing a completely correct short label.

## What this is and is not evidence for

It is evidence that **at 7,599 training crops, splitting an OCR head three ways is expensive and
splitting it by style is worse than splitting it arbitrarily.** That is the third independent
measurement of the same shape in this project, and the numbers line up: 7.4.7 found per-scribe
Gaussians costing 7.93 nats against a pooled one; 7.4.3 found mixture components collapsing onto
outliers as K rose; here a per-style head costs 24-31% relative CER. **Every time this corpus has
been asked to support a specialised model, the pooled model has won, and the reason has always
been the same - there is not enough of it.**

It is not evidence that style conditioning cannot work in principle. Three things would have to be
tried before saying that, and none is in scope here: a base model strong enough that 0.66 CER is
not the ceiling; per-cluster *adaptation* from the global head rather than independent fine-tuning
from the pretrained one, so each head inherits all 7,599 crops of training before specialising;
and an order of magnitude more data. The second of those is the one this result most directly
recommends, because it removes the 23.9% arithmetic penalty and leaves only the question 8.6
actually asked.

**And 8.6's clusters are not thereby worthless.** They passed their own controls - AMI -0.0080
against the exercise, 0.7927 routing accuracy - so they are a real description of how these 105
people write. What this task establishes is that a real description of handwriting style is not,
at this data scale, a useful thing to condition an OCR head on.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SEED = 42

ROOT = Path(__file__).resolve().parents[2]
IR_DIR = ROOT / "data" / "processed" / "ir" / "hdbpmn"
CROPS = ROOT / "data" / "features" / "text_crops"
INDEX = CROPS / "index.parquet"
FIGURE = ROOT / "reports" / "figures" / "p8_style_ocr.png"

MODEL = "microsoft/trocr-small-handwritten"

ARMS = ("zero_shot", "global", "style", "random")

#: Cached at a fixed height; the processor resizes to its own square anyway, and storing the
#: full-resolution crop would be four gigabytes for no benefit.
CROP_HEIGHT = 64
CROP_MAX_WIDTH = 512

#: Fraction of the box trimmed on each side, to leave the writing and drop the drawn outline.
INSET = 0.10

MIN_CROP = (20, 12)

EPOCHS = 3
BATCH = 24
LEARNING_RATE = 5e-5

#: Held-out share of *writers*, not of crops.
TEST_SCRIBES = 0.25


# -- the crops -------------------------------------------------------------------------------------


def page_crops(ir_path: Path) -> list[dict]:
    """Every labelled text crop on one page, written to the cache."""
    import cv2

    from src.ir.model import Diagram
    from src.preprocess.exif import load

    diagram = Diagram.load(ir_path)
    image_path = ROOT / diagram.meta["image"]
    if not image_path.is_file():
        return []
    image = load(image_path, grayscale=True)
    page = ir_path.name.replace(".ir.json", "")

    rows = []
    for node in diagram.nodes:
        text = (node.text or "").strip()
        if not text or not node.bbox:
            continue
        x, y, w, h = node.bbox
        ix, iy = INSET * w, INSET * h
        x0, y0 = max(0, int(x + ix)), max(0, int(y + iy))
        x1 = min(image.shape[1], int(x + w - ix))
        y1 = min(image.shape[0], int(y + h - iy))
        if x1 - x0 < MIN_CROP[0] or y1 - y0 < MIN_CROP[1]:
            continue

        crop = image[y0:y1, x0:x1]
        scale = CROP_HEIGHT / crop.shape[0]
        width = min(CROP_MAX_WIDTH, max(8, int(round(crop.shape[1] * scale))))
        crop = cv2.resize(crop, (width, CROP_HEIGHT), interpolation=cv2.INTER_AREA)

        name = f"{page}__{node.id}.png"
        cv2.imwrite(str(CROPS / name), crop)
        rows.append(
            {
                "file": name,
                "page": page,
                "scribe": diagram.meta.get("scribe_id"),
                "exercise": diagram.meta.get("exercise"),
                "text": text,
            }
        )
    return rows


def build(limit: int | None = None):
    import pandas as pd

    from src.utils.parallel import pmap

    CROPS.mkdir(parents=True, exist_ok=True)
    paths = sorted(IR_DIR.glob("*.ir.json"))[:limit]
    rows = [row for page in pmap(page_crops, paths) for row in page]
    table = pd.DataFrame(rows)
    table.to_parquet(INDEX, index=False)
    return table


def load_index():
    import pandas as pd

    if not INDEX.is_file():
        raise FileNotFoundError(f"{INDEX} not built; run --build")
    return pd.read_parquet(INDEX)


# -- grouping --------------------------------------------------------------------------------------


def style_groups() -> dict:
    """8.6's cluster for every scribe."""
    from src.cluster import styles

    table = styles.load_table()
    X, scribes, _ = styles.per_scribe(table)
    assignment = styles.cluster(X, 3).named_steps["kmeans"].labels_
    return {str(s): int(c) for s, c in zip(scribes, assignment, strict=True)}


def random_groups(assignment: dict, seed: int = SEED) -> dict:
    """The same group sizes, allocated to writers at random.

    Same sizes rather than equal sizes: an arbitrary partition has to match the real one in every
    respect except *which* writers are together, or the comparison prices two things at once.
    """
    scribes = sorted(assignment)
    sizes = np.bincount(list(assignment.values()))
    labels = np.repeat(np.arange(len(sizes)), sizes)
    np.random.default_rng(seed).shuffle(labels)
    return {s: int(c) for s, c in zip(scribes, labels, strict=True)}


def split(table, seed: int = SEED):
    """Hold out whole writers, so no test hand appears in any training set."""
    scribes = np.array(sorted(table["scribe"].dropna().unique()))
    rng = np.random.default_rng(seed)
    held = set(rng.choice(scribes, size=int(round(TEST_SCRIBES * len(scribes))), replace=False))
    test = table["scribe"].isin(held).to_numpy()
    return table[~test].reset_index(drop=True), table[test].reset_index(drop=True), sorted(held)


# -- the model -------------------------------------------------------------------------------------


def processor():
    from transformers import TrOCRProcessor

    return TrOCRProcessor.from_pretrained(MODEL)


def fresh_model(device: str = "cuda"):
    import torch
    from transformers import VisionEncoderDecoderModel

    model = VisionEncoderDecoderModel.from_pretrained(MODEL)
    proc = processor()
    model.config.decoder_start_token_id = proc.tokenizer.cls_token_id
    model.config.pad_token_id = proc.tokenizer.pad_token_id
    model.config.eos_token_id = proc.tokenizer.sep_token_id
    return model.to(torch.device(device))


def _images(files):
    import cv2

    out = []
    for name in files:
        crop = cv2.imread(str(CROPS / name), cv2.IMREAD_GRAYSCALE)
        out.append(cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB))
    return out


def finetune(table, proc, device: str = "cuda", epochs: int = EPOCHS, batch: int = BATCH):
    """One head, fine-tuned on `table`."""
    import torch

    model = fresh_model(device)
    model.train()
    optimiser = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    rng = np.random.default_rng(SEED)
    files = table["file"].to_numpy()
    texts = table["text"].to_numpy()

    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    for _ in range(epochs):
        order = rng.permutation(len(files))
        for start in range(0, len(order), batch):
            chunk = order[start : start + batch]
            pixels = proc(images=_images(files[chunk]), return_tensors="pt").pixel_values.to(device)
            labels = proc.tokenizer(
                list(texts[chunk]),
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt",
            ).input_ids.to(device)
            labels[labels == proc.tokenizer.pad_token_id] = -100

            optimiser.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                loss = model(pixel_values=pixels, labels=labels).loss
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
    model.eval()
    return model


def predict(model, table, proc, device: str = "cuda", batch: int = 32) -> list[str]:
    import torch

    files = table["file"].to_numpy()
    out: list[str] = []
    for start in range(0, len(files), batch):
        chunk = files[start : start + batch]
        pixels = proc(images=_images(chunk), return_tensors="pt").pixel_values.to(device)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=device == "cuda"):
            ids = model.generate(pixels, max_new_tokens=48, num_beams=1)
        out += proc.batch_decode(ids, skip_special_tokens=True)
    return out


def score(truth, predicted) -> dict:
    import jiwer

    truth = [t if t.strip() else " " for t in truth]
    predicted = [p if p.strip() else " " for p in predicted]
    return {
        "cer": round(float(jiwer.cer(truth, predicted)), 4),
        "wer": round(float(jiwer.wer(truth, predicted)), 4),
        "exact": round(
            float(np.mean([a.strip() == b.strip() for a, b in zip(truth, predicted, strict=True)])),
            4,
        ),
        "n": len(truth),
    }


def relative_gain(baseline: float, candidate: float) -> float:
    """Relative CER reduction, the quantity the plan's >= 15% is stated in."""
    return round(float((baseline - candidate) / baseline), 4) if baseline else 0.0


# -- the arms --------------------------------------------------------------------------------------


def grouped_arm(train, test, groups: dict, proc, device: str) -> dict:
    """One head per group; each test crop scored by the head its page routes to."""
    predicted = np.empty(len(test), dtype=object)
    per_group = {}
    # -1 for a writer 8.6 never saw, so those crops are visibly unroutable rather than silently
    # folded into group 0.
    train_group = train["scribe"].map(groups).fillna(-1).astype(int).to_numpy()
    test_group = test["scribe"].map(groups).fillna(-1).astype(int).to_numpy()

    for group in sorted(set(train_group.tolist()) - {-1}):
        rows = train[train_group == group]
        targets = np.flatnonzero(test_group == group)
        if not len(rows) or not len(targets):
            per_group[int(group)] = {"train_crops": int(len(rows)), "test_crops": int(len(targets))}
            continue
        model = finetune(rows, proc, device)
        subset = test.iloc[targets]
        said = predict(model, subset, proc, device)
        predicted[targets] = said
        per_group[int(group)] = {
            "train_crops": int(len(rows)),
            **score(subset["text"].tolist(), said),
        }
        del model
        _free(device)

    # A test crop whose group had no training data has no head to route to; it is scored as an
    # empty prediction rather than dropped, because dropping it would flatter the arm.
    predicted = [p if isinstance(p, str) else "" for p in predicted]
    return {"overall": score(test["text"].tolist(), predicted), "per_group": per_group}


def _free(device: str) -> None:
    import gc

    import torch

    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


def run(device: str = "cuda", epochs: int = EPOCHS) -> dict:
    table = load_index()
    train, test, held = split(table)
    proc = processor()

    styles_map = style_groups()
    randoms_map = random_groups(styles_map)

    results: dict = {
        "crops": int(len(table)),
        "train_crops": int(len(train)),
        "test_crops": int(len(test)),
        "held_out_scribes": len(held),
        "train_scribes": int(train["scribe"].nunique()),
        "epochs": epochs,
        "model": MODEL,
    }

    zero = fresh_model(device).eval()
    results["zero_shot"] = score(test["text"].tolist(), predict(zero, test, proc, device))
    del zero
    _free(device)

    world = finetune(train, proc, device, epochs)
    results["global"] = score(test["text"].tolist(), predict(world, test, proc, device))
    del world
    _free(device)

    results["style"] = grouped_arm(train, test, styles_map, proc, device)
    results["random"] = grouped_arm(train, test, randoms_map, proc, device)

    base = results["zero_shot"]["cer"]
    world_cer = results["global"]["cer"]
    style_cer = results["style"]["overall"]["cer"]
    random_cer = results["random"]["overall"]["cer"]
    results["gains"] = {
        "global_vs_zero_shot": relative_gain(base, world_cer),
        "style_vs_zero_shot": relative_gain(base, style_cer),
        "style_vs_global": relative_gain(world_cer, style_cer),
        "style_vs_random": relative_gain(random_cer, style_cer),
        "random_vs_global": relative_gain(world_cer, random_cer),
        "meets_fifteen_percent_against_global": relative_gain(world_cer, style_cer) >= 0.15,
        "meets_fifteen_percent_against_random": relative_gain(random_cer, style_cer) >= 0.15,
    }
    return results


def figure(results, path: Path = FIGURE) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["zero_shot", "global", "style", "random"]
    values = [
        results["zero_shot"]["cer"],
        results["global"]["cer"],
        results["style"]["overall"]["cer"],
        results["random"]["overall"]["cer"],
    ]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    bars = ax.bar(names, values, color=["#7f7f7f", "#1f77b4", "#2ca02c", "#d62728"])
    for bar, value in zip(bars, values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom")
    ax.set_ylabel("character error rate")
    ax.set_title("8.7 - style routing against a same-shape random partition")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    if args.build:
        table = build(args.limit)
        print(f"built {len(table)} crops -> {CROPS}")
        return 0

    result = run(args.device, args.epochs)
    result["figure"] = str(figure(result))
    text = json.dumps(result, indent=2)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            fh.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
