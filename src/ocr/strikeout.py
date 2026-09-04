"""Phase 9.3.8 - detecting crossed-out elements, and the control that stops it counting ink.

    python -m src.ocr.strikeout --build      # synthesise the corpus
    python -m src.ocr.strikeout              # train and score

An element the writer struck through is an instruction to ignore it, and an IR that carries it as
a node produces code for something the author deleted. The IR schema has had a `crossed_out` list
since 2.1, and this is what fills it.

## There is no supervision for this, and that is the first finding

**Not one page in this project is annotated with a struck-out element.** `crossed_out` is empty in
all 5,796 IR files and `low_conf_text` with it; hdbpmn's transcriptions come from the BPMN model
the writer was copying, so anything they crossed out never entered the annotation at all. So the
positive class has to be **synthesised**, using 9.1.7's `strike_out` - the same three gestures,
single diagonal, cross, and zig-zag scribble - drawn onto real crops of real handwriting. The
negatives are the untouched crops.

That makes the reported precision a statement about **this damage model**, not about how the
twelve people who actually cross things out do it, and no amount of tuning changes that. It is
stated here rather than in a limitations paragraph because it is the main thing a reader needs to
discount.

## The control, which is the reason this task is not trivial

A synthetic strike adds ink. A classifier can reach a very high score by counting dark pixels and
never looking at what shape they are in, and it would then fail on the first page whose writer
simply presses hard or writes a long label. So there is a third class:

    clean       the crop as written
    struck      `strike_out`'s gesture drawn across the writing
    inked       **the control** - the same quantity of ink added as `doodle`-style marks that do
                *not* cross the text: strokes in the margin, a bracket beside it, an underline.

`inked` is scored as a negative. A model that has learned "more ink means struck" scores near
zero precision against it; a model that has learned "a long stroke crossing the writing at an
angle" is unaffected. The headline number is precision on the three-way corpus, and the
two-way-only number is reported beside it so the size of the difference is visible.

Ink is matched between `struck` and `inked` within a tolerance, so the two classes differ in the
*placement* of the added ink and not in how much there is.

## What it measured

33,687 crops - 11,229 each of `clean`, `struck` and `inked` - from the annotated hdbpmn and
fa_bresler node crops, split 22,080 / 5,955 / 5,652 by 9.3.1's writer-disjoint assignment.

    arm                              precision   recall      F1
    three-way (clean+inked+struck)      0.9990   1.0000   0.9995
    two-way (clean+struck only)         0.9990   1.0000   0.9995
    ink-share baseline                  0.4062   0.6096   0.4875
    **held-out gesture**                1.0000   0.0163   0.0320

**The headline is 0.999 precision against a 0.85 target, and it is worth almost nothing.** The
last row is the finding: hold the zig-zag scribble out of training entirely and ask the model for
it at validation, and recall collapses from 1.0000 to **0.0163 - 10 of 615 scribbles found**. The
classifier has not learned "a stroke crossing the writing"; it has learned the two specific
gestures it was shown, and a third gesture drawn by the same code with the same pen is invisible
to it. **If a synthetic gesture it has never seen defeats it completely, a real one will too**,
and the 0.999 should be read as a statement about `strike()` rather than about crossing-out.

**The ink control did its job and rules out the cheap explanation.** A threshold on the crop's ink
share reaches precision **0.4062**, barely above the 0.333 a positive is worth by chance, because
`inked` was built to carry the same added ink as `struck` - median ink 0.3307 against 0.3122, the
control slightly *overshooting*, which is the safe direction since it means a pixel counter would
prefer the negative class. So the model is not counting ink. It is reading geometry - just far
more specific geometry than the task wanted.

**Two earlier versions of this measurement were wrong in the same way and the fix is instructive.**
The first run scored precision **1.000** exactly, with a constant ink value of grey 20 and a
constant stroke thickness: a watermark, not a gesture. Drawing the pen from the crop's own 5th
percentile with a jittered width removed it, and the score barely moved - which by itself would
have read as "the watermark was not being used". It took the held-out gesture to show that
something almost as narrow had been learned instead. **A control that the model passes tells you
much less than one it fails**, and this task has one of each.

**The 2 false positives are both `clean`, and none is `inked`** - so on the classes it was trained
on there is no confusion between "ink was added beside the writing" and "the writing was crossed
out" at all.

What Phase 16 should take from this is not the precision number. It is that **crossed-out
detection cannot be validated on this corpus**, because the corpus contains no crossed-out
elements; the code and the corpus are here, the gate is passed, and the honest recommendation is
that the first real annotation campaign should collect struck-out pages before anything downstream
relies on this flag.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from src.ocr.metrics import RUNS
from src.ocr.textcrops import OUT as CROPS
from src.ocr.textcrops import load_index
from src.utils.config import ROOT

OUT = ROOT / "data" / "processed" / "strikeout"
SEED = 42

#: Input to the classifier. Wider than it is tall because a strike is a long stroke and squashing
#: a 512-wide crop to a square would turn a diagonal line into a nearly vertical one.
SIZE = (48, 192)

CLASSES = ("clean", "struck", "inked")
#: `inked` is a negative. The label a prediction is scored against is `struck` vs not-struck.
POSITIVE = "struck"

EPOCHS = 12
BATCH = 64
LR = 1e-3

TARGET_PRECISION = 0.85


# ------------------------------------------------------------------------------------------
# synthesis
# ------------------------------------------------------------------------------------------


def ink_share(image: np.ndarray) -> float:
    import cv2

    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return float((binary > 0).mean())


def pen(image: np.ndarray, rng) -> tuple[int, int]:
    """An ink value and a stroke width drawn from *this crop's own* writing.

    The first version of this module used a constant grey 20 and a constant thickness, and the
    classifier reached precision 1.000. A constant is a watermark: the model can key on the
    presence of a pixel whose value is exactly 20 and never look at the stroke's shape at all.
    The ink value is now the crop's own 5th percentile jittered, and the width is drawn around a
    plausible nib, so `struck` and `clean` are not separable by a histogram.
    """
    dark = int(np.percentile(image, 5))
    value = int(np.clip(dark + rng.integers(-15, 16), 0, 90))
    thickness = int(max(2, round(image.shape[0] * float(rng.uniform(0.035, 0.09)))))
    return value, thickness


def strike(image: np.ndarray, rng, style: int | None = None) -> tuple[np.ndarray, int]:
    """9.1.7's three gestures, drawn across the writing rather than across a node box.

    Returns the picture and the gesture index, because 9.3.8's strongest control holds one
    gesture out of training entirely.
    """
    import cv2

    out = image.copy()
    h, w = out.shape
    value, thickness = pen(out, rng)
    margin = int(0.08 * w)
    x0, x1 = margin, max(margin + 4, w - margin)
    jitter = lambda: int(rng.integers(-h // 12, h // 12 + 1))  # noqa: E731
    y0, y1 = int(0.2 * h) + jitter(), int(0.8 * h) + jitter()
    style = int(rng.integers(0, 3)) if style is None else int(style)
    if style == 0:
        cv2.line(out, (x0, y1), (x1, y0), value, thickness, cv2.LINE_AA)
    elif style == 1:
        cv2.line(out, (x0, y0), (x1, y1), value, thickness, cv2.LINE_AA)
        cv2.line(out, (x1, y0), (x0, y1), value, thickness, cv2.LINE_AA)
    else:
        steps = int(rng.integers(6, 12))
        xs = np.linspace(x0, x1, steps)
        ys = np.where(np.arange(steps) % 2 == 0, y0, y1)
        points = np.stack([xs, ys], 1).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [points], False, value, thickness, cv2.LINE_AA)
    return out, style


def inked(image: np.ndarray, rng, target: float) -> np.ndarray:
    """The control: the same amount of ink, placed so that it does not cross the writing.

    Marks go above, below, or to one side - an underline, a bracket, a margin tick. The loop adds
    strokes until the added ink share reaches `target`, which is what the matching `strike` added,
    so the two classes are separated by geometry and not by quantity.
    """
    import cv2

    out = image.copy()
    h, w = out.shape
    before = ink_share(out)
    value, thickness = pen(out, rng)
    for _ in range(24):
        if ink_share(out) - before >= target:
            break
        style = int(rng.integers(0, 3))
        if style == 0:  # underline
            y = int(rng.integers(int(0.88 * h), h - 1))
            cv2.line(out, (int(0.05 * w), y), (int(0.95 * w), y), value, thickness, cv2.LINE_AA)
        elif style == 1:  # a bracket in the margin
            x = int(rng.integers(0, max(1, int(0.08 * w))))
            cv2.line(out, (x, int(0.1 * h)), (x, int(0.9 * h)), value, thickness, cv2.LINE_AA)
            cv2.line(out, (x, int(0.1 * h)), (x + thickness * 3, int(0.1 * h)), value, thickness)
            cv2.line(out, (x, int(0.9 * h)), (x + thickness * 3, int(0.9 * h)), value, thickness)
        else:  # a tick above the line
            y = int(rng.integers(1, max(2, int(0.1 * h))))
            x = int(rng.integers(0, max(1, w - 40)))
            cv2.line(out, (x, y), (x + 30, y), value, thickness, cv2.LINE_AA)
    return out


def build(out: Path = OUT, limit: int | None = None, seed: int = SEED) -> dict:
    """One `clean`, one `struck` and one `inked` version of every node crop with ink."""
    import cv2

    frame = load_index()
    # Annotated node crops only. The detected arm is the same nodes cropped a second time, and
    # including both would put two near-identical pictures of the same handwriting into the
    # corpus - one of which can land in train while the other lands in val.
    frame = frame[
        (frame["kind"] == "node") & frame["has_ink"] & (frame["provenance"] == "annotated")
    ]
    if limit:
        frame = frame.head(limit)

    if out.exists():
        shutil.rmtree(out)
    rows = []
    rng = np.random.default_rng(seed)
    for name in CLASSES:
        (out / name).mkdir(parents=True, exist_ok=True)

    for _, row in frame.iterrows():
        image = cv2.imread(str(CROPS / "images" / row["file"]), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        base = ink_share(image)
        struck, style = strike(image, rng)
        added = max(0.005, ink_share(struck) - base)
        control = inked(image, rng, added)
        for label, picture in (("clean", image), ("struck", struck), ("inked", control)):
            path = out / label / f"{label}__{row['file']}"
            cv2.imwrite(str(path), picture)
            rows.append(
                {
                    "file": str(path.relative_to(out)),
                    "label": label,
                    "split": row["split"],
                    "source": row["source"],
                    "scribe": row["scribe"],
                    "ink": round(ink_share(picture), 5),
                    "style": style if label == "struck" else -1,
                }
            )

    import pandas as pd

    pd.DataFrame(rows).to_parquet(out / "index.parquet", index=False)
    return summary(out)


def summary(out: Path = OUT) -> dict:
    import pandas as pd

    frame = pd.read_parquet(out / "index.parquet")
    ink = frame.groupby("label")["ink"].median().to_dict()
    return {
        "crops": int(len(frame)),
        "by_label": {k: int(v) for k, v in frame["label"].value_counts().items()},
        "by_split": {k: int(v) for k, v in frame["split"].value_counts().items()},
        "median_ink": {k: round(float(v), 5) for k, v in ink.items()},
        # If these two are far apart the control has failed and any precision below is suspect.
        "ink_match_struck_vs_inked": round(float(ink.get("struck", 0) - ink.get("inked", 0)), 5),
    }


# ------------------------------------------------------------------------------------------
# model
# ------------------------------------------------------------------------------------------


def build_model():
    import torch.nn as nn

    def block(cin, cout):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    return nn.Sequential(
        block(1, 16),
        block(16, 32),
        block(32, 64),
        block(64, 64),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Dropout(0.2),
        nn.Linear(64, 2),
    )


def load_split(split: str, out: Path = OUT, labels=CLASSES, styles=None, exclude_styles=None):
    """`(x, y, frame)` for one split, optionally restricted by which gesture was drawn.

    `styles` and `exclude_styles` are what the held-out-gesture arm is built from: a training set
    that has never contained a scribble, and a validation set made only of scribbles.
    """
    import cv2
    import pandas as pd

    frame = pd.read_parquet(out / "index.parquet")
    frame = frame[(frame["split"] == split) & frame["label"].isin(labels)]
    if styles is not None:
        frame = frame[(frame["label"] != POSITIVE) | frame["style"].isin(styles)]
    if exclude_styles is not None:
        frame = frame[(frame["label"] != POSITIVE) | ~frame["style"].isin(exclude_styles)]
    images, targets = [], []
    for _, row in frame.iterrows():
        image = cv2.imread(str(out / row["file"]), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        images.append(cv2.resize(image, (SIZE[1], SIZE[0]), interpolation=cv2.INTER_AREA))
        targets.append(int(row["label"] == POSITIVE))
    x = np.stack(images).astype(np.float32)[:, None] / 255.0 - 0.5
    return x, np.asarray(targets, dtype=np.int64), frame.reset_index(drop=True)


def precision_recall(truth, predicted) -> dict:
    tp = int(((predicted == 1) & (truth == 1)).sum())
    fp = int(((predicted == 1) & (truth == 0)).sum())
    fn = int(((predicted == 0) & (truth == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if tp else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def fit(x_train, y_train, x_val, epochs: int, device):
    """Train the classifier and return it with the validation scores."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    model = build_model().to(device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=BATCH,
        shuffle=True,
    )
    criterion = torch.nn.CrossEntropyLoss()
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            loss = criterion(model(xb.to(device)), yb.to(device))
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
        schedule.step()
        print(f"[strikeout] epoch {epoch + 1}/{epochs} loss {float(loss):.4f}", flush=True)
    seconds = time.perf_counter() - started

    model.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, len(x_val), 256):
            chunk = torch.from_numpy(x_val[start : start + 256]).to(device)
            scores.append(model(chunk).softmax(-1)[:, 1].cpu().numpy())
    return model, np.concatenate(scores), round(seconds, 1)


def run(epochs: int = EPOCHS, out: Path = OUT) -> dict:
    import torch

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_train, y_train, _ = load_split("train", out)
    x_val, y_val, val_frame = load_split("val", out)
    model, probability, seconds = fit(x_train, y_train, x_val, epochs, device)
    predicted = (probability >= 0.5).astype(np.int64)

    labels = val_frame["label"].to_numpy()
    two_way = np.isin(labels, ["clean", "struck"])
    ink_baseline = (val_frame["ink"].to_numpy() >= np.median(val_frame["ink"].to_numpy())).astype(
        int
    )

    # The strongest control in this task: the scribble is removed from training entirely and
    # the model is asked for it at validation. A detector that has learned "a long stroke
    # crossing the writing" transfers; one that has memorised this damage model does not.
    x_seen, y_seen, _ = load_split("train", out, exclude_styles=[2])
    x_unseen, y_unseen, unseen_frame = load_split("val", out, styles=[2])
    _, unseen_probability, _ = fit(x_seen, y_seen, x_unseen, epochs, device)
    unseen = precision_recall(y_unseen, (unseen_probability >= 0.5).astype(np.int64))

    torch.save(model.state_dict(), out / "model.pt")
    return {
        "epochs": epochs,
        "train_seconds": seconds,
        "held_out_gesture": {
            "held_out": "zig-zag scribble",
            "train_crops": int(len(x_seen)),
            "val_crops": int(len(x_unseen)),
            **unseen,
        },
        "train_crops": int(len(x_train)),
        "val_crops": int(len(x_val)),
        "three_way": precision_recall(y_val, predicted),
        # The same model scored on the easy corpus the control was added to eliminate.
        "two_way": precision_recall(y_val[two_way], predicted[two_way]),
        # What a pixel counter gets. If this is close to `three_way`, the model learned nothing
        # the threshold did not already know.
        "ink_baseline_three_way": precision_recall(y_val, ink_baseline),
        "false_positives_by_label": {
            str(k): int(v)
            for k, v in dict(
                zip(
                    *np.unique(labels[(predicted == 1) & (y_val == 0)], return_counts=True),
                    strict=False,
                )
            ).items()
        },
        "by_style": {
            str(int(k)): int(v)
            for k, v in dict(
                zip(
                    *np.unique(
                        val_frame.loc[val_frame["label"] == POSITIVE, "style"], return_counts=True
                    ),
                    strict=False,
                )
            ).items()
        },
        "target_precision": TARGET_PRECISION,
        "meets_target": bool(precision_recall(y_val, predicted)["precision"] >= TARGET_PRECISION),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--result", type=Path, default=RUNS / "strikeout.json")
    args = ap.parse_args(argv)

    if args.build:
        print(json.dumps(build(args.out, args.limit), indent=2))
        return 0
    result = {"corpus": summary(args.out), **run(args.epochs, args.out)}
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
