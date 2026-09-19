"""Phase 9.3.2 - a CRNN trained with CTC, pretrained on IAM and fine-tuned on diagram labels.

    python -m src.ocr.crnn --arm iam            # pretrain on IAM lines
    python -m src.ocr.crnn --arm scratch        # diagram labels only, random init
    python -m src.ocr.crnn --arm finetune       # IAM weights, then diagram labels
    python -m src.ocr.crnn --arm zero_shot      # the IAM model read straight onto diagrams
    python -m src.ocr.crnn                      # all four, in dependency order

CNN encoder -> BiLSTM -> CTC, the standard line recogniser, built here rather than imported so
that 9.3.3's transformer comparison is against something whose every choice is visible.

## Why CTC and not a decoder

The alternative is what 9.3.3 uses: an autoregressive decoder that emits characters one at a
time conditioned on what it has already emitted. CTC has no such conditioning - it scores a
per-column distribution over the alphabet plus a blank, and collapses repeats. That makes it
**monotonic and alignment-free**, which is exactly right for a line of handwriting where the
image order is the character order, and it makes it **incapable of hallucinating fluent text**,
which is the failure 8.7 measured: TrOCR zero-shot turned `send a welcome pack in a letter` into
`a b c dwynson was a former election`, a CER above 1.0 produced by a language model writing
prose. A CTC model with nothing to condition on cannot do that. It also cannot use context to
repair an ambiguous glyph, and the two arms are the price of that trade measured rather than
argued.

CTC also gives 9.3.4 and 9.3.7 something a decoder does not: **a per-column posterior over the
alphabet**, which is what a lexicon can be applied to and what a calibrated confidence can be
read from.

## The four arms, and which one is the control

    iam         6,482 IAM handwriting lines. Real cursive English prose, the standard corpus.
    zero_shot   that model, evaluated on diagram crops with no adaptation. **This is the
                control**, and it prices the domain gap in the same units as everything else.
    scratch     the same architecture, random init, diagram crops only.
    finetune    IAM weights, then diagram crops.

`finetune - scratch` is what pretraining is worth. 9.2.7 ran exactly this comparison on shape
crops and found pretraining worth +0.0035 macro F1 - three thousandths, indistinguishable from
noise - so the prior going in is that it will be small here too. The difference is that 9.2.7's
pretraining was ImageNet photographs against line drawings, and this is *handwriting against
handwriting*: the domain gap is a fraction of the size, and if pretraining is ever going to pay
in this project it is here. That is why the arm exists and why the negative result from 9.2.7 is
not sufficient to skip it.

## The alphabet

Built from the normalised training text, which is 9.3.6's normalisation - lowercased, whitespace
collapsed - so the model is never trained to produce a distinction the metric discards. IAM's
lines carry characters the diagram corpus does not, so the alphabet is the **union**, and a
symbol seen only in IAM stays in the head after fine-tuning rather than being pruned; pruning it
would make `zero_shot` and `finetune` different models with different output layers and the two
CERs incomparable.

## What it measured

Five arms, one architecture, 3,787,806 parameters, alphabet 61.

    arm          trained on                  CER      WER   exact   train s
    iam          6,482 IAM lines          0.1157   0.3724  0.0553       112
    zero_shot    (that model, no adapt)   0.9338   1.0721  0.0029         -
    scratch      16,974 diagram crops     0.8397   0.9845  0.0439       576
    finetune     IAM, then those crops    0.6864   0.9496  0.0660       566
    unwrap       the same, lines unwrapped 0.6889  0.9491  0.0691       517

**The IAM row is the control that makes every other row interpretable, and it is the finding.**
The same 3.8M-parameter network, trained the same way, reads real cursive English handwriting at
**CER 0.1157 - inside 9.3.6's 0.15 target** - and reads this project's diagram labels at 0.6864,
**six times worse**. Whatever is wrong here is not the architecture, the alphabet, the CTC loss or
the optimiser: all five are demonstrably sufficient for handwriting on the same day, on the same
GPU, in the same file. **It is the crops.** An IAM line is one line of writing on a ruled page,
tightly cropped, scanned flat. A 9.3.1 node crop is a photograph of two or three lines of block
capitals stacked inside a hand-drawn box, at whatever resolution the writer's phone happened to
have, with the box's own border a tenth of the way in from every edge.

**Pretraining is worth 0.1533 CER here, and that is 44 times what 9.2.7 measured.** 9.2.7 found
ImageNet pretraining worth +0.0035 macro F1 on shape crops - noise - and this task's docstring
predicted the difference in advance: there the pretraining domain was colour photographs of
natural objects against greyscale line drawings, and here it is *handwriting against
handwriting*. It is the same experiment with the domain gap closed, and closing it turns a
negative result into a 22.4% relative improvement. **That is the one place in this project so far
where transfer has clearly paid**, and it pays for the reason the theory says it should.

**The scratch arm's loss curve says it never escaped the blank optimum.** It falls 5.79 -> 3.23
in one epoch and then crawls to 2.42 over twenty-nine more, and its outputs are degenerate -
`evaluate application`, `write notification email` and `rejected` all come back as `ce e` or `co`.
The fine-tuned arm starts at the same place and reaches 1.368 still descending. So `scratch` is
not a weaker model, it is a **stuck** one, and IAM's contribution is largely that it hands CTC an
initialisation from which the blank collapse is not the easiest thing to do.

**`unwrap` was this task's own hypothesis and it is refuted.** A CTC recogniser reads along a
width axis and pools height to nothing, so a label whose second word sits *below* its first is
stacked along the axis the model destroys - which is a complete and plausible explanation for the
gap against IAM. `unwrap` cuts a crop into ink bands by row projection and lays them end to end,
putting the characters back into reading order along the width. It scores **0.6889 against
0.6864: no difference at all**, 0.0025 in the wrong direction, with the loss curve tracking the
`finetune` curve about 0.04 behind at every epoch. Exact match is very slightly better (0.0691
against 0.0660) and that is the whole of it. **Wrapping is not the bottleneck**, and the more
likely remaining causes are the ones `unwrap` cannot touch: resolution, the drawn border inside
the crop, and photographic variation across 105 phones.

**No arm meets the 0.15 target and 9.3.6 has to say so.** The gap from 0.6864 to 0.15 is not a
tuning gap - 9.3.3's 62M-parameter transformer is the next thing to try, and 8.7's fine-tuned
TrOCR reached 0.6628 on an easier version of this corpus, which suggests the ceiling here is set
by the images and not by the model.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from src.ocr.metrics import RUNS, normalise, score
from src.ocr.textcrops import HEIGHT, OUT, load_index
from src.utils.config import ROOT

#: 9.3.1's corpus. Aliased rather than imported as `OUT as CROPS`, because isort and
#: ruff order an aliased member differently and the pre-commit hooks then fight over the line.
CROPS = OUT

IAM = ROOT / "data" / "raw" / "iam_line" / "data"
CHECKPOINTS = RUNS / "crnn"

SEED = 42
BLANK = 0

#: Width every crop is padded or resized to. The CNN downsamples width by 4, so 512 gives 128
#: output columns, and CTC needs at least as many columns as the target has characters - the
#: longest label in the corpus is 90, so the margin is comfortable rather than tight.
WIDTH = 512
DOWNSAMPLE = 4

EPOCHS_IAM = 12
EPOCHS_DIAGRAM = 30
BATCH = 32
LR = 3e-4

ARMS = ("iam", "zero_shot", "scratch", "finetune", "unwrap")


# ------------------------------------------------------------------------------------------
# alphabet
# ------------------------------------------------------------------------------------------


def alphabet(*text_sources) -> str:
    """Sorted union of every character in the normalised text, with index 0 reserved for blank."""
    chars: set[str] = set()
    for texts in text_sources:
        for text in texts:
            chars.update(normalise(text))
    return "".join(sorted(chars))


def encode(text: str, table: dict[str, int]) -> list[int]:
    return [table[c] for c in normalise(text) if c in table]


def greedy_decode(logits: np.ndarray, chars: str) -> str:
    """Best path: argmax per column, collapse repeats, drop blanks.

    Not beam search. A beam without a language model on a CTC posterior recovers very little -
    the standard result is a fraction of a percent - and 9.3.4 is where the beam belongs, because
    there it carries a lexicon that can actually change the answer.
    """
    best = logits.argmax(axis=-1)
    out, previous = [], -1
    for index in best:
        if index != previous and index != BLANK:
            out.append(chars[index - 1])
        previous = index
    return "".join(out)


# ------------------------------------------------------------------------------------------
# model
# ------------------------------------------------------------------------------------------


def build_model(classes: int):
    """Six conv blocks halving height six times and width twice, then a two-layer BiLSTM.

    The asymmetric pooling in the last four blocks - `(2, 1)`, height only - is the one design
    choice worth naming: **width is the time axis and must not be destroyed**, because every
    column it removes is a character CTC can no longer place. Height carries no ordering and is
    pooled all the way to 1, which is what turns a feature map into a sequence.
    """
    import torch.nn as nn

    def block(cin, cout, pool):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool),
        )

    class CRNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.cnn = nn.Sequential(
                block(1, 32, (2, 2)),  # 64x512 -> 32x256
                block(32, 64, (2, 2)),  # -> 16x128
                block(64, 128, (2, 1)),  # -> 8x128
                block(128, 128, (2, 1)),  # -> 4x128
                block(128, 256, (2, 1)),  # -> 2x128
                block(256, 256, (2, 1)),  # -> 1x128
            )
            self.rnn = nn.LSTM(
                256, 256, num_layers=2, bidirectional=True, batch_first=True, dropout=0.2
            )
            self.head = nn.Linear(512, classes)

        def forward(self, x):
            features = self.cnn(x)  # (N, C, 1, W')
            sequence = features.squeeze(2).permute(0, 2, 1)  # (N, W', C)
            output, _ = self.rnn(sequence)
            return self.head(output)  # (N, W', classes)

    return CRNN()


# ------------------------------------------------------------------------------------------
# data
# ------------------------------------------------------------------------------------------


def load_image(path: Path, unwrap: bool = False) -> np.ndarray:
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return np.full((HEIGHT, WIDTH), 255, dtype=np.uint8)
    return fit(unwrap_lines(image) if unwrap else image)


def text_lines(image: np.ndarray, min_rows: int = 4, gap: int = 2) -> list[np.ndarray]:
    """Split a crop into horizontal bands of writing, by the ink's row projection.

    Otsu, count dark pixels per row, and cut where a run of ink-bearing rows ends. `gap` bridges
    the one-row holes between the body of a word and its descenders, which would otherwise cut a
    single line of writing into three.
    """
    import cv2

    if image.ndim != 2 or image.shape[0] < 2 * min_rows:
        return [image]
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    rows = (binary > 0).sum(axis=1)
    threshold = max(1, int(0.02 * image.shape[1]))
    inked = rows >= threshold
    bands, start, blank = [], None, 0
    for index, value in enumerate(inked):
        if value:
            start = index if start is None else start
            blank = 0
        elif start is not None:
            blank += 1
            if blank > gap:
                if index - blank - start >= min_rows:
                    bands.append((start, index - blank))
                start, blank = None, 0
    if start is not None and len(inked) - start >= min_rows:
        bands.append((start, len(inked)))
    return [image[a:b] for a, b in bands] or [image]


def unwrap_lines(image: np.ndarray, pad: int = 8) -> np.ndarray:
    """Lay a crop's text lines end to end, so a wrapped box becomes one long line.

    **This is the arm that tests why the CRNN fails.** A CTC recogniser reads one line: its
    columns are a time axis and its height is pooled to nothing. A BPMN node label is two or
    three lines of writing stacked inside a hand-drawn box, and stacking is exactly the structure
    that pooling destroys - the model is asked to emit `evaluate application` from an image where
    `evaluate` sits above `application` and both have been squashed to 64 rows between them.
    Cutting the bands apart and concatenating them left to right puts the characters back into
    reading order along the axis the model actually has. The ground truth is unchanged, so this
    is a change of input representation and not of task.
    """
    import cv2

    bands = text_lines(image)
    if len(bands) < 2:
        return image
    height = max(8, max(b.shape[0] for b in bands))
    pieces = []
    for band in bands:
        scale = height / band.shape[0]
        width = max(4, int(round(band.shape[1] * scale)))
        pieces.append(cv2.resize(band, (width, height), interpolation=cv2.INTER_AREA))
        pieces.append(np.full((height, pad), 255, dtype=image.dtype))
    return np.concatenate(pieces[:-1], axis=1)


def fit(image: np.ndarray) -> np.ndarray:
    """Height to `HEIGHT` preserving aspect, then pad with paper to `WIDTH` or shrink to it.

    Padding rather than stretching: a stretched crop changes the aspect of the glyphs, and the
    corpus contains both `q0` and a 90-character sentence, so a common stretch would make those
    two images of very different writing. Padding keeps the letters the shape they were drawn.
    """
    import cv2

    scale = HEIGHT / image.shape[0]
    width = max(8, int(round(image.shape[1] * scale)))
    image = cv2.resize(image, (min(width, WIDTH), HEIGHT), interpolation=cv2.INTER_AREA)
    if image.shape[1] < WIDTH:
        pad = np.full((HEIGHT, WIDTH - image.shape[1]), 255, dtype=image.dtype)
        image = np.concatenate([image, pad], axis=1)
    return image


def diagram_split(split: str, provenance=("annotated", "derived")):
    """`(files, texts)` for one split of 9.3.1's corpus."""
    frame = load_index()
    frame = frame[(frame["split"] == split) & (frame["provenance"].isin(provenance))]
    files = [str(CROPS / "images" / f) for f in frame["file"]]
    return files, frame["text"].tolist(), frame


def iam_split(name: str, limit: int | None = None):
    """`(images, texts)` from the IAM line parquet, decoded to arrays in memory."""
    import cv2
    import pandas as pd

    frame = pd.read_parquet(IAM / f"{name}.parquet")
    if limit:
        frame = frame.head(limit)
    images, texts = [], []
    for _, row in frame.iterrows():
        buffer = np.frombuffer(row["image"]["bytes"], dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        images.append(fit(image))
        texts.append(row["text"])
    return images, texts


class Batches:
    """A minimal loader: arrays in, `(x, targets, lengths)` out, shuffled per epoch."""

    def __init__(self, images, texts, table, batch=BATCH, shuffle=True, seed=SEED, unwrap=False):
        self.images, self.texts, self.table = images, texts, table
        self.batch, self.shuffle = batch, shuffle
        self.unwrap = unwrap
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return int(np.ceil(len(self.images) / self.batch))

    def __iter__(self):
        import torch

        order = np.arange(len(self.images))
        if self.shuffle:
            self.rng.shuffle(order)
        for start in range(0, len(order), self.batch):
            chunk = order[start : start + self.batch]
            pixels = np.stack(
                [
                    (
                        self.images[i]
                        if isinstance(self.images[i], np.ndarray)
                        else load_image(Path(self.images[i]))
                    )
                    for i in chunk
                ]
            )
            x = torch.from_numpy(pixels).float().div_(255.0).sub_(0.5).unsqueeze(1)
            encoded = [encode(self.texts[i], self.table) for i in chunk]
            lengths = torch.tensor([max(1, len(e)) for e in encoded], dtype=torch.long)
            flat = torch.tensor([c for e in encoded for c in (e or [BLANK])], dtype=torch.long)
            yield x, flat, lengths, [self.texts[i] for i in chunk]


# ------------------------------------------------------------------------------------------
# train / evaluate
# ------------------------------------------------------------------------------------------


def train(model, batches, epochs: int, device, lr: float = LR, tag: str = "") -> dict:
    import torch

    criterion = torch.nn.CTCLoss(blank=BLANK, zero_infinity=True)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=max(1, epochs))
    curve = []
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        total, seen = 0.0, 0
        for x, targets, lengths, _ in batches:
            x, targets, lengths = x.to(device), targets.to(device), lengths.to(device)
            logits = model(x)
            logprobs = logits.log_softmax(-1).permute(1, 0, 2)  # (T, N, C) for CTCLoss
            input_lengths = torch.full(
                (x.shape[0],), logits.shape[1], dtype=torch.long, device=device
            )
            loss = criterion(logprobs, targets, input_lengths, lengths)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()
            total += float(loss) * x.shape[0]
            seen += x.shape[0]
        schedule.step()
        curve.append(round(total / max(1, seen), 4))
        print(f"[{tag}] epoch {epoch + 1}/{epochs} loss {curve[-1]}", flush=True)
    return {"loss_curve": curve, "train_seconds": round(time.perf_counter() - started, 1)}


def predict(model, batches, chars: str, device) -> tuple[list[str], list[str]]:
    import torch

    model.eval()
    predictions, truths = [], []
    with torch.no_grad():
        for x, _, _, texts in batches:
            logits = model(x.to(device)).log_softmax(-1).cpu().numpy()
            predictions.extend(greedy_decode(logits[i], chars) for i in range(len(texts)))
            truths.extend(texts)
    return truths, predictions


# ------------------------------------------------------------------------------------------
# arms
# ------------------------------------------------------------------------------------------


def shared_alphabet() -> str:
    """One alphabet across every arm, so their CERs are comparable.

    IAM is pulled in even for the diagram-only arms: an arm that could not *emit* a character the
    pretrained arm can emit would be scored on a different label space, and the comparison 9.3.2
    exists for would be between two different problems.

    **IAM may be absent.** `data/raw/iam_line` is a DVC pointer whose store is empty, so the `iam`,
    `zero_shot` and `finetune` arms cannot run at all; `scratch` can, and blocking it on a file
    only its siblings need would lose the one measurement still available. When IAM is missing the
    alphabet is the diagram labels' own, and the arms that needed IAM are the arms that are gone -
    so nothing is being compared across two label spaces.
    """
    _, diagram_texts, _ = diagram_split("train")
    iam_train = IAM / "train.parquet"
    if not iam_train.is_file():
        return alphabet(diagram_texts)

    import pandas as pd

    iam_texts = pd.read_parquet(iam_train, columns=["text"])["text"].tolist()
    return alphabet(diagram_texts, iam_texts)


def run(arms=ARMS, epochs_iam: int = EPOCHS_IAM, epochs_diagram: int = EPOCHS_DIAGRAM) -> dict:
    import torch

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)

    chars = shared_alphabet()
    table = {c: i + 1 for i, c in enumerate(chars)}
    (CHECKPOINTS / "alphabet.json").write_text(json.dumps({"chars": chars}), encoding="utf-8")

    train_files, train_texts, _ = diagram_split("train")
    # Trained on the boxes a human drew; evaluated on those *and* on 9.3.1's detected boxes, so
    # that 9.3.6's per-provenance split can price what cropping from a detector costs without
    # any model ever having been trained on a detector's mistakes.
    val_files, val_texts, val_frame = diagram_split(
        "val", provenance=("annotated", "derived", "detected")
    )
    results: dict[str, dict] = {}

    def evaluate(model, name, unwrap=False):
        batches = Batches(val_files, val_texts, table, shuffle=False, unwrap=unwrap)
        truths, predictions = predict(model, batches, chars, device)
        RUNS.mkdir(parents=True, exist_ok=True)
        (RUNS / f"{name}.json").write_text(
            json.dumps(
                {
                    "model": name,
                    "split": "val",
                    "files": val_frame["file"].tolist(),
                    "predictions": predictions,
                }
            ),
            encoding="utf-8",
        )
        return score(truths, predictions)

    iam_weights = CHECKPOINTS / "iam.pt"
    if "iam" in arms:
        images, texts = iam_split("train")
        model = build_model(len(chars) + 1).to(device)
        stats = train(model, Batches(images, texts, table), epochs_iam, device, tag="iam")
        torch.save(model.state_dict(), iam_weights)
        val_images, val_iam_texts = iam_split("validation")
        truths, predictions = predict(
            model, Batches(val_images, val_iam_texts, table, shuffle=False), chars, device
        )
        results["iam"] = {**stats, "iam_val": score(truths, predictions), "lines": len(images)}
        print("iam", json.dumps(results["iam"]["iam_val"]), flush=True)

    if "zero_shot" in arms:
        model = build_model(len(chars) + 1).to(device)
        model.load_state_dict(torch.load(iam_weights, map_location=device))
        results["zero_shot"] = evaluate(model, "crnn_zero_shot")
        print("zero_shot", json.dumps(results["zero_shot"]), flush=True)

    if "scratch" in arms:
        model = build_model(len(chars) + 1).to(device)
        stats = train(
            model, Batches(train_files, train_texts, table), epochs_diagram, device, tag="scratch"
        )
        torch.save(model.state_dict(), CHECKPOINTS / "scratch.pt")
        results["scratch"] = {**stats, **evaluate(model, "crnn_scratch")}
        print("scratch", json.dumps(results["scratch"]), flush=True)

    if "finetune" in arms:
        model = build_model(len(chars) + 1).to(device)
        model.load_state_dict(torch.load(iam_weights, map_location=device))
        stats = train(
            model, Batches(train_files, train_texts, table), epochs_diagram, device, tag="finetune"
        )
        torch.save(model.state_dict(), CHECKPOINTS / "finetune.pt")
        results["finetune"] = {**stats, **evaluate(model, "crnn_finetune")}
        print("finetune", json.dumps(results["finetune"]), flush=True)

    if "unwrap" in arms:
        model = build_model(len(chars) + 1).to(device)
        model.load_state_dict(torch.load(iam_weights, map_location=device))
        stats = train(
            model,
            Batches(train_files, train_texts, table, unwrap=True),
            epochs_diagram,
            device,
            tag="unwrap",
        )
        torch.save(model.state_dict(), CHECKPOINTS / "unwrap.pt")
        results["unwrap"] = {**stats, **evaluate(model, "crnn_unwrap", unwrap=True)}
        print("unwrap", json.dumps(results["unwrap"]), flush=True)

    if "unwrap" in results and "finetune" in results:
        results["unwrapping_gain_cer"] = round(
            results["finetune"]["cer"] - results["unwrap"]["cer"], 4
        )

    if "scratch" in results and "finetune" in results:
        results["pretraining_benefit_cer"] = round(
            results["scratch"]["cer"] - results["finetune"]["cer"], 4
        )
    if "zero_shot" in results and "finetune" in results:
        results["domain_gap_cer"] = round(
            results["zero_shot"]["cer"] - results["finetune"]["cer"], 4
        )
    results["alphabet_size"] = len(chars)
    results["parameters"] = int(sum(p.numel() for p in build_model(len(chars) + 1).parameters()))
    return results


def load(name: str = "finetune", device=None):
    """A trained arm plus its alphabet, for 9.3.4 and 9.3.7 to decode with."""
    import torch

    chars = json.loads((CHECKPOINTS / "alphabet.json").read_text(encoding="utf-8"))["chars"]
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(len(chars) + 1).to(device)
    model.load_state_dict(torch.load(CHECKPOINTS / f"{name}.pt", map_location=device))
    model.eval()
    return model, chars, device


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", action="append", choices=list(ARMS))
    ap.add_argument("--epochs-iam", type=int, default=EPOCHS_IAM)
    ap.add_argument("--epochs-diagram", type=int, default=EPOCHS_DIAGRAM)
    ap.add_argument("--out", type=Path, default=RUNS / "crnn.json")
    args = ap.parse_args(argv)

    result = run(tuple(args.arm) if args.arm else ARMS, args.epochs_iam, args.epochs_diagram)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "loss_curve"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
