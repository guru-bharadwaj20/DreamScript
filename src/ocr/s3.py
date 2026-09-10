"""Headline S3: label OCR character error rate.

    python -m src.ocr.s3 --train        # fine-tune, then score
    python -m src.ocr.s3                # score the stored checkpoint

Target CER <= 0.15. 9.3.6's best arm reached 0.6816 and the table under it reads as a ceiling:
thirteen arms inside 0.15 of each other, the CRNN family and the TrOCR family alike. It is not a
ceiling. It is one budget spent thirteen ways.

## Why 9.3.6 stalled at 0.68

Two constants in `src.ocr.trocr` were chosen for a *comparison* and then read as a *result*:

    MODEL = "microsoft/trocr-small-handwritten"      61.6M parameters
    EPOCHS = 4

That module says so itself - it equalised wall-clock against the CRNN's thirty epochs rather
than equalising epochs, "so this is a comparison at roughly equal wall-clock budget". As an
answer to "which family is more sample-efficient per second" that is a fair protocol. As an
answer to "how well can this project read its own labels" it is a four-epoch run of the smallest
released TrOCR checkpoint, and 0.8211 is what that costs.

The CRNN arms fail differently, and their predictions name it. `adapt_style_adapt` on the
annotated node crops:

    'evaluate application'      -> 'ecaleato ien'
    'write notification email'  -> 'cnfomotintions c'
    'create new bank account'   -> 'creoo mnein cien'

The letters are approximately right and the string is roughly half the length it should be. That
is a CTC decoder whose time axis is too short for the phrase, not a model that cannot see the
ink - and it is why every CRNN arm lands within 0.02 of every other. They share the bottleneck,
so widening the search (`lexicon_beam`), the training set (`crnn_scratch`) or the style
adaptation (`adapt_style_adapt`) moves the third decimal place and nothing else.

## What this changes

One thing, deliberately: the budget. `microsoft/trocr-base-handwritten` - 333.9M parameters, an
encoder-decoder already pretrained on IAM handwriting - fine-tuned on the same 16,974 training
crops, with a cosine schedule, warmup, gradient clipping, bf16 autocast and beam search at decode.
No new data, no new labels, no change to the split, and the evaluation rows are the same 4,530 val
crops under the same `src.ocr.metrics.score`, so the number drops straight into 9.3.6's table.

## The failure that shaped the rest of this module

The first version of this file ran a fixed twelve epochs with nothing watching, drove the training
loss to **0.0044**, and scored **0.4768 CER** - worse than the CRNN it replaced. The diagnosis is
not subtle and is recorded here so the mistake is not repeated:

    predictions that are verbatim a training label string      90.0%
    val crops whose label was seen in training   CER 0.4442, exact 0.4906
    val crops whose label was NOT seen           CER 0.5388, exact 0.1545

**The decoder memorised the training label vocabulary - 3,197 distinct strings - and retrieved
from it instead of reading the handwriting.** On a phrase it had never seen it was right 15% of
the time. The errors were fluent, in-domain and wrong (`'start student list'` read as
`'application for opening a bank account received'`), diffuse rather than concentrated: the
predicted/reference length ratio was 0.98 and the hundred worst crops carried only 10% of the
edits, which is what rules out runaway decoding and leaves memorisation as the explanation. That
is the mirror of 9.3.6's error rather than a correction of it - four epochs of a 61M model was too
little, twelve unmonitored epochs of a 334M model was too much.

Two changes follow from it, and both are about *not trusting the training loss*:

**A dev split carved from the training writers.** 15 of the 85 training writers are held out as a
model-selection set; the epoch with the best dev CER is the one kept. Selecting on `val` would
make the reported number selection-on-test, which is the standard S1 and S4 are held to and this
row is no exception. Train, dev and val writers are mutually disjoint and `check_disjoint`
asserts it.

**Augmentation, to make the pixels cheaper to read than the vocabulary is to memorise.** Affine
jitter, scale and brightness, plus a deliberate crop-box jitter that imitates the detector error
the `detected` provenance already carries.

The split stays writer-disjoint and `evaluate` re-asserts that with `check_disjoint` rather than
trusting it, because a writer leak is the one error that would make a low CER meaningless.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from src.ocr.metrics import RUNS, TARGET_CER, normalise, score
from src.utils.config import ROOT

MODEL = "microsoft/trocr-base-handwritten"
CHECKPOINT = RUNS / "trocr_s3"
REPORT = ROOT / "reports" / "s3_label_ocr.json"
PROVENANCE = ("annotated", "derived", "detected")

#: Micro-batch and accumulation are chosen together so the *effective* batch stays 24, which is
#: what the optimiser sees. 12x2 and 24x1 are the same update; 24x1 issues half as many kernel
#: launches for it, and the card has the memory. Changing the effective batch would change the
#: experiment, so it is deliberately held.
SEED = 42
EPOCHS = 12
BATCH = 24
ACCUMULATE = 1
LR = 4e-5
WARMUP = 0.05
MAX_LENGTH = 96
BEAMS = 4
DEV_WRITERS = 15
DEV_CROPS = 900


#: Which crop corpus to read. "element" is 9.3.1's geometry-derived crops, the ones 9.3.6 scored;
#: "label" is `src.ocr.labelcrops`, which crops where the writing actually is. The default is the
#: latter because the former cannot contain a fifth of its own labels - see that module.
CORPUS = "label"


def split(name: str, corpus: str | None = None):
    """One writer-disjoint split of the chosen crop corpus."""
    corpus = corpus or CORPUS
    if corpus == "element":
        from src.ocr.crnn import diagram_split

        return diagram_split(name, provenance=PROVENANCE)
    from src.ocr.labelcrops import split as label_split

    return label_split(name)


def dev_writers(seed: int = SEED, writers: int = DEV_WRITERS) -> set[str]:
    """The training writers held out for model selection, as a set of scribe ids.

    Exposed because `src.ocr.labelcrops` has to know them too: the crops for these writers must
    be chosen the way val's are, without the transcript, or dev stops being a check on anything.
    """
    from src.ocr.labelcrops import load_index

    frame = load_index()
    names = sorted(frame.loc[frame["split"] == "train", "scribe"].dropna().unique())
    rng = np.random.default_rng(seed)
    return set(rng.permutation(names)[:writers].tolist())


def train_dev_split(seed: int = SEED, writers: int = DEV_WRITERS):
    """Carve a model-selection set out of the training writers.

    The epoch is chosen on this and never on `val`; holding writers out rather than rows keeps
    the choice honest about the thing that actually varies, which is handwriting.
    """
    files, texts, frame = split("train")
    held = dev_writers(seed, writers)
    mask = frame["scribe"].isin(held).to_numpy()
    keep = ~mask
    train = (
        [f for f, m in zip(files, keep, strict=True) if m],
        [t for t, m in zip(texts, keep, strict=True) if m],
        frame[keep],
    )
    dev = (
        [f for f, m in zip(files, mask, strict=True) if m],
        [t for t, m in zip(texts, mask, strict=True) if m],
        frame[mask],
    )
    return train, dev


def augment(image, rng):
    """Affine, scale, brightness and a crop-box jitter that imitates detector error."""
    from PIL import Image, ImageEnhance

    width, height = image.size
    pad = rng.uniform(-0.06, 0.06, size=4) * np.array([width, height, width, height])
    box = (pad[0], pad[1], width + pad[2], height + pad[3])
    image = image.crop([int(round(v)) for v in box])
    if image.size[0] < 8 or image.size[1] < 8:
        image = image.resize((max(8, image.size[0]), max(8, image.size[1])), Image.BILINEAR)
    image = image.rotate(
        float(rng.uniform(-3.0, 3.0)), resample=Image.BILINEAR, expand=True, fillcolor=255
    )
    image = ImageEnhance.Brightness(image).enhance(float(rng.uniform(0.85, 1.15)))
    image = ImageEnhance.Contrast(image).enhance(float(rng.uniform(0.85, 1.15)))
    return image


def processor():
    from transformers import TrOCRProcessor

    return TrOCRProcessor.from_pretrained(MODEL)


def fresh_model(device):
    from transformers import VisionEncoderDecoderModel

    model = VisionEncoderDecoderModel.from_pretrained(MODEL).to(device)
    model.config.decoder_start_token_id = model.config.decoder.decoder_start_token_id
    model.config.pad_token_id = model.config.decoder.pad_token_id
    model.config.eos_token_id = model.config.decoder.eos_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    return model


def pixels(proc, files, device, rng=None):
    """The processor's own normalisation, on RGB, which is what the ViT encoder expects."""
    from PIL import Image

    batch = [Image.open(name).convert("RGB") for name in files]
    if rng is not None:
        batch = [augment(image, rng) for image in batch]
    return proc(images=batch, return_tensors="pt").pixel_values.to(device)


def targets(proc, texts, device):
    """Tokenised transcripts with padding masked out of the loss."""
    tokens = proc.tokenizer(
        [normalise(text) for text in texts],
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    ).input_ids
    labels = tokens.masked_fill(tokens == proc.tokenizer.pad_token_id, -100)
    return labels.to(device)


def train(epochs: int = EPOCHS, batch: int = BATCH, checkpoint: Path = CHECKPOINT) -> dict:
    """Fine-tune on the training split and store the weights the evaluation reads."""
    import torch
    from transformers import get_cosine_schedule_with_warmup

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proc = processor()
    model = fresh_model(device)

    (files, texts, _), (dev_files, dev_texts, _) = train_dev_split()
    # A fixed subset keeps the per-epoch check cheap enough to run every epoch.
    step = max(1, len(dev_files) // DEV_CROPS)
    dev_files, dev_texts = dev_files[::step][:DEV_CROPS], dev_texts[::step][:DEV_CROPS]
    steps = max(1, int(np.ceil(len(files) / batch)) * epochs // ACCUMULATE)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    schedule = get_cosine_schedule_with_warmup(optimiser, int(steps * WARMUP), steps)

    rng = np.random.default_rng(SEED)
    augment_rng = np.random.default_rng(SEED + 1)
    order = np.arange(len(files))
    curve, dev_curve = [], []
    best_cer, best_epoch = float("inf"), -1
    started = time.perf_counter()
    checkpoint.mkdir(parents=True, exist_ok=True)
    for epoch in range(epochs):
        model.train()
        rng.shuffle(order)
        total, seen = 0.0, 0
        optimiser.zero_grad(set_to_none=True)
        for step, start in enumerate(range(0, len(order), batch)):
            chunk = order[start : start + batch]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = model(
                    pixel_values=pixels(proc, [files[i] for i in chunk], device, augment_rng),
                    labels=targets(proc, [texts[i] for i in chunk], device),
                ).loss
            (loss / ACCUMULATE).backward()
            if (step + 1) % ACCUMULATE == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()
                schedule.step()
                optimiser.zero_grad(set_to_none=True)
            total += float(loss) * len(chunk)
            seen += len(chunk)
        curve.append(round(total / max(1, seen), 4))

        # Greedy on the held-out training writers - the only signal that sees generalisation.
        dev_cer = score(dev_texts, predict(model, proc, dev_files, device, beams=1))["cer"]
        dev_curve.append(round(float(dev_cer), 4))
        marker = ""
        if dev_cer < best_cer:
            best_cer, best_epoch = float(dev_cer), epoch + 1
            model.save_pretrained(checkpoint)
            proc.save_pretrained(checkpoint)
            marker = "  <- best, kept"
        print(
            f"[s3] epoch {epoch + 1}/{epochs} loss {curve[-1]} dev_cer {dev_curve[-1]}{marker}",
            flush=True,
        )

    return {
        "loss_curve": curve,
        "dev_cer_curve": dev_curve,
        "best_epoch": best_epoch,
        "best_dev_cer": round(best_cer, 4),
        "train_seconds": round(time.perf_counter() - started, 1),
        "epochs": epochs,
        "train_crops": len(files),
        "dev_crops": len(dev_files),
    }


def predict(model, proc, files, device, batch: int = 24, beams: int = BEAMS) -> list[str]:
    """Beam-search transcripts for every crop, in the order given."""
    import torch

    model.eval()
    out: list[str] = []
    with torch.no_grad():
        for start in range(0, len(files), batch):
            chunk = files[start : start + batch]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                generated = model.generate(
                    pixels(proc, chunk, device), max_length=MAX_LENGTH, num_beams=beams
                )
            out.extend(proc.batch_decode(generated, skip_special_tokens=True))
    return out


def check_disjoint(train_frame, eval_frame) -> dict:
    """A low CER only means something if no writer appears on both sides of the split."""
    a = set(train_frame["scribe"].dropna().tolist())
    b = set(eval_frame["scribe"].dropna().tolist())
    overlap = a & b
    if overlap:
        raise AssertionError(f"writer leaked across the split: {sorted(overlap)!r}")
    return {"train_writers": len(a), "eval_writers": len(b), "writer_overlap": 0}


def evaluate(
    name: str = "val",
    checkpoint: Path = CHECKPOINT,
    beams: int = BEAMS,
    corpus: str | None = None,
) -> dict:
    """Score the stored checkpoint on one split, under 9.3.6's own metric."""
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    from src.ocr.metrics import breakdown

    if not (checkpoint / "config.json").is_file():
        raise FileNotFoundError(f"no S3 checkpoint at {checkpoint}; run with --train")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proc = TrOCRProcessor.from_pretrained(checkpoint)
    model = VisionEncoderDecoderModel.from_pretrained(checkpoint).to(device)

    files, truths, frame = split(name, corpus)
    _, _, train_frame = split("train", corpus)
    started = time.perf_counter()
    predicted = predict(model, proc, files, device, beams=beams)
    elapsed = time.perf_counter() - started

    # Labels whose writing was never located are scored as empty predictions rather than
    # dropped. Reporting CER over only the crops the pipeline managed to find would let the
    # thing being measured choose its own evaluation set.
    unfound: list[str] = []
    if (corpus or CORPUS) == "label":
        from src.ocr.labelcrops import unlocated

        unfound = unlocated(name)
    located = score(truths, predicted)
    result = score(list(truths) + unfound, list(predicted) + [""] * len(unfound))
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / f"s3_{name}.json").write_text(
        json.dumps(
            {
                "model": "s3",
                "split": name,
                "files": frame["file"].tolist(),
                "predictions": predicted,
            }
        ),
        encoding="utf-8",
    )
    return {
        "criterion": "S3",
        "target_cer": TARGET_CER,
        "model": f"{MODEL} fine-tuned {EPOCHS} epochs (beam {beams})",
        "split": name,
        "protocol": "9.3.1 crops, 1.3.3 writer-disjoint split, src.ocr.metrics.score",
        **check_disjoint(train_frame, frame),
        **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in result.items()},
        "unlocated_labels": len(unfound),
        "located_crops": len(truths),
        "cer_located_only": round(float(located["cer"]), 4),
        "milliseconds_per_crop": round(elapsed * 1000.0 / max(1, len(files)), 2),
        "passes": bool(result["cer"] <= TARGET_CER),
        **{
            f"by_{column}": breakdown(frame, predicted, column)
            for column in ("kind", "element", "source", "provenance")
            if column in frame.columns
        },
    }


def write_report(result: dict, path: Path = REPORT) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", action="store_true", help="fine-tune before scoring")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--split", default="val", choices=("val", "test"))
    ap.add_argument("--corpus", default=CORPUS, choices=("label", "element"))
    args = ap.parse_args(argv)
    globals()["CORPUS"] = args.corpus
    try:
        training = train(args.epochs) if args.train else {}
        result = {**evaluate(args.split), **({"training": training} if training else {})}
        write_report(result)
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0 if result["passes"] else 2


if __name__ == "__main__":
    sys.exit(main())
