"""Phase 9.3.3 - a transformer OCR baseline against 9.3.2's CRNN, on identical crops.

    python -m src.ocr.trocr --arm zero_shot
    python -m src.ocr.trocr --arm finetune
    python -m src.ocr.trocr

`microsoft/trocr-small-handwritten`: a ViT encoder and an autoregressive text decoder, 62M
parameters against the CRNN's 3.8M, pretrained on far more handwriting than IAM alone. The
comparison is run on 9.3.1's corpus with 9.3.6's evaluator, so the only thing that differs
between the two rows is the model.

## What is actually being compared

Not "transformer against RNN". The architectural difference that matters here is **conditioning**:

    CRNN + CTC     each column is scored independently; the output cannot depend on the output.
    TrOCR          each character is generated conditioned on the characters before it.

Conditioning is a language model, and a language model is exactly the right thing to have when
reading `evaluate the Application` and exactly the wrong thing when reading `q0`. 8.7 measured
the failure mode directly - zero-shot TrOCR turned `send a welcome pack in a letter` into
`a b c dwynson was a former election`, a **CER of 1.0573**, above 1.0 because a fluent decoder
inserts more than it gets right. So the interesting comparison is not the headline CER but
**where in the corpus the two models differ**, and this task reports the split that 9.3.1's
corpus makes possible and 8.7's did not:

    by length      short labels (fa_bresler's single symbols) against long BPMN phrases
    by kind        node labels against edge labels
    by provenance  annotated boxes against 9.1.3's detected boxes

The prediction going in is that the transformer wins on long phrases, loses on single symbols,
and that a single pooled CER will hide both.

## Cost is reported, because it is part of the comparison

Parameters, training seconds and inference milliseconds per crop. A model that is better by 0.02
CER and 30x more expensive is a different recommendation from one that is better and free, and
9.2.7 already found the 30x-larger model losing outright on this project's other corpus.

## What it measured

`microsoft/trocr-small-handwritten`, 61,596,672 parameters, four epochs, against 9.3.2's
3,787,806-parameter CRNN on identical crops and the same evaluator.

    model               CER      WER    exact   params   ms/crop
    crnn finetune    0.6864   0.9496   0.0660    3.8M         -
    trocr finetune   0.8211   1.0100   0.0704   61.6M     11.91
    trocr zero_shot  1.1702   1.5427   0.0055   61.6M     95.58

**The transformer loses by 0.1347 CER with 16.3 times the parameters.** That is the third time
this project has run a large model against a small one on the same data - 9.2.7's ResNet-18
against a hand-built CNN, 7.4.7's per-scribe against pooled, 8.7's per-style against global - and
the fourth time the smaller model has won.

**The length breakdown is the finding, and it confirms the mechanism this task predicted.**

    band     crops   CRNN CER   TrOCR CER   CRNN exact   TrOCR exact
    13+      2,800     0.6916      0.7532       0.0018        0.0221
    4-12     1,058     0.6545      1.1733       0.0189        0.0813
    1-3        672     0.6855      1.4052       0.4077        0.2545

**The CRNN is flat across label length - 0.65 to 0.69 - and TrOCR doubles its error as labels get
shorter, reaching a CER of 1.4052 on one-to-three-character labels.** A CER above 1.0 means the
model emits more wrong characters than the truth contains, and that is precisely what an
autoregressive decoder does when asked for `a` or `q0`: it has a language model, the language
model wants a sentence, and there is nothing in the architecture that can decline. The CTC model
cannot hallucinate because it has nothing to condition on, and on 672 crops that difference is
worth **0.72 CER**. 8.7 saw the same failure zero-shot and this shows it **surviving four epochs
of fine-tuning**.

**The prediction was half wrong and that half matters too.** This docstring predicted the
transformer would *win* on long phrases and lose on short ones. It loses on long phrases as well -
0.7532 against 0.6916 - just by much less. So the language model is not paying for itself anywhere
on this corpus, not even where the labels are full English phrases like `evaluate the
Application`.

**Where TrOCR does win is exact match, and only in aggregate: 0.0704 against 0.0660.** Per band it
wins on the two longer bands (0.0221 against 0.0018, and 0.0813 against 0.0189) and loses badly on
the short one (0.2545 against 0.4077). So the decoder does produce *completely correct* long
labels more often - which is the thing a language model should buy - while being worse on
characters overall. If Phase 12 needed only long identifiers this comparison would be closer.

**The fairness caveat is stated rather than buried: TrOCR had four epochs and the CRNN thirty**,
and TrOCR's loss was still falling (2.95, 2.14, 1.90, 1.75). Four was chosen because an epoch
costs 250 s against the CRNN's 19 s, and equalising *epochs* would have meant 125 minutes for one
arm. So this is a comparison at roughly equal wall-clock budget rather than equal epochs, and the
gap of 0.1347 should be read with that attached. What it does not affect is the length breakdown:
no amount of further training turns an autoregressive decoder into one that will emit two
characters when asked.

**Inference cost moves in the interesting direction.** Zero-shot TrOCR takes **95.58 ms a crop**
and the fine-tuned model **11.91 ms**, an eight-fold speedup from the same weights and the same
hardware, because generation stops when the model emits its end token and fine-tuning taught it
that these labels are short. The 95.58 ms is the model writing prose nobody asked for, timed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from src.ocr.metrics import RUNS, normalise, score
from src.ocr.textcrops import OUT as CROPS

MODEL = "microsoft/trocr-small-handwritten"
CHECKPOINT = RUNS / "trocr"

SEED = 42
EPOCHS = 4
BATCH = 16
LR = 5e-5
MAX_LENGTH = 96

ARMS = ("zero_shot", "finetune")


def processor():
    from transformers import TrOCRProcessor

    return TrOCRProcessor.from_pretrained(MODEL)


def fresh_model(device):
    from transformers import VisionEncoderDecoderModel

    model = VisionEncoderDecoderModel.from_pretrained(MODEL).to(device)
    model.config.decoder_start_token_id = model.config.decoder.decoder_start_token_id
    model.config.pad_token_id = model.config.decoder.pad_token_id
    return model


def images(files):
    """RGB PIL images, because the processor's normalisation expects three channels."""
    from PIL import Image

    return [Image.open(f).convert("RGB") for f in files]


def batches(files, texts, batch: int):
    order = np.arange(len(files))
    for start in range(0, len(order), batch):
        chunk = order[start : start + batch]
        yield [files[i] for i in chunk], [texts[i] for i in chunk]


def finetune(files, texts, proc, device, epochs: int = EPOCHS, batch: int = BATCH) -> dict:
    import torch

    model = fresh_model(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    order = np.arange(len(files))
    rng = np.random.default_rng(SEED)
    started = time.perf_counter()
    curve = []
    for epoch in range(epochs):
        model.train()
        rng.shuffle(order)
        total, seen = 0.0, 0
        for start in range(0, len(order), batch):
            chunk = order[start : start + batch]
            pixels = proc(
                images=images([files[i] for i in chunk]), return_tensors="pt"
            ).pixel_values.to(device)
            labels = proc.tokenizer(
                [normalise(texts[i]) for i in chunk],
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            ).input_ids.to(device)
            labels[labels == proc.tokenizer.pad_token_id] = -100
            loss = model(pixel_values=pixels, labels=labels).loss
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            total += float(loss) * len(chunk)
            seen += len(chunk)
        curve.append(round(total / max(1, seen), 4))
        print(f"[trocr] epoch {epoch + 1}/{epochs} loss {curve[-1]}", flush=True)
    return {
        "model": model,
        "loss_curve": curve,
        "train_seconds": round(time.perf_counter() - started, 1),
    }


def predict(model, files, proc, device, batch: int = 32) -> tuple[list[str], float]:
    import torch

    model.eval()
    out: list[str] = []
    started = time.perf_counter()
    with torch.no_grad():
        for chunk, _ in batches(files, files, batch):
            pixels = proc(images=images(chunk), return_tensors="pt").pixel_values.to(device)
            generated = model.generate(pixels, max_length=MAX_LENGTH)
            out.extend(proc.batch_decode(generated, skip_special_tokens=True))
    milliseconds = (time.perf_counter() - started) * 1000.0 / max(1, len(files))
    return out, round(milliseconds, 2)


def slices(frame, texts, predictions) -> dict:
    """Where the two families differ: length, kind, provenance."""
    from src.ocr.metrics import breakdown

    frame = frame.reset_index(drop=True).copy()
    lengths = frame["text"].str.len()
    frame["length_band"] = np.where(lengths <= 3, "1-3", np.where(lengths <= 12, "4-12", "13+"))
    return {
        "by_length": breakdown(frame, predictions, "length_band"),
        "by_kind": breakdown(frame, predictions, "kind"),
        "by_provenance": breakdown(frame, predictions, "provenance"),
    }


def run(arms=ARMS, epochs: int = EPOCHS) -> dict:
    import torch

    from src.ocr.crnn import diagram_split

    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proc = processor()
    CHECKPOINT.mkdir(parents=True, exist_ok=True)

    train_files, train_texts, _ = diagram_split("train")
    val_files, val_texts, val_frame = diagram_split(
        "val", provenance=("annotated", "derived", "detected")
    )
    results: dict[str, dict] = {}

    def record(name, model):
        predictions, milliseconds = predict(model, val_files, proc, device)
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
        return {
            **score(val_texts, predictions),
            "inference_ms_per_crop": milliseconds,
            "parameters": int(sum(p.numel() for p in model.parameters())),
            **slices(val_frame, val_texts, predictions),
        }

    if "zero_shot" in arms:
        model = fresh_model(device)
        results["zero_shot"] = record("trocr_zero_shot", model)
        print("trocr zero_shot", results["zero_shot"]["cer"], flush=True)
        del model
        torch.cuda.empty_cache()

    if "finetune" in arms:
        trained = finetune(train_files, train_texts, proc, device, epochs)
        model = trained.pop("model")
        results["finetune"] = {**trained, **record("trocr_finetune", model)}
        model.save_pretrained(CHECKPOINT)
        print("trocr finetune", results["finetune"]["cer"], flush=True)

    crnn = (
        json.loads((RUNS / "crnn.json").read_text(encoding="utf-8-sig"))
        if (RUNS / "crnn.json").is_file()
        else {}
    )
    if crnn.get("finetune") and results.get("finetune"):
        results["crnn_cer"] = crnn["finetune"]["cer"]
        results["trocr_minus_crnn_cer"] = round(
            results["finetune"]["cer"] - crnn["finetune"]["cer"], 4
        )
        results["parameter_ratio"] = round(
            results["finetune"]["parameters"] / crnn["parameters"], 1
        )
    results["crops_root"] = str(CROPS)
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", action="append", choices=list(ARMS))
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--out", type=Path, default=RUNS / "trocr.json")
    args = ap.parse_args(argv)

    result = run(tuple(args.arm) if args.arm else ARMS, args.epochs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
