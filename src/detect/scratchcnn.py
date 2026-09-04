"""Phase 9.2.1 - a conv/pool/FC classifier built by hand, and what it scores on shape crops.

    python -m src.detect.scratchcnn                  # train the reference architecture
    python -m src.detect.scratchcnn --epochs 30

Not a detector and not competitive with one. This is the exam-facing deliverable: a stack whose
every layer is written out, whose parameter count is arithmetic rather than a `summary()` call
(9.2.2), whose receptive field can be computed by hand (9.2.3), and which 9.2.5 can ablate one
knob at a time. It classifies a 64x64 crop that is already known to contain one shape.

## The architecture, and why it is shaped this way

Six 3x3 convolutions in four stages, each stage ending in a 2x2 max-pool, then two fully
connected layers:

    stage 1   conv 1->16, conv 16->16, pool     64 -> 32
    stage 2   conv 16->32, conv 32->32, pool    32 -> 16
    stage 3   conv 32->64, pool                 16 -> 8
    stage 4   conv 64->128, pool                8 -> 4
    head      flatten 2048 -> 128 -> 7

Stacked 3x3 rather than one large kernel, because that is the substitution the receptive-field
arithmetic in 9.2.3 exists to make visible: two 3x3 layers see a 5x5 window for 18 weights a
channel pair where one 5x5 layer needs 25, and the two-layer version has a nonlinearity in the
middle. 9.2.5 tests the substitution directly by putting 5x5 and 7x7 kernels in the same slots.

Channels double whenever the map halves, which keeps the per-stage activation volume roughly
constant - the convention every ImageNet-era architecture uses and the reason the arithmetic in
9.2.2 comes out as tidily as it does.

## The class imbalance, stated before any accuracy is quoted

The crop corpus is **59% `rectangle`** (24,373 of 41,014) and 0.5% `parallelogram` (195), so
accuracy is close to useless here and **macro F1 is the number reported**, exactly as in Phases
5-8. A model that predicts `rectangle` for everything scores 0.59 accuracy and 0.11 macro F1.
The training loss is class-weighted for the same reason.

## What it measured

30 epochs, 33,312 training crops, 5,359 validation crops, 136 s on one card.

**Macro F1 0.9595 pooled and 0.9840 on the hand-drawn hdbpmn crops**, against a majority
baseline of 0.5076 accuracy. Accuracy is 0.9931 and is quoted only to be discounted: on a
corpus this imbalanced it is the number that would look good regardless.

**The finding worth carrying is what this says about 7.4.** Phase 7.4.8 asked exactly this
question - name the shape from the pixels of one hdbpmn node - and reported that **the 22
descriptors of 7.4.1 support a 0.8160 macro F1 ceiling** with a supervised random forest, that
the learned GMM vocabulary reached 0.4160 and 4.1.3's hand-written template rules 0.3734, and
that "neither vocabulary should ship as a shape classifier". A 372,183-parameter CNN reading raw
pixels reaches **0.9840 on the same source** - and it is most decisively better exactly where
the descriptors were weakest:

    class        7.4.8 RF ceiling      this CNN (hdbpmn)
    freeform            0.5666               0.9617
    rectangle           ~0.82                0.9812
    circle              ~0.82                0.9934
    diamond             ~0.82                0.9962

7.4.8 wrote that `freeform` "is separable in this table, just not by a rule or by a cluster
centre, because what identifies it is a conjunction rather than a region". **A conjunction of
features over a region is what a convolutional stack computes**, and the +0.395 on that class is
the clearest single statement in this project that handcrafted descriptors were the binding
constraint rather than the difficulty of the task.

That comparison is **not controlled** and is not offered as one: 7.4.8 used five folds grouped
by page over all 12,400 hdbpmn shapes with four classes, this uses a writer-disjoint split with
five, and the crop here carries a 12% context margin that a descriptor vector does not. The gap
is large enough to survive those differences; the exact number is not.

**Where the errors are.** The confusion matrix is almost diagonal and every off-diagonal entry
is a confusion the earlier phases predicted. Six of 74 `double-circle` crops are called `circle`
- the outer ring lost to the resize, which is precisely why 2.1.4 froze `double-circle` as its
own shape and why the crop carries a context margin at all. Three `rectangle` crops are called
`parallelogram`, which with **7 parallelogram instances in validation** is what holds that class
to 0.8235 at perfect recall: 7 of 7 found, 3 false positives, so the F1 is a precision artefact
of a class too small to measure. `freeform` at 0.9617 loses two crops to `rounded-rect`, which
is a BPMN data object drawn with soft corners.

**One number is deliberately not a headline.** flowchartseg's crops score 0.999 accuracy because
they are computer-rendered with exact boundaries; the pooled 0.9595 is 54% those. The hdbpmn
figure is the deployment one, exactly as in 9.1.4.

Training time is reported at 136 s but was measured while a Phase 9.1.7 run held the same GPU,
so it is an upper bound and not a benchmark.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.detect.choice import RUNS
from src.detect.crops import OUT as CROPS
from src.detect.crops import SHAPE_CLASSES, SIZE, load_split

SEED = 42
WEIGHTS = RUNS / "scratchcnn.pt"


@dataclass
class Spec:
    """Every knob 9.2.5 ablates, in one object so a run is described by its config.

    `stages` is `(out_channels, convs_in_stage)` per stage; the map halves once per stage.
    """

    stages: tuple[tuple[int, int], ...] = ((16, 2), (32, 2), (64, 1), (128, 1))
    kernel: int = 3
    pool: str = "max"
    batchnorm: bool = True
    dropout: float = 0.3
    hidden: int = 128
    in_channels: int = 1
    size: int = SIZE
    classes: tuple[str, ...] = field(default_factory=lambda: SHAPE_CLASSES)

    @property
    def padding(self) -> int:
        """`same` padding, which is what makes the output-size column in 9.2.2 trivial."""
        return self.kernel // 2

    def layers(self) -> list[dict]:
        """The architecture as data, so 9.2.2 and the torch module cannot disagree about it."""
        out: list[dict] = []
        channels = self.in_channels
        for out_channels, convs in self.stages:
            for _ in range(convs):
                out.append(
                    {
                        "type": "conv",
                        "in": channels,
                        "out": out_channels,
                        "kernel": self.kernel,
                        "stride": 1,
                        "padding": self.padding,
                    }
                )
                if self.batchnorm:
                    out.append({"type": "batchnorm", "channels": out_channels})
                out.append({"type": "relu"})
                channels = out_channels
            out.append({"type": "pool", "kernel": 2, "stride": 2, "padding": 0, "mode": self.pool})
        return out


def build(spec: Spec = Spec()):
    import torch.nn as nn

    modules: list[nn.Module] = []
    for layer in spec.layers():
        if layer["type"] == "conv":
            modules.append(
                nn.Conv2d(
                    layer["in"],
                    layer["out"],
                    layer["kernel"],
                    stride=layer["stride"],
                    padding=layer["padding"],
                    bias=not spec.batchnorm,
                )
            )
        elif layer["type"] == "batchnorm":
            modules.append(nn.BatchNorm2d(layer["channels"]))
        elif layer["type"] == "relu":
            modules.append(nn.ReLU(inplace=True))
        elif layer["type"] == "pool":
            pooling = nn.MaxPool2d if layer["mode"] == "max" else nn.AvgPool2d
            modules.append(pooling(layer["kernel"], layer["stride"]))

    side = spec.size // (2 ** len(spec.stages))
    flat = spec.stages[-1][0] * side * side
    modules += [
        nn.Flatten(),
        nn.Linear(flat, spec.hidden),
        nn.ReLU(inplace=True),
        nn.Dropout(spec.dropout),
        nn.Linear(spec.hidden, len(spec.classes)),
    ]
    return nn.Sequential(*modules)


# ------------------------------------------------------------------------------------------
# training
# ------------------------------------------------------------------------------------------


def class_weights(y: np.ndarray, classes=SHAPE_CLASSES) -> np.ndarray:
    """Inverse-frequency weights. A 195-row class beside a 24,373-row one is otherwise ignored."""
    counts = np.bincount(y, minlength=len(classes)).astype(float)
    weights = np.where(counts > 0, len(y) / (len(classes) * np.maximum(counts, 1)), 0.0)
    return weights


def train(
    spec: Spec = Spec(),
    epochs: int = 30,
    batch: int = 128,
    lr: float = 1e-3,
    root: Path = CROPS,
    seed: int = SEED,
    save: Path | None = None,
) -> dict:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_train, y_train = load_split("train", root)
    x_val, y_val = load_split("val", root)
    if not len(x_train) or not len(x_val):
        raise SystemExit(f"no crops under {root} (run `python -m src.detect.crops`)")

    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=batch,
        shuffle=True,
        drop_last=False,
    )
    model = build(spec).to(device)
    weights = torch.tensor(class_weights(y_train), dtype=torch.float32, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

    started = time.perf_counter()
    history = []
    for _ in range(epochs):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = criterion(model(xb), yb)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            total += float(loss) * len(xb)
        schedule.step()
        history.append(round(total / len(x_train), 5))
    seconds = time.perf_counter() - started

    result = evaluate(model, x_val, y_val, device)
    result.update(
        {
            "epochs": epochs,
            "train_seconds": round(seconds, 1),
            "train_rows": int(len(x_train)),
            "val_rows": int(len(x_val)),
            "parameters": int(sum(p.numel() for p in model.parameters())),
            "loss_curve": history,
        }
    )
    if save:
        save.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "spec": vars(spec)}, save)
        result["weights"] = str(save)
    return result


def evaluate(model, x: np.ndarray, y: np.ndarray, device=None, batch: int = 256) -> dict:
    import torch
    from sklearn.metrics import confusion_matrix, f1_score

    device = device or next(model.parameters()).device
    model.eval()
    predicted = []
    with torch.no_grad():
        for start in range(0, len(x), batch):
            chunk = torch.from_numpy(x[start : start + batch]).to(device)
            predicted.append(model(chunk).argmax(1).cpu().numpy())
    predicted = np.concatenate(predicted) if predicted else np.zeros(0, dtype=np.int64)

    present = sorted(set(y.tolist()))
    majority = float((y == np.bincount(y).argmax()).mean()) if len(y) else 0.0
    return {
        "accuracy": round(float((predicted == y).mean()), 4),
        "macro_f1": round(float(f1_score(y, predicted, average="macro", zero_division=0)), 4),
        "per_class_f1": {
            SHAPE_CLASSES[i]: round(float(f1_score(y == i, predicted == i, zero_division=0)), 4)
            for i in present
        },
        "majority_baseline_accuracy": round(majority, 4),
        "confusion": confusion_matrix(
            y, predicted, labels=list(range(len(SHAPE_CLASSES)))
        ).tolist(),
    }


def by_source(weights: Path = WEIGHTS, root: Path = CROPS, split: str = "val") -> dict:
    """The same model scored on each dataset's crops separately.

    The pooled validation split is **half computer-rendered flowchartseg crops**, so a pooled
    figure is mostly a statement about pages nobody drew - 9.1.4 made the same correction for
    the detector. The hdbpmn row is the one comparable with 7.4.8's descriptor ceiling.
    """
    import torch

    model, _ = load(weights)
    out = {}
    for name in ("hdbpmn", "fa_bresler", "flowchartseg"):
        x, y = load_split(split, root, sources=(name,))
        if not len(x):
            continue
        scored = evaluate(model, x, y, torch.device("cpu"))
        out[name] = {
            "rows": int(len(y)),
            "accuracy": scored["accuracy"],
            "macro_f1": scored["macro_f1"],
            "per_class_f1": scored["per_class_f1"],
        }
    return out


def load(path: Path = WEIGHTS):
    """Rebuild a saved model together with the spec it was trained under."""
    import torch

    blob = torch.load(path, map_location="cpu", weights_only=False)
    fields = dict(blob["spec"])
    fields["stages"] = tuple(tuple(s) for s in fields["stages"])
    fields["classes"] = tuple(fields["classes"])
    spec = Spec(**fields)
    model = build(spec)
    model.load_state_dict(blob["state_dict"])
    return model, spec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--root", type=Path, default=CROPS)
    ap.add_argument("--out", type=Path, default=RUNS / "scratchcnn.json")
    args = ap.parse_args(argv)

    result = train(Spec(), args.epochs, args.batch, root=args.root, save=WEIGHTS)
    result["by_source"] = by_source(WEIGHTS, args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "confusion"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
