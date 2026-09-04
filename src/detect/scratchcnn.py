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

FILLED_IN_BELOW
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
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "confusion"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
