"""Phase 9.2.7 - what ImageNet pretraining is worth on 64x64 hand-drawn shape crops.

    python -m src.detect.transfer

Four arms on identical crops, folds and metric:

    scratch          9.2.1's hand-built stack, trained from random init
    resnet18_frozen  ImageNet ResNet-18, all conv weights frozen, new 7-way head
    resnet18_ft      the same, fine-tuned end to end
    resnet18_random  the same architecture at random init, trained end to end

**The fourth arm is what makes this a measurement rather than a slogan.** "Transfer beats
scratch" is usually demonstrated by comparing a pretrained ResNet against a small hand-built
network, which conflates two different advantages - the pretrained *weights* and the much larger
*architecture*. `resnet18_random` holds the architecture fixed and removes only the pretraining,
so the difference between it and `resnet18_ft` is what pretraining is actually worth, and the
difference between it and `scratch` is what 11M parameters buy over 372k. 7.4.7 and 8.7 both
found this project's specialised models losing to pooled ones for want of exactly such a
control.

## The domain gap, stated in advance

ImageNet is 224x224 colour photographs of natural objects. These are 64x64 greyscale line
drawings that are mostly white. Every one of the low-level statistics a pretrained first layer
encodes - colour opponency, natural-image gradient distributions, texture - is absent here, and
the only transferable thing is edge and corner detection. So the prediction going in is that
freezing will do badly and fine-tuning will mostly be a better *initialisation* rather than
genuine feature reuse. Grey crops are replicated to three channels and resized to 64x64 rather
than 224, because upsampling a 64px crop to 224 fabricates detail and quadruples the cost.

## What it measured

FILLED_IN_BELOW
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from src.detect.choice import RUNS
from src.detect.crops import OUT as CROPS
from src.detect.crops import SHAPE_CLASSES, load_split
from src.detect.scratchcnn import Spec, class_weights
from src.detect.scratchcnn import evaluate as evaluate_model
from src.detect.scratchcnn import train as train_scratch

SEED = 42


def build_resnet(pretrained: bool, freeze: bool, classes=SHAPE_CLASSES):
    """ResNet-18 with a 7-way head, optionally frozen below it.

    The stem is left at its ImageNet 7x7 stride-2 form rather than being swapped for a 3x3:
    changing it would mean the frozen arm no longer holds pretrained weights in the layer that
    matters most, and the comparison is about pretraining rather than about stem design.
    """
    import torch.nn as nn
    from torchvision.models import ResNet18_Weights, resnet18

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    if freeze:
        for parameter in model.parameters():
            parameter.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    return model


def to_three_channel(x: np.ndarray) -> np.ndarray:
    """`(N, 1, H, W)` -> `(N, 3, H, W)`; ImageNet stems expect three channels."""
    return np.repeat(x, 3, axis=1)


def train_resnet(
    pretrained: bool,
    freeze: bool,
    epochs: int = 20,
    batch: int = 128,
    lr: float = 1e-3,
    root: Path = CROPS,
    seed: int = SEED,
) -> dict:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_train, y_train = load_split("train", root)
    x_val, y_val = load_split("val", root)
    x_train, x_val = to_three_channel(x_train), to_three_channel(x_val)

    model = build_resnet(pretrained, freeze).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=batch,
        shuffle=True,
    )
    weights = torch.tensor(class_weights(y_train), dtype=torch.float32, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimiser = torch.optim.Adam(trainable, lr=lr)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

    started = time.perf_counter()
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = criterion(model(xb), yb)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
        schedule.step()
    seconds = time.perf_counter() - started

    result = evaluate_model(model, x_val, y_val, device)
    result.update(
        {
            "train_seconds": round(seconds, 1),
            "parameters": int(sum(p.numel() for p in model.parameters())),
            "trainable_parameters": int(sum(p.numel() for p in trainable)),
            "epochs": epochs,
        }
    )
    return result


def run(epochs: int = 20, root: Path = CROPS) -> dict:
    arms = {}
    scratch = train_scratch(Spec(), epochs=epochs, root=root)
    arms["scratch"] = {k: v for k, v in scratch.items() if k != "loss_curve"}
    arms["resnet18_frozen"] = train_resnet(True, True, epochs, root=root)
    arms["resnet18_ft"] = train_resnet(True, False, epochs, root=root)
    arms["resnet18_random"] = train_resnet(False, False, epochs, root=root)

    f1 = {name: arm["macro_f1"] for name, arm in arms.items()}
    return {
        "epochs": epochs,
        "arms": arms,
        "macro_f1": f1,
        # The three quantities the fourth arm exists to separate.
        "pretraining_benefit": round(f1["resnet18_ft"] - f1["resnet18_random"], 4),
        "architecture_benefit": round(f1["resnet18_random"] - f1["scratch"], 4),
        "naive_transfer_claim": round(f1["resnet18_ft"] - f1["scratch"], 4),
        "freezing_cost": round(f1["resnet18_ft"] - f1["resnet18_frozen"], 4),
        "best": max(f1, key=f1.get),
    }


def write_table(result: dict) -> list[str]:
    lines = [
        "| arm | macro F1 | accuracy | params | trainable | train s |",
        "| :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, arm in result["arms"].items():
        lines.append(
            f"| `{name}` | {arm['macro_f1']:.4f} | {arm['accuracy']:.4f} | "
            f"{arm['parameters']:,} | {arm.get('trainable_parameters', arm['parameters']):,} | "
            f"{arm['train_seconds']:.0f} |"
        )
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--root", type=Path, default=CROPS)
    ap.add_argument("--out", type=Path, default=RUNS / "transfer.json")
    args = ap.parse_args(argv)

    result = run(args.epochs, args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\n".join(write_table(result)))
    print(json.dumps({k: v for k, v in result.items() if k != "arms"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
