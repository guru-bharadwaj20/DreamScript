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

Four arms, 20 epochs, identical crops and metric.

    arm                macro F1   accuracy      params   trainable
    scratch              0.9574     0.9922     372,183     372,183
    resnet18_ft          0.9475     0.9927  11,180,103  11,180,103
    resnet18_random      0.9440     0.9897  11,180,103  11,180,103
    resnet18_frozen      0.7928     0.9410  11,180,103       3,591

**"Transfer beats scratch" is false here, and the fourth arm is what turns that from an
anecdote into a decomposition.** The naive comparison - fine-tuned ResNet against the hand-built
network - reads **-0.0099**: transfer *loses* to a model with **30 times fewer parameters**.
Split into its two parts:

    pretraining benefit    resnet18_ft - resnet18_random   =  +0.0035
    architecture benefit   resnet18_random - scratch       =  -0.0134

So **ImageNet pretraining is worth +0.0035 macro F1 - three thousandths, indistinguishable from
run noise - and the larger architecture is worth minus 0.0134.** Without `resnet18_random` the
only available reading would have been "transfer loses by 0.01", with no way to tell whether the
weights or the architecture were responsible; the answer is that the weights help imperceptibly
and the architecture hurts. 7.4.7 priced per-scribe adaptation and 8.7 priced per-style OCR
heads with exactly this kind of control, and both found the same shape: **on this corpus, the
bigger or more specialised model loses to the smaller general one.**

**Freezing is a disaster and it is the most informative arm.** At **0.7928, a cost of 0.1547**
against fine-tuning, with only 3,591 trainable parameters, the frozen ResNet is being asked to
classify line drawings using features learned from natural photographs, and the per-class
breakdown says precisely which features are missing: `parallelogram` **0.2500**, `freeform`
**0.7489**, `double-circle` **0.7543**, against `diamond` 0.9847 and `rectangle` 0.9676. The
classes it can still do are the ones with strong oriented edges - which is the one thing
ImageNet's early layers genuinely transfer - and the classes it cannot are the ones needing a
*shape-specific conjunction*: a slanted pair of sides, an irregular outline, one ring inside
another. That is the domain gap this task predicted, measured rather than asserted.

**What fine-tuning actually is here.** `resnet18_ft` and `resnet18_random` differ by 0.0035
after 20 epochs on 33,312 crops, so the pretrained weights are functioning as **a marginally
better initialisation and not as reusable features**. With this much in-domain data there is
little for pretraining to add; the frozen arm shows what happens when it is forced to add
everything.

**The recommendation is the scratch network**, which is best on macro F1, 30x smaller, and
trains in 48 s against 64 s. The one number favouring the ResNets is accuracy - `resnet18_ft` is
0.9927 against 0.9922 - and that is the imbalance talking: on a corpus 59% `rectangle`, accuracy
rewards the majority class and macro F1 is what separates these models.
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
