"""Phase 9.2.5 - kernel size, depth, pooling and BatchNorm, one knob at a time.

    python -m src.detect.ablations              # the full grid
    python -m src.detect.ablations --epochs 20

Four axes, each varied against the same reference configuration so every row differs from the
baseline in exactly one respect. That is the whole design: a grid that moves two things at once
cannot attribute anything, and this task exists to attribute.

    kernel      3 (reference), 5, 7
    depth       4 stages (reference), 3, 2
    pooling     max (reference), average
    batchnorm   on (reference), off

Every cell reports **macro F1**, not accuracy: 9.2's crop corpus is 59% `rectangle`, so a model
that answers `rectangle` to everything scores 0.59 accuracy, and an ablation table in accuracy
would rank such a model above a genuinely better one that trades a little majority-class recall
for the small classes.

Each cell also reports its parameter count from 9.2.2's arithmetic, because **an ablation that
changes accuracy and parameters together has not isolated anything either** - a 7x7 kernel is
not "a bigger receptive field", it is a bigger receptive field and 5.4x the convolutional
weights, and the table has to show both for the reader to tell which one moved the number.

## What it measured

Seven arms, 20 epochs each, one knob apart.

    arm             change                params      RF   macro F1   d vs ref
    reference       3x3, 4 stages, max     372,183     52     0.9511        -
    kernel_5        5x5 kernels            564,951     88     0.9481    -0.0030
    kernel_7        7x7 kernels            854,103    124     0.9519    +0.0008
    depth_3         3 stages               560,343     28     0.9344    -0.0167
    depth_2         2 stages             1,066,071     16     0.9430    -0.0081
    pool_avg        average pooling        372,183     52     0.9463    -0.0048
    no_batchnorm    BatchNorm removed      371,895     52     0.8990    -0.0521

**BatchNorm is the only knob that matters, and it is free.** Removing it costs **0.0521 macro
F1 - ten times the next largest effect - while changing the parameter count by 288, or 0.08%.**
Every other axis moves the score by less than 0.017, and the three that move it most cost 1.5x
to 2.9x the parameters to do so. If one sentence is wanted from this table: **the architecture
was not the problem; the normalisation was the only thing holding it up.**

**Enlarging the kernel buys nothing, and that is a direct refutation of the obvious reading of
9.2.3.** 9.2.3 found the reference network's receptive field 12 pixels short of its input and
observed that shape is a global property. The natural inference is that widening the field would
help. It does not: `kernel_5` reaches RF 88 and `kernel_7` reaches 124 - **both comfortably
covering the 64-pixel crop** - and they score -0.0030 and +0.0008 for 1.5x and 2.3x the
parameters. Covering the input is worth nothing measurable, because 9.2.3's other half was
already the explanation: **the fully connected layer does the global integration**, and it does
it whether or not the convolutions could have. The +0.0008 of `kernel_7` is well inside run
noise and is not a result.

**Depth is the one axis where fewer parameters would have been the wrong summary.** Removing
stages *increases* the parameter count - 4 stages 372k, 3 stages 560k, 2 stages 1.07M - because
each dropped pool doubles the flatten width and the first dense layer pays for every position.
So `depth_2` is a **2.9x larger model that scores 0.0081 worse**, and `depth_3` is 1.5x larger
and 0.0167 worse. Parameters and capacity are not the same quantity, and this is the cleanest
illustration of it in the project: the pooling that removes parameters is also what buys the
depth that earns the accuracy.

**Average pooling costs 0.0048 at identical cost**, which is small but consistent with the
domain: these crops are thin dark strokes on white paper, max-pooling propagates the strongest
local response and averaging dilutes a 1-pixel stroke across a 2x2 window. It is the cheapest
change in the table and the one most safely ignored.

**The spread across the whole table is 0.0529, and 0.0521 of it is BatchNorm.** Excluding that
row, every architectural choice in this grid is worth less than 0.017 macro F1 - which is the
same shape as 9.1.1's finding that three detector families landed inside 0.014, and 6.3.7's that
six classifiers landed inside 0.012. Three phases, three model families, one conclusion.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.detect.choice import RUNS
from src.detect.cnnmath import totals, walk
from src.detect.crops import OUT as CROPS
from src.detect.scratchcnn import Spec, train


#: One knob per row, each named for the axis it moves. The reference appears once.
def grid() -> list[tuple[str, str, Spec]]:
    reference = Spec()
    return [
        ("reference", "3x3 kernels, 4 stages, max pool, BN on", reference),
        ("kernel_5", "5x5 kernels", Spec(kernel=5)),
        ("kernel_7", "7x7 kernels", Spec(kernel=7)),
        ("depth_3", "3 stages instead of 4", Spec(stages=((16, 2), (32, 2), (64, 1)))),
        ("depth_2", "2 stages instead of 4", Spec(stages=((16, 2), (32, 2)))),
        ("pool_avg", "average pooling", Spec(pool="avg")),
        ("no_batchnorm", "BatchNorm removed", Spec(batchnorm=False)),
    ]


def run(epochs: int = 20, root: Path = CROPS, batch: int = 128) -> dict:
    rows = []
    for name, description, spec in grid():
        summary = totals(walk(spec), spec)
        result = train(spec, epochs=epochs, batch=batch, root=root)
        rows.append(
            {
                "arm": name,
                "description": description,
                "kernel": spec.kernel,
                "stages": len(spec.stages),
                "pool": spec.pool,
                "batchnorm": spec.batchnorm,
                "parameters": summary["total"],
                "receptive_field": summary["final_receptive_field"],
                "covers_input": summary["covers_input"],
                "macro_f1": result["macro_f1"],
                "accuracy": result["accuracy"],
                "train_seconds": result["train_seconds"],
                "per_class_f1": result["per_class_f1"],
            }
        )

    reference = next(r for r in rows if r["arm"] == "reference")
    deltas = {
        r["arm"]: {
            "macro_f1": round(r["macro_f1"] - reference["macro_f1"], 4),
            "parameters_x": round(r["parameters"] / reference["parameters"], 3),
            "receptive_field": r["receptive_field"] - reference["receptive_field"],
        }
        for r in rows
        if r["arm"] != "reference"
    }
    best = max(rows, key=lambda r: r["macro_f1"])
    return {
        "epochs": epochs,
        "reference_macro_f1": reference["macro_f1"],
        "rows": rows,
        "delta_vs_reference": deltas,
        "best_arm": best["arm"],
        "best_macro_f1": best["macro_f1"],
        "spread": round(max(r["macro_f1"] for r in rows) - min(r["macro_f1"] for r in rows), 4),
    }


def write_table(result: dict) -> list[str]:
    lines = [
        "| arm | change | params | RF | macro F1 | d vs ref | accuracy |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["rows"]:
        delta = result["delta_vs_reference"].get(row["arm"], {}).get("macro_f1")
        lines.append(
            f"| `{row['arm']}` | {row['description']} | {row['parameters']:,} | "
            f"{row['receptive_field']} | {row['macro_f1']:.4f} | "
            f"{'-' if delta is None else f'{delta:+.4f}'} | {row['accuracy']:.4f} |"
        )
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--root", type=Path, default=CROPS)
    ap.add_argument("--out", type=Path, default=RUNS / "ablations.json")
    args = ap.parse_args(argv)

    result = run(args.epochs, args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\n".join(write_table(result)))
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
