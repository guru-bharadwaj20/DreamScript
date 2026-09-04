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

FILLED_IN_BELOW
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
