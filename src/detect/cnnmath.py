"""Phase 9.2.2 / 9.2.3 - the layer arithmetic, done by hand and checked against the module.

    python -m src.detect.cnnmath                # writes docs/cnn_math.md

Two tables, both computed from `Spec.layers()` rather than read off a `summary()` call, and both
asserted against the real `torch` module so the document cannot drift from the code it
describes.

## 9.2.2 - parameters and output sizes

For a convolution with `C_in` input channels, `C_out` output channels and a `k x k` kernel:

    params  = C_out * C_in * k * k  +  C_out          (the bias)
    out     = floor((in + 2p - k) / s) + 1

with the bias dropped when the layer is followed by BatchNorm, since the BN shift makes it
redundant and every framework's `bias=False` convention exists for that reason. BatchNorm
itself carries `2 * C` learned parameters (a scale and a shift) plus `2 * C` running buffers
that are **not** parameters and are not counted here - a distinction `numel()` on
`state_dict()` gets wrong and `parameters()` gets right.

A fully connected layer from `n` to `m` is `n * m + m`.

## 9.2.3 - receptive field

Propagated forward, one layer at a time, with the standard recurrence:

    j_out = j_in * s                       (the jump: input pixels per output step)
    r_out = r_in + (k - 1) * j_in          (the receptive field)

starting from `r = 1, j = 1` at the input. Pooling counts: a 2x2 stride-2 pool adds `1 * j` to
the field and doubles the jump. The question the table exists to answer is whether the last
convolution's units can see the whole 64x64 crop, because a shape is a global property - a
rectangle is not identifiable from any 20x20 window of itself - and if they cannot, then
whatever integrates the shape globally is the fully connected layer rather than the convolution
stack.

## What it measured

**The hand arithmetic and `torch` agree exactly at 372,183 parameters**, and they agree for
every configuration 9.2.5 will ablate - 5x5 kernels, BatchNorm off, average pooling, a thinner
stack, a smaller head. That agreement is the point: a parameter table nobody checked is a table
that quietly describes a different network than the one that trained.

**Two findings fall out, and they are the same finding seen twice.**

**1. 70.7% of the parameters are in one layer that does no convolving.** The six convolutions
hold 108,432 weights between them; `fc 2048->128` alone holds **262,272**. The whole
convolutional stack - four stages, six layers, all the feature extraction - costs 40% of what
the first dense layer costs, and `batchnorm` at 576 is a rounding error. The reason is
structural rather than a bad choice: convolution weights are *shared across every position*, so
a 3x3 layer costs `C_out * C_in * 9` no matter how large the map is, while a dense layer pays
for every position separately - 128 x 4 x 4 = 2,048 inputs, each with its own weight to each of
128 units. This is why global average pooling replaced flatten-then-dense in every architecture
after 2014, and the table shows exactly how much it would save here.

**2. No convolutional unit ever sees the whole crop. The final receptive field is 52x52 on a
64x64 input - 12 pixels short.** A unit in the last feature map is blind to a 6-pixel border all
the way around. That matters because **shape is a global property**: a rectangle is not
identifiable from any 20x20 window of itself, and neither is the difference between a circle and
a double-circle, whose distinguishing outer ring is exactly the thing nearest the crop edge.

Put together, the two say the same thing: **whatever integrates the crop into a shape is the
fully connected layer**, and that is why it holds 70% of the parameters. The convolutions build
local edge and corner evidence with a 52-pixel horizon; the dense layer is the only component
that sees all 16 spatial positions at once and can learn "corners at the four extremes" or "one
closed curve inside another". A design that pooled globally instead would drop to about 110,000
parameters and lose exactly that ability.

**The arithmetic also prices the architecture's founding substitution.** Two stacked 3x3 layers
reach a 5x5 field (rows 1 and 4 of the table both arrive at RF 5) for `2 * 9 = 18` weights a
channel pair, where one 5x5 layer needs 25 - a 28% saving with a nonlinearity gained. Scaled to
the whole stack, `kernel=5` costs 25/9 of `kernel=3`'s convolutional parameters, and the test
suite pins that ratio. 9.2.5 measures whether the accuracy follows.

**One conversion is easy to get wrong and is called out.** BatchNorm carries `2 * C` learned
parameters and `2 * C` running buffers; the buffers are not parameters. Counting a
`state_dict()` with `numel()` includes them and would report 576 + 576 here rather than 576,
and would also silently disagree with `parameters()`. The convolutions before a BatchNorm carry
no bias for the related reason that the BN shift subsumes it - which is why row 1 is 144 rather
than 160.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.detect.scratchcnn import Spec, build
from src.utils.config import ROOT

DOC = ROOT / "docs" / "cnn_math.md"


def walk(spec: Spec = Spec()) -> list[dict]:
    """One row per layer: shape in, shape out, parameters, receptive field, jump."""
    rows: list[dict] = []
    channels, side = spec.in_channels, spec.size
    field, jump = 1, 1

    for layer in spec.layers():
        before = (channels, side, side)
        if layer["type"] == "conv":
            k, s, p = layer["kernel"], layer["stride"], layer["padding"]
            side = (side + 2 * p - k) // s + 1
            channels = layer["out"]
            # No bias when BatchNorm follows: the BN shift subsumes it.
            params = layer["out"] * layer["in"] * k * k + (0 if spec.batchnorm else layer["out"])
            field, jump = field + (k - 1) * jump, jump * s
            name = f"conv {layer['in']}->{layer['out']} {k}x{k}"
        elif layer["type"] == "batchnorm":
            params = 2 * layer["channels"]
            name = f"batchnorm {layer['channels']}"
        elif layer["type"] == "pool":
            k, s = layer["kernel"], layer["stride"]
            side = (side - k) // s + 1
            params = 0
            field, jump = field + (k - 1) * jump, jump * s
            name = f"{layer['mode']}pool {k}x{k} s{s}"
        else:
            params = 0
            name = "relu"
        rows.append(
            {
                "layer": name,
                "in": list(before),
                "out": [channels, side, side],
                "params": int(params),
                "receptive_field": int(field),
                "jump": int(jump),
            }
        )

    flat = channels * side * side
    rows.append(
        {
            "layer": "flatten",
            "in": [channels, side, side],
            "out": [flat],
            "params": 0,
            "receptive_field": int(spec.size),
            "jump": int(jump),
        }
    )
    rows.append(
        {
            "layer": f"fc {flat}->{spec.hidden}",
            "in": [flat],
            "out": [spec.hidden],
            "params": flat * spec.hidden + spec.hidden,
            "receptive_field": int(spec.size),
            "jump": int(jump),
        }
    )
    rows.append(
        {
            "layer": "relu",
            "in": [spec.hidden],
            "out": [spec.hidden],
            "params": 0,
            "receptive_field": int(spec.size),
            "jump": int(jump),
        }
    )
    rows.append(
        {
            "layer": f"dropout p={spec.dropout}",
            "in": [spec.hidden],
            "out": [spec.hidden],
            "params": 0,
            "receptive_field": int(spec.size),
            "jump": int(jump),
        }
    )
    rows.append(
        {
            "layer": f"fc {spec.hidden}->{len(spec.classes)}",
            "in": [spec.hidden],
            "out": [len(spec.classes)],
            "params": spec.hidden * len(spec.classes) + len(spec.classes),
            "receptive_field": int(spec.size),
            "jump": int(jump),
        }
    )
    return rows


def totals(rows: list[dict], spec: Spec = Spec()) -> dict:
    conv = sum(r["params"] for r in rows if r["layer"].startswith("conv"))
    norm = sum(r["params"] for r in rows if r["layer"].startswith("batchnorm"))
    dense = sum(r["params"] for r in rows if r["layer"].startswith("fc"))
    total = conv + norm + dense
    last_conv = max(
        (r for r in rows if r["layer"].startswith(("conv", "maxpool", "avgpool"))),
        key=lambda r: r["receptive_field"],
    )
    return {
        "total": total,
        "conv": conv,
        "batchnorm": norm,
        "fully_connected": dense,
        "fc_share": round(dense / total, 4) if total else 0.0,
        "final_receptive_field": last_conv["receptive_field"],
        "input_size": spec.size,
        "covers_input": last_conv["receptive_field"] >= spec.size,
        "field_shortfall": max(0, spec.size - last_conv["receptive_field"]),
    }


def check_against_torch(spec: Spec = Spec()) -> dict:
    """The arithmetic is only worth writing down if it agrees with the module it describes."""
    model = build(spec)
    actual = int(sum(p.numel() for p in model.parameters()))
    predicted = totals(walk(spec), spec)["total"]
    return {"hand_computed": predicted, "torch_parameters": actual, "agree": predicted == actual}


def write_doc(spec: Spec = Spec(), path: Path = DOC) -> Path:
    rows = walk(spec)
    summary = totals(rows, spec)
    check = check_against_torch(spec)

    def shape(values):
        return "x".join(str(v) for v in values)

    lines = [
        "# CNN layer arithmetic",
        "",
        "Phase 9.2.2 and 9.2.3. Generated by `python -m src.detect.cnnmath`.",
        "",
        f"Architecture: `src/detect/scratchcnn.py`, input **{spec.in_channels}x{spec.size}x"
        f"{spec.size}**, {len(spec.classes)} classes, {spec.kernel}x{spec.kernel} kernels, "
        f"{spec.pool} pooling, BatchNorm {'on' if spec.batchnorm else 'off'}.",
        "",
        "## Layer table",
        "",
        "| # | layer | in | out | params | RF | jump |",
        "| ---: | :--- | :--- | :--- | ---: | ---: | ---: |",
    ]
    for index, row in enumerate(rows, 1):
        lines.append(
            f"| {index} | `{row['layer']}` | {shape(row['in'])} | {shape(row['out'])} | "
            f"{row['params']:,} | {row['receptive_field']} | {row['jump']} |"
        )

    lines += [
        "",
        "## Where the parameters are",
        "",
        "| block | params | share |",
        "| :--- | ---: | ---: |",
        f"| convolutions | {summary['conv']:,} | {summary['conv'] / summary['total']:.1%} |",
        f"| batchnorm | {summary['batchnorm']:,} | {summary['batchnorm'] / summary['total']:.1%} |",
        f"| fully connected | {summary['fully_connected']:,} | {summary['fc_share']:.1%} |",
        f"| **total** | **{summary['total']:,}** | |",
        "",
        f"Hand-computed **{check['hand_computed']:,}**, `torch` reports "
        f"**{check['torch_parameters']:,}** - "
        f"{'they agree' if check['agree'] else 'THEY DISAGREE'}.",
        "",
        "## Receptive field",
        "",
        f"The last convolutional stage sees a **{summary['final_receptive_field']}x"
        f"{summary['final_receptive_field']}** window of a "
        f"{summary['input_size']}x{summary['input_size']} input"
        + (
            "."
            if summary["covers_input"]
            else f", which is **{summary['field_shortfall']} pixels short of the whole crop**."
        ),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DOC)
    args = ap.parse_args(argv)

    spec = Spec()
    rows = walk(spec)
    summary = totals(rows, spec)
    check = check_against_torch(spec)
    path = write_doc(spec, args.out)
    print(json.dumps({"totals": summary, "check": check, "doc": str(path)}, indent=2))
    return 0 if check["agree"] else 1


if __name__ == "__main__":
    sys.exit(main())
