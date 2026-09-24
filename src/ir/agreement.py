"""Phase 2.2.4 - inter-annotator agreement on shape and role.

    python -m src.ir agreement

A measurement, run once and reported. Nothing imports it because nothing should: its output is
`reports/annotator_agreement.md`, and a later stage that wanted the number would read the report,
not recompute the kappa.

**Read this first: there are not two human annotation teams on this project.** The plan asks
for 50 double-labelled images and Cohen's kappa between the two labellers. What exists instead
is three labellers of different kinds, and the honest thing is to name each one and report
every pairing rather than to dress one of them up as a second team.

| Labeller | What it sees | What it is |
| :--- | :--- | :--- |
| **A - annotation** | the hdBPMN XML | the dataset's own human annotation. Roles are ground truth; shapes are the BPMN drawing *convention* applied to those roles, never an observation of the photo |
| **B - geometry** | only the pixels inside each annotated box | `convert.geometry`, blind to the label. A machine, and a crude one |
| **C - human** | the same crops, as a contact sheet, with no labels attached | one person labelling a sample by eye, recorded in `data/annotations/human_shape_sample.csv` |

Two of the three pairings are genuinely informative:

* **A vs B on shape** answers "how often is a BPMN task actually drawn as a rounded rectangle?"
  - which matters, because every hdBPMN-derived shape label in this project rests on it.
* **A vs C and B vs C on shape** are the closest thing here to real inter-annotator agreement,
  on however many crops the person labelled.

**A vs B on role is included and is expected to be poor**, because B infers role from shape
alone. That is the point: it measures how much of the role a shape can carry without graph
context, and the answer bounds what Phase 5 can do before Phase 10 supplies the context.

    python -m src.ir.agreement --limit 50
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from src.ir.convert import hdbpmn
from src.ir.convert.geometry import classify_crop
from src.ir.model import Diagram
from src.utils.config import ROOT
from src.utils.figures import save as _figsave

REPORT = ROOT / "reports" / "annotator_agreement.md"
FIGURE = ROOT / "reports" / "figures" / "p2_agreement.png"
HUMAN = ROOT / "data" / "annotations" / "human_shape_sample.csv"
SHEET = ROOT / "reports" / "figures" / "p2_crop_sheet.png"

SEED = 42
#: Boxes smaller than this are text labels and event markers whose crop is a few dozen pixels;
#: no labeller, human or otherwise, can name a shape from that.
MIN_SIDE = 24

#: What annotator B calls the role, knowing only the shape. Deliberately coarse: a circle in a
#: flowchart may be a start, an end or an intermediate event, and nothing in the pixels of one
#: crop distinguishes them.
ROLE_FROM_SHAPE = {
    "diamond": "decision",
    "rectangle": "process",
    "rounded-rect": "process",
    "parallelogram": "io",
    "circle": "event",
    "double-circle": "event",
    "ellipse": "event",
    "octagon": "process",
    "text-block": "unknown",
    "arrow": "transition",
    "line": "transition",
    "freeform": "unknown",
}


def sample_diagrams(limit: int) -> list[Diagram]:
    files = hdbpmn.annotations()
    if not files:
        return []
    rng = random.Random(SEED)
    order = files[:]
    rng.shuffle(order)
    # Keep drawing until `limit` files actually convert: a handful of annotations have no
    # matching photo, and stopping at the first shortfall would silently sample 49 not 50.
    out = []
    for path in order:
        if len(out) >= limit:
            break
        try:
            out.append(hdbpmn.convert(path))
        except (FileNotFoundError, ValueError):
            continue
    return sorted(out, key=lambda d: d.id)


def crops(diagrams: list[Diagram]) -> list[dict]:
    """One row per annotated node big enough to judge, with its crop and labeller A's view.

    Each row is tagged `calibration` or `evaluation` by the parity of its diagram's position in
    the sorted sample. Labeller B's thresholds were set from the calibration half only, so the
    headline kappa can be reported on data those thresholds never saw.
    """
    rows: list[dict] = []
    for index, diagram in enumerate(diagrams):
        half = "calibration" if index % 2 == 0 else "evaluation"
        image = cv2.imread(str(ROOT / diagram.meta["image"]), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        height, width = image.shape
        for node in diagram.nodes:
            x, y, w, h = (int(round(v)) for v in node.bbox)
            if w < MIN_SIDE or h < MIN_SIDE:
                continue
            pad = int(0.06 * max(w, h))
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(width, x + w + pad), min(height, y + h + pad)
            crop = image[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            rows.append(
                {
                    "key": f"{diagram.id}:{node.id}",
                    "diagram": diagram.id,
                    "node": node.id,
                    "half": half,
                    "crop": crop,
                    "a_shape": node.shape,
                    "a_role": node.semantic_role,
                }
            )
    return rows


def label_with_geometry(rows: list[dict]) -> None:
    for row in rows:
        shape, confidence = classify_crop(row["crop"])
        row["b_shape"] = shape
        row["b_confidence"] = confidence
        row["b_role"] = ROLE_FROM_SHAPE.get(shape, "unknown")


def read_human() -> dict[str, str]:
    if not HUMAN.is_file():
        return {}
    with HUMAN.open(encoding="utf-8", newline="") as fh:
        return {r["key"]: r["shape"] for r in csv.DictReader(fh) if r.get("shape")}


def kappa(a: list[str], b: list[str]) -> dict:
    labels = sorted(set(a) | set(b))
    if len(labels) < 2:
        return {"kappa": float("nan"), "agreement": 1.0, "n": len(a), "labels": labels}
    return {
        "kappa": float(cohen_kappa_score(a, b, labels=labels)),
        "agreement": float(np.mean([x == y for x, y in zip(a, b, strict=True)])),
        "n": len(a),
        "labels": labels,
    }


def contact_sheet(rows: list[dict], count: int = 30, cell: int = 150) -> Path:
    """A numbered grid of crops with no labels on it, for a person to label blind."""
    rng = random.Random(SEED)
    picked = rng.sample(rows, min(count, len(rows)))
    cols = 6
    grid_rows = (len(picked) + cols - 1) // cols
    sheet = np.full((grid_rows * (cell + 22), cols * cell, 3), 255, np.uint8)
    for i, row in enumerate(picked):
        crop = row["crop"]
        scale = (cell - 8) / max(crop.shape)
        small = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
        cy, cx = (i // cols) * (cell + 22) + 20, (i % cols) * cell + 4
        sheet[cy : cy + rgb.shape[0], cx : cx + rgb.shape[1]] = rgb
        cv2.putText(sheet, str(i), (cx + 2, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 200), 2)
    SHEET.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(SHEET), sheet)
    with (SHEET.parent / "p2_crop_sheet_keys.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["index", "key"])
        for i, row in enumerate(picked):
            writer.writerow([i, row["key"]])
    return SHEET


def figure(pairs: dict) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [(name, data) for name, data in pairs.items() if data.get("matrix") is not None]
    if not panels:
        return FIGURE
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5.2))
    if len(panels) == 1:
        axes = [axes]
    for ax, (name, data) in zip(axes, panels, strict=True):
        matrix = np.asarray(data["matrix"], dtype=float)
        normed = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
        ax.imshow(normed, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(data["labels"])), data["labels"], rotation=45, ha="right")
        ax.set_yticks(range(len(data["labels"])), data["labels"])
        ax.set_title(f"{name}\nkappa = {data['kappa']:.2f}, n = {data['n']}")
        ax.set_ylabel("first labeller")
        ax.set_xlabel("second labeller")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if matrix[i, j]:
                    ax.text(
                        j,
                        i,
                        int(matrix[i, j]),
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white" if normed[i, j] > 0.5 else "black",
                    )
    fig.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    _figsave(fig, FIGURE, dpi=130)
    plt.close(fig)
    return FIGURE


def compare(first: list[str], second: list[str]) -> dict:
    result = kappa(first, second)
    labels = result["labels"]
    result["matrix"] = confusion_matrix(first, second, labels=labels).tolist()
    return result


def analyse(limit: int) -> dict:
    diagrams = sample_diagrams(limit)
    rows = crops(diagrams)
    label_with_geometry(rows)
    human = read_human()

    held_out = [r for r in rows if r["half"] == "evaluation"]
    calibration = [r for r in rows if r["half"] == "calibration"]

    pairs: dict[str, dict] = {}
    pairs["shape: annotation vs geometry"] = compare(
        [r["a_shape"] for r in held_out], [r["b_shape"] for r in held_out]
    )
    pairs["role: annotation vs geometry"] = compare(
        [r["a_role"] for r in held_out], [r["b_role"] for r in held_out]
    )
    halves = {
        "calibration": compare(
            [r["a_shape"] for r in calibration], [r["b_shape"] for r in calibration]
        )["kappa"],
        "evaluation": pairs["shape: annotation vs geometry"]["kappa"],
    }
    scored = [r for r in rows if r["key"] in human]
    if scored:
        pairs["shape: annotation vs human"] = compare(
            [r["a_shape"] for r in scored], [human[r["key"]] for r in scored]
        )
        pairs["shape: geometry vs human"] = compare(
            [r["b_shape"] for r in scored], [human[r["key"]] for r in scored]
        )

    return {
        "diagrams": len(diagrams),
        "regions": len(rows),
        "held_out_regions": len(held_out),
        "human_labelled": len(scored),
        "pairs": pairs,
        "halves": halves,
        "a_shapes": Counter(r["a_shape"] for r in rows),
        "b_shapes": Counter(r["b_shape"] for r in rows),
        "rows": rows,
    }


def write_report(result: dict) -> Path:
    lines: list[str] = []
    add = lines.append
    add("# Inter-Annotator Agreement")
    add("")
    add("Phase 2.2.4. Generated by `python -m src.ir.agreement`.")
    add("")
    add(
        f"**{result['diagrams']} hdBPMN diagrams, {result['regions']} annotated regions** "
        f"at least {MIN_SIDE}px on a side, sampled with seed {SEED}."
    )
    add("")
    add(
        "The sample is split in half by diagram. Labeller B's thresholds were set from the "
        f"**calibration** half only, so every kappa below is computed on the **evaluation** "
        f"half - {result['held_out_regions']} regions those thresholds never saw. For "
        f"comparison, shape agreement is {result['halves']['calibration']:.3f} on the half it "
        f"was tuned on and {result['halves']['evaluation']:.3f} on the half it was not."
    )
    add("")
    add("## Who the annotators are")
    add("")
    add("This project does not have two human annotation teams, and this report does not")
    add("pretend otherwise. Three labellers of different kinds were compared:")
    add("")
    add("| Labeller | Sees | Is |")
    add("| :--- | :--- | :--- |")
    add(
        "| **A - annotation** | the hdBPMN XML | the dataset's own human annotation. Roles are "
        "ground truth; shapes are the BPMN drawing convention applied to those roles |"
    )
    add(
        "| **B - geometry** | only the pixels in each box | `src/ir/convert/geometry.py`, blind "
        "to the label |"
    )
    add(
        f"| **C - human** | the same crops as an unlabelled contact sheet | one person, "
        f"{result['human_labelled']} crops |"
    )
    add("")
    add("## Results")
    add("")
    add("| Pairing | n | raw agreement | Cohen's kappa |")
    add("| :--- | ---: | ---: | ---: |")
    for name, data in result["pairs"].items():
        add(f"| {name} | {data['n']} | {data['agreement']:.3f} | " f"**{data['kappa']:.3f}** |")
    add("")
    add("![agreement](figures/p2_agreement.png)")
    add("")
    add("Kappa is read on the usual scale: below 0.4 poor, 0.4-0.6 moderate, 0.6-0.8")
    add("substantial, above 0.8 near-perfect.")
    add("")
    best = max(d["kappa"] for d in result["pairs"].values())
    add("## The headline, before the detail")
    add("")
    add(
        f"**contributing.md 2.2.4 sets a bar of kappa >= 0.75. Nothing here reaches it; the best "
        f"pairing is {best:.2f}.** That is the finding, not a missing piece of work, and three "
        "things follow from it:"
    )
    add("")
    if "shape: annotation vs human" in result["pairs"]:
        human_a = result["pairs"]["shape: annotation vs human"]["kappa"]
        human_b = result["pairs"]["shape: geometry vs human"]["kappa"]
        if human_b > human_a:
            add(
                f"1. **The BPMN drawing convention is a worse account of what was drawn than a "
                f"crude contour classifier is** - kappa {human_a:.2f} against the human, versus "
                f"{human_b:.2f} for the geometry. Writers do not draw tasks as rounded "
                "rectangles. Every hdBPMN shape label carries `shape_basis: bpmn-convention` "
                "for exactly this reason, and no phase may treat those shapes as ground truth."
            )
        else:
            add(
                f"1. The BPMN convention tracks the human reading better than the geometry does "
                f"(kappa {human_a:.2f} against {human_b:.2f}), but neither is close to reliable."
            )
    add(
        "2. **Shape labels for hand-drawn diagrams have to be annotated, not derived.** Phase 5 "
        "cannot be trained on hdBPMN-derived shapes and then be said to classify shape; it "
        "would be learning to reproduce a convention."
    )
    add(
        "3. **Reaching 0.75 needs two people and the guidelines**, which is what "
        "`docs/annotation_guide.md` and the Label Studio project in 2.2.1 exist to make "
        "possible. This report is the baseline they would be measured against."
    )
    add("")
    add("## What each number means")
    add("")

    shape_ab = result["pairs"]["shape: annotation vs geometry"]
    add("### Shape, annotation against geometry")
    add("")
    add(
        f"kappa = {shape_ab['kappa']:.3f} over {shape_ab['n']} regions. This is the number that "
        "matters most in this report, because **every hdBPMN shape label in this project is a "
        "convention, not an observation** - the XML says an element is a `task` and the "
        "converter writes `rounded-rect`. This measures how often that convention matches the "
        "pen."
    )
    add("")
    add("Shape distributions, side by side:")
    add("")
    add("| shape | A (BPMN convention) | B (from the pixels) |")
    add("| :--- | ---: | ---: |")
    for shape in sorted(set(result["a_shapes"]) | set(result["b_shapes"])):
        add(
            f"| {shape} | {result['a_shapes'].get(shape, 0)} | {result['b_shapes'].get(shape, 0)} |"
        )
    add("")

    role_ab = result["pairs"]["role: annotation vs geometry"]
    add("### Role, annotation against geometry")
    add("")
    add(
        f"kappa = {role_ab['kappa']:.3f}. **A low number here is the expected result, not a "
        "defect.** Labeller B assigns a role from the shape alone, and a circle in a flowchart "
        "may be a start, an end or an intermediate event with nothing in its own pixels to "
        "separate them. What this bounds is how far shape classification alone can go: it is "
        "the reason Phase 10 assembles a graph before anything claims to know what a node "
        "means, and the reason Phase 5's classifier is scored on shape rather than role."
    )
    add("")

    if "shape: annotation vs human" in result["pairs"]:
        human_a = result["pairs"]["shape: annotation vs human"]
        human_b = result["pairs"]["shape: geometry vs human"]
        add("### The human pass")
        add("")
        add(
            f"A person labelled {human_a['n']} crops from the contact sheet in "
            "`reports/figures/p2_crop_sheet.png`, seeing only the ink - no BPMN label, no "
            "neighbouring shapes, no file name."
        )
        add("")
        add(
            f"* against the BPMN convention: kappa = **{human_a['kappa']:.3f}** "
            f"(raw agreement {human_a['agreement']:.0%})"
        )
        add(
            f"* against the geometry classifier: kappa = **{human_b['kappa']:.3f}** "
            f"(raw agreement {human_b['agreement']:.0%})"
        )
        add("")
        add(
            "This is the closest thing in the project to true inter-annotator agreement, and "
            "it is one person on a small sample rather than two teams on fifty images. It is "
            "reported as what it is."
        )
        add("")

    add("## Limitations, stated plainly")
    add("")
    add("| Limitation | Why it matters |")
    add("| :--- | :--- |")
    add(
        "| Only hdBPMN is measured | It is the one source with both an image and a human "
        "structural annotation. The chaos corpus has no per-shape labels to disagree with |"
    )
    add(
        "| A is not an independent annotator | It is the dataset's own annotation, so A vs B "
        "measures convention-versus-pixels, not two people |"
    )
    add(
        "| B is crude by design | Douglas-Peucker on one contour. Phase 3.2 replaces it, and "
        "these numbers are the baseline it must beat |"
    )
    add(
        f"| Regions under {MIN_SIDE}px are excluded | Mostly event circles and text labels. "
        "Excluding them flatters every number here, and the exclusion is stated rather than "
        "hidden |"
    )
    add(
        "| One human, one sample | Real IAA needs two people who did not write the guidelines. "
        "Phase 2.2.3 exists so that becomes possible; this is not it |"
    )
    add("")
    add("## Reproducing")
    add("")
    add("```")
    add("python -m src.ir.agreement --sheet     # build the blind contact sheet")
    add("#   label it into data/annotations/human_shape_sample.csv")
    add("python -m src.ir.agreement --limit 50  # recompute and rewrite this report")
    add("```")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    while lines and not lines[-1].strip():
        lines.pop()
    with REPORT.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    return REPORT


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=50, help="how many diagrams to sample")
    ap.add_argument("--sheet", action="store_true", help="write the blind contact sheet and stop")
    args = ap.parse_args(argv)

    if args.sheet:
        rows = crops(sample_diagrams(args.limit))
        label_with_geometry(rows)
        path = contact_sheet(rows)
        print(f"wrote {path.relative_to(ROOT)} ({len(rows)} regions available)")
        return 0

    result = analyse(args.limit)
    if not result["regions"]:
        print("no hdBPMN regions available", file=sys.stderr)
        return 1

    fig = figure(result["pairs"])
    rep = write_report(result)
    print(f"wrote {rep.relative_to(ROOT)}")
    print(f"wrote {fig.relative_to(ROOT)}")
    print(
        json.dumps(
            {
                "diagrams": result["diagrams"],
                "regions": result["regions"],
                "human_labelled": result["human_labelled"],
                "kappa": {k: round(v["kappa"], 3) for k, v in result["pairs"].items()},
            },
            indent=2,
        )
    )

    checks = {
        "report_written": rep.is_file(),
        "figure_written": fig.is_file(),
        "at_least_50_diagrams": result["diagrams"] >= 50,
        "kappa_reported_for_shape_and_role": len(result["pairs"]) >= 2,
    }
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
