"""Phase 7.4.5 - giving the discovered components names, and admitting what they are named after.

    python -m src.parse.shapenames      # writes reports/figures/p7_gmm_montage.png
                                        #    and  reports/shape_vocabulary.md

7.4.2 fitted the mixture and 7.4.3 chose K. This task asks what the components *are*, which the
plan phrases as mapping them onto `rectangle / diamond / oval / arrow / blob`.

## The naming is a measurement, not a decision

Two names are produced for every component and they are allowed to differ:

    label_name    the hdbpmn class that most of the component's nodes carry, with the share.
                  This is what the component *is*, against the only ground truth available.
    shape_name    the plan's vocabulary, assigned from the component's mean descriptor by the
                  rules in `NAME_RULES` below - roundness, elongation, dentedness.

When they agree, the component has found a shape a person would name. When they disagree, the
component has found something real that the label set has no word for, and 7.4.1 already predicted
where that happens: rectangles average `rect_aspect` 3.26 against ~1.2 elsewhere, so a component of
wide BPMN tasks and a component of square ones are both `rectangle` by label and are genuinely
different things.

Two of the plan's five names cannot be earned on this corpus and that is stated rather than
worked around. **`oval` has no rows** - 7.4.1 measured zero `ellipse` labels in hdbpmn - and
**`arrow` is not a node**: connectors are edges in the IR and never get a bounding box, so no
descriptor row exists for one. The vocabulary that can actually be learned here is
`rectangle / diamond / round / blob`, and `name_from` returns exactly those.

A third name is nominally reachable and in practice is not. **No threshold on these descriptors
names `freeform`**, because 7.4.1's measured means put it *between* the other classes on every
column that separates anything: circularity 0.435 against circle's 0.520 and diamond's 0.310,
solidity 0.724 against circle's 0.830 and rectangle's 0.574, extent 0.609 against circle's 0.637
and rectangle's 0.490. It is not a shape with a signature; it is the residue left when the other
three are taken out. The rules below therefore do not try to name it, and `label_name` is what
reports when a component is mostly freeform - which is the case the two names exist to expose.

## Why purity is reported per component rather than pooled

A pooled purity hides the interesting case. A vocabulary where three components are 90% pure and
two are 40% pure is a much better result than one where all five are 66%, because the first has
found three real shapes and the second has found nothing - and both average the same.

## The montage

Nine real crops per component, drawn from the pages themselves, ordered by responsibility so the
most typical members come first. It is the only artefact in 7.4 that can be checked by eye, and it
is the reason the naming rules are applied to component *means* rather than tuned until the
numbers look good: a rule that calls something a diamond can be disagreed with by looking.

## What it measured

K = 6 (7.4.3's elbow), `full` covariance, 12,400 shapes:

     c   size   share  shape name  label name  purity   circ  extent  solid  aspect
     0    796   0.064  rectangle   diamond      0.445  0.277   0.512  0.657   1.18
     1  3,855   0.311  rectangle   circle       0.467  0.456   0.591  0.797   1.15
     2  1,236   0.100  blob        rectangle    0.649  0.056   0.235  0.299   1.77
     3  1,225   0.099  blob        rectangle    0.622  0.060   0.238  0.304   1.82
     4  3,702   0.298  round       rectangle    0.661  0.502   0.734  0.876   1.71
     5  1,586   0.128  blob        rectangle    0.950  0.072   0.335  0.410   6.97

**Not one component's two names agree.** Four of the six are majority-`rectangle`, only three of
the four labels ever appear as a dominant name, and `freeform` is dominant nowhere.

## The mixture split the corpus by outline quality, not by shape

That is the finding, and it is visible in one column. Components 2, 3 and 5 have a mean
circularity of **0.064**; components 0, 1 and 4 have **0.459**. That is a **7x gap**, against the
2.4x that separates 7.4.1's roundest class from its least round (circle 0.520, rectangle 0.215).
Their solidity tells the same story - 0.30 against a lowest *class* mean of 0.574, far outside the
range any shape occupies.

A circularity of 0.06 does not describe a shape at all. It means the perimeter is enormous
relative to the area: a ragged, broken, or fragmented outline. **32.6% of the corpus - 4,047
nodes - lives in that condition**, and its label mix is 3,071 rectangles, 365 diamonds, 314
circles and 297 freeforms, which is roughly the corpus distribution. The three components are not
three shapes; they are one damage mode, cut three ways.

So the mixture's dominant axis of variation is **how well 3.2 extracted the outline**, not what
was drawn inside it. 7.4.2's adjusted Rand of 0.1499 and 7.4.3's monotonically falling purity are
both this fact measured indirectly - the mixture is describing the corpus faithfully and the
strongest signal in the corpus is not shape.

## The naming rules are not what failed

It would be easy to read `names_agree = 0` as broken thresholds. It is not. Components 2, 3 and 5
*are* blobs by any geometric reading - solidity 0.30, circularity 0.06 - and the rules call them
blobs correctly. What disagrees is the label, because **a fragmented rectangle is still annotated
`rectangle`**: the human labelled the intent, and the descriptor measured the pixels that survived
binarization. Component 5 is the sharpest case - 95% rectangle, the purest component in the table,
with a mean aspect of 6.97 and a solidity of 0.410. Those are long thin BPMN pools whose outlines
broke up, and they are simultaneously the most label-pure component and the least shape-like one.

## What that means for 7.4.6 and 7.4.8

The responsibilities 7.4.6 is about to feed into the HMM carry **real information that is mostly
not shape identity**. That is not necessarily useless - a node whose outline fragmented is a node
the rest of the pipeline should perhaps trust less - but it is not the "soft shape evidence" the
plan describes, and 7.4.6 has to be read as measuring what the responsibilities actually contain.

For 7.4.8 the implication is sharper. A template matcher is asked to compete against a vocabulary
that has largely learned image quality; if it loses, it loses to something other than a better
account of shape.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from src.features.descriptors import NAMES
from src.parse.vocab import DEFAULT_K, fit, matrix
from src.utils.figures import save as _figsave

ROOT = Path(__file__).resolve().parents[2]
FIGURE = ROOT / "reports" / "figures" / "p7_gmm_montage.png"
REPORT = ROOT / "reports" / "shape_vocabulary.md"

#: Crops shown per component in the montage.
PER_COMPONENT = 9

#: The names reachable on this corpus. `oval` and `arrow` from the plan's list are unreachable;
#: see the module docstring.
VOCABULARY = ("rectangle", "diamond", "round", "blob")


def name_from(mean: dict) -> str:
    """The plan's vocabulary, from a component's mean descriptor.

    Thresholds are read off 7.4.1's measured class means rather than from geometry textbooks,
    because 7.4.1 established that drawn shapes sit far from their ideal values - a rule keyed on
    circularity 1.0 for a circle would name nothing.
    """
    circularity = mean["circularity"]
    extent = mean["extent"]
    solidity = mean["solidity"]
    aspect = mean["rect_aspect"]

    # Below every class mean in 7.4.1's table, so only an unusually dented component reaches it.
    # The obvious threshold - "less solid than a shape should be" - cannot be used: 7.4.1
    # measured `rectangle` at 0.574 solidity, the *lowest* of the four classes, because a drawn
    # box has gaps at its corners, while `freeform` measured 0.724. Solidity does not detect
    # blobs on this corpus; it detects rectangles.
    if solidity < 0.50:
        return "blob"
    # Roundest of the four (circle 0.520 against freeform 0.435 and rectangle 0.215).
    if circularity >= 0.47:
        return "round"
    # A diamond fills its axis-aligned box about half (0.414) where a rectangle fills 0.490 and
    # a circle 0.637. The aspect guard keeps a long thin box from being read as a diamond.
    if extent < 0.46 and aspect < 2.0:
        return "diamond"
    return "rectangle"


def components(k: int = DEFAULT_K, covariance: str = "full"):
    """The fitted mixture, its hard assignment, and the responsibility of each node."""
    X, labels, keys, _ = matrix()
    model = fit(X, k, covariance)
    scaled = model.named_steps["scale"].transform(X)
    gmm = model.named_steps["gmm"]
    return X, labels, keys, gmm.predict(scaled), gmm.predict_proba(scaled)


def describe_components(X, labels, assignment, responsibility, k: int) -> list[dict]:
    """One row per component: size, what it is, what it looks like, and how pure it is."""
    rows = []
    for component in range(k):
        members = assignment == component
        size = int(members.sum())
        if not size:
            rows.append({"component": component, "size": 0, "label_name": None, "shape_name": None})
            continue

        mean = dict(zip(NAMES, X[members].mean(axis=0), strict=True))
        counts: dict[str, int] = {}
        for label in labels[members]:
            counts[label] = counts.get(label, 0) + 1
        dominant = max(counts, key=counts.get)

        rows.append(
            {
                "component": component,
                "size": size,
                "share_of_corpus": round(size / len(labels), 4),
                "label_name": dominant,
                "label_purity": round(counts[dominant] / size, 4),
                "label_mix": {k2: round(v / size, 3) for k2, v in sorted(counts.items())},
                "shape_name": name_from(mean),
                "mean_confidence": round(float(responsibility[members].max(axis=1).mean()), 4),
                "mean": {name: round(float(mean[name]), 4) for name in NAMES},
            }
        )
    return rows


def montage(keys, assignment, responsibility, rows: list[dict], path: Path = FIGURE) -> Path:
    """Real crops of the most typical members of each component."""
    import matplotlib

    matplotlib.use("Agg")
    import cv2
    import matplotlib.pyplot as plt

    from src.features.descriptors import CROP_PAD
    from src.preprocess import layers as ly
    from src.preprocess.exif import load

    from src.ir.model import Diagram  # isort: skip

    live = [row for row in rows if row["size"]]
    fig, axes = plt.subplots(
        len(live), PER_COMPONENT, figsize=(PER_COMPONENT * 1.1, len(live) * 1.25)
    )
    axes = np.atleast_2d(axes)

    for axis_row, row in zip(axes, live, strict=True):
        component = row["component"]
        members = np.flatnonzero(assignment == component)
        # Most typical first: the members this component is surest of.
        members = members[np.argsort(-responsibility[members, component])][:PER_COMPONENT]

        # One decode per page rather than per crop.
        wanted: dict[str, list[str]] = {}
        for index in members:
            page, element = keys[index].split(":", 1)
            wanted.setdefault(page, []).append(element)

        patches = []
        for page, elements in wanted.items():
            path_ir = ROOT / "data" / "processed" / "ir" / "hdbpmn" / f"{page}.ir.json"
            if not path_ir.is_file():
                continue
            diagram = Diagram.load(path_ir)
            image_path = ROOT / diagram.meta["image"]
            if not image_path.is_file():
                continue
            original = load(image_path, grayscale=True)
            gray, mask = ly.prepare(image_path)
            scale = gray.shape[1] / original.shape[1]
            boxes = {n.id: n.bbox for n in diagram.nodes if n.bbox}
            for element in elements:
                if element not in boxes:
                    continue
                x, y, w, h = (v * scale for v in boxes[element])
                px, py = CROP_PAD * w, CROP_PAD * h
                x0, y0 = max(0, int(x - px)), max(0, int(y - py))
                x1 = min(mask.shape[1], int(x + w + px))
                y1 = min(mask.shape[0], int(y + h + py))
                if x1 - x0 > 4 and y1 - y0 > 4:
                    patches.append(cv2.resize(mask[y0:y1, x0:x1].astype(np.uint8) * 255, (64, 64)))

        for axis, patch in zip(axis_row, patches + [None] * PER_COMPONENT, strict=False):
            axis.axis("off")
            if patch is not None:
                axis.imshow(255 - patch, cmap="gray", vmin=0, vmax=255)
        axis_row[0].set_ylabel(f"{component}", rotation=0, labelpad=14)
        axis_row[0].axis("on")
        axis_row[0].set_xticks([])
        axis_row[0].set_yticks([])
        axis_row[0].set_title(
            f"c{component}: {row['shape_name']} / {row['label_name']} {row['label_purity']:.2f}",
            fontsize=7,
            loc="left",
        )

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    _figsave(fig, path, dpi=150)
    plt.close(fig)
    return path


def markdown(result: dict) -> str:
    lines = [
        "# Learned shape vocabulary (Phase 7.4.5)",
        "",
        f"K = {result['k']}, {result['rows']} shapes, `{result['covariance']}` covariance.",
        "",
        "| component | size | share | shape name | label name | purity | mix |",
        "| ---: | ---: | ---: | :--- | :--- | ---: | :--- |",
    ]
    for row in result["components"]:
        if not row["size"]:
            lines.append(f"| {row['component']} | 0 | - | - | - | - | - |")
            continue
        mix = ", ".join(f"{k} {v:.2f}" for k, v in row["label_mix"].items())
        lines.append(
            f"| {row['component']} | {row['size']} | {row['share_of_corpus']:.3f} | "
            f"{row['shape_name']} | {row['label_name']} | {row['label_purity']:.3f} | {mix} |"
        )
    lines += [
        "",
        f"- names the vocabulary can reach: {', '.join(VOCABULARY)}",
        f"- distinct label names claimed: {result['distinct_label_names']}",
        f"- distinct shape names used: {result['distinct_shape_names']}",
        f"- components whose two names agree: {result['names_agree']} of {result['k']}",
    ]
    return "\n".join(lines) + "\n"


def run(k: int = DEFAULT_K, covariance: str = "full", write: bool = True) -> dict:
    X, labels, keys, assignment, responsibility = components(k, covariance)
    rows = describe_components(X, labels, assignment, responsibility, k)
    live = [row for row in rows if row["size"]]

    agree = sum(
        1
        for row in live
        if row["shape_name"] == ("round" if row["label_name"] == "circle" else row["label_name"])
    )
    result = {
        "k": k,
        "covariance": covariance,
        "rows": int(len(X)),
        "vocabulary": list(VOCABULARY),
        "distinct_label_names": len({row["label_name"] for row in live}),
        "distinct_shape_names": len({row["shape_name"] for row in live}),
        "names_agree": agree,
        "purest": max(live, key=lambda r: r["label_purity"])["component"],
        "least_pure": min(live, key=lambda r: r["label_purity"])["component"],
        "components": rows,
    }
    if write:
        result["figure"] = str(montage(keys, assignment, responsibility, rows).relative_to(ROOT))
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(markdown(result), encoding="utf-8")
        result["report"] = str(REPORT.relative_to(ROOT))
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--covariance", default="full")
    args = ap.parse_args(argv)
    try:
        result = run(args.k, args.covariance)
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 1
    for row in result["components"]:
        row.pop("mean", None)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
