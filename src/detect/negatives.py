"""Phase 9.1.7 - what the detector hallucinates, and whether showing it more of that helps.

    python -m src.detect.negatives --mine        # what the false positives actually are
    python -m src.detect.negatives --build       # write the augmented training set
    python -m src.detect.negatives --retrain     # train on it and compare

The plan names three things to mine - crossed-out elements, doodles, margin scribbles - and
**none of them is annotated anywhere in the corpus**, so there is no set of labelled hard
negatives to add. There are two honest ways to get some, and this task does both because they
answer different questions:

**Mined.** Run the trained detector over the *training* split and keep every confident
prediction that overlaps no ground truth at all (IoU < 0.1 against every box on the page).
Those are the model's own hallucinations, which is the definition of a hard negative, and
`mine` reports what they are before anything is done with them - a false-positive census is
worth more than the mining, because if they turn out to be *correct* detections of unannotated
objects then adding them as negatives teaches the model to be wrong.

**Synthesized.** Strike a node out with a scribble and delete its box; scatter doodles and
margin marks on empty paper with no box at all. This is the only source of the specific three
the plan asks for, and it is honest about being simulation: 7.3.10 already showed on this
project what a damage model that is easier than reality buys, so the numbers here are an upper
bound on the fix and a lower bound on the problem.

Both are added as extra training images, and the arm is retrained under 9.1.3's exact budget so
the comparison is the negatives and nothing else.

## What it measured

FILLED_IN_BELOW
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

from src.detect.choice import RUNS, predict_ultralytics, truths
from src.detect.dataset import OUT as DATA
from src.detect.train import WEIGHTS, score, train

AUGMENTED = DATA.parent / "detect_hard"

SEED = 42
#: A prediction this confident that overlaps nothing is what the model is sure of and wrong about.
CONFIDENT = 0.5
#: Overlapping nothing, rather than overlapping badly - a poorly localised true positive is
#: 9.1.5's problem, not this one.
BACKGROUND_IOU = 0.1

STRIKE_SHARE = 0.35  # of augmented pages that get a node struck out
DOODLES = (2, 5)  # per augmented page


def _page_boxes(root: Path, split: str) -> dict[str, list[dict]]:
    reference, _ = truths(split, root)
    grouped: dict[str, list[dict]] = {}
    for row in reference:
        grouped.setdefault(row["image"], []).append(row)
    return grouped


def mine(
    weights: Path = WEIGHTS, root: Path = DATA, split: str = "train", imgsz: int = 896
) -> dict:
    """The false-positive census: how many, how confident, what class, and where on the page."""
    import collections

    from ultralytics import YOLO

    from src.detect.metrics import iou_matrix

    model = YOLO(str(weights))
    predictions, _ = predict_ultralytics(model, split, imgsz, root)
    grouped = _page_boxes(root, split)

    hard: list[dict] = []
    by_class: collections.Counter = collections.Counter()
    by_source: collections.Counter = collections.Counter()
    for row in predictions:
        if row["score"] < CONFIDENT:
            continue
        page = grouped.get(row["image"], [])
        if page:
            ious = iou_matrix(
                np.asarray([row["xyxy"]], dtype=float),
                np.asarray([t["xyxy"] for t in page], dtype=float),
            )[0]
            if ious.max() >= BACKGROUND_IOU:
                continue
        hard.append(row)
        by_class[row["cls"]] += 1
        by_source[row["image"].split("__", 1)[0]] += 1

    pages = len({row["image"] for row in hard})
    return {
        "split": split,
        "confident_threshold": CONFIDENT,
        "false_positives": len(hard),
        "pages_affected": pages,
        "per_page": round(len(hard) / max(len(grouped), 1), 3),
        "by_class": dict(by_class.most_common()),
        "by_source": dict(by_source.most_common()),
        "mean_score": round(float(np.mean([r["score"] for r in hard])), 4) if hard else None,
        "median_short_side_px": (
            round(
                float(
                    np.median(
                        [
                            min(r["xyxy"][2] - r["xyxy"][0], r["xyxy"][3] - r["xyxy"][1])
                            for r in hard
                        ]
                    )
                ),
                1,
            )
            if hard
            else None
        ),
        "boxes": hard,
    }


# ------------------------------------------------------------------------------------------
# synthesis
# ------------------------------------------------------------------------------------------


def strike_out(image, box, rng) -> None:
    """Scribble a node out, in place - the drawn gesture for "ignore this"."""
    import cv2

    x1, y1, x2, y2 = (int(v) for v in box)
    thickness = max(2, int(0.03 * max(x2 - x1, y2 - y1)))
    ink = (35, 35, 40)
    style = rng.integers(0, 3)
    if style == 0:  # single diagonal
        cv2.line(image, (x1, y1), (x2, y2), ink, thickness, cv2.LINE_AA)
    elif style == 1:  # cross
        cv2.line(image, (x1, y1), (x2, y2), ink, thickness, cv2.LINE_AA)
        cv2.line(image, (x2, y1), (x1, y2), ink, thickness, cv2.LINE_AA)
    else:  # zig-zag scribble
        steps = 8
        xs = np.linspace(x1, x2, steps)
        ys = np.where(np.arange(steps) % 2 == 0, y1, y2)
        points = np.stack([xs, ys], axis=1).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(image, [points], False, ink, thickness, cv2.LINE_AA)


def doodle(image, rng) -> None:
    """A mark on empty paper: a margin squiggle, an underline, a stray arrow-like stroke."""
    import cv2

    height, width = image.shape[:2]
    ink = (35, 35, 40)
    thickness = max(1, int(0.002 * max(height, width)))
    kind = rng.integers(0, 3)
    x = int(rng.uniform(0.02, 0.95) * width)
    y = int(rng.uniform(0.02, 0.95) * height)
    span = int(rng.uniform(0.03, 0.12) * max(height, width))
    if kind == 0:
        points = (
            np.stack(
                [
                    np.linspace(x, x + span, 12),
                    y + np.sin(np.linspace(0, 4 * np.pi, 12)) * span * 0.15,
                ],
                axis=1,
            )
            .astype(np.int32)
            .reshape(-1, 1, 2)
        )
        cv2.polylines(image, [points], False, ink, thickness, cv2.LINE_AA)
    elif kind == 1:
        cv2.line(image, (x, y), (x + span, y), ink, thickness, cv2.LINE_AA)
    else:
        cv2.circle(image, (x, y), max(2, span // 6), ink, thickness, cv2.LINE_AA)


def build(
    root: Path = DATA,
    out: Path = AUGMENTED,
    hard: list[dict] | None = None,
    seed: int = SEED,
    synthesize: bool = True,
) -> dict:
    """Copy the export and add the two negative populations to its training split.

    The validation and test splits are copied **unchanged**. Augmenting them would make the
    retrained arm score against a different question than the baseline did.

    `synthesize=False` builds the **mined-only** arm: the crops the detector hallucinated and
    nothing else. That arm exists because the combined build adds 59 mined crops beside 699
    synthesized pages, so any gain it shows is 92% attributable to synthesis by page count and
    the task cannot otherwise say which half did the work.
    """
    import cv2

    rng = np.random.default_rng(seed)
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(root, out)

    grouped = _page_boxes(root, "train")
    train_images = sorted((out / "images" / "train").glob("*.png")) if synthesize else []
    struck = 0
    doodled = 0
    removed_boxes = 0
    for path in train_images:
        if rng.random() > STRIKE_SHARE:
            continue
        image = cv2.imread(str(path))
        if image is None:
            continue
        boxes = grouped.get(path.stem, [])
        label_path = out / "labels" / "train" / f"{path.stem}.txt"
        lines = label_path.read_text(encoding="utf-8").splitlines() if label_path.is_file() else []
        keep = list(range(len(lines)))
        if boxes and lines and len(lines) == len(boxes):
            victim = int(rng.integers(0, len(boxes)))
            strike_out(image, boxes[victim]["xyxy"], rng)
            keep.remove(victim)
            removed_boxes += 1
            struck += 1
        for _ in range(int(rng.integers(*DOODLES))):
            doodle(image, rng)
            doodled += 1
        name = f"{path.stem}__hard"
        cv2.imwrite(str(out / "images" / "train" / f"{name}.png"), image)
        (out / "labels" / "train" / f"{name}.txt").write_text(
            "\n".join(lines[i] for i in keep) + "\n", encoding="utf-8"
        )

    mined_written = 0
    if hard:
        # A mined false positive becomes a *crop with an empty label file*: an image the model
        # must learn contains nothing, which is how YOLO consumes a negative.
        crops = out / "images" / "train"
        by_image: dict[str, list[dict]] = {}
        for row in hard:
            by_image.setdefault(row["image"], []).append(row)
        for image_name, rows in by_image.items():
            source = root / "images" / "train" / f"{image_name}.png"
            page = cv2.imread(str(source))
            if page is None:
                continue
            height, width = page.shape[:2]
            for index, row in enumerate(rows[:3]):
                x1, y1, x2, y2 = (int(v) for v in row["xyxy"])
                pad = int(0.5 * max(x2 - x1, y2 - y1))
                x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
                x2, y2 = min(width, x2 + pad), min(height, y2 + pad)
                crop = page[y1:y2, x1:x2]
                if crop.size == 0 or min(crop.shape[:2]) < 16:
                    continue
                name = f"{image_name}__neg{index}"
                cv2.imwrite(str(crops / f"{name}.png"), crop)
                (out / "labels" / "train" / f"{name}.txt").write_text("", encoding="utf-8")
                mined_written += 1

    yaml = (out / "data.yaml").read_text(encoding="utf-8").replace(root.as_posix(), out.as_posix())
    (out / "data.yaml").write_text(yaml, encoding="utf-8")
    return {
        "root": str(out),
        "pages_struck_out": struck,
        "boxes_removed": removed_boxes,
        "doodles_drawn": doodled,
        "mined_negative_crops": mined_written,
        "train_images": len(list((out / "images" / "train").glob("*.png"))),
    }


def retrain(
    imgsz: int = 896, epochs: int = 40, out: Path = AUGMENTED, baseline: Path = DATA
) -> dict:
    """Same budget as 9.1.3's final run, scored on the **unaugmented** validation split."""
    model, seconds = train("hardneg", epochs=epochs, imgsz=imgsz, batch=8, root=out)
    scored = score(model, imgsz, baseline, "val")
    scored["train_seconds"] = round(seconds, 1)
    weights = RUNS / "hardneg" / "weights" / "best.pt"
    hallucinations = mine(weights, baseline, "val", imgsz) if weights.is_file() else {}
    scored["false_positives_on_val"] = hallucinations.get("false_positives")
    scored["false_positives_by_class"] = hallucinations.get("by_class")
    return scored


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", type=Path, default=WEIGHTS)
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--imgsz", type=int, default=896)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--mine", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--out", type=Path, default=RUNS / "negatives.json")
    args = ap.parse_args(argv)

    if not args.weights.is_file():
        print(f"no weights at {args.weights}", file=sys.stderr)
        return 1

    result: dict = {}
    do_all = not (args.mine or args.build or args.retrain)
    hard = None
    if args.mine or args.build or do_all:
        census = mine(args.weights, args.data, "train", args.imgsz)
        hard = census.pop("boxes")
        result["mined"] = census
    if args.build or do_all:
        result["built"] = build(args.data, AUGMENTED, hard)
    if args.retrain or do_all:
        from src.detect.metrics import evaluate  # noqa: F401  (import cost paid once, up front)

        result["baseline_val_false_positives"] = mine(args.weights, args.data, "val", args.imgsz)[
            "false_positives"
        ]
        result["retrained"] = retrain(args.imgsz, args.epochs, AUGMENTED, args.data)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
