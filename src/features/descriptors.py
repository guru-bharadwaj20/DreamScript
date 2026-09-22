"""Phase 7.4.1 - a shape descriptor vector, and the invariances it is and is not allowed.

    python -m src.features.descriptors --build   # writes data/features/descriptors.parquet
    python -m src.features.descriptors           # the summary, from the built table

7.4 asks whether a shape vocabulary can be *learned* rather than declared. 4.1.3's `classify()`
is the declared version: a decision tree of hand-set thresholds on extent, corner count and
aspect that returns one of five names. This task builds the table a GMM is fitted to in 7.4.2,
and the design question is entirely about **which invariances the descriptor should have**.

## The vector

    hu_0 .. hu_6         Hu moments, log-scaled and signed. Invariant to translation, scale and
                         rotation; the classical shape signature.
    vertices             Douglas-Peucker vertex count at 2% of the perimeter - a rectangle is 4,
                         a circle is many.
    rect_aspect          long side / short side of the *minimum-area* rectangle, not the
                         axis-aligned one. 4.1.3 learned this the hard way: on a photograph the
                         page is not square to the camera, so an axis-aligned box overstates a
                         tilted rectangle's aspect badly.
    solidity             area / convex hull area. A blob with a dent is less solid than a box.
    circularity          4 pi A / P^2 - exactly 1 for a circle and lower for anything with
                         corners. The most direct round/cornered discriminator here.
    rect_fill            area / minimum-area rectangle area. ~1.0 for any convex quadrilateral
                         and ~0.79 for a circle, so this measures how *polygonal* an outline is,
                         not which polygon it is.
    extent               area / axis-aligned bounding box area. The one deliberately
                         rotation-sensitive column; see below.
    defect_count         convexity defects deeper than 2% of the hull perimeter.
    defect_max           the deepest such defect, as a share of the hull perimeter. An arrow
                         has one deep notch; a rectangle has none.
    fourier_1 .. _7      normalised Fourier magnitudes of the contour, harmonics 2-8.

22 columns. Everything is a ratio or a count, so nothing carries the size of the drawing.

## Why the Fourier descriptors are normalised the way they are

The contour is resampled to 64 equally spaced points and read as a complex signal, `x + iy`.
The DFT of that signal has a standard set of invariances that have to be *taken*, not assumed:

    translation   drop coefficient 0, which is the centroid.
    scale         divide by the magnitude of coefficient 1, the fundamental.
    rotation      take magnitudes and discard phase.
    start point   also phase, so the same discard covers it.

The last one is the one that matters in practice. `findContours` starts wherever it starts, and
two identical rectangles traced from different corners have completely different phase spectra
and identical magnitude spectra. Keeping phase would make the descriptor depend on an
implementation detail of OpenCV.

## The invariance that is deliberately NOT taken

**Rotation is discarded, and that is a real loss on this corpus.** A diamond is a square rotated
45 degrees, so every rotation-invariant column sees the two as one shape - the Hu moments and the
Fourier magnitudes by construction, and `rect_fill`, `solidity` and `circularity` too, because a
minimum-area rectangle and a convex hull both turn with the shape. On the synthetic pair a square
and a diamond agree on all of them to three decimals.

`extent` is in the vector for exactly that reason. Measured against the *axis-aligned* box a
square reads 1.00 and a diamond 0.50, and it is the only column in the table that separates them.
4.1.3 reached the same conclusion from the other direction - its `DIAMOND_EXTENT` threshold is a
band on this quantity - so the learned vocabulary and the declared one at least agree about which
measurement carries the distinction.

Keeping absolute orientation instead would be worse. These are photographs of paper, and the page
tilt is a property of the camera rather than of the drawing; 3.1.3 already declined to deskew for
that reason. So the descriptor is rotation-invariant with one deliberate rotation-*sensitive*
column added back, chosen because it answers the question rotation invariance destroys.

## Where the rows come from

hdbpmn, the one corpus whose IR carries a `shape` label on every node. Each node's bounding box is
cropped out of its own page's shape mask and the largest closed outline inside it is the shape.
The page is decoded once and every node on it cropped from that single decode - 4.1.3's
`_crop_regions` reloads and re-binarizes the page per node, which is fine for the 30 hand-labelled
crops it was written for and hopeless for twelve thousand.

The label is `shape` mapped through 4.1.3's `LABEL_MAP`, so `rounded-rect` is a rectangle: no
geometric rule recovers a corner radius from a drawn box, and asking a GMM to find one would be
asking it to model the annotator rather than the page.

## What it measured

**12,400 shapes over 693 hdbpmn pages, 22 columns, no missing values and no constant columns.**
Six of the 12,406 labelled nodes produced no closed outline inside their own box and were dropped.
The label distribution is what hdbpmn draws, not a balanced set:

    rectangle   5,947     (4,181 of them `rounded-rect`)
    circle      3,231
    diamond     2,141
    freeform    1,081

`ellipse` is in 4.1.3's vocabulary and has zero rows here, so the learned vocabulary of 7.4.5 has
four names to find and not five.

## Hand-drawn shapes are nowhere near their textbook values

    class        circularity   solidity   extent   vertices   rect_aspect
    circle          0.520        0.830     0.637      9.7         1.16
    freeform        0.435        0.724     0.609      8.0         1.36
    diamond         0.310        0.631     0.414      9.9         1.20
    rectangle       0.215        0.574     0.490      8.0         3.26

An ideal circle scores 1.0 on circularity and an ideal rectangle 0.785. **The drawn circles here
average 0.520 and the drawn rectangles 0.215** - every class lands far below its ideal and in the
same direction, because a photographed pen outline is rough, and perimeter appears squared in the
denominator so roughness is punished twice. Solidity tells the same story: a rectangle is convex
by definition and these average 0.574.

That is the single most important fact for 7.4.8. **A template matcher keyed on ideal values would
classify almost nothing correctly**, and the reason a learned vocabulary is worth fitting is not
that it is more expressive but that it is calibrated to what people actually draw. The *ordering*
survives - circles really are the roundest and rectangles the least round - so the information is
there; it is the absolute scale that the textbook gets wrong.

`vertices` is the clearest casualty: a rectangle should be 4 and averages **8.0**, because
Douglas-Peucker at 2% of a wobbly perimeter keeps the wobble. It still separates (F = 475.7),
just not by meaning what its name suggests.

## What separates the classes

ANOVA F against the four labels, over all 22 columns:

    hu_1          2693.9        rect_fill      489.5
    circularity   1411.6        vertices       475.7
    hu_0           884.4        hu_2           408.4
    hu_3           723.3        defect_count   384.3
    solidity       708.9        ...
    extent         530.5        rect_aspect     84.8
                                hu_6             1.1

**`hu_1` is the strongest column in the table by a factor of two.** The second Hu moment is
essentially the elongation of the shape, and it beats `rect_aspect` - the column built to measure
exactly that - by 32x, because `rect_aspect` is computed on a minimum-area rectangle that a rough
outline makes unstable while the moment integrates over the whole region.

**`hu_6` is inert at F = 1.1** and is the one column here worth naming as dead weight. It is the
skew-invariant moment, which changes sign under reflection and is otherwise near zero for every
roughly-symmetric outline; nothing in this corpus is distinguished by its handedness. It is kept
so the Hu block stays the standard seven rather than a subset chosen on this corpus's labels, and
7.4.2's covariance comparison is where a column that carries nothing gets priced.

`rect_aspect` at 84.8 is the other surprise, and it is a corpus fact: BPMN tasks are wide rounded
rectangles averaging an aspect of 3.26 while every other class sits near 1.2, so the column is
strong at picking rectangles out and useless among the rest.

## Two bugs the build caught, recorded because the table was wrong until they were fixed

**The convexity-defect columns were dead.** `defect_count` and `defect_max` read zero for all
12,400 rows in the first build, and `summary()` reported them under `constant_columns`, which is
what surfaced it. Two causes: the hull indices were re-sorted before being handed to
`convexityDefects`, which makes OpenCV raise rather than answer, and the guard required a hull of
more than three points, which skips exactly the triangular hulls an arrowhead has. Fixed, they
rank 10th and 18th of 22 - not decisive, but `diamond` averages 4.04 defects against `freeform`'s
2.20, which is real signal that was being discarded.

**The Hu log transform amplified sign noise into the largest distance in the table.** The textbook
`sign(h) * log10(|h|)` on a numerically-zero moment turns the sign of floating-point dust into a
swing of tens: measured at **60.0 between a square and the same square rotated 45 degrees, against
22.5 between a square and a circle**. A rotation was three times more different than a change of
shape. The symmetric log now used maps a zero moment to zero from either side, and the same pair
of synthetic shapes agrees to 0.0 while square-versus-circle still reads 4.47.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TABLE = ROOT / "data" / "features" / "descriptors.parquet"

#: Harmonics kept from the contour DFT, after the fundamental is spent on scale normalisation.
HARMONICS = 7

#: Points the contour is resampled to before the DFT. Comfortably above twice the top harmonic.
RESAMPLE = 64

#: Douglas-Peucker tolerance, as a share of the contour perimeter.
EPSILON = 0.02

#: A convexity defect shallower than this share of the hull perimeter is pen noise.
DEFECT_MIN = 0.02

#: Scale of the symmetric log applied to the Hu moments. A moment smaller than this is treated
#: as zero rather than as a very large negative logarithm; see `describe`.
HU_FLOOR = 1e-12

#: Padding around a cropped node, as a share of its own size, so the outline is not clipped.
CROP_PAD = 0.12

HU = tuple(f"hu_{i}" for i in range(7))
FOURIER = tuple(f"fourier_{i}" for i in range(1, HARMONICS + 1))
NAMES: tuple[str, ...] = (
    *HU,
    "vertices",
    "rect_aspect",
    "solidity",
    "circularity",
    "rect_fill",
    "extent",
    "defect_count",
    "defect_max",
    *FOURIER,
)


def fourier(points: np.ndarray, harmonics: int = HARMONICS) -> np.ndarray:
    """Normalised Fourier magnitudes of a closed contour.

    Invariant to translation, scale, rotation and - the one that bites in practice - the point
    `findContours` happened to start at. See the module docstring.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(pts) < 4:
        return np.zeros(harmonics)

    # Resample by arc length, so a shape stored as 4 points and the same shape stored as 400
    # give the same spectrum rather than differing by the encoding.
    closed = np.vstack([pts, pts[:1]])
    steps = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    distance = np.concatenate([[0.0], np.cumsum(steps)])
    if distance[-1] <= 0:
        return np.zeros(harmonics)
    wanted = np.linspace(0, distance[-1], RESAMPLE, endpoint=False)
    x = np.interp(wanted, distance, closed[:, 0])
    y = np.interp(wanted, distance, closed[:, 1])

    spectrum = np.fft.fft(x + 1j * y)
    fundamental = abs(spectrum[1])
    if fundamental <= 0:
        return np.zeros(harmonics)
    # Coefficient 0 is the centroid and 1 is the scale; magnitudes discard rotation and phase.
    return np.abs(spectrum[2 : 2 + harmonics]) / fundamental


def describe(contour: np.ndarray) -> np.ndarray | None:
    """The 24-column descriptor for one closed contour, or None if it is too small to describe."""
    import cv2

    points = np.asarray(contour, dtype=np.int32).reshape(-1, 1, 2)
    if len(points) < 4:
        return None
    area = float(cv2.contourArea(points))
    perimeter = float(cv2.arcLength(points, True))
    if area <= 0 or perimeter <= 0:
        return None

    hull = cv2.convexHull(points)
    hull_area = float(cv2.contourArea(hull)) or area
    hull_perimeter = float(cv2.arcLength(hull, True)) or perimeter

    # Hu moments span many orders of magnitude and need a log to be usable as GMM features.
    # The textbook `sign(h) * log10(|h|)` is unusable here: for a symmetric shape the higher
    # moments are numerically zero, their sign is noise, and the transform turns that noise into
    # a swing of tens of units - measured at 60.0 between a square and the same square rotated
    # 45 degrees, which is three times the distance between a square and a circle. A symmetric
    # log is continuous through zero, so a moment that is zero maps to zero whichever side it
    # approaches from, and the ordering of the genuinely non-zero moments is unchanged.
    raw = cv2.HuMoments(cv2.moments(points)).ravel()
    hu = np.sign(raw) * np.log10(1.0 + np.abs(raw) / HU_FLOOR)

    approx = cv2.approxPolyDP(points, EPSILON * perimeter, True)
    _, _, box_w, box_h = cv2.boundingRect(points)
    box_area = float(box_w * box_h) or area
    (_, (rect_w, rect_h), _) = cv2.minAreaRect(points.astype(np.float32))
    long_side = max(rect_w, rect_h)
    short_side = max(min(rect_w, rect_h), 1e-6)
    rect_area = float(rect_w * rect_h) or area

    defect_count, defect_max = 0, 0.0
    # `convexityDefects` wants the hull as indices, in the order `convexHull` returns them.
    # Re-sorting them - a common suggestion - makes OpenCV raise, and a hull of exactly three
    # points is valid: an arrowhead's hull is a triangle, and it is the shape whose defect
    # matters most here. Both mistakes were made in the first draft, and between them they left
    # these two columns reading zero for every shape in the corpus.
    hull_indices = cv2.convexHull(points, returnPoints=False)
    if hull_indices is not None and len(hull_indices) >= 3:
        try:
            defects = cv2.convexityDefects(points, hull_indices)
        except cv2.error:
            defects = None
        if defects is not None:
            # Column 3 is the depth, in fixed-point units of 1/256 of a pixel. The array comes
            # back `(n, 1, 4)` on some OpenCV builds and `(n, 4)` on others - this one returns
            # the flat form, which indexed as `[:, 0, 3]` raises rather than reading a wrong
            # column, so the shape is normalised instead of assumed.
            depths = np.asarray(defects).reshape(-1, 4)[:, 3] / 256.0 / hull_perimeter
            deep = depths[depths > DEFECT_MIN]
            defect_count = int(len(deep))
            defect_max = float(deep.max()) if len(deep) else 0.0

    return np.array(
        [
            *hu,
            float(len(approx)),
            long_side / short_side,
            min(area / hull_area, 1.0),
            4 * np.pi * area / (perimeter**2),
            min(area / rect_area, 1.0),
            min(area / box_area, 1.0),
            float(defect_count),
            defect_max,
            *fourier(points.reshape(-1, 2)),
        ],
        dtype=float,
    )


def page_contours(page_id: str, boxes: dict[str, list[float]]) -> dict:
    """The largest closed outline inside each node's box, from a single decode of the page."""
    import cv2

    from src.preprocess import layers as ly
    from src.preprocess.exif import load

    from src.ir.model import Diagram  # isort: skip

    path = ROOT / "data" / "processed" / "ir" / "hdbpmn" / f"{page_id}.ir.json"
    if not path.is_file():
        return {}
    diagram = Diagram.load(path)
    image_path = ROOT / diagram.meta["image"]
    if not image_path.is_file():
        return {}

    original = load(image_path, grayscale=True)
    gray, mask = ly.prepare(image_path)
    # The IR is in original pixels and the mask is at the working width: move the box, not the
    # pixels. 3.1.3 and 4.1.3 both made this choice for the same reason.
    scale = gray.shape[1] / original.shape[1]

    out = {}
    for element_id, bbox in boxes.items():
        x, y, w, h = (value * scale for value in bbox)
        pad_x, pad_y = CROP_PAD * w, CROP_PAD * h
        x0, y0 = max(0, int(x - pad_x)), max(0, int(y - pad_y))
        x1 = min(mask.shape[1], int(x + w + pad_x))
        y1 = min(mask.shape[0], int(y + h + pad_y))
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue
        patch = mask[y0:y1, x0:x1].astype(np.uint8)
        # Cropping can cut a stroke at the border, leaving the outline open; a border of
        # background closes nothing but keeps `findContours` off the image edge.
        patch = cv2.copyMakeBorder(patch, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
        found, _ = cv2.findContours(patch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not found:
            continue
        out[element_id] = max(found, key=cv2.contourArea)
    return out


def _one_page(page_id: str) -> list[dict]:
    from src.features.shapes import LABEL_MAP

    from src.ir.model import Diagram  # isort: skip

    path = ROOT / "data" / "processed" / "ir" / "hdbpmn" / f"{page_id}.ir.json"
    diagram = Diagram.load(path)
    wanted = {
        node.id: (node.bbox, LABEL_MAP[node.shape])
        for node in diagram.nodes
        if node.bbox and node.shape in LABEL_MAP
    }
    if not wanted:
        return []
    contours = page_contours(page_id, {i: b for i, (b, _) in wanted.items()})

    rows = []
    for element_id, (_, label) in wanted.items():
        contour = contours.get(element_id)
        if contour is None:
            continue
        vector = describe(contour)
        if vector is None:
            continue
        rows.append(
            {
                "key": f"{page_id}:{element_id}",
                "page": page_id,
                "label": label,
                **dict(zip(NAMES, vector.tolist(), strict=True)),
            }
        )
    return rows


def build(limit: int | None = None, n_jobs: int | None = None):
    """The descriptor table over every labelled hdbpmn node."""
    import pandas as pd

    from src.utils.parallel import pmap

    page_ids = sorted(
        p.name[: -len(".ir.json")]
        for p in (ROOT / "data" / "processed" / "ir" / "hdbpmn").glob("*.ir.json")
    )[:limit]
    pages = pmap(_one_page, page_ids, n_jobs=n_jobs)
    table = pd.DataFrame([row for page in pages for row in page])
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(TABLE, index=False)
    return table


def load_table():
    import pandas as pd

    if not TABLE.is_file():
        raise FileNotFoundError(f"{TABLE} not built; run --build")
    return pd.read_parquet(TABLE)


def summary(table=None) -> dict:
    """Row counts, per-class support, and which columns separate the declared labels."""
    from scipy.stats import f_oneway

    table = load_table() if table is None else table
    finite = table[list(NAMES)].replace([np.inf, -np.inf], np.nan)

    separation = {}
    for name in NAMES:
        groups = [
            group[name].replace([np.inf, -np.inf], np.nan).dropna()
            for _, group in table.groupby("label")
        ]
        groups = [g for g in groups if len(g) > 2 and g.std() > 0]
        if len(groups) > 1:
            statistic = f_oneway(*groups).statistic
            if np.isfinite(statistic):
                separation[name] = round(float(statistic), 1)

    return {
        "rows": int(len(table)),
        "pages": int(table["page"].nunique()),
        "columns": len(NAMES),
        "per_label": {k: int(v) for k, v in table["label"].value_counts().items()},
        "missing_by_column": {
            name: int(finite[name].isna().sum()) for name in NAMES if int(finite[name].isna().sum())
        },
        "constant_columns": [name for name in NAMES if finite[name].nunique(dropna=True) <= 1],
        "anova_f": dict(sorted(separation.items(), key=lambda kv: -kv[1])),
        "class_means": {
            label: {name: round(float(group[name].mean()), 4) for name in NAMES}
            for label, group in table.groupby("label")
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=None)
    args = ap.parse_args(argv)

    table = build(args.limit, args.jobs) if args.build else None
    try:
        print(json.dumps(summary(table), indent=2))
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
