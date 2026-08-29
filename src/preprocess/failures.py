"""Phase 3.3.3 - the twenty worst pages, and what went wrong on each.

3.3.1 and 3.3.2 measure the pipeline on rendered strokes, where the ground truth is exact and
the material is clean. Both say so about themselves. This module does the opposite: it runs the
pipeline over real photographs, ranks them by how badly it did, and writes the twenty worst out
as a gallery with a named diagnosis each - `reports/preproc_failures.md`.

## Scoring a page with no ground truth

Nobody has traced these photographs, so "how badly did it do" has to be assembled from things
the annotation *does* know, plus a few properties any sane binarization of a page of ink has.
Six signals, each mapping to a failure a person would recognise on sight:

    empty_boxes        annotated node boxes with almost no ink inside them   strokes lost
    ink_share          share of the page that came out as ink                paper survived,
                                                                             or the ink did not
    fragmentation      components per thousand ink pixels                    strokes broken up
    shadow_left        quadrant brightness spread after 3.1.4                lighting unfixed
    off_page_ink       ink outside the page quadrilateral 3.1.2 finds        the desk was in shot
    crop_loss          annotated content outside that same quadrilateral     bad crop

`empty_boxes` is the one that would carry the ranking if it ever fired, and it is worth saying
why it is fair: every one of those boxes is a shape a person drew and an annotator saw. If there
is no ink where the annotation says a shape is, the pipeline lost it - there is no interpretation
in which that is correct. **It never fires. Across 120 photographs, not one annotated shape came
back empty**, which is the strongest single statement this phase can make about its own
binarization and is worth more than the gallery below it.

The score is the maximum of the normalised signals rather than their sum. A page that fails one
way badly is a more useful thing to look at than a page that is mildly odd in six ways, and
summing would rank it lower.

## What the gallery found

Of 120 pages, 93 fire no signal at all. The 27 that do divide as:

    background_survived  9    shadow_left  8    off_page_background  6    fragmented  4

The two worst diagnoses are both about **what surrounds the drawing, not the drawing itself**,
and neither was anticipated when Phase 3 was planned:

* **`off_page_background`** - the photograph caught the desk. On the worst page, 84% of the
  "ink" is the wood grain of a table above and below the sheet. Phase 3.1.2 detects the page
  correctly on these images; nothing crops to it, because the 3.2 pipeline deliberately runs on
  the whole frame. That is a wiring decision this gallery argues against, and it is a one-line
  fix rather than a research problem.
* **`background_survived`** - squared notebook paper. The printed grid survives, at up to 33% of
  all ink lying in straight runs longer than a tenth of the page. 3.1.8 removes a grid from a
  *flat* page - it measured 99.98% removal on exactly that - but a photographed notebook is
  curved, its printed lines bow, and directional morphology with a straight structuring element
  does not see a curved rule. The stage is not broken; its assumption is.

`shadow_left` and `fragmented` are the expected ones and are the smaller half.

The pages are named so a later phase can go and look at the same ones, and the two headline
failures both belong to capture and wiring rather than to thresholding - which is consistent
with `empty_boxes` never firing.

    python -m src.preprocess.failures --limit 120
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from src.utils.config import ROOT

REPORT = ROOT / "reports" / "preproc_failures.md"
FIGURES = ROOT / "reports" / "figures" / "preproc_failures"

#: A node box with less than this share of ink inside it has lost its shape.
EMPTY_BOX_INK = 0.004

#: Ink share of the page outside this band is a threshold that went wrong in one direction or
#: the other. Measured over 60 hdBPMN pages rather than guessed: the distribution runs
#: p05 1.8%, median 6.7%, p90 16.9%, so the floor is set just under p05 and the ceiling at p90.
#: The first draft of this module used 0.5%-12% from intuition and called half the corpus a
#: failure.
MIN_INK_SHARE = 0.01
MAX_INK_SHARE = 0.17

#: Connected components per thousand ink pixels. Same 60 pages: median 2.2, p98 5.2.
FRAGMENTATION_LIMIT = 5.0

#: Quadrant brightness spread, *after* 3.1.4 has had its go. Corpus p90 is 17.8.
SHADOW_SPREAD = 20.0

#: Share of the ink lying outside the detected page. Some is normal - the quadrilateral is
#: approximate and a stroke can run to the paper's edge - but a photograph that caught the desk
#: puts most of its "ink" there.
OFF_PAGE_INK = 0.15

GALLERY_SIZE = 20


def rule_residue(mask: np.ndarray, span_frac: float = 0.10) -> float:
    """Share of ink lying in straight runs longer than `span_frac` of the page's long side.

    A morphological opening with a long horizontal or vertical structuring element keeps only
    ink that continues in a straight line for that distance. Handwriting does not; a drawn box
    edge occasionally does, which is why the threshold is calibrated on the corpus rather than
    set to zero. Printed rules and squared-paper grids do it everywhere, and that is what this
    measures - what 3.1.8 left behind.
    """
    binary = mask.astype(np.uint8)
    total = int(binary.sum())
    if not total:
        return 0.0
    span = max(15, int(span_frac * max(mask.shape)))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((1, span), np.uint8))
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((span, 1), np.uint8))
    return float(np.count_nonzero(horizontal | vertical) / total)


@dataclass
class Case:
    id: str
    image: str
    empty_boxes: float
    ink_share: float
    fragmentation: float
    shadow_left: float
    crop_loss: float
    boxes: int
    rule_residue: float = 0.0
    off_page_ink: float = 0.0
    diagnosis: str = ""
    score: float = 0.0
    thumbnail: str = ""
    signals: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "score": round(self.score, 4),
            "diagnosis": self.diagnosis,
            "empty_boxes": round(self.empty_boxes, 4),
            "ink_share": round(self.ink_share, 4),
            "fragmentation": round(self.fragmentation, 2),
            "shadow_left": round(self.shadow_left, 1),
            "crop_loss": round(self.crop_loss, 4),
            "rule_residue": round(self.rule_residue, 3),
            "off_page_ink": round(self.off_page_ink, 3),
        }


def signals(case: Case) -> dict[str, float]:
    """Each signal normalised so that 1.0 is "as bad as this signal gets before it saturates"."""
    return {
        "ink_lost": case.empty_boxes,
        "threshold_too_high": max(0.0, (MIN_INK_SHARE - case.ink_share) / MIN_INK_SHARE),
        "background_survived": max(0.0, (case.ink_share - MAX_INK_SHARE) / MAX_INK_SHARE),
        "fragmented": max(0.0, (case.fragmentation - FRAGMENTATION_LIMIT) / FRAGMENTATION_LIMIT),
        "shadow_left": max(0.0, (case.shadow_left - SHADOW_SPREAD) / SHADOW_SPREAD),
        "crop_lost_content": case.crop_loss,
        "off_page_background": max(0.0, (case.off_page_ink - OFF_PAGE_INK) / OFF_PAGE_INK),
    }


def diagnose(case: Case) -> tuple[str, float, dict]:
    """The worst single signal names the failure. See the docstring for why not the sum."""
    scored = signals(case)
    name = max(scored, key=lambda key: scored[key])
    return (name if scored[name] > 0 else "clean"), scored[name], scored


def examine(page_id: str, image_path: Path, boxes: list[tuple[int, int, int, int]]) -> Case | None:
    """Run the pipeline on one photograph and collect the signals."""
    from src.preprocess import layers as ly
    from src.preprocess.denoise import count_components
    from src.preprocess.illumination import correct, quadrant_spread
    from src.preprocess.page import detect

    gray, mask = ly.prepare(image_path)
    scale = ly.WORKING_WIDTH / max(cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE).shape)
    scaled = [tuple(int(v * min(scale, 1.0)) for v in box) for box in boxes]

    empty = 0
    for x, y, w, h in scaled:
        if w < 10 or h < 10:
            continue
        patch = mask[max(0, y) : y + h, max(0, x) : x + w]
        if patch.size and patch.mean() < EMPTY_BOX_INK:
            empty += 1
    counted = max(1, sum(1 for _, _, w, h in scaled if w >= 10 and h >= 10))

    ink = float(mask.mean())
    fragmentation = 1000.0 * count_components(mask) / max(1, int(mask.sum()))

    quad = detect(gray)
    off_page = 0.0
    if quad is not None:
        page_mask = np.zeros(mask.shape, np.uint8)
        cv2.fillPoly(page_mask, [np.asarray(quad, np.int32).reshape(-1, 2)], 1)
        outside = int(np.count_nonzero(mask & (page_mask == 0)))
        off_page = outside / max(1, int(mask.sum()))

    crop_loss = 0.0
    if quad is not None and scaled:
        polygon = np.asarray(quad, np.float32).reshape(-1, 2)
        outside = sum(
            1
            for x, y, w, h in scaled
            if cv2.pointPolygonTest(polygon, (float(x + w / 2), float(y + h / 2)), False) < 0
        )
        crop_loss = outside / len(scaled)

    return Case(
        id=page_id,
        image=(
            str(image_path.relative_to(ROOT))
            if image_path.is_relative_to(ROOT)
            else str(image_path)
        ),
        empty_boxes=empty / counted,
        ink_share=ink,
        fragmentation=fragmentation,
        shadow_left=quadrant_spread(correct(gray)),
        crop_loss=crop_loss,
        boxes=counted,
        rule_residue=rule_residue(mask),
        off_page_ink=off_page,
    )


def collect(limit: int = 120) -> list[Case]:
    from src.ir.model import SUFFIX, Diagram
    from src.utils.parallel import pmap

    paths = sorted((ROOT / "data" / "processed" / "ir" / "hdbpmn").glob(f"*{SUFFIX}"))[:limit]

    def one(path):
        diagram = Diagram.load(path)
        image_path = ROOT / diagram.meta["image"]
        if not image_path.is_file():
            return None
        boxes = [tuple(int(v) for v in n.bbox) for n in diagram.nodes if n.bbox]
        if not boxes:
            return None
        case = examine(path.stem, image_path, boxes)
        if case is None:
            return None
        case.diagnosis, case.score, case.signals = diagnose(case)
        return case

    return [c for c in pmap(one, paths, prefer="threads") if c]


def thumbnail(case: Case, width: int = 520) -> Path:
    """Photograph beside the mask the pipeline produced, so the failure is visible."""
    from src.preprocess import layers as ly

    gray, mask = ly.prepare(ROOT / case.image)
    scale = width / gray.shape[1]
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    binary = cv2.resize(
        np.where(mask, 0, 255).astype(np.uint8),
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA,
    )
    gap = np.full((small.shape[0], 8), 128, np.uint8)
    FIGURES.mkdir(parents=True, exist_ok=True)
    target = FIGURES / f"{case.id}.png"
    cv2.imwrite(str(target), np.hstack([small, gap, binary]))
    return target


def report(cases: list[Case], gallery: list[Case]) -> Path:
    from collections import Counter

    counts = Counter(c.diagnosis for c in cases)
    lines = [
        "# Phase 3.3.3 — preprocessing failure gallery",
        "",
        f"{len(cases)} hdBPMN photographs through the full Phase 3 pipeline, ranked by the ",
        "worst of five signals. There is no pixel ground truth for these images — the ranking ",
        "is built from the annotation (a node box with no ink inside it is a lost shape) and ",
        "from properties any sane binarization of a page of ink has. `src/preprocess/failures.py` ",
        "explains each signal and why the score is a maximum rather than a sum.",
        "",
        "Each thumbnail is the photograph beside the mask the pipeline produced.",
        "",
        "## How the corpus divides",
        "",
        "| diagnosis | pages |",
        "| :--- | ---: |",
    ]
    lines += [f"| `{name}` | {count} |" for name, count in counts.most_common()]
    lines += [
        "",
        f"## The {len(gallery)} worst",
        "",
        "| # | page | diagnosis | score | shapes with no ink | ink share | fragmentation | "
        "straight-run residue |",
        "| ---: | :--- | :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for index, case in enumerate(gallery, 1):
        lines.append(
            f"| {index} | `{case.id}` | `{case.diagnosis}` | {case.score:.2f} | "
            f"{case.empty_boxes:.0%} of {case.boxes} | {case.ink_share:.1%} | "
            f"{case.fragmentation:.1f} | {case.rule_residue:.2f} |"
        )
    lines += ["", "## The images", ""]
    for index, case in enumerate(gallery, 1):
        lines += [
            f"### {index}. `{case.id}` — {case.diagnosis}",
            "",
            f"![{case.id}](figures/preproc_failures/{case.id}.png)",
            "",
            _explain(case),
            "",
        ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return REPORT


def _explain(case: Case) -> str:
    """One sentence saying what the numbers mean for this page."""
    reasons = {
        "ink_lost": (
            f"{case.empty_boxes:.0%} of the {case.boxes} annotated shapes have essentially no "
            f"ink inside them after binarization — the strokes were there and the threshold "
            f"did not keep them."
        ),
        "threshold_too_high": (
            f"Only {case.ink_share:.2%} of the page survived as ink, well under the {MIN_INK_SHARE:.1%} "
            f"a drawn page carries; the threshold sat above the pen."
        ),
        "background_survived": (
            f"{case.ink_share:.1%} of the page came out as ink, above the {MAX_INK_SHARE:.0%} "
            f"ceiling"
            + (
                f", and {case.rule_residue:.0%} of it lies in straight runs longer than a tenth "
                f"of the page: this is squared paper whose grid survived 3.1.8. The thumbnails "
                f"show why — the notebook is not flat, the printed lines bow with the page, and "
                f"a suppression built on straight structuring elements does not see a curved rule."
                if case.rule_residue >= 0.20
                else " — paper texture or a shadow edge was thresholded in as stroke."
            )
        ),
        "off_page_background": (
            f"{case.off_page_ink:.0%} of the ink falls outside the detected sheet of paper: the "
            f"photograph caught the desk around the page and the dark surroundings binarized as "
            f"stroke. Phase 3.1.2 finds the page correctly here — nothing crops to it, because "
            f"the primitive pipeline of 3.2 deliberately runs on the whole frame. This is the "
            f"gallery's argument for wiring page detection into the default chain."
        ),
        "fragmented": (
            f"{case.fragmentation:.1f} components per thousand ink pixels against a healthy 1.5: "
            f"the ink is present but broken into beads, which is what breaks contour extraction "
            f"and text grouping downstream."
        ),
        "shadow_left": (
            f"Quadrant brightness spread is still {case.shadow_left:.0f} after illumination "
            f"correction, above Phase 1's own threshold of {SHADOW_SPREAD:.0f} for calling a "
            f"photograph shadowed."
        ),
        "crop_lost_content": (
            f"{case.crop_loss:.0%} of the annotated shapes fall outside the detected page "
            f"quadrilateral — the page detector cropped into the drawing."
        ),
        "clean": "No signal fired; this page is in the gallery only because twenty were asked for.",
    }
    return reasons.get(case.diagnosis, "")


def main(argv: list[str] | None = None) -> int:
    from src.utils.parallel import pmap

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=120)
    args = ap.parse_args(argv)

    cases = collect(args.limit)
    if not cases:
        print("no pages; run the hdBPMN converter first", file=sys.stderr)
        return 1
    gallery = sorted(cases, key=lambda c: -c.score)[:GALLERY_SIZE]
    written = pmap(thumbnail, gallery, prefer="threads")
    for case, path in zip(gallery, written, strict=True):
        case.thumbnail = str(path.relative_to(ROOT))
    path = report(cases, gallery)

    print(json.dumps({"pages": len(cases), "gallery": len(gallery)}, indent=2))
    for case in gallery[:8]:
        print(f"  {case.to_dict()}")

    checks = {
        "twenty_cases_in_the_gallery": len(gallery) == GALLERY_SIZE,
        "every_case_has_a_diagnosis": all(c.diagnosis for c in gallery),
        "every_thumbnail_written": all((ROOT / c.thumbnail).is_file() for c in gallery),
        "report_written": path.is_file(),
    }
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
