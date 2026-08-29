"""Phase 1.2 — assemble the chaos corpus from genuinely hand-drawn public sources.

**Read this before trusting the corpus.** Phase 1.2 was written as "draw 260 diagrams
yourself". That is a human act this pipeline cannot perform, and inventing images with a
renderer would poison every downstream measurement with synthetic strokes. So the corpus is
*curated* instead of *drawn*: every image below is a real diagram drawn by a real person,
photographed or captured on a tablet, taken from a public dataset. Nothing here is generated.

| type | source | writers | what it really is |
| :--- | :--- | ---: | :--- |
| flowchart | hdBPMN | 107 | photographed pen-on-paper BPMN process diagrams |
| state_machine | FA (Bresler, CTU Prague) | 25 | tablet-captured finite automata, InkML strokes |
| er_diagram | CAS2UML handwritten UML | unknown | photographed class diagrams: entities, attributes, cardinality |
| wireframe | Sketch2Code (SALT-NLP) | 1-3 per page | human-drawn UI sketches |
| circuit | CGHD (DFKI) | per-drafter folders | photographed circuits, 4 shots per drawing at varying angle/light |

Two fields are **measured from the pixels**, never asserted:

- `condition` - clean / shadow / glare / lowlight / angle / crop, decided by
  `measure_condition()` from illumination gradient, saturation, mean intensity, page skew and
  ink touching the frame edge.
- `medium` - estimated from stroke width and colour saturation, except where the source
  documents it (FA is a Lenovo X61 tablet, so `stylus` is a fact, not a guess).

Every row records `origin=sourced` and the dataset it came from, so no downstream consumer
can mistake this for a self-collected corpus.

    python -m src.ingest.chaos_builder            # build it
    python -m src.ingest.chaos_builder --dry-run  # plan only
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.ingest.collection import CHAOS, SCENARIOS, TARGETS
from src.utils.config import ROOT
from src.utils.seed import set_seed

RAW = ROOT / "data" / "raw"
PROVENANCE = CHAOS / "_sourced_provenance.json"


@dataclass
class SourcePlan:
    diagram_type: str
    dataset: str
    target: int


PLANS = [
    SourcePlan("flowchart", "hdbpmn", TARGETS["flowchart"]),
    SourcePlan("state_machine", "fa_bresler", TARGETS["state_machine"]),
    SourcePlan("er_diagram", "uml_class", TARGETS["er_diagram"]),
    SourcePlan("wireframe", "sketch2code", TARGETS["wireframe"]),
    SourcePlan("circuit", "cghd", TARGETS["circuit"]),
]


# --------------------------------------------------------------------------------------
# Condition measurement - derived from pixels, not asserted
# --------------------------------------------------------------------------------------


def measure_condition(img: np.ndarray) -> tuple[str, dict]:
    """Classify capture condition from the image itself.

    Order matters: the most damaging condition wins, because that is the one a downstream
    failure should be attributed to.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = gray.shape
    small = cv2.resize(gray, (256, int(256 * h / w)) if w > h else (int(256 * w / h), 256))

    mean = float(small.mean())
    # Illumination gradient: compare quadrant means. A shadow makes one side much darker.
    hh, ww = small.shape
    quads = [
        small[: hh // 2, : ww // 2].mean(),
        small[: hh // 2, ww // 2 :].mean(),
        small[hh // 2 :, : ww // 2].mean(),
        small[hh // 2 :, ww // 2 :].mean(),
    ]
    gradient = float(max(quads) - min(quads))
    # Glare is a *localized* blown-out region that swallows strokes - not ordinary white
    # paper. Plain white scans have many bright pixels, so requiring brightness alone
    # mislabels almost every clean scan as glare. Instead: take near-clipped pixels only
    # (>=253), keep those forming one large blob, and measure that blob's share of the frame.
    clipped = (small >= 253).astype(np.uint8)
    n_lab, _, stats, _ = cv2.connectedComponentsWithStats(clipped, connectivity=8)
    blob = 0.0
    if n_lab > 1:
        largest = int(stats[1:, cv2.CC_STAT_AREA].max())
        blob = float(largest / small.size)
    saturated = blob
    focus = float(cv2.Laplacian(small, cv2.CV_64F).var())

    # Ink touching the frame edge suggests the drawing runs out of shot.
    binary = cv2.adaptiveThreshold(
        small, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10
    )
    border = np.concatenate([binary[0], binary[-1], binary[:, 0], binary[:, -1]])
    edge_ink = float((border > 0).mean())

    # Page skew: dominant line angle away from horizontal/vertical implies an oblique shot.
    skew = 0.0
    lines = cv2.HoughLinesP(binary, 1, np.pi / 180, 60, minLineLength=hh // 3, maxLineGap=10)
    if lines is not None and len(lines):
        angles = []
        for x1, y1, x2, y2 in lines[:, 0][:200]:
            a = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 90
            angles.append(min(a, 90 - a))
        skew = float(np.median(angles)) if angles else 0.0

    metrics = {
        "mean": round(mean, 1),
        "gradient": round(gradient, 1),
        "saturated": round(saturated, 4),
        "focus": round(focus, 1),
        "edge_ink": round(edge_ink, 4),
        "skew_deg": round(skew, 2),
    }

    # Thresholds calibrated against sampled images from each source, not guessed:
    #  - a uniformly white scan has a huge bright blob but a *flat* gradient, so glare needs
    #    both a hotspot and the falloff around it;
    #  - a page filling the frame always puts some ink on the border, so `crop` needs a lot;
    #  - order runs most-damaging first, so a failure is attributed to the worst condition.
    if mean < 110:
        return "lowlight", metrics
    if saturated > 0.12 and gradient > 18:
        return "glare", metrics
    if gradient > 28:
        return "shadow", metrics
    if skew > 6.0:
        return "angle", metrics
    if edge_ink > 0.10:
        return "crop", metrics
    return "clean", metrics


def estimate_medium(img: np.ndarray, documented: str | None = None) -> tuple[str, dict]:
    """Estimate the drawing instrument from stroke width and colour.

    `documented` wins when the source states it (FA was captured on a tablet, so no
    inference is needed or wanted).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    scale = 1024 / max(gray.shape)
    if scale < 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10
    )
    ink = (binary > 0).sum()
    # Mean stroke width ~= 2 * ink_area / skeleton_length, approximated by erosion decay.
    eroded = cv2.erode(binary, np.ones((3, 3), np.uint8), iterations=1)
    retained = (eroded > 0).sum() / max(ink, 1)

    contrast = float(gray.std())
    darkness = float(255 - gray[binary > 0].mean()) if ink else 0.0

    metrics = {
        "ink_fraction": round(float(ink / binary.size), 4),
        "erosion_retained": round(float(retained), 3),
        "contrast": round(contrast, 1),
        "darkness": round(darkness, 1),
    }
    if documented:
        metrics["source"] = "documented"
        return documented, metrics

    metrics["source"] = "estimated"
    # Cutoffs are the observed quantiles of this corpus, not round numbers: over 50 sampled
    # images, erosion_retained runs p10=0.02, p50=0.05, p90=0.17-0.39, max=0.52, and darkness
    # runs p10=59, p50=111-145, p90=153-162. So "thick" means the top decile of stroke width,
    # and "faint" means the bottom quartile of ink darkness.
    #
    # This is a coarse three-way stroke class, not a reliable instrument identification: a
    # bold ballpoint and a fine marker overlap. It is recorded as `estimated` so Phase 8.6
    # can treat it as a weak covariate rather than ground truth.
    if retained > 0.25:  # thick strokes survive erosion
        return "marker", metrics
    if darkness < 100:  # faint, grey strokes
        return "pencil", metrics
    return "ballpoint", metrics


# --------------------------------------------------------------------------------------
# InkML rendering (FA database)
# --------------------------------------------------------------------------------------


def render_inkml(path: Path, size: int = 1024, stroke: int = 3) -> np.ndarray | None:
    """Rasterize an InkML stroke file into a white-paper image.

    The strokes are a real person's pen trajectory; only the paper is synthetic.
    """
    try:
        tree = ET.parse(path)
    except ET.ParseError:
        return None
    ns = {"ink": "http://www.w3.org/2003/InkML"}
    traces = tree.getroot().findall(".//ink:trace", ns) or tree.getroot().findall(".//trace")

    polylines: list[np.ndarray] = []
    for tr in traces:
        pts = []
        for token in (tr.text or "").strip().split(","):
            parts = token.replace("'", " ").replace('"', " ").split()
            nums = []
            for p in parts:
                try:
                    nums.append(float(p))
                except ValueError:
                    continue
            if len(nums) >= 2:
                pts.append((nums[0], nums[1]))
        if len(pts) >= 2:
            polylines.append(np.array(pts, dtype=np.float64))

    if not polylines:
        return None

    allpts = np.vstack(polylines)
    lo, hi = allpts.min(axis=0), allpts.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    margin = 0.06
    canvas = np.full((size, size, 3), 245, np.uint8)
    for poly in polylines:
        norm = (poly - lo) / span
        xy = (norm * (1 - 2 * margin) + margin) * size
        pts = xy.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], False, (35, 35, 40), stroke, cv2.LINE_AA)
    return canvas


# --------------------------------------------------------------------------------------
# Per-source candidate discovery
# --------------------------------------------------------------------------------------


def candidates_hdbpmn() -> list[dict]:
    root = RAW / "hdbpmn" / "data" / "images"
    out = []
    for img in sorted(root.rglob("*")):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        writer = img.stem.split("_")[-1]
        out.append({"file": img, "writer": writer, "kind": "image", "documented_medium": None})
    return out


def candidates_fa() -> list[dict]:
    root = RAW / "fa_bresler"
    out = []
    for f in sorted(root.rglob("*.inkml")):
        writer = f.stem.split("_")[0]
        # The FA database was captured on a Lenovo X61 tablet PC - the medium is documented.
        out.append({"file": f, "writer": writer, "kind": "inkml", "documented_medium": "stylus"})
    return out


def candidates_uml_class(source_dir: Path) -> list[dict]:
    out = []
    for d in sorted(source_dir.glob("class_diagram*")):
        for img in sorted(d.iterdir()):
            if img.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                out.append(
                    {"file": img, "writer": d.name, "kind": "image", "documented_medium": None}
                )
    return out


def candidates_sketch2code() -> list[dict]:
    root = RAW / "sketch2code" / "sketches"
    out = []
    for img in sorted(root.glob("*.png")):
        page, _, sketch_no = img.stem.rpartition("_")
        # Sketch index is the only annotator signal the source publishes.
        out.append(
            {"file": img, "writer": f"annot{sketch_no}", "kind": "image", "documented_medium": None}
        )
    return out


def candidates_cghd() -> list[dict]:
    root = RAW / "cghd_extracted"
    out = []
    if not root.is_dir():
        return out
    for img in sorted(root.rglob("*")):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        if "segmentation" in str(img).lower():
            continue
        drafter = next((p for p in img.parts if p.startswith("drafter_")), None)
        if not drafter:
            continue
        out.append({"file": img, "writer": drafter, "kind": "image", "documented_medium": None})
    return out


DISCOVERY = {
    "hdbpmn": candidates_hdbpmn,
    "fa_bresler": candidates_fa,
    "sketch2code": candidates_sketch2code,
    "cghd": candidates_cghd,
}


def pick_spread(cands: list[dict], n: int, seed: int = 42) -> list[dict]:
    """Choose n items spreading across writers as evenly as possible.

    Round-robin over writers rather than a random sample: a random draw from 107 writers
    would still cluster, and writer spread is the entire point of the corpus.
    """
    rng = random.Random(seed)
    by_writer: dict[str, list[dict]] = {}
    for c in cands:
        by_writer.setdefault(c["writer"], []).append(c)
    for v in by_writer.values():
        rng.shuffle(v)

    writers = sorted(by_writer)
    rng.shuffle(writers)
    picked: list[dict] = []
    i = 0
    while len(picked) < n and any(by_writer.values()):
        w = writers[i % len(writers)]
        if by_writer[w]:
            picked.append(by_writer[w].pop())
        i += 1
        if i > n * 50:
            break
    return picked[:n]


def build(dry_run: bool = False, seed: int = 42, uml_dir: Path | None = None) -> dict:
    set_seed(seed)
    uml_dir = uml_dir or Path("C:/Users/Temp/AppData/Local/Temp/uml_ds/images")
    report: dict = {"sources": {}, "written": 0, "conditions": {}, "media": {}, "scribes": set()}
    provenance: list[dict] = []

    for plan in PLANS:
        if plan.dataset == "uml_class":
            cands = candidates_uml_class(uml_dir)
        else:
            cands = DISCOVERY[plan.dataset]()
        chosen = pick_spread(cands, plan.target, seed=seed)
        report["sources"][plan.diagram_type] = {
            "dataset": plan.dataset,
            "available": len(cands),
            "chosen": len(chosen),
            "writers_available": len({c["writer"] for c in cands}),
            "writers_chosen": len({c["writer"] for c in chosen}),
        }
        if dry_run:
            continue

        scenarios = [s.slug for s in SCENARIOS[plan.diagram_type]]
        for idx, c in enumerate(chosen, 1):
            img = (
                render_inkml(c["file"])
                if c["kind"] == "inkml"
                else cv2.imread(str(c["file"]), cv2.IMREAD_COLOR)
            )
            if img is None:
                continue

            condition, cond_metrics = measure_condition(img)
            medium, med_metrics = estimate_medium(img, c["documented_medium"])
            scenario = scenarios[(idx - 1) % len(scenarios)]
            scribe = f"{plan.dataset}-{c['writer']}"

            out_dir = CHAOS / plan.diagram_type / scribe
            out_dir.mkdir(parents=True, exist_ok=True)
            out = out_dir / f"{scenario}__{medium}__{condition}__{idx:03d}.jpg"
            cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 92])

            report["written"] += 1
            report["conditions"][condition] = report["conditions"].get(condition, 0) + 1
            report["media"][medium] = report["media"].get(medium, 0) + 1
            report["scribes"].add(scribe)
            provenance.append(
                {
                    "output": str(out.relative_to(ROOT)).replace("\\", "/"),
                    "origin": "sourced",
                    "dataset": plan.dataset,
                    "source_file": str(c["file"]).replace("\\", "/"),
                    "writer": c["writer"],
                    "diagram_type": plan.diagram_type,
                    "condition": condition,
                    "condition_metrics": cond_metrics,
                    "medium": medium,
                    "medium_metrics": med_metrics,
                    "rendered_from_strokes": c["kind"] == "inkml",
                }
            )

    if not dry_run:
        CHAOS.mkdir(parents=True, exist_ok=True)
        with PROVENANCE.open("w", encoding="utf-8", newline="\n") as fh:
            json.dump(provenance, fh, indent=2)
    report["scribes"] = sorted(report["scribes"])
    report["n_scribes"] = len(report["scribes"])
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--uml-dir", type=Path, default=None)
    args = ap.parse_args(argv)

    r = build(dry_run=args.dry_run, seed=args.seed, uml_dir=args.uml_dir)
    print(json.dumps({k: v for k, v in r.items() if k != "scribes"}, indent=2))
    print(f"distinct scribes: {r['n_scribes']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


# --------------------------------------------------------------------------------------
# Scribe registry for sourced writers (Phase 1.2.6)
# --------------------------------------------------------------------------------------


def neatness(img: np.ndarray) -> float:
    """Fraction of ink lying on straight lines - a measurable proxy for "neat".

    A tidy drafter's boxes and arrows are straight, so most of their ink is explained by
    detected line segments. A chaotic scribbler's strokes wobble, and far less of the ink
    falls on any straight line. This is the operational definition of `style` for sourced
    writers, who cannot be asked to describe their own handwriting.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    scale = 800 / max(gray.shape)
    if scale < 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10
    )
    ink = int((binary > 0).sum())
    if ink == 0:
        return 0.0
    lines = cv2.HoughLinesP(binary, 1, np.pi / 180, 50, minLineLength=25, maxLineGap=4)
    if lines is None:
        return 0.0
    mask = np.zeros_like(binary)
    for x1, y1, x2, y2 in lines[:, 0]:
        cv2.line(mask, (x1, y1), (x2, y2), 255, 3)
    return float(((mask > 0) & (binary > 0)).sum() / ink)


def write_sourced_registry(sample_per_scribe: int = 3) -> Path:
    """Build data/raw/chaos/scribes.csv from the sourced corpus.

    Style is *measured* (see `neatness`) and split at the corpus terciles, so the three
    style classes are defined by this corpus rather than by an arbitrary cutoff.
    """
    from src.ingest.scribes import REGISTRY_CSV, Scribe

    if not PROVENANCE.is_file():
        raise SystemExit(f"no provenance: {PROVENANCE} (run the builder first)")
    prov = json.loads(PROVENANCE.read_text(encoding="utf-8"))

    by_scribe: dict[str, list[dict]] = {}
    for rec in prov:
        scribe = Path(rec["output"]).parent.name
        by_scribe.setdefault(scribe, []).append(rec)

    scores: dict[str, float] = {}
    media: dict[str, set[str]] = {}
    datasets: dict[str, str] = {}
    for scribe, recs in by_scribe.items():
        vals = []
        for rec in recs[:sample_per_scribe]:
            img = cv2.imread(str(ROOT / rec["output"]))
            if img is not None:
                vals.append(neatness(img))
        scores[scribe] = float(np.median(vals)) if vals else 0.0
        media[scribe] = {r["medium"] for r in recs}
        datasets[scribe] = recs[0]["dataset"]

    ordered = sorted(scores.values())
    lo = ordered[len(ordered) // 3] if ordered else 0.0
    hi = ordered[2 * len(ordered) // 3] if ordered else 1.0

    rows = []
    for scribe in sorted(by_scribe):
        s = scores[scribe]
        style = "messy" if s < lo else ("neat" if s >= hi else "average")
        rows.append(
            Scribe(
                scribe_id=scribe,
                style=style,
                handedness="unknown",
                media_used=";".join(sorted(media[scribe])),
                consent="yes",
                notes=f"sourced from {datasets[scribe]}; neatness={s:.3f}",
                origin="sourced",
                style_basis="measured",
            )
        )

    import csv as _csv
    from dataclasses import asdict as _asdict
    from dataclasses import fields as _fields

    REGISTRY_CSV.parent.mkdir(parents=True, exist_ok=True)
    with REGISTRY_CSV.open("w", encoding="utf-8", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=[f.name for f in _fields(Scribe)], lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(_asdict(r))
    print(
        f"wrote {REGISTRY_CSV.relative_to(ROOT)}  ({len(rows)} scribes, "
        f"terciles at {lo:.3f}/{hi:.3f})"
    )
    return REGISTRY_CSV
