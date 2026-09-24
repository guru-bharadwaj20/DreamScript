"""Phase 14.5 - the robustness suite: what the pipeline does as the photograph gets worse.

    python -m src.eval.robust                      # reports/robustness_pipeline.{md,json} + figure
    python -m src.eval.robust --pages 6 --quick    # a short sweep while developing

3.3.2 already swept blur, rotation and lighting - but it measured **stroke IoU after
preprocessing**, which is the first stage of eight. A binarisation that survives a shadow tells
you nothing about whether the detector still finds the diamond underneath it. This row sweeps the
same degradations (plus occlusion and resolution) through **the stages that consume the
photograph**: the component detector, the arrow detector, assembly, and the routing decision.

## What is measured, and against what

Each degraded page is run through the real detector - `corpus.predict` on the degraded pixels,
never the cached boxes, which is the mistake that would make every curve flat - and then through
S5's own assembly, and scored with S5's own `decompose` against the unchanged ground truth. So a
point on a curve is directly comparable to the S5 headline, and severity 0 reproduces it.

Four curves per degradation:

    node_f1      did the detector still find the shapes
    edge_f1      did the arrows still land on the right pair of them
    median_ged   the criterion S5 is scored on
    routed       did 13.3's naive-Bayes router still name the diagram type

## Severity 0 is measured, not assumed

Every sweep includes its own clean control, re-run through the identical code path rather than
read from the S5 report. A curve whose baseline came from somewhere else cannot distinguish "the
degradation hurt" from "this subsample is easier".

## The sample is small and stratified, and that is stated rather than hidden

Assembly costs seconds per page, and the matrix is five degradations x five severities. The
default is a stratified sample of held-out pages (hdbpmn and fa_bresler in their test-split
proportion), with `--pages` to widen it. Every reported point names the number of pages behind
it, and the per-page rows are kept in the JSON so a curve can be re-aggregated without a rerun.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from src.utils.config import ROOT
from src.utils.figures import save as _figsave

REPORT_JSON = ROOT / "reports" / "robustness_pipeline.json"
REPORT_MD = ROOT / "reports" / "robustness_pipeline.md"
FIGURE = ROOT / "reports" / "figures" / "p14_robustness.png"

#: Severities per degradation. The first entry of each is the clean control and is always run.
SWEEPS: dict[str, tuple[float, ...]] = {
    "blur": (0, 3, 5, 7, 9),
    "rotation": (0.0, 3.0, 6.0, 9.0, 12.0),
    "lighting": (0.0, 0.4, 0.7, 0.85, 0.95),
    "occlusion": (0.0, 0.05, 0.1, 0.2, 0.35),
    "resolution": (1.0, 0.75, 0.5, 0.35, 0.25),
}

METRICS = ("node_f1", "edge_f1", "median_ged", "routed")


def degrade(gray, kind: str, severity: float):
    """One degradation at one severity, reusing 3.3.2's and 7.1.6's own implementations."""
    import cv2

    from src.classify.occlusion import occlude
    from src.preprocess.robustness import blur, light, rotate

    if kind == "blur":
        return blur(gray, int(severity))
    if kind == "rotation":
        return rotate(gray, float(severity))
    if kind == "lighting":
        return light(gray, float(severity))
    if kind == "occlusion":
        return occlude(gray, "coffee_stain", float(severity))
    if kind == "resolution":
        if severity >= 1.0:
            return gray
        height, width = gray.shape[:2]
        small = cv2.resize(
            gray,
            (max(8, int(width * severity)), max(8, int(height * severity))),
            interpolation=cv2.INTER_AREA,
        )
        # Back to the detector's expected frame: a real low-resolution capture is upsampled
        # somewhere, and doing it here keeps every geometry in one coordinate system.
        return cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)
    raise ValueError(f"unknown degradation {kind!r}")


class DegradedPage:
    """A `Page` whose pixels are a degraded copy, and whose truth is the original's.

    Everything that identifies the page to the rest of the pipeline - source, id, scale, the IR
    path - is the original's, so the truth this is scored against is unchanged. Only `name` and
    `image` differ, and `name` differs so the detection cache cannot hand back the clean boxes.
    """

    def __init__(self, page, image: Path, suffix: str):
        self._page = page
        self._image = image
        self.name = f"{page.name}@{suffix}"

    def __getattr__(self, item):
        return getattr(self._page, item)

    @property
    def image(self) -> Path:
        return self._image


def _install_detections(page: DegradedPage, weights=None) -> None:
    """Run the real detector on the degraded pixels and put the boxes in the cache by name."""
    from src.assemble import corpus

    payload = corpus.predict((page,), weights or corpus.WEIGHTS)
    corpus._cache(corpus.WEIGHTS, False)["pages"][page.name] = payload["pages"][page.name]


def score_variant(page, kind: str, severity: float, directory: Path) -> dict[str, Any]:
    """One page at one severity: detect, assemble, route, and diff against the truth."""
    import cv2

    from src.assemble.corpus import truth
    from src.assemble.s5 import assemble, decompose

    gray = cv2.imread(str(page.image), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return {"page": page.name, "kind": kind, "severity": severity, "error": "unreadable"}
    out = directory / f"{page.name}_{kind}_{severity}.png"
    cv2.imwrite(str(out), degrade(gray, kind, severity))

    degraded = DegradedPage(page, out, f"{kind}{severity}")
    _install_detections(degraded)
    row: dict[str, Any] = {
        "page": page.name,
        "source": page.source,
        "kind": kind,
        "severity": severity,
    }
    try:
        predicted = assemble(
            degraded,
            state_text=True,
            arrow_edges=True,
            page_text=True,
            direct=True,
            prune_loops=True,
        )
        row.update(decompose(predicted, truth(page)))
    # A stage that cannot answer is a data point, not a crash.
    except Exception as error:  # noqa: BLE001
        row["error"] = f"{type(error).__name__}: {error}"
        return row

    try:
        from src.assemble.corpus import detections
        from src.pipeline.routing import route

        predicted_type, confidence = route(detections(degraded))
        row["routed_type"] = predicted_type
        row["routed_confidence"] = round(float(confidence), 4)
        row["routed"] = int(predicted_type == truth(page).diagram_type)
    # The routing half fails independently of the assembly half.
    except Exception as error:  # noqa: BLE001
        row["routed_error"] = f"{type(error).__name__}: {error}"
    return row


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return float((ordered[middle - 1] + ordered[middle]) / 2)


def aggregate(rows: list[dict]) -> dict[str, Any]:
    """Curves as `{degradation: {severity: {metric: value}}}`, with the page count behind each."""
    curves: dict[str, dict[str, Any]] = {}
    for row in rows:
        if "ged" not in row:
            continue
        point = curves.setdefault(row["kind"], {}).setdefault(str(row["severity"]), {"rows": []})
        point["rows"].append(row)
    for points in curves.values():
        for point in points.values():
            scored = point.pop("rows")
            point["pages"] = len(scored)
            point["node_f1"] = round(sum(r["node_f1"] for r in scored) / len(scored), 4)
            point["edge_f1"] = round(sum(r["edge_f1"] for r in scored) / len(scored), 4)
            point["median_ged"] = _median([r["ged"] for r in scored])
            routed = [r["routed"] for r in scored if "routed" in r]
            point["routed"] = round(sum(routed) / len(routed), 4) if routed else None
            point["errors"] = sum(1 for r in scored if r.get("error"))
    return curves


def _sample(limit: int, split: str) -> tuple:
    """A stratified sample: sources in the order the split lists them, round-robin by source."""
    from src.assemble.corpus import pages

    available = pages((split,))
    by_source: dict[str, list] = {}
    for page in available:
        by_source.setdefault(page.source, []).append(page)
    picked: list = []
    index = 0
    while len(picked) < min(limit, len(available)):
        added = False
        for source in sorted(by_source):
            if index < len(by_source[source]) and len(picked) < limit:
                picked.append(by_source[source][index])
                added = True
        if not added:
            break
        index += 1
    return tuple(picked)


def run(limit: int = 12, split: str = "test", kinds: tuple[str, ...] = tuple(SWEEPS)) -> dict:
    selected = _sample(limit, split)
    started = time.time()
    rows: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="dreamscript_robust_") as name:
        directory = Path(name)
        for kind in kinds:
            for severity in SWEEPS[kind]:
                for page in selected:
                    rows.append(score_variant(page, kind, severity, directory))
                print(f"  {kind} {severity} done", file=sys.stderr, flush=True)
    curves = aggregate(rows)
    return {
        "split": split,
        "pages": [page.name for page in selected],
        "sources": sorted({page.source for page in selected}),
        "seconds": round(time.time() - started, 1),
        "curves": curves,
        "rows": rows,
    }


def figure(curves: dict, path: Path = FIGURE) -> Path | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kinds = [kind for kind in SWEEPS if kind in curves]
    if not kinds:
        return None
    fig, axes = plt.subplots(1, len(kinds), figsize=(3.2 * len(kinds), 3.4), sharey=True)
    axes = axes if len(kinds) > 1 else [axes]
    for axis, kind in zip(axes, kinds, strict=True):
        severities = sorted(curves[kind], key=float)
        x = [float(value) for value in severities]
        for metric, style in (("node_f1", "-o"), ("edge_f1", "-s"), ("routed", "--^")):
            y = [curves[kind][s].get(metric) for s in severities]
            if any(value is None for value in y):
                continue
            axis.plot(x, y, style, label=metric, markersize=4)
        axis.set_title(kind)
        axis.set_xlabel("severity")
        axis.set_ylim(0, 1.02)
        axis.grid(alpha=0.3)
    axes[0].set_ylabel("score")
    axes[-1].legend(fontsize=7)
    fig.suptitle("Phase 14.5 - pipeline robustness (detector, arrows, routing)", fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    _figsave(fig, path, dpi=150)
    plt.close(fig)
    return path


def render(result: dict) -> str:
    lines = [
        "# Phase 14.5 - robustness of the pipeline, not of the binariser",
        "",
        f"Generated by `python -m src.eval.robust` over {len(result['pages'])} held-out"
        f" {result['split']} pages ({', '.join(result['sources'])}) in {result['seconds']} s."
        " Every point re-runs the real detector on the degraded pixels and then S5's own"
        " assembly, scored against the unchanged truth, so severity 0 reproduces the S5 number"
        " on this sample rather than importing it.",
        "",
        f"![degradation curves](figures/{FIGURE.name})",
        "",
    ]
    for kind, points in result["curves"].items():
        severities = sorted(points, key=float)
        lines += [
            f"## {kind}",
            "",
            "| severity | " + " | ".join(severities) + " |",
            "| :--- |" + " ---: |" * len(severities),
        ]
        for metric in METRICS:
            values = []
            for severity in severities:
                value = points[severity].get(metric)
                values.append("-" if value is None else f"{value:.4f}".rstrip("0").rstrip("."))
            lines.append(f"| {metric} | " + " | ".join(values) + " |")
        errors = sum(points[severity].get("errors", 0) for severity in severities)
        lines += [
            "",
            f"pages per point: {points[severities[0]]['pages']}; stage errors: {errors}",
            "",
        ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pages", type=int, default=12)
    ap.add_argument("--split", default="test")
    ap.add_argument("--kinds", nargs="*", default=list(SWEEPS))
    ap.add_argument("--quick", action="store_true", help="two severities per degradation")
    ap.add_argument("--out", type=Path, default=REPORT_MD)
    ap.add_argument("--json", dest="out_json", type=Path, default=REPORT_JSON)
    args = ap.parse_args(argv)

    if args.quick:
        for kind in SWEEPS:
            SWEEPS[kind] = SWEEPS[kind][:2]
    result = run(args.pages, args.split, tuple(args.kinds))
    result["figure"] = str(figure(result["curves"]) or "")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(result), encoding="utf-8")
    args.out_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["curves"], indent=2)[:1500])
    print(f"-> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
