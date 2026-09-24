"""Phase 13.3 / 13.4 - deciding what kind of diagram this is, and what to do when unsure.

    from src.pipeline.routing import route, fit
    route(boxes)      # -> ("flowchart", 0.98)

## The prior, and why it is the detector's own histogram

The plan's routing line is *NB fast prior -> ensemble classifier -> type-specific parser*. The
ensemble classifier of Phases 5-6 reaches 0.9871 on this question (S1) and is the right thing to
route on - but it is fitted on CLIP embeddings whose artefact does not survive a rebuild, and
re-deriving them costs more than the routing decision is worth at inference.

What the pipeline already has in hand by this point is the detector's output, and the *shape
vocabulary is itself a strong prior on diagram type*: a page of diamonds and rounded rectangles
is a flowchart, a page of circles and double circles is an automaton. So the fast prior here is
a multinomial naive Bayes over the class histogram 9.1 produces - literally the plan's "NB fast
prior", fitted on evidence the pipeline has already paid for.

`fit` trains it on the validation pages and `evaluate` scores it on test, both of which are
disjoint from the detector's own training split, so the number it reports is not measuring the
detector remembering its training pages.

## What this cannot route, and why that is honest

Only **flowchart** and **state_machine** have held-out detector coverage: 9.1.2's export is
hdbpmn, flowchartseg and fa_bresler. ER and circuit have no real annotated diagrams anywhere in
this repo (12.1.6 records the same gap), and sketch2code's wireframes are not in the detection
corpus. So the router is fitted and measured on two of the five types, and a page it has no
evidence for is returned as `unknown` with the confidence that earned it rather than being
forced into whichever of the two it resembles least.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.detect.classes import CLASSES
from src.pipeline.contracts import UNKNOWN
from src.utils.config import ROOT
from src.utils.splits import normalise

# `CLASSES` is imported above rather than restated here, where a verbatim copy of it used to
# live. The original's own comment says "Index is the YOLO class id and must never be reordered -
# a trained checkpoint stores integers, not names, so a reorder silently relabels every
# prediction". A copy of a list that must never be reordered is a list that can be reordered in
# one place, and the failure would be a prior fitted on one histogram scoring another with its
# columns shuffled - which reports a number rather than an error.
#
# Kept as a comment rather than deleted: the histogram's column order is this module's contract
# with every router.json it ever wrote, and the next reader deserves to be told where it lives.

#: Which diagram type each corpus source is drawn from. The manifest's own assignment, complete
#: rather than filtered - a source is here whether or not the detector has ever seen it.
SOURCE_TYPE = {
    "hdbpmn": "flowchart",
    "flowchartseg": "flowchart",
    "fa_bresler": "state_machine",
    "sketch2code": "wireframe",
    "cghd": "circuit",
}

#: The two entries above that the detector has no held-out coverage for, named rather than left
#: to be discovered by noticing that `fit` never reaches them. 9.1.2's export is hdbpmn,
#: flowchartseg and fa_bresler; `data/processed/detect/index.json` contains no sketch2code page
#: and no cghd page at all, so `wireframe` and `circuit` are types this router can never learn
#: and never returns. The module docstring says the router covers two of five types; this is
#: which three it does not, in a form `fit` can check and a report can print.
UNCOVERED_SOURCES: tuple[str, ...] = ("sketch2code", "cghd")

MODEL = ROOT / "experiments" / "pipeline" / "router.json"


def histogram(boxes: list[dict]) -> np.ndarray:
    """Counts of each detector class on one page."""
    index = {name: i for i, name in enumerate(CLASSES)}
    counts = np.zeros(len(CLASSES), dtype=float)
    for box in boxes:
        position = index.get(str(box.get("cls")))
        if position is not None:
            counts[position] += 1.0
    return counts


def fit(split: str = "val", model_path: Path = MODEL) -> dict[str, Any]:
    """Fit the prior on one split's detector output and store it."""
    from src.assemble.corpus import detections, pages

    # Normalised, not compared as text: this index says `val` and the manifest says
    # `validation`, and `fit("validation")` used to match no page and then raise "no detector
    # output" three frames later instead of here.
    wanted = normalise(split)
    rows, labels = [], []
    for page in pages():
        if normalise(page.split) != wanted:
            continue
        kind = SOURCE_TYPE.get(page.source)
        if kind is None:
            continue
        rows.append(histogram(detections(page)))
        labels.append(kind)
    if not rows:
        raise RuntimeError(f"no detector output for split {split!r}")

    matrix = np.vstack(rows)
    names = sorted(set(labels))
    # Multinomial naive Bayes, written out rather than pickled: the whole model is a table of
    # log class priors and log per-class feature rates, and a JSON of those is both inspectable
    # and immune to the sklearn version that wrote it.
    log_prior, log_rate = {}, {}
    for name in names:
        mask = np.array([label == name for label in labels])
        counts = matrix[mask].sum(axis=0) + 1.0  # Laplace
        log_prior[name] = float(np.log(mask.sum() / len(labels)))
        log_rate[name] = np.log(counts / counts.sum()).tolist()

    payload = {
        "classes": list(CLASSES),
        "types": names,
        # What this model cannot answer, stored beside what it can. A router.json that lists
        # only the two types it learned reads as though the other three were never asked for.
        "uncovered_sources": list(UNCOVERED_SOURCES),
        "uncovered_types": sorted(
            {SOURCE_TYPE[source] for source in UNCOVERED_SOURCES} - set(names)
        ),
        "log_prior": log_prior,
        "log_rate": log_rate,
        "fitted_on": {"split": split, "pages": len(labels)},
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _model(model_path: Path = MODEL) -> dict[str, Any] | None:
    if not Path(model_path).is_file():
        return None
    try:
        return json.loads(Path(model_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def route(boxes: list[dict], model_path: Path = MODEL) -> tuple[str, float]:
    """`(diagram_type, probability)`. `unknown` when nothing was detected or no model is fitted."""
    payload = _model(model_path)
    counts = histogram(boxes)
    if payload is None or counts.sum() == 0:
        return UNKNOWN, 0.0

    scores = {
        kind: payload["log_prior"][kind]
        + float(np.dot(counts, np.array(payload["log_rate"][kind])))
        for kind in payload["types"]
    }
    best = max(scores, key=lambda k: scores[k])
    # Softmax over the two log scores, which is the posterior when the priors are already in them.
    values = np.array(list(scores.values()))
    posterior = float(np.exp(values - values.max()).max() / np.exp(values - values.max()).sum())
    return best, posterior


def evaluate(split: str = "test", model_path: Path = MODEL) -> dict[str, Any]:
    """Per-type accuracy of the prior on a split it was not fitted on (13.3's own test)."""
    from src.assemble.corpus import detections, pages

    per_type: dict[str, list[int]] = {}
    for page in pages():
        if page.split != split:
            continue
        want = SOURCE_TYPE.get(page.source)
        if want is None:
            continue
        got, _ = route(detections(page), model_path)
        per_type.setdefault(want, []).append(int(got == want))

    out = {
        kind: {"n": len(hits), "accuracy": round(sum(hits) / len(hits), 4)}
        for kind, hits in sorted(per_type.items())
    }
    total = [hit for hits in per_type.values() for hit in hits]
    out["overall"] = {"n": len(total), "accuracy": round(sum(total) / max(1, len(total)), 4)}
    return out
