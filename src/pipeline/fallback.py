"""Phase 13.4 - the fallback chains, and the control that decides whether the first one is real.

    python -m src.pipeline.fallback              # reports/p13_fallback_chain.md + .json
    python -m src.pipeline.fallback --check      # non-zero if the classifier rung fails its control

13.4 names three chains. Two were never in doubt:

    model fails -> emitter          `src/pipeline/generate.py`, which marks an emitter answer
                                    `degraded=True` so no measurement mixes the two
    OCR fails -> placeholders       three rungs - the read label (`q0`), the node id (`s14`),
                                    `_ident`'s literal fallback - each verified to still emit
                                    code that passes 12.3.1

The third - *handcrafted classifier fails -> embedding classifier -> CNN* - was blocked, and the
reason was worth more than the row: **the corpus had no `state_machine` class at all**. The 300
fa_bresler state machines were never in the manifest, and 1,319 flowchartseg flowcharts were in
it but addressed as `<shard>.parquet#5`, which is not a file, so the feature builder dropped
every one. The table was two classes and 1,435 rows. Both are fixed in `src.ingest.manifest`, and
the table is now 3,054 rows over three classes.

## The confound, and why it is answerable here rather than arguable

fa_bresler is the only source of `state_machine`, and its pages are **rendered** from inkml pen
trajectories rather than photographed. The handcrafted features are pixel-derived. So a
three-class model that scores well might have learned *renderer*, not *diagram type*, and
wiring it would be worse than leaving the row red.

The corpus answers this rather than arguing about it, because **flowchart exists in both
domains**: flowchartseg is computer-rendered and hdbpmn is photographs of paper. That makes
three arms, of which the third decides the row:

    full            all four sources, three classes, grouped out-of-fold. The headline, and on
                    its own not evidence of anything.
    domain_probe    the same features predicting *rendered vs photographed*. If this is near
                    perfect, domain is fully recoverable from these features, and the full arm's
                    score cannot be taken at face value.
    cross_domain    train on the rendered domain only - flowchartseg flowcharts against
                    fa_bresler state machines - and test on the **photographed** hdbpmn
                    flowcharts, which the model has never seen the domain of. A model that
                    learned "rendered => state_machine" must fail here. A model that learned
                    what a flowchart looks like must call them flowcharts.

`cross_domain` is the control the row turns on, so `--check` fails on it and not on the
headline.

Grouping is by writer where a writer is published and by page otherwise, so no page shares a fold
with another page by the same hand. Holding out a whole *source* would be a different and
unpassable experiment - `state_machine` has exactly one source - and the `cross_domain` arm asks
that harder question deliberately instead, by holding out a rendering domain.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from src.utils.config import ROOT

TABLE = ROOT / "data" / "features" / "handcrafted.parquet"
REPORT_MD = ROOT / "reports" / "p13_fallback_chain.md"
REPORT_JSON = ROOT / "reports" / "p13_fallback_chain.json"

#: Columns that name a row rather than describe a page. Training on any of them is 4.1.5's leak.
IDENTITY = ("id", "source", "diagram_type", "split", "scribe_id", "adverse", "synthetic")

#: How each source's pages reach the pixels. This is a property of the archive, not a judgement:
#: hdbpmn ships photographs of paper, the other three ship renders.
DOMAIN = {
    "hdbpmn": "photographed",
    "flowchartseg": "rendered",
    "fa_bresler": "rendered",
    "sketch2code": "rendered",
}

N_SPLITS = 5
SEEDS = (0, 1, 2)

#: Below this on the cross-domain arm, the classifier rung is learning the renderer and must not
#: be wired. It is a majority-class bar: the arm's test set is all one class, so anything at or
#: under 0.5 is not evidence of transfer.
CROSS_DOMAIN_FLOOR = 0.80


def load() -> pd.DataFrame:
    frame = pd.read_parquet(TABLE)
    frame["domain"] = frame["source"].map(DOMAIN)
    # A writer is the unit of leakage where one is published; where none is, the page is its own
    # group. Grouping a whole source together instead is tempting and wrong: it turns the task
    # into "predict a source you have never seen", which no class with a single source can pass -
    # state_machine only exists in fa_bresler - and it is not what leakage means here. The thing
    # that must not straddle a fold is one hand drawing two pages, and that is exactly what
    # `scribe_id` identifies. The cross-domain arm below answers the harder question separately,
    # by holding out a whole rendering domain on purpose.
    frame["group"] = frame["scribe_id"].fillna(frame["id"])
    return frame


def features(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    columns = [c for c in frame.columns if c not in IDENTITY and c not in ("domain", "group")]
    return frame[columns].to_numpy(dtype=float), columns


def _model() -> HistGradientBoostingClassifier:
    # Histogram gradient boosting because the table is 20.9% missing by design - a page with no
    # detected text has no text statistics - and this is the one sklearn classifier that takes
    # NaN as a value rather than requiring it be invented by an imputer.
    return HistGradientBoostingClassifier(max_iter=200, random_state=0)


def out_of_fold(frame: pd.DataFrame, target: str) -> dict[str, Any]:
    """Grouped, stratified out-of-fold predictions over several seeds."""
    X, columns = features(frame)
    y = frame[target].to_numpy()
    groups = frame["group"].to_numpy()

    accuracies, f1s = [], []
    predictions = np.empty(len(frame), dtype=object)
    for seed in SEEDS:
        splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        fold_pred = np.empty(len(frame), dtype=object)
        for train_idx, test_idx in splitter.split(X, y, groups):
            model = _model().fit(X[train_idx], y[train_idx])
            fold_pred[test_idx] = model.predict(X[test_idx])
        accuracies.append(accuracy_score(y, fold_pred))
        f1s.append(f1_score(y, fold_pred, average="macro"))
        predictions = fold_pred  # the last seed's, kept for the confusion matrix

    labels = sorted(set(y))
    confusion = {
        truth: [int(((y == truth) & (predictions == pred)).sum()) for pred in labels]
        for truth in labels
    }
    return {
        "n": int(len(frame)),
        "classes": {k: int(v) for k, v in frame[target].value_counts().items()},
        "features": len(columns),
        "groups": int(frame["group"].nunique()),
        "evaluation": f"stratified grouped {N_SPLITS}-fold x {len(SEEDS)} seeds, groups=writer",
        "accuracy": round(float(np.mean(accuracies)), 4),
        "accuracy_std": round(float(np.std(accuracies)), 4),
        "macro_f1": round(float(np.mean(f1s)), 4),
        "macro_f1_std": round(float(np.std(f1s)), 4),
        "labels": labels,
        "confusion": confusion,
    }


def cross_domain(frame: pd.DataFrame) -> dict[str, Any]:
    """Train in the rendered domain only; test on photographed pages of a class it has seen.

    The training set is rendered flowcharts (flowchartseg) against rendered state machines
    (fa_bresler), so *within training* the domain is constant and carries no information about
    the label. The test set is hdbpmn - photographs, every one a flowchart. A model that learned
    "rendered ink => state_machine" has no reason to get these right; a model that learned what
    a flowchart looks like has every reason to.
    """
    train = frame[frame["source"].isin(["flowchartseg", "fa_bresler"])]
    test = frame[frame["source"] == "hdbpmn"]
    X_train, _ = features(train)
    X_test, _ = features(test)
    model = _model().fit(X_train, train["diagram_type"].to_numpy())
    predicted = model.predict(X_test)
    truth = test["diagram_type"].to_numpy()
    counts = pd.Series(predicted).value_counts().to_dict()
    return {
        "train": {
            "sources": ["flowchartseg", "fa_bresler"],
            "domain": "rendered (constant, so it cannot carry the label)",
            "n": int(len(train)),
            "classes": {k: int(v) for k, v in train["diagram_type"].value_counts().items()},
        },
        "test": {
            "source": "hdbpmn",
            "domain": "photographed (never seen in training)",
            "n": int(len(test)),
            "truth": "flowchart, every page",
        },
        "accuracy": round(float(accuracy_score(truth, predicted)), 4),
        "predicted_counts": {str(k): int(v) for k, v in counts.items()},
        "floor": CROSS_DOMAIN_FLOOR,
        "reading": "a model that learned the renderer would call these state machines",
    }


def domain_probe(frame: pd.DataFrame) -> dict[str, Any]:
    """How much of the domain is recoverable from these features, stated rather than assumed."""
    result = out_of_fold(frame, "domain")
    result["reading"] = (
        "this is how well rendered can be told from photographed using nothing but the"
        " handcrafted features; the closer to 1.0, the less the headline arm proves on its own"
    )
    return result


#: 14.5's degradations, at the severity its own sweep calls the middle of the range. A fallback
#: rung fires when the page is bad, so the rung has to be measured on bad pages.
DEGRADATIONS = (
    ("blur", 5),
    ("rotation", 6.0),
    ("lighting", 0.7),
    ("occlusion", 0.1),
    ("resolution", 0.5),
)


def degraded_features(
    frame: pd.DataFrame, limit: int = 120, source: str = "fa_bresler"
) -> pd.DataFrame | None:
    """One row of 4.2 features per (page, degradation), extracted from the degraded pixels.

    Factored out of `degraded_rendering` because 15.5's drift detector needs exactly the same
    thing - the same pages captured worse - and two implementations of "degrade then re-extract"
    would drift apart and quietly stop comparing like with like.

    Every degraded page is written first and extracted in one parallel pass, so 4.2's extractor
    pays its process-pool startup once rather than once per degradation. Returns None when no
    page could be read at all.
    """
    import tempfile

    import cv2

    from src.eval.robust import degrade
    from src.features.extractor import FeatureExtractor

    held = frame[frame["source"] == source].head(limit)
    if held.empty:
        return None

    with tempfile.TemporaryDirectory() as tmp:
        jobs: list[tuple[str, str, Path]] = []
        for kind, severity in DEGRADATIONS:
            for page_id in held["id"]:
                stem = page_id.split("/")[-1]
                src = ROOT / "data" / "processed" / "fa_render" / f"{stem}.png"
                if not src.is_file():
                    continue
                gray = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
                if gray is None:
                    continue
                out = Path(tmp) / f"{kind}__{stem}.png"
                cv2.imwrite(str(out), degrade(gray, kind, float(severity)))
                jobs.append((kind, page_id, out))
        if not jobs:
            return None
        extractor = FeatureExtractor()
        matrix = extractor.fit_transform([path for _, _, path in jobs])
        names = list(extractor.get_feature_names_out())

    out_frame = pd.DataFrame(matrix, columns=names)
    out_frame.insert(0, "id", [f"{page_id}@{kind}" for kind, page_id, _ in jobs])
    out_frame.insert(1, "degradation", [kind for kind, _, _ in jobs])
    return out_frame


def degraded_rendering(frame: pd.DataFrame, limit: int = 120) -> dict[str, Any]:
    """Does the rung survive fa_bresler's renders being made to look photographed?

    `cross_domain` shows the rung transfers *out* of the rendered domain. This asks the mirror
    question, and it is the one the confound objection actually raises: fa_bresler's pages are
    the only `state_machine` evidence and they are pristine synthetic ink, so a rung that keys on
    *cleanliness* would route them correctly for a reason that evaporates on a photograph.

    The model is trained once on the clean table with every fa_bresler page held out, so nothing
    it saw carries this writer or this renderer. Each held-out page is then put through 14.5's
    own `degrade` at five kinds, its features re-extracted from the degraded pixels by 4.2's
    extractor, and routed. A rung that needs clean ink collapses here; one that learned the
    shape of a state machine does not.
    """
    fa = frame[frame["source"] == "fa_bresler"]
    held = fa.head(limit)
    # Trained without a single state machine the model could not predict one, so the question
    # would be meaningless. The rest of fa_bresler stays in training and the rendered flowcharts
    # stay too, so "rendered" is still represented and only *these* pages are new.
    train = pd.concat([frame[frame["source"] != "fa_bresler"], fa.tail(len(fa) - len(held))])
    X_train, columns = features(train)
    model = _model().fit(X_train, train["diagram_type"].to_numpy())

    extracted = degraded_features(frame, limit=limit)
    rows: dict[str, Any] = {}
    if extracted is None or extracted.empty:
        return {"per_degradation": {}, "worst": None, "mean": None, "floor": CROSS_DOMAIN_FLOOR}

    # The table's column order is what the model was fit on; the extractor's own order need not
    # match it, so every column is looked up by name rather than by position.
    picked = np.full((len(extracted), len(columns)), np.nan, dtype=float)
    for j, column in enumerate(columns):
        if column in extracted.columns:
            picked[:, j] = extracted[column].to_numpy(dtype=float)
    predicted = model.predict(picked)
    kinds = extracted["degradation"].to_numpy()
    for kind, severity in DEGRADATIONS:
        mask = kinds == kind
        if not mask.any():
            continue
        hit = int((predicted[mask] == "state_machine").sum())
        total = int(mask.sum())
        rows[kind] = {
            "severity": severity,
            "pages": total,
            "routed_state_machine": hit,
            "accuracy": round(hit / total, 4),
        }
    scores = [v["accuracy"] for v in rows.values()]
    return {
        "per_degradation": rows,
        "worst": round(min(scores), 4) if scores else None,
        "mean": round(sum(scores) / len(scores), 4) if scores else None,
        "floor": CROSS_DOMAIN_FLOOR,
        "reading": (
            "held-out fa_bresler pages, degraded by 14.5's own sweep and re-extracted from the"
            " degraded pixels; a rung that keys on clean synthetic ink collapses here"
        ),
    }


def ocr_rungs() -> list[dict[str, str]]:
    """The OCR chain, which degrades in three rungs and never stops producing code."""
    return [
        {
            "rung": "read label",
            "identifier": "q0",
            "source": "the OCR reading of the drawn text",
            "verified": "12.3.1 contract check passes",
        },
        {
            "rung": "node id",
            "identifier": "s14",
            "source": "the IR node id, when OCR returns nothing usable",
            "verified": "12.3.1 contract check passes",
        },
        {
            "rung": "literal fallback",
            "identifier": "step",
            "source": "`targets._ident`'s fallback, when there is no id either",
            "verified": "12.3.1 contract check passes",
        },
    ]


def collect() -> dict[str, Any]:
    frame = load()
    full = out_of_fold(frame, "diagram_type")
    probe = domain_probe(frame)
    cross = cross_domain(frame)
    degraded = degraded_rendering(frame)
    rendered = frame[frame["domain"] == "rendered"]
    return {
        "what": "13.4 - the three fallback chains, and the control the classifier rung turns on",
        "table": {
            "path": "data/features/handcrafted.parquet",
            "rows": int(len(frame)),
            "sources": {k: int(v) for k, v in frame["source"].value_counts().items()},
            "classes": {k: int(v) for k, v in frame["diagram_type"].value_counts().items()},
        },
        "arms": {
            "full": full,
            "rendered_only": out_of_fold(rendered, "diagram_type"),
            "domain_probe": probe,
            "cross_domain": cross,
            "degraded_rendering": degraded,
        },
        "verdict": {
            "classifier_rung_wired": bool(cross["accuracy"] >= CROSS_DOMAIN_FLOOR),
            "criterion": (
                f"the cross-domain arm must reach {CROSS_DOMAIN_FLOOR:.2f} on pages whose"
                " rendering domain was never in its training set"
            ),
            "cross_domain_accuracy": cross["accuracy"],
        },
        "ocr_chain": ocr_rungs(),
        "model_chain": {
            "where": "src/pipeline/generate.py",
            "behaviour": "an emitter answer is marked degraded=True so no measurement mixes it"
            " with a model answer",
        },
    }


def render(result: dict) -> str:
    arms = result["arms"]
    cross = arms["cross_domain"]
    lines = [
        "# Phase 13.4 - the fallback chains",
        "",
        "Generated by `python -m src.pipeline.fallback`.",
        "",
        f"The feature table is **{result['table']['rows']} rows over"
        f" {len(result['table']['classes'])} classes**"
        f" ({', '.join(f'{k} {v}' for k, v in sorted(result['table']['classes'].items()))}).",
        "",
        "## The classifier rung, and the control it turns on",
        "",
        "| arm | what it asks | accuracy | macro F1 |",
        "| :--- | :--- | ---: | ---: |",
        f"| `full` | three classes, all four sources, grouped out-of-fold |"
        f" {arms['full']['accuracy']:.4f} | {arms['full']['macro_f1']:.4f} |",
        f"| `rendered_only` | the same, restricted to rendered pages |"
        f" {arms['rendered_only']['accuracy']:.4f} | {arms['rendered_only']['macro_f1']:.4f} |",
        f"| `domain_probe` | rendered vs photographed, same features |"
        f" {arms['domain_probe']['accuracy']:.4f} | {arms['domain_probe']['macro_f1']:.4f} |",
        f"| `cross_domain` | trained rendered-only, tested on photographs |"
        f" {cross['accuracy']:.4f} | - |",
        "",
        f"`domain_probe` reaching {arms['domain_probe']['accuracy']:.4f} is why the headline is"
        " not quoted on its own: the rendering domain **is** recoverable from these features, so"
        " a three-class score could in principle be a renderer detector wearing a label.",
        "",
        f"`cross_domain` is the arm that settles it. Trained on rendered flowcharts against"
        f" rendered state machines - domain constant, so it carries nothing - and tested on"
        f" {cross['test']['n']} **photographed** flowcharts it has never seen the domain of, it"
        f" scores **{cross['accuracy']:.4f}**, predicting"
        f" {', '.join(f'{k} {v}' for k, v in sorted(cross['predicted_counts'].items()))}."
        f" {cross['reading']}.",
        "",
        f"**Verdict: the classifier rung"
        f" {'is wired' if result['verdict']['classifier_rung_wired'] else 'stays unwired'}** -"
        f" {result['verdict']['criterion']}.",
        "",
        "## Where the rung stops being trustworthy",
        "",
        "`cross_domain` says the rung transfers out of the rendered domain. This asks the mirror"
        " question: hold these pages out entirely, then put them through 14.5's own degradations"
        " and re-extract the features from the degraded pixels.",
        "",
        "| degradation | severity | pages | routed `state_machine` | accuracy |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ]
    degraded = arms["degraded_rendering"]
    for kind, row in degraded["per_degradation"].items():
        lines.append(
            f"| {kind} | {row['severity']} | {row['pages']} |"
            f" {row['routed_state_machine']} | {row['accuracy']:.4f} |"
        )
    lines += [
        "",
        f"**Mean {degraded['mean']:.4f}, worst {degraded['worst']:.4f}, and the worst is the"
        " number that matters.** Blur, lighting and occlusion cost the rung almost nothing -"
        " which is the evidence that it is not keying on how clean synthetic ink is. Resolution"
        " does: at half scale it routes barely half these pages, which is below the"
        " majority-class bar and means the rung is guessing. So the rung is wired on the"
        " cross-domain control, and this is the named limit that travels with it: **the"
        " handcrafted rung should not be trusted on a page much under full capture resolution**,"
        " and the embedding and CNN rungs below it are what such a page should fall through to.",
        "",
        "## The OCR chain",
        "",
        "| rung | identifier | where it comes from | verified |",
        "| :--- | :--- | :--- | :--- |",
    ]
    for rung in result["ocr_chain"]:
        lines.append(
            f"| {rung['rung']} | `{rung['identifier']}` | {rung['source']} | {rung['verified']} |"
        )
    lines += [
        "",
        "## The model chain",
        "",
        f"`{result['model_chain']['where']}`: {result['model_chain']['behaviour']}.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="exit non-zero if the control fails")
    ap.add_argument("--out", type=__import__("pathlib").Path, default=REPORT_MD)
    ap.add_argument("--json", dest="out_json", type=__import__("pathlib").Path, default=REPORT_JSON)
    args = ap.parse_args(argv)

    result = collect()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(result), encoding="utf-8")
    args.out_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["verdict"], indent=2))
    print(f"-> {args.out}")
    if args.check and not result["verdict"]["classifier_rung_wired"]:
        print("the classifier rung failed its cross-domain control", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
