"""Phase 7.4.8 - the learned vocabulary against the hand-written one, and against the ceiling.

    python -m src.parse.templates

4.1.3's `classify()` is the declared shape vocabulary: a short decision tree of thresholds on
extent, corner count and aspect, written by hand from the 2.2.4 crops. 7.4.2 fitted a learned one.
This task scores them against the same 12,400 hdbpmn labels.

## Three predictors, because two would be unreadable

    template     4.1.3's `classify()`, unchanged, on regions rebuilt from each node's own crop.
                 No fitting of any kind.
    learned      7.4.2's mixture. Fitted unsupervised on the training fold, its components then
                 named by a Hungarian assignment against the training labels, and applied to the
                 held-out fold. The naming step uses labels and is declared as such - a mixture
                 with anonymous components cannot be scored against named ones at all.
    supervised   a random forest on the same 22 descriptors. Not part of the plan's comparison;
                 it is the **ceiling**, and without it the other two numbers cannot be read.

The ceiling is the point. If template and learned both score far below supervised, the finding is
about the descriptors rather than about either vocabulary, and reporting "learned beats hand-written
by X" would be reporting noise between two things that are both failing.

## Folds are grouped by page

A page contributes many nodes drawn by one person in one sitting. Splitting by node would put a
scribe's own rectangles on both sides of the fold and flatter every fitted model; 7.4.7 measured
that scribe identity is recoverable from geometry at 8.6x chance, so the leak would be real.
`template` is unaffected - it fits nothing - which is exactly why it is the honest baseline.

## The `ellipse` problem, stated rather than patched

`classify()` can return `ellipse`, and 7.4.1 measured **zero `ellipse` labels in hdbpmn**. Every
such prediction is therefore wrong by construction. Both figures are reported: `strict` scores the
raw output, and `collapsed` maps `ellipse` to `circle` before scoring, on the grounds that the two
are one round family and the distinction is not one this corpus can express. Quoting only the
collapsed number would be flattering the template; quoting only the strict one would be punishing
it for a vocabulary mismatch rather than for a bad rule.

## What it measured

12,400 shapes, 693 pages, K = 6, five folds grouped by page. Macro F1, `collapsed`:

    predictor     accuracy   macro F1   share of ceiling
    supervised     0.8815     0.8160         -
    learned        0.5218     0.4160        0.510
    template       0.4071     0.3734        0.458

    majority baseline (always `rectangle`)  0.4796 accuracy

**The learned vocabulary beats the hand-written one by 0.0426 macro F1 - and both leave about half
the available signal on the floor.** A random forest on the *same 22 descriptors* reaches 0.8160.
So the honest headline is not "learned beats template"; it is that **the descriptors support a
0.82 vocabulary and neither candidate delivers one.**

That is what the ceiling row was included for. Reporting +0.0426 on its own would be reporting the
gap between two things that are both failing, and inviting the conclusion that shape naming on this
corpus is simply hard. It is not hard: it is unsolved by these two methods.

## Where each one's win comes from, and they are different places

    class        template   learned   supervised
    rectangle     0.5001    0.7086     0.9256
    diamond       0.6245    0.3275     0.8967
    circle        0.2758    0.4961     0.8750
    freeform      0.0932    0.1319     0.5666

**The template is the better diamond detector (0.6245 against 0.4160's 0.3275)** and it is the only
place a hand-written rule clearly wins. 4.1.3's `DIAMOND_EXTENT` band is a threshold on exactly the
axis-aligned `extent` that 7.4.1 identified as the one rotation-sensitive column, applied to the one
shape whose signature is rotation - a rule written by someone who had looked at the pictures. The
mixture has no such prior and splits diamonds across components that are mostly about outline
quality, so it loses nearly half the class.

The learned vocabulary wins on `rectangle` (+0.21) and `circle` (+0.22), which are the two largest
classes and the two whose descriptor clouds are big enough for a component to land on.

**Both fail on `freeform` - 0.09 and 0.13 - and the supervised model gets 0.5666.** 7.4.5 predicted
the first half: freeform's measured means sit between every other class on every separating column,
so no threshold and no component centre can claim it. What 7.4.5 could not predict is that a
supervised model finds it anyway, at five times either score. Freeform is separable in this table;
it is just not separable by a *rule* or by a *cluster centre*, because what identifies it is a
conjunction rather than a region.

## The `ellipse` cost, reported both ways

    template, strict     0.2934
    template, collapsed  0.3734

**Predicting a class the corpus does not contain costs the template 0.08 macro F1** - a fifth of its
score - and none of that is a bad shape rule. It is a vocabulary written against a different corpus,
which is a real deployment failure mode and a different one from being wrong about geometry. The
learned predictor cannot make this mistake by construction: its names come from a Hungarian
assignment against training labels, so it can only emit names that exist.

## What this settles

Neither vocabulary should be shipped as a shape classifier. **If Phase 10 needs a shape name from
pixels, the supervised model on 7.4.1's descriptors is the thing to use** - it roughly doubles
both candidates, at 0.8160 against 0.4160 and 0.3734. The
plan's framing, learned against hardcoded, turns out to be a comparison between two ways of *not*
using the labels, on a table where the labels are what makes the problem tractable.

7.4.6 already showed that Phase 10 does not need this at all where an annotation exists. This task
prices what to do where one does not.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from src.features.descriptors import CROP_PAD, NAMES

SEED = 42

#: `classify()` names that have no rows in hdbpmn, and what they collapse onto when `collapsed`.
COLLAPSE = {"ellipse": "circle"}

PREDICTORS = ("template", "learned", "supervised")


def _one_page_regions(page_id: str) -> dict:
    """`element id` -> 4.1.3's `classify()` verdict, one decode of the page."""
    import cv2

    from src.features import context as ctx
    from src.features.descriptors import ROOT
    from src.features.shapes import classify
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
    scale = gray.shape[1] / original.shape[1]

    out = {}
    for node in diagram.nodes:
        if not node.bbox:
            continue
        x, y, w, h = (value * scale for value in node.bbox)
        pad_x, pad_y = CROP_PAD * w, CROP_PAD * h
        x0, y0 = max(0, int(x - pad_x)), max(0, int(y - pad_y))
        x1 = min(mask.shape[1], int(x + w + pad_x))
        y1 = min(mask.shape[0], int(y + h + pad_y))
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue
        patch = mask[y0:y1, x0:x1].astype(np.uint8)
        patch = cv2.copyMakeBorder(patch, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
        regions = ctx.build_regions(patch.astype(bool), min_area_frac=0.02)
        if not regions:
            # 4.1.3 does the same: nothing closed means nothing recognisable.
            out[node.id] = "freeform"
            continue
        out[node.id] = classify(max(regions, key=lambda r: r.area))
    return out


def template_predictions(pages, n_jobs: int | None = None) -> dict:
    """`page:element` -> template verdict, over every page named."""
    from src.utils.parallel import pmap

    results = pmap(_one_page_regions, list(pages), n_jobs=n_jobs)
    return {
        f"{page}:{element}": verdict
        for page, verdicts in zip(pages, results, strict=True)
        for element, verdict in verdicts.items()
    }


def learned_predictions(X, labels, groups, k: int, folds: int = 5) -> np.ndarray:
    """Cross-validated mixture predictions, components named on the training fold only."""
    from scipy.optimize import linear_sum_assignment
    from sklearn.model_selection import GroupKFold

    from src.parse.vocab import fit

    names = sorted(set(labels))
    predicted = np.empty(len(labels), dtype=object)
    for train, test in GroupKFold(folds).split(X, labels, groups=groups):
        model = fit(X[train], k, "full")
        scaler, gmm = model.named_steps["scale"], model.named_steps["gmm"]
        assigned = gmm.predict(scaler.transform(X[train]))

        table = np.zeros((k, len(names)))
        for component, label in zip(assigned, labels[train], strict=True):
            table[component, names.index(label)] += 1
        # Hungarian rather than greedy, for 7.4.2's reason: two components must not both claim
        # the same name. With k > len(names) the surplus components take their own best label.
        rows, columns = linear_sum_assignment(-table)
        mapping = {int(r): names[c] for r, c in zip(rows, columns, strict=True)}
        for component in range(k):
            mapping.setdefault(component, names[int(table[component].argmax())])

        for position, component in zip(test, gmm.predict(scaler.transform(X[test])), strict=True):
            predicted[position] = mapping[int(component)]
    return predicted


def supervised_predictions(X, labels, groups, folds: int = 5) -> np.ndarray:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", RandomForestClassifier(n_estimators=400, random_state=SEED, n_jobs=-1)),
        ]
    )
    return cross_val_predict(model, X, labels, cv=GroupKFold(folds), groups=groups, n_jobs=1)


def score(predicted, truth, collapsed: bool = False) -> dict:
    from sklearn.metrics import accuracy_score, f1_score

    if collapsed:
        predicted = np.array([COLLAPSE.get(p, p) for p in predicted], dtype=object)
    labels = sorted(set(truth))
    return {
        "accuracy": round(float(accuracy_score(truth, predicted)), 4),
        "macro_f1": round(float(f1_score(truth, predicted, average="macro", zero_division=0)), 4),
        "per_class_f1": {
            name: round(float(v), 4)
            for name, v in zip(
                labels,
                f1_score(truth, predicted, average=None, labels=labels, zero_division=0),
                strict=True,
            )
        },
        "predicted_names": sorted({str(p) for p in predicted}),
    }


def run(k: int = 6, folds: int = 5, n_jobs: int | None = None) -> dict:
    from src.features.descriptors import load_table

    table = load_table()
    X = table[list(NAMES)].to_numpy(dtype=float)
    keep = np.isfinite(X).all(axis=1)
    table, X = table[keep], X[keep]
    labels = table["label"].to_numpy()
    groups = table["page"].to_numpy()
    keys = table["key"].to_numpy()

    verdicts = template_predictions(sorted(set(groups)), n_jobs)
    template = np.array([verdicts.get(key, "freeform") for key in keys], dtype=object)
    missing = int(sum(1 for key in keys if key not in verdicts))

    predictions = {
        "template": template,
        "learned": learned_predictions(X, labels, groups, k, folds),
        "supervised": supervised_predictions(X, labels, groups, folds),
    }

    rows = {}
    for name, predicted in predictions.items():
        rows[name] = {
            "strict": score(predicted, labels),
            "collapsed": score(predicted, labels, collapsed=True),
        }

    best = max(rows, key=lambda n: rows[n]["collapsed"]["macro_f1"])
    ceiling = rows["supervised"]["collapsed"]["macro_f1"]
    return {
        "rows": int(len(labels)),
        "pages": int(len(set(groups))),
        "k": k,
        "template_nodes_without_a_region": missing,
        "majority_baseline": round(float(max(np.mean(labels == c) for c in set(labels))), 4),
        "best": best,
        "learned_minus_template": round(
            rows["learned"]["collapsed"]["macro_f1"] - rows["template"]["collapsed"]["macro_f1"], 4
        ),
        "template_share_of_ceiling": round(
            rows["template"]["collapsed"]["macro_f1"] / ceiling if ceiling else 0.0, 4
        ),
        "learned_share_of_ceiling": round(
            rows["learned"]["collapsed"]["macro_f1"] / ceiling if ceiling else 0.0, 4
        ),
        "by_predictor": rows,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=None)
    args = ap.parse_args(argv)
    try:
        print(json.dumps(run(args.k, args.folds, args.jobs), indent=2))
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
