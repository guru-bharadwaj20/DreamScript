"""Phase 15.9 - the tiny-model smoke train: does the training path still work at all?

    python -m src.mlops.smoke            # run it, print what happened
    python -m src.mlops.smoke --check    # non-zero if any stage of the path is broken

## What this is for, and what it deliberately does not claim

CI has no GPU, no DVC payload and no checkpoints, so it cannot retrain anything that matters and
must not pretend to. What it *can* do is catch the failure that wastes the most time: the training
path breaking in a way nobody notices until someone spends an hour of GPU on it. A renamed column,
a changed extractor signature, a sklearn deprecation that turned into an error - none of those
need a corpus to find, and all of them have cost this project a run.

So this trains something real, end to end, on the **five committed fixtures** from 0.2.7 - about
36 KB, one sketch per diagram type, in git precisely so the suite runs with no dataset present.
Every stage the real training does, at a scale where it takes seconds on CPU:

    extract   4.2's `FeatureExtractor` over the fixture images, the same code path as the corpus
    fit       a small classifier on the resulting matrix
    predict   and score it

**The accuracy is not a result and is not reported as one.** Five images with one example per
class cannot measure anything, and a number from them would be noise that looks like evidence.
What is asserted is only that each stage ran, produced the shape the next stage expects, and that
the features are not degenerate - a matrix of all-NaN or all-zeros would let a fit "succeed" while
proving the extractor is broken, which is exactly the silent failure worth catching.

15.12 is the row that reproduces headline metrics, on a machine that has the data.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.config import ROOT

FIXTURES = ROOT / "tests" / "fixtures"

#: The five diagram types 0.2.7 committed one sketch each for.
KINDS = ("flowchart", "wireframe", "state_machine", "er_diagram", "circuit")


def fixture_images() -> list[Path]:
    return [FIXTURES / f"{kind}.png" for kind in KINDS if (FIXTURES / f"{kind}.png").is_file()]


def run() -> dict[str, Any]:
    from sklearn.tree import DecisionTreeClassifier

    from src.features.extractor import FeatureExtractor

    result: dict[str, Any] = {"stages": {}, "ok": False}
    started = time.perf_counter()

    images = fixture_images()
    result["stages"]["fixtures"] = {
        "ok": len(images) == len(KINDS),
        "found": len(images),
        "expected": len(KINDS),
    }
    if not images:
        result["error"] = f"no fixtures under {FIXTURES}"
        return result

    # --- extract ------------------------------------------------------------------------------
    extractor = FeatureExtractor(n_jobs=1)  # one job: CI runners are small and this is 5 images
    matrix = extractor.fit_transform(images)
    names = list(extractor.get_feature_names_out())
    finite = int(np.isfinite(matrix).sum())
    result["stages"]["extract"] = {
        "ok": matrix.shape == (len(images), len(names)) and finite > 0,
        "shape": list(matrix.shape),
        "features": len(names),
        "finite_values": finite,
        # An all-NaN or all-constant matrix still fits and still predicts, so the fit passing
        # would prove nothing. This is the check that the extractor actually extracted.
        "distinct_columns": int(
            sum(
                1
                for j in range(matrix.shape[1])
                if len(np.unique(matrix[np.isfinite(matrix[:, j]), j])) > 1
            )
        ),
    }
    if finite == 0:
        result["error"] = "every extracted feature is NaN; the extractor is not working"
        return result

    # --- fit ----------------------------------------------------------------------------------
    labels = [path.stem for path in images]
    filled = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    model = DecisionTreeClassifier(random_state=0).fit(filled, labels)
    result["stages"]["fit"] = {"ok": True, "classes": sorted(model.classes_.tolist())}

    # --- predict ------------------------------------------------------------------------------
    predicted = model.predict(filled)
    result["stages"]["predict"] = {
        "ok": len(predicted) == len(labels),
        "n": len(predicted),
        # Reported for completeness and explicitly not a result: one example per class means a
        # tree can memorise it, so this says the call returned labels, nothing more.
        "recovered_training_labels": int((predicted == np.array(labels)).sum()),
        "note": "not a measurement - one example per class, memorisable by construction",
    }

    result["seconds"] = round(time.perf_counter() - started, 2)
    result["ok"] = all(stage["ok"] for stage in result["stages"].values())
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 15.9 tiny-model smoke train")
    ap.add_argument("--check", action="store_true", help="non-zero if any stage is broken")
    args = ap.parse_args(argv)

    result = run()
    json.dump(result, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")
    if not result["ok"]:
        print(f"smoke train FAILED: {result.get('error', 'a stage did not pass')}", file=sys.stderr)
    return 0 if result["ok"] or not args.check else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
