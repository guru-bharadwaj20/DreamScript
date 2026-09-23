"""Phase 14.2 - every stage scored on its own, with the input it would get from a perfect upstream.

    python -m src.eval.stagewise                 # reports/stagewise.md + reports/stagewise.json
    python -m src.eval.stagewise --skip-fit      # read-only: no cross-validation, no GPU

A pipeline number is a product of six stages, and a product tells you nothing about which factor
is small. This module reports each stage against the input it would receive if everything before
it were perfect - the ground-truth IR for codegen, the annotated crops for OCR, the raw page for
classification and detection - so the numbers are comparable to each other and to the published
per-stage literature. **What an error upstream costs the stage after it is 14.3's question, not
this one**, and the two are kept apart deliberately: a stage-wise table that quietly mixed
predicted and gold inputs would answer neither.

Every entry therefore records `input:` - what the stage was handed - beside its score. That
field is the whole contract of this report.

## Classification is re-measured; the rest is read

Five of the six stages have a canonical scorer whose artefact is already on disk and was
produced by the protocol this row would otherwise repeat, so they are read (and named) rather
than re-run. Classification is re-measured here, and the reason is a finding rather than a
preference: **`reports/s1_heldout_scribes.json` was computed on a corpus that no longer exists**.
It reports 1,340 rows over five classes; `data/features/handcrafted.parquet` after the DVC
rebuild holds **1,435 rows over two** - flowchart and wireframe - because the circuit, ER and
state-machine photographs did not survive. Quoting 0.9871 as this stage's accuracy today would
be quoting a measurement of a different problem, so this module re-runs the same protocol
(repeated GroupKFold by writer, 5 folds x 3 seeds) on the data that exists and reports both, with
the difference stated rather than smoothed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from src.utils.config import ROOT

REPORT_JSON = ROOT / "reports" / "stagewise.json"
REPORT_MD = ROOT / "reports" / "stagewise.md"

#: The models re-measured for the classification stage, by their Phase 5/6 names.
HANDCRAFTED_MODELS = ("majority", "logreg", "knn", "tree")


def _read(path: str, *keys: str) -> Any:
    file = ROOT / path
    if not file.is_file():
        return None
    value: Any = json.loads(file.read_text(encoding="utf-8"))
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _cross_validate(data, factory, seeds, folds: int) -> dict:
    """Out-of-fold accuracy and macro F1, writer-grouped, averaged over seeded repeats."""
    import numpy as np
    from sklearn.base import clone
    from sklearn.metrics import accuracy_score, f1_score

    from src.classify.cv import splitter

    accuracies, macro_f1s = [], []
    for seed in seeds:
        predicted = np.empty(len(data.y), dtype=object)
        for train, test in splitter(data, "grouped", seed, folds):
            model = clone(factory())
            model.fit(data.X[train], data.y[train])
            predicted[test] = model.predict(data.X[test])
        accuracies.append(float(accuracy_score(data.y, predicted)))
        macro_f1s.append(float(f1_score(data.y, predicted, average="macro", zero_division=0)))
    return {
        "accuracy": round(float(np.mean(accuracies)), 4),
        "accuracy_std": round(float(np.std(accuracies)), 4),
        "macro_f1": round(float(np.mean(macro_f1s)), 4),
        "macro_f1_std": round(float(np.std(macro_f1s)), 4),
        "worst_repeat_accuracy": round(float(np.min(accuracies)), 4),
        "seeds": [int(seed) for seed in seeds],
    }


def classify_stage(fit: bool = True) -> dict:
    """The diagram-type stage, on the raw page - the one stage with no upstream to be perfect."""
    from src.classify.cv import SEEDS

    entry: dict[str, Any] = {
        "input": "raw page (no upstream)",
        "published": {
            "s1_accuracy": _read("reports/s1_heldout_scribes.json", "accuracy"),
            "s1_rows": _read("reports/s1_heldout_scribes.json", "rows"),
            "provenance": "reports/s1_heldout_scribes.json",
        },
        "models": {},
    }
    if not fit:
        entry["note"] = "not re-measured (--skip-fit)"
        return entry

    from src.classify.cv import MODELS
    from src.classify.data import load

    handcrafted = load("real")
    entry["corpus"] = handcrafted.summary()
    started = time.time()
    for name in HANDCRAFTED_MODELS:
        entry["models"][name] = {
            "features": "33 handcrafted (4.2.1)",
            **_cross_validate(handcrafted, MODELS[name], SEEDS, 5),
        }

    try:
        from src.classify.s1 import estimator
        from src.embed.hybrid import dataset

        embedded = dataset("embedding", "real")
        entry["models"]["embedding_svm"] = {
            "features": "CLIP ViT-B/32 grayscale, 128 PCA components",
            **_cross_validate(embedded, estimator, SEEDS, 5),
        }
    except (FileNotFoundError, KeyError) as error:
        entry["models"]["embedding_svm"] = {"unavailable": str(error)}

    try:
        from src.pipeline.routing import evaluate

        routed = evaluate("test")
        entry["models"]["router"] = {
            "features": "detector class histogram (13.3 naive Bayes)",
            "accuracy": routed["overall"]["accuracy"],
            "n": routed["overall"]["n"],
            "by_type": {k: v for k, v in routed.items() if k != "overall"},
            "note": "two classes only; the router's label space is the pipeline's, not Phase 5's",
        }
    except Exception as error:  # a missing detector or corpus must not lose the rest
        entry["models"]["router"] = {"unavailable": f"{type(error).__name__}: {error}"}

    entry["fit_seconds"] = round(time.time() - started, 1)
    return entry


def detect_stage() -> dict:
    return {
        "input": "raw page (no upstream)",
        "models": {
            "yolo_components": {
                "map50": _read("reports/s2_component_detection.json", "map50"),
                "hand_drawn_map50": _read(
                    "reports/s2_component_detection.json", "hand_drawn_map50"
                ),
                "pages": _read("reports/s2_component_detection.json", "pages"),
                "target": _read("reports/s2_component_detection.json", "target_map50"),
                "provenance": "reports/s2_component_detection.json",
            },
            "yolov8m_pose_arrows": {
                "box_map50": _read("experiments/detect/arrows/train_pose_m.json", "box_map50"),
                "pose_map50": _read("experiments/detect/arrows/train_pose_m.json", "pose_map50"),
                "epochs": _read("experiments/detect/arrows/train_pose_m.json", "epochs"),
                "provenance": "experiments/detect/arrows/train_pose_m.json",
            },
        },
    }


def ocr_stage() -> dict:
    return {
        "input": "annotated label crops (9.3.1 protocol, writer-disjoint val)",
        "models": {
            "trocr_large_selected": {
                "cer": _read("reports/s3_label_ocr.json", "cer"),
                "exact_match": _read("reports/s3_label_ocr.json", "exact_match"),
                "wer": _read("reports/s3_label_ocr.json", "wer"),
                "crops": _read("reports/s3_label_ocr.json", "crops"),
                "target": _read("reports/s3_label_ocr.json", "target_cer"),
                "provenance": "reports/s3_label_ocr.json",
            },
            "trocr_large_old_selection": {
                "cer": _read("reports/s3_label_ocr_large.json", "cer"),
                "provenance": "reports/s3_label_ocr_large.json",
            },
        },
    }


def parse_stage() -> dict:
    return {
        "input": "ground-truth IR sequences (gold nodes and edges)",
        "models": {
            "hmm_roles": {
                "macro_f1": _read("reports/s4_role_labelling.json", "macro_f1"),
                "accuracy": _read("reports/s4_role_labelling.json", "accuracy"),
                "sequences": _read("reports/s4_role_labelling.json", "sequences"),
                "target": _read("reports/s4_role_labelling.json", "target_macro_f1"),
                "provenance": "reports/s4_role_labelling.json",
            }
        },
    }


def assemble_stage() -> dict:
    return {
        "input": "predicted detections and predicted text (no gold upstream available)",
        "note": "the only stage whose canonical scorer already runs on predicted input; 14.3"
        " prices what that costs against the gold-IR codegen number below",
        "models": {
            "assembly_test": {
                "median_ged": _read("reports/s5_test_large.json", "overall", "median_ged"),
                "mean_ged": _read("reports/s5_test_large.json", "overall", "mean_ged"),
                "pass_share": _read("reports/s5_test_large.json", "overall", "pass_share"),
                "pages": _read("reports/s5_test_large.json", "overall", "pages"),
                "target": _read("reports/s5_test_large.json", "target_median_ged"),
                "provenance": "reports/s5_test_large.json",
            },
            "assembly_val": {
                "median_ged": _read("reports/s5_graph_ged.json", "overall", "median_ged"),
                "provenance": "reports/s5_graph_ged.json",
            },
        },
    }


def codegen_stage() -> dict:
    return {
        "input": "ground-truth IR + traversal (12.1.1 pairs), never a predicted graph",
        "models": {
            "lora_r64": {
                "functional": _read("experiments/llm/quality/done.json", "lora", "functional"),
                "executes": _read("experiments/llm/quality/done.json", "lora", "executes"),
                "syntax": _read("experiments/llm/quality/done.json", "lora", "syntax"),
                "structural": _read("experiments/llm/structural/done.json", "lora", "structural"),
                "n": _read("experiments/llm/quality/done.json", "lora", "n"),
                "provenance": "experiments/llm/quality/done.json",
            },
            "reference_emitter": {
                "functional": _read("experiments/llm/quality/done.json", "reference", "functional"),
                "provenance": "experiments/llm/quality/done.json",
            },
            "zero_shot": {
                "functional": _read(
                    "experiments/llm/compare/done.json",
                    "results",
                    "zero-shot",
                    "summary",
                    "functional",
                ),
                "provenance": "experiments/llm/compare/done.json",
            },
            "few_shot_k2": {
                "functional": _read(
                    "experiments/llm/compare/done.json",
                    "results",
                    "few-shot (k=2)",
                    "summary",
                    "functional",
                ),
                "provenance": "experiments/llm/compare/done.json",
            },
        },
    }


def collect(fit: bool = True) -> dict:
    return {
        "classify": classify_stage(fit),
        "detect": detect_stage(),
        "ocr": ocr_stage(),
        "parse": parse_stage(),
        "assemble": assemble_stage(),
        "codegen": codegen_stage(),
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "*missing*"
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def render(result: dict) -> str:
    lines = [
        "# Phase 14.2 - stage-wise accuracy, each stage on its own input",
        "",
        "Generated by `python -m src.eval.stagewise`. Each stage is scored against the input it"
        " would receive from a perfect upstream, and the `input` line says which that is. What an"
        " upstream error costs downstream is 14.3, deliberately not mixed in here.",
        "",
    ]
    for stage, entry in result.items():
        lines += [f"## {stage}", "", f"*input:* {entry['input']}", ""]
        if entry.get("note"):
            lines += [entry["note"], ""]
        metrics: list[str] = []
        for model in entry["models"].values():
            for key in model:
                if key in ("provenance", "features", "note", "by_type", "seeds", "unavailable"):
                    continue
                if key not in metrics:
                    metrics.append(key)
        lines += [
            "| model | " + " | ".join(metrics) + " | provenance |",
            "| :--- |" + " ---: |" * len(metrics) + " :--- |",
        ]
        for name, model in entry["models"].items():
            if "unavailable" in model:
                lines.append(
                    f"| {name} |" + " *unavailable* |" * len(metrics) + f" {model['unavailable']} |"
                )
                continue
            cells = " | ".join(_fmt(model.get(key)) for key in metrics)
            source = model.get("provenance", "re-measured here")
            lines.append(f"| {name} | {cells} | `{source}` |")
        lines.append("")
        if stage == "classify" and entry.get("corpus"):
            published = entry["published"]
            lines += [
                f"**The published S1 figure is {_fmt(published['s1_accuracy'])} over"
                f" {published['s1_rows']} rows and five classes; the corpus that survived the DVC"
                f" rebuild is {entry['corpus']['rows']} rows over"
                f" {len(entry['corpus']['classes'])}"
                f" ({', '.join(entry['corpus']['classes'])}).** Every accuracy in the table above"
                " is the two-class problem, which is easier, and is reported here rather than"
                " compared to a number measured on a corpus that no longer exists.",
                "",
            ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-fit", action="store_true", help="read artefacts only; no CV, no GPU")
    ap.add_argument("--out", type=Path, default=REPORT_MD)
    ap.add_argument("--json", dest="out_json", type=Path, default=REPORT_JSON)
    args = ap.parse_args(argv)

    result = collect(fit=not args.skip_fit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(result), encoding="utf-8")
    args.out_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({stage: list(entry["models"]) for stage, entry in result.items()}, indent=2))
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
