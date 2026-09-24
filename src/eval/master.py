"""Phase 14.1 - the master results table: every model this project trained, every headline metric.

    python -m src.eval.master                 # reports/master_results.md + .json

Fourteen phases produced numbers in fourteen different files, and the only place they have ever
appeared together is prose in `contributing.md`. Prose drifts: a rerun updates the JSON and the
sentence keeps the old figure. So this module does not *contain* any result. It declares where
each number lives - a file, and a path inside it - reads them at run time, and prints the table.
A number that moves on disk moves here on the next run, and a number whose artefact is missing
is printed as `missing` with the section that says how to rebuild it, never as a blank or a
stale value.

## Two kinds of source, both provenance-carrying

* `Json(path, *keys)` - the headline criteria (`reports/s*.json`) and the Phase 12 experiment
  ledgers (`experiments/llm/*/done.json`) are already machine-readable.
* `MdCell(path, header, row, column)` - the Phase 5 and 9.3 model comparisons exist only as a
  markdown table in a report. Parsing is strict: the header must match and the row must be
  present, or the cell reads `missing`, because a silently-renamed column is exactly the drift
  this module exists to catch.

Every emitted row carries its own provenance string, so the table can be audited back to the
file that produced it without reading this source.

## What is deliberately absent

The Phase 6-8 model sweeps (MLP, SVM kernels, forests, boosting, clustering) wrote their
artefacts to gitignored `experiments/` directories that the DVC loss took with them, so they
have no durable file to read. They are listed in the report's own "not durable" section with
the command that rebuilds each, rather than transcribed from prose into a table that would then
claim an authority it does not have. 14.2 re-measures the ones that are cheap to rerun and
writes `reports/stagewise.json`; entries reading from it appear here automatically once it
exists.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils.config import ROOT

REPORT_MD = ROOT / "reports" / "master_results.md"
REPORT_JSON = ROOT / "reports" / "master_results.json"

MISSING = "missing"


@dataclass(frozen=True)
class Json:
    """A value at a key path inside a JSON artefact."""

    path: str
    keys: tuple[str, ...]

    def __init__(self, path: str, *keys: str) -> None:
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "keys", tuple(keys))

    @property
    def provenance(self) -> str:
        return f"{self.path}:{'.'.join(self.keys)}"

    def read(self, root: Path) -> Any:
        file = root / self.path
        if not file.is_file():
            return None
        value: Any = json.loads(file.read_text(encoding="utf-8"))
        for key in self.keys:
            if not isinstance(value, dict) or key not in value:
                return None
            value = value[key]
        return value


@dataclass(frozen=True)
class MdCell:
    """A cell of a markdown table, located by its header row, its first column and a column name."""

    path: str
    header: str
    row: str
    column: str

    @property
    def provenance(self) -> str:
        return f"{self.path}:[{self.row}].{self.column}"

    def read(self, root: Path) -> Any:
        file = root / self.path
        if not file.is_file():
            return None
        table = _find_table(file.read_text(encoding="utf-8"), self.header)
        if table is None:
            return None
        head, rows = table
        if self.column not in head:
            return None
        index = head.index(self.column)
        for cells in rows:
            if _plain(cells[0]) != self.row or index >= len(cells):
                continue
            return _number(cells[index])
        return None


def _split_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _plain(text: str) -> str:
    return text.strip().strip("`*").strip()


def _number(text: str) -> float | str | None:
    body = _plain(text).replace(",", "")
    percent = body.endswith("%")
    body = body.rstrip("%")
    try:
        value = float(body)
    except ValueError:
        return _plain(text) or None
    return value / 100.0 if percent else value


def _is_rule(line: str) -> bool:
    body = line.replace("|", "").strip()
    return bool(body) and not (set(body) - set(" -:"))


def _find_table(text: str, header: str) -> tuple[list[str], list[list[str]]] | None:
    """The first table whose header row starts with ``header``, as (columns, rows)."""
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if not line.startswith("|"):
            continue
        cells = _split_row(line)
        if not cells or _plain(cells[0]) != header:
            continue
        if i + 1 >= len(lines) or not _is_rule(lines[i + 1]):
            continue
        rows = []
        for body in lines[i + 2 :]:
            if not body.startswith("|"):
                break
            rows.append(_split_row(body))
        return [_plain(cell) for cell in cells], rows
    return None


@dataclass(frozen=True)
class Row:
    stage: str
    model: str
    metric: str
    source: Json | MdCell
    target: float | None = None
    higher_is_better: bool = True
    criterion: str | None = None


# --------------------------------------------------------------------------------------
# The registry. Each row names a file and a path inside it; no result is written here.
# --------------------------------------------------------------------------------------

S1 = "reports/s1_heldout_scribes.json"
S2 = "reports/s2_component_detection.json"
S3 = "reports/s3_label_ocr.json"
S4 = "reports/s4_role_labelling.json"
S5V = "reports/s5_graph_ged.json"
S5T = "reports/s5_test_large.json"
QUALITY = "experiments/llm/quality/done.json"
COMPARE = "experiments/llm/compare/done.json"
STRUCT = "experiments/llm/structural/done.json"
SIM = "experiments/llm/similarity/done.json"
REPAIR = "experiments/llm/repair/done.json"
LATENCY = "experiments/llm/latency/done.json"
GOLDEN = "runs/p13_golden/summary.json"
CACHE = "reports/p13_cache.json"
CLFREPORT = "reports/classification_report.md"
OCRMD = "reports/ocr.md"
STAGEWISE = "reports/stagewise.json"

ROWS: tuple[Row, ...] = (
    # ---- Stage 1: diagram-type classification -----------------------------------------
    Row(
        "classify",
        "CLIP ViT-B/32 + OvO RBF SVM",
        "accuracy",
        Json(S1, "accuracy"),
        0.92,
        criterion="S1",
    ),
    Row(
        "classify",
        "CLIP ViT-B/32 + OvO RBF SVM",
        "accuracy std",
        Json(S1, "accuracy_std"),
        higher_is_better=False,
    ),
    Row(
        "classify",
        "CLIP ViT-B/32 + OvO RBF SVM",
        "worst repeat accuracy",
        Json(S1, "minimum_repeat_accuracy"),
    ),
    Row("classify", "CLIP ViT-B/32 + OvO RBF SVM", "macro F1", Json(S1, "macro_f1")),
    Row(
        "classify",
        "handcrafted logreg",
        "macro F1",
        MdCell(CLFREPORT, "model", "logreg", "macro F1"),
    ),
    Row(
        "classify",
        "handcrafted logreg",
        "balanced accuracy",
        MdCell(CLFREPORT, "model", "logreg", "balanced accuracy"),
    ),
    Row("classify", "handcrafted knn", "macro F1", MdCell(CLFREPORT, "model", "knn", "macro F1")),
    Row(
        "classify",
        "handcrafted knn",
        "balanced accuracy",
        MdCell(CLFREPORT, "model", "knn", "balanced accuracy"),
    ),
    Row("classify", "handcrafted tree", "macro F1", MdCell(CLFREPORT, "model", "tree", "macro F1")),
    Row(
        "classify",
        "handcrafted tree",
        "balanced accuracy",
        MdCell(CLFREPORT, "model", "tree", "balanced accuracy"),
    ),
    Row(
        "classify",
        "majority baseline",
        "macro F1",
        MdCell(CLFREPORT, "model", "majority", "macro F1"),
    ),
    Row(
        "classify",
        "NB pipeline router",
        "test accuracy",
        Json(STAGEWISE, "classify", "models", "router", "accuracy"),
    ),
    # ---- Stage 2: component detection --------------------------------------------------
    Row("detect", "YOLO component detector", "mAP@0.5", Json(S2, "map50"), 0.80, criterion="S2"),
    Row("detect", "YOLO component detector", "hand-drawn mAP@0.5", Json(S2, "hand_drawn_map50")),
    # ---- Stage 3: label OCR ------------------------------------------------------------
    Row(
        "ocr",
        "trocr_large + learned selection (incumbent)",
        "CER",
        Json(S3, "cer"),
        0.15,
        higher_is_better=False,
        criterion="S3",
    ),
    Row(
        "ocr", "trocr_large + learned selection (incumbent)", "exact match", Json(S3, "exact_match")
    ),
    Row(
        "ocr",
        "trocr_large + learned selection (incumbent)",
        "WER",
        Json(S3, "wer"),
        higher_is_better=False,
    ),
    Row(
        "ocr",
        "trocr_large, old selection",
        "CER",
        Json("reports/s3_label_ocr_large.json", "cer"),
        higher_is_better=False,
    ),
    Row(
        "ocr",
        "trocr_large, 28 epochs",
        "CER",
        Json("reports/s3_label_ocr_large28.json", "cer"),
        higher_is_better=False,
    ),
    Row(
        "ocr",
        "trocr_large, strong augmentation",
        "CER",
        Json("reports/s3_label_ocr_aug.json", "cer"),
        higher_is_better=False,
    ),
    Row(
        "ocr",
        "trocr_large, retrained on reselection",
        "CER",
        Json("reports/s3_label_ocr_resel_retrain.json", "cer"),
        higher_is_better=False,
    ),
    Row(
        "ocr",
        "trocr_fa (fa_bresler specialist)",
        "best dev CER",
        Json("reports/s3_fa_specialist.json", "best_dev_cer"),
        higher_is_better=False,
    ),
    Row(
        "ocr",
        "crnn_finetune (9.3.6)",
        "CER",
        MdCell(OCRMD, "model", "crnn_finetune", "CER"),
        higher_is_better=False,
    ),
    Row(
        "ocr",
        "crnn_scratch (9.3.6)",
        "CER",
        MdCell(OCRMD, "model", "crnn_scratch", "CER"),
        higher_is_better=False,
    ),
    Row(
        "ocr",
        "trocr_zero_shot (9.3.6)",
        "CER",
        MdCell(OCRMD, "model", "trocr_zero_shot", "CER"),
        higher_is_better=False,
    ),
    # ---- Stage 4: semantic roles -------------------------------------------------------
    Row(
        "parse",
        "HMM + per-component Viterbi",
        "macro F1",
        Json(S4, "macro_f1"),
        0.80,
        criterion="S4",
    ),
    Row(
        "parse",
        "HMM + per-component Viterbi",
        "macro F1 std",
        Json(S4, "macro_f1_std"),
        higher_is_better=False,
    ),
    Row("parse", "HMM + per-component Viterbi", "accuracy", Json(S4, "accuracy")),
    # ---- Stage 5: graph assembly -------------------------------------------------------
    Row(
        "assemble",
        "full assembly (val)",
        "median GED",
        Json(S5V, "overall", "median_ged"),
        3.0,
        higher_is_better=False,
        criterion="S5",
    ),
    Row(
        "assemble",
        "full assembly (test)",
        "median GED",
        Json(S5T, "overall", "median_ged"),
        3.0,
        higher_is_better=False,
        criterion="S5",
    ),
    Row(
        "assemble",
        "full assembly (test)",
        "mean GED",
        Json(S5T, "overall", "mean_ged"),
        higher_is_better=False,
    ),
    Row(
        "assemble", "full assembly (test)", "pass share at <= 3", Json(S5T, "overall", "pass_share")
    ),
    Row(
        "assemble",
        "yolov8m-pose arrows (test)",
        "median GED",
        Json("reports/s5_test_posem.json", "overall", "median_ged"),
        higher_is_better=False,
    ),
    Row(
        "assemble",
        "yolov8s arrows (test)",
        "median GED",
        Json("reports/s5_test_ranker.json", "overall", "median_ged"),
        higher_is_better=False,
    ),
    # ---- Stage 6: code synthesis -------------------------------------------------------
    Row(
        "codegen",
        "Qwen2.5-Coder-7B zero-shot",
        "functional pass@1",
        Json(COMPARE, "results", "zero-shot", "summary", "functional"),
    ),
    Row(
        "codegen",
        "Qwen2.5-Coder-7B few-shot k=2",
        "functional pass@1",
        Json(COMPARE, "results", "few-shot (k=2)", "summary", "functional"),
    ),
    Row(
        "codegen",
        "LoRA r=64",
        "functional pass@1",
        Json(QUALITY, "lora", "functional"),
        0.70,
        criterion="S7",
    ),
    Row(
        "codegen", "LoRA r=64", "executes", Json(QUALITY, "lora", "executes"), 0.85, criterion="S6"
    ),
    Row("codegen", "LoRA r=64", "syntax", Json(QUALITY, "lora", "syntax"), 0.95),
    Row("codegen", "LoRA r=64", "contract", Json(QUALITY, "lora", "contract")),
    Row(
        "codegen",
        "LoRA r=64",
        "dropped-node rate",
        Json(QUALITY, "lora", "dropped_node_rate"),
        higher_is_better=False,
    ),
    Row(
        "codegen",
        "LoRA r=64",
        "invented per program",
        Json(QUALITY, "lora", "invented_per_program"),
        higher_is_better=False,
    ),
    Row("codegen", "LoRA r=64", "structural fidelity", Json(STRUCT, "lora", "structural")),
    Row("codegen", "LoRA r=64", "CodeBLEU", Json(SIM, "lora", "codebleu")),
    Row("codegen", "LoRA r=64", "exact match", Json(SIM, "lora", "exact_match")),
    Row("codegen", "LoRA r=64", "edit similarity", Json(SIM, "lora", "edit_similarity")),
    Row("codegen", "LoRA r=64 + one repair turn", "functional pass@1", Json(REPAIR, "pass_after")),
    Row(
        "codegen",
        "12.1.6 reference emitter",
        "functional pass@1",
        Json(QUALITY, "reference", "functional"),
    ),
    Row(
        "codegen",
        "12.1.6 reference emitter",
        "structural fidelity",
        Json(STRUCT, "reference", "structural"),
    ),
    # ---- Serving and the whole pipeline ------------------------------------------------
    Row(
        "serve",
        "HF NF4 + LoRA",
        "total p50 s",
        Json(LATENCY, "summary", "HF NF4 + LoRA (batch 1, streaming)", "total_p50_s"),
        higher_is_better=False,
    ),
    Row(
        "serve",
        "llama.cpp Q4_K_M",
        "total p50 s",
        Json(LATENCY, "summary", "llama.cpp Q4_K_M (batch 1, streaming)", "total_p50_s"),
        8.0,
        higher_is_better=False,
    ),
    Row(
        "serve",
        "llama.cpp Q4_K_M",
        "functional pass@1",
        Json(LATENCY, "summary", "llama.cpp Q4_K_M (batch 1, streaming)", "functional"),
    ),
    Row(
        "pipeline",
        "DreamScriptPipeline (golden, warm)",
        "median s",
        Json(GOLDEN, "seconds", "median"),
        10.0,
        higher_is_better=False,
    ),
    Row(
        "pipeline",
        "DreamScriptPipeline (golden, cold)",
        "median s",
        Json(CACHE, "cold", "median_s"),
        10.0,
        higher_is_better=False,
        criterion="S8",
    ),
    Row("pipeline", "DreamScriptPipeline (golden)", "pages producing code", Json(GOLDEN, "rate")),
    # ---- 14.2's stage-wise rerun, read if it exists -------------------------------------
    Row(
        "classify",
        "handcrafted logreg (14.2 rerun)",
        "macro F1",
        Json(STAGEWISE, "classify", "models", "logreg", "macro_f1"),
    ),
    Row(
        "classify",
        "embedding SVM (14.2 rerun)",
        "macro F1",
        Json(STAGEWISE, "classify", "models", "embedding_svm", "macro_f1"),
    ),
)

NOT_DURABLE: tuple[tuple[str, str], ...] = (
    ("Phase 6 MLP / SVM kernel sweep", "python -m src.classify.mlp ; python -m src.classify.rbf"),
    (
        "Phase 7 forests, boosting, naive Bayes, GMM+EM",
        "python -m src.classify.forest ; python -m src.classify.boosting",
    ),
    (
        "Phase 8 clustering and style adaptation",
        "python -m src.cluster.kmeans ; python -m src.cluster.styles",
    ),
    ("Phase 11 RL policy vs. topological / DFS baselines", "python -m src.rl.baselines --write"),
)


def collect(root: Path = ROOT, rows: tuple[Row, ...] = ROWS) -> list[dict]:
    """Read every declared cell. A cell that cannot be read is ``None``, never a stale value."""
    out = []
    for row in rows:
        value = row.source.read(root)
        passes: bool | None = None
        if (
            isinstance(value, int | float)
            and not isinstance(value, bool)
            and row.target is not None
        ):
            passes = value >= row.target if row.higher_is_better else value <= row.target
        out.append(
            {
                "stage": row.stage,
                "model": row.model,
                "metric": row.metric,
                "value": value,
                "target": row.target,
                "higher_is_better": row.higher_is_better,
                "passes": passes,
                "criterion": row.criterion,
                "provenance": row.source.provenance,
                "available": value is not None,
            }
        )
    return out


def _fmt(value: Any) -> str:
    if value is None:
        return f"*{MISSING}*"
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def render(cells: list[dict]) -> str:
    stages: list[str] = []
    for cell in cells:
        if cell["stage"] not in stages:
            stages.append(cell["stage"])
    available = sum(1 for cell in cells if cell["available"])
    graded = [cell for cell in cells if cell["passes"] is not None]
    lines = [
        "# Phase 14.1 - master results table",
        "",
        "Generated by `python -m src.eval.master`. Every number below is read from the artefact"
        " named in its provenance column at generation time; none is stored in the generator."
        " A cell reading *missing* means the artefact is absent, not that the measurement is"
        " zero.",
        "",
        f"**{available} of {len(cells)} declared cells resolved**;"
        f" {sum(1 for cell in graded if cell['passes'])} of {len(graded)} graded cells meet"
        " their target.",
        "",
    ]
    for stage in stages:
        lines += [
            f"## {stage}",
            "",
            "| model | metric | value | target | verdict | criterion | provenance |",
            "| :--- | :--- | ---: | ---: | :---: | :---: | :--- |",
        ]
        for cell in cells:
            if cell["stage"] != stage:
                continue
            verdict = "-" if cell["passes"] is None else ("pass" if cell["passes"] else "**fail**")
            target = "-" if cell["target"] is None else _fmt(cell["target"])
            bound = (
                "" if cell["target"] is None else (" (>=)" if cell["higher_is_better"] else " (<=)")
            )
            lines.append(
                f"| {cell['model']} | {cell['metric']} | {_fmt(cell['value'])} |"
                f" {target}{bound} | {verdict} | {cell['criterion'] or '-'} |"
                f" `{cell['provenance']}` |"
            )
        lines.append("")
    lines += ["## Not durable on disk", "", "| model family | regenerated by |", "| :--- | :--- |"]
    lines += [f"| {name} | `{command}` |" for name, command in NOT_DURABLE]
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--out", type=Path, default=REPORT_MD)
    ap.add_argument("--json", dest="out_json", type=Path, default=REPORT_JSON)
    args = ap.parse_args(argv)

    cells = collect(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(cells), encoding="utf-8")
    args.out_json.write_text(json.dumps({"cells": cells}, indent=2) + "\n", encoding="utf-8")
    missing = [cell["provenance"] for cell in cells if not cell["available"]]
    print(f"{len(cells) - len(missing)}/{len(cells)} cells resolved -> {args.out}")
    for provenance in missing:
        print(f"  missing: {provenance}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
