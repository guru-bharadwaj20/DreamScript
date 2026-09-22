"""Phase 14.4 - the full ablation matrix: HMM, RL, GMM+EM, LoRA, style adaptation, one at a time.

    python -m src.eval.ablate                 # reports/ablation_matrix.{md,json}
    python -m src.eval.ablate --skip-fit      # read the published artefacts, fit nothing

The plan asks for one row per component and a number saying what removing it costs. Two things
have to be true for such a table to mean anything, and both are enforced here.

**A removal is only informative against a stated alternative.** "Without the HMM" is not a
number; "the HMM replaced by the majority role per shape class" is. Every row therefore names
what the component was replaced *by*, and the delta is between those two specific arms on
identical folds, not between a published headline and a memory.

**A component that the deployed pipeline never calls cannot cost the pipeline anything.** So
every row also carries `on_deployed_path`, decided by looking for the component's call
sites in `src/pipeline/*.py` rather than by what the plan intended. This is where the matrix earns its keep:
**four of the five components are not on that path**, and a table that quietly reported their
stage-local deltas as pipeline deltas would be wrong by a whole pipeline. Their numbers are
reported as what they are - the effect on the stage that owns them.

## Where each number comes from

| component | replaced by | measured on | how |
| :--- | :--- | :--- | :--- |
| HMM | majority role per shape class (7.3.11's bag of shapes) | role macro F1, 993 sequences | re-run |
| GMM+EM | the annotated `shape` field (7.4.6) | role macro F1, hdbpmn | re-run |
| RL | the best heuristic order (DFS), 11.2.10 | semantic score + sandbox pass | published artefact |
| LoRA | the same base model zero-shot, 12.2.7 | functional pass@1, 162 test pairs | published artefact |
| style adaptation | one global adaptation, 9.3.6 | OCR CER | published artefact |

The two re-runs are cheap cross-validations; the three reads are GPU runs whose artefacts exist,
and each is quoted with its file so the row can be audited rather than believed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from src.utils.config import ROOT

REPORT_JSON = ROOT / "reports" / "ablation_matrix.json"
REPORT_MD = ROOT / "reports" / "ablation_matrix.md"

PIPELINE_PACKAGE = ROOT / "src" / "pipeline"

#: What a component's presence in the pipeline package's own source looks like. Markers are
#: call-site names, not words from a docstring: `core.py` mentions "adapter" in its module
#: header while `generate.py` is the file that actually asks a served model.
PATH_MARKERS = {
    "hmm": ("src.parse.roles", "src.parse.viterbi", "decode_sequence", "semantic_role="),
    "gmm": ("softshapes", "GaussianMixture", "responsibilities("),
    "rl": ("src.rl.qlearning", "src.rl.dqn", "policy.order("),
    "lora": ("src.pipeline.generate", "ask_model", "llama"),
    "style": ("src.ocr.adapt", "src.ocr.styled", "style_cluster"),
}


def on_deployed_path(component: str) -> dict[str, Any]:
    """Whether the pipeline package reaches this component, by reading its own source."""
    hits = []
    for file in sorted(PIPELINE_PACKAGE.glob("*.py")):
        source = file.read_text(encoding="utf-8")
        hits += [f"{file.name}:{marker}" for marker in PATH_MARKERS[component] if marker in source]
    return {
        "on_deployed_path": bool(hits),
        "evidence": hits or f"no {component} call site in src/pipeline/*.py",
    }


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


def _md_cell(path: str, header: str, row: str, column: str) -> Any:
    from src.eval.master import MdCell

    return MdCell(path, header, row, column).read(ROOT)


def hmm_row(fit: bool = True) -> dict:
    entry: dict[str, Any] = {
        "component": "HMM (semantic roles, 7.3)",
        "replaced_by": "majority role per shape class (7.3.11 bag of shapes)",
        "metric": "role macro F1 (5-fold, 993 sequences)",
        "source": "re-run: src.parse.ablation",
        **on_deployed_path("hmm"),
    }
    if not fit:
        entry["note"] = "not re-measured (--skip-fit)"
        return entry
    from src.parse.ablation import run

    try:
        result = run()
    except FileNotFoundError as error:  # a missing input is a reportable gap, not a lost matrix
        entry["note"] = f"could not re-measure: {error}"
        return entry
    ladder = result["ladder"]
    entry.update(
        with_component=round(ladder["hmm_per_type"]["macro_f1"], 4),
        without_component=round(ladder["bag_of_shapes"]["macro_f1"], 4),
        delta=result["contributions"]["total"],
        detail={
            "alphabet_beyond_shape": result["contributions"]["alphabet_beyond_shape"],
            "sequence_model": result["contributions"]["sequence_model"],
            "diagram_type": result["contributions"]["diagram_type"],
            "nodes": result["nodes"],
        },
    )
    return entry


def gmm_row(fit: bool = True) -> dict:
    entry: dict[str, Any] = {
        "component": "GMM + EM (learned shape evidence, 7.4)",
        "replaced_by": "the annotated `shape` field (7.4.6 `annotated` arm)",
        "metric": "role macro F1 (5-fold, hdbpmn)",
        "source": "re-run: src.parse.softshapes",
        **on_deployed_path("gmm"),
    }
    if not fit:
        entry["note"] = "not re-measured (--skip-fit)"
        return entry
    from src.parse.softshapes import run

    try:
        result = run()
    except FileNotFoundError as error:
        entry["note"] = f"could not re-measure: {error}"
        return entry
    modes = result["by_mode"]
    # "With the component" is the better of the two *learned* arms, not the better of all three:
    # `best_mode` can be `annotated`, which is the arm this row is measuring the removal of, and
    # scoring it against itself would print a delta of zero for a component that lost by 0.24.
    learned = max(("hard", "soft"), key=lambda mode: modes[mode]["macro_f1"])
    entry.update(
        with_component=round(modes[learned]["macro_f1"], 4),
        without_component=round(modes["annotated"]["macro_f1"], 4),
        delta=round(modes[learned]["macro_f1"] - modes["annotated"]["macro_f1"], 4),
        detail={
            "best_learned_arm": learned,
            "best_mode_overall": result["best_mode"],
            "hard_minus_annotated": result["hard_minus_annotated"],
            "soft_minus_annotated": result["soft_minus_annotated"],
            "sequences": result["sequences"],
        },
    )
    return entry


def rl_row() -> dict:
    """11.2.10's end-to-end arms, read from the report that ran them on the sandbox."""
    best_learned = _md_cell("reports/rl_ablation.md", "arm", "dqn+dfs", "semantic")
    heuristic = _md_cell("reports/rl_ablation.md", "arm", "dfs", "semantic")
    entry: dict[str, Any] = {
        "component": "RL traversal agent (11.1-11.2)",
        "replaced_by": "the DFS reading order (11.2.7's best heuristic)",
        "metric": "semantic score of the emitted program",
        "source": "reports/rl_ablation.md",
        **on_deployed_path("rl"),
    }
    if best_learned is None or heuristic is None:
        entry["note"] = "reports/rl_ablation.md does not carry the arms table"
        return entry
    entry.update(
        with_component=best_learned,
        without_component=heuristic,
        delta=round(float(best_learned) - float(heuristic), 4),
        detail={
            "pure_dqn_semantic": _md_cell("reports/rl_ablation.md", "arm", "dqn", "semantic"),
            "pure_dqn_sandbox_pass": _md_cell(
                "reports/rl_ablation.md", "arm", "dqn", "sandbox pass"
            ),
            "note": "the learned arm that beats DFS is the hybrid dqn+dfs; the pure policy"
            " cannot cross a connected component and loses badly",
        },
    )
    return entry


def lora_row() -> dict:
    with_lora = _read("experiments/llm/quality/done.json", "lora", "functional")
    zero_shot = _read(
        "experiments/llm/compare/done.json", "results", "zero-shot", "summary", "functional"
    )
    few_shot = _read(
        "experiments/llm/compare/done.json", "results", "few-shot (k=2)", "summary", "functional"
    )
    entry: dict[str, Any] = {
        "component": "LoRA fine-tune (12.2)",
        "replaced_by": "the same base model, zero-shot (12.2.7)",
        "metric": "functional pass@1 over 162 test pairs",
        "source": "experiments/llm/{quality,compare}/done.json",
        **on_deployed_path("lora"),
    }
    if with_lora is None or zero_shot is None:
        entry["note"] = "the Phase 12 comparison artefacts are absent"
        return entry
    entry.update(
        with_component=with_lora,
        without_component=zero_shot,
        delta=round(with_lora - zero_shot, 4),
        detail={
            "few_shot_k2": few_shot,
            "base_model": "Qwen2.5-Coder-7B-Instruct",
            "served_at_measurement_time": "13.4 records that no llama.cpp server was reachable"
            " for the golden run, so the pipeline answered from 12.1.6's emitter - being on the"
            " path and being the answerer are different facts and both are recorded",
        },
    )
    return entry


def style_row() -> dict:
    adapted = _md_cell("reports/ocr.md", "model", "adapt_style_adapt", "CER")
    globally = _md_cell("reports/ocr.md", "model", "adapt_global", "CER")
    randomly = _md_cell("reports/ocr.md", "model", "adapt_random_adapt", "CER")
    entry: dict[str, Any] = {
        "component": "style adaptation (Phase 8 clusters -> 9.3.5)",
        "replaced_by": "one global adaptation over every writer",
        "metric": "OCR CER (lower is better)",  # the direction is in the metric name
        "lower_is_better": True,
        "source": "reports/ocr.md",
        **on_deployed_path("style"),
    }
    if adapted is None or globally is None:
        entry["note"] = "reports/ocr.md does not carry the 9.3.6 model table"
        return entry
    entry.update(
        with_component=adapted,
        without_component=globally,
        delta=round(float(adapted) - float(globally), 4),
        detail={
            "random_cluster_control": randomly,
            "note": "the control matters: adapting to a *random* partition scores within 0.0002"
            " of adapting to the style clusters, so the gain is adaptation, not style",
            "deployed_reader": "trocr_large (S3), which is not a CRNN and carries no"
            " style-adapted variant - these rows are the 9.3.6 generation",
        },
    )
    return entry


def collect(fit: bool = True) -> dict:
    started = time.time()
    rows = [hmm_row(fit), gmm_row(fit), rl_row(), lora_row(), style_row()]
    return {
        "rows": rows,
        "components": len(rows),
        "on_deployed_path": sum(1 for row in rows if row.get("on_deployed_path")),
        "seconds": round(time.time() - started, 1),
    }


def _fmt(value: Any, *, signed: bool = False) -> str:
    """Values print plainly; only a delta carries a sign, because only a delta has a direction."""
    if value is None:
        return "*missing*"
    if isinstance(value, float):
        return f"{value:+.4f}" if signed else f"{value:.4f}"
    return str(value)


def render(result: dict) -> str:
    lines = [
        "# Phase 14.4 - ablation matrix",
        "",
        "Generated by `python -m src.eval.ablate`. One row per component, each naming what it was"
        " replaced *by* - a removal measured against an unstated alternative is not a number - and"
        " whether the deployed pipeline calls it at all.",
        "",
        "| component | replaced by | metric | with | without | delta | on the deployed path |",
        "| :--- | :--- | :--- | ---: | ---: | ---: | :---: |",
    ]
    for row in result["rows"]:
        delta = row.get("delta")
        lines.append(
            f"| {row['component']} | {row['replaced_by']} | {row['metric']} |"
            f" {_fmt(row.get('with_component'))} | {_fmt(row.get('without_component'))} |"
            f" {_fmt(delta, signed=True)} | {'yes' if row.get('on_deployed_path') else 'no'} |"
        )
    lines += [
        "",
        f"**{result['on_deployed_path']} of {result['components']} components have a call site"
        " in `src/pipeline/*.py`.** For the others the delta is the effect on the"
        " stage that owns the component, not on the pipeline, and is reported as such rather than"
        " promoted into an end-to-end claim.",
        "",
        "## Row detail",
        "",
    ]
    for row in result["rows"]:
        lines.append(f"### {row['component']}")
        lines.append("")
        lines.append(f"* replaced by: {row['replaced_by']}")
        lines.append(f"* measured on: {row['metric']}")
        lines.append(f"* source: `{row['source']}`")
        lines.append(f"* on the deployed path: {row.get('evidence')}")
        if row.get("note"):
            lines.append(f"* note: {row['note']}")
        for key, value in (row.get("detail") or {}).items():
            lines.append(f"* {key}: {value}")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-fit", action="store_true", help="read published artefacts only")
    ap.add_argument("--out", type=Path, default=REPORT_MD)
    ap.add_argument("--json", dest="out_json", type=Path, default=REPORT_JSON)
    args = ap.parse_args(argv)

    result = collect(fit=not args.skip_fit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(result), encoding="utf-8")
    args.out_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2)[:2500])
    print(f"-> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
