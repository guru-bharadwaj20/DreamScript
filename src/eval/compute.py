"""Phase 14.11 - compute accounting: GPU-hours, energy and cost, per phase.

    python -m src.eval.compute                    # reports/compute.md + .json
    python -m src.eval.compute --rate 0.60        # a different cloud price per GPU-hour

Every training run in this project wrote its wall time somewhere, and nothing has ever added
them up. This does, by **reading the durations out of the artefacts** rather than re-timing
anything: `stage_wall_s` in the Phase 12 ledgers, `seconds` in the detector and arrow training
records, `train_seconds` in the OCR reports, and so on. Each entry carries the file it came
from, so a number here can be traced to the run that produced it.

## Three honest limits, stated rather than buried

**This is the compute that survived.** The DVC store was lost and rebuilt, and several runs
(the Phase 5-8 sweeps, the CRNN generation, the first detector training) left no duration behind.
They appear in the report as *unrecorded* with what is known about them, and the total is
therefore a **lower bound** on what the project actually cost. Presenting it as the total would
be the same error as quoting a mean without its denominator.

**Energy is modelled, not metered.** No wattmeter was attached. Energy is GPU-hours times the
card's rated board power times a utilisation factor times a datacentre overhead, and all three
numbers are parameters printed in the report. A run that idles between batches draws less than
its rating, which is exactly why the factor is named and swept rather than assumed to be 1.0.

**Cost is a rental equivalent.** The hardware is owned, so nothing was paid per hour; the cost
column answers "what would this have cost to rent", at a stated price, and says so.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from src.utils.config import ROOT

REPORT_JSON = ROOT / "reports" / "compute.json"
REPORT_MD = ROOT / "reports" / "compute.md"

#: RTX 4500 Ada Generation, the card every GPU run in this project used.
BOARD_POWER_W = 210.0
#: A training run does not hold the board at its rating; this is the fraction assumed.
UTILISATION = 0.75
#: Datacentre overhead (cooling, PSU losses). 1.0 would be a machine under a desk with no cooling.
PUE = 1.15
#: Rental equivalent, USD per GPU-hour for a card of this class.
RATE_USD_PER_HOUR = 0.50

#: (phase, label, file, key path). Durations are read, never typed.
SOURCES: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
    ("9 - detection", "arrow detector, yolov8s 40 epochs", "experiments/detect/arrows/train.json", ("seconds",)),
    ("9 - detection", "arrow detector, yolov8m-pose 150 epochs", "experiments/detect/arrows/train_pose_m.json", ("seconds",)),
    ("9 - OCR", "trocr_large, 16 epochs", "reports/s3_label_ocr_large.json", ("train_seconds",)),
    ("9 - OCR", "trocr_large, 28 epochs", "reports/s3_label_ocr_large28.json", ("train_seconds",)),
    ("9 - OCR", "trocr, strong augmentation", "reports/s3_label_ocr_aug.json", ("train_seconds",)),
    ("9 - OCR", "trocr_fa specialist", "reports/s3_fa_specialist.json", ("train_seconds",)),
    ("12 - LLM", "base-model benchmark", "experiments/llm/benchmark/done.json", ("stage_wall_s",)),
    ("12 - LLM", "LoRA rank sweep", "experiments/llm/lora_sweep/done.json", ("stage_wall_s",)),
    ("12 - LLM", "hyperparameter sweep", "experiments/llm/hparam_sweep/done.json", ("stage_wall_s",)),
    ("12 - LLM", "final training run", "experiments/llm/train/done.json", ("stage_wall_s",)),
    ("12 - LLM", "three-way comparison", "experiments/llm/compare/done.json", ("stage_wall_s",)),
    ("12 - LLM", "quality scoring", "experiments/llm/quality/done.json", ("stage_wall_s",)),
    ("12 - LLM", "repair loop", "experiments/llm/repair/done.json", ("stage_wall_s",)),
    ("12 - LLM", "latency study", "experiments/llm/latency/done.json", ("stage_wall_s",)),
    ("12 - LLM", "merge and export", "experiments/llm/export/done.json", ("stage_wall_s",)),
    ("12 - LLM", "similarity scoring", "experiments/llm/similarity/done.json", ("stage_wall_s",)),
    ("14 - evaluation", "error propagation ladder", "reports/error_propagation.json", ("seconds",)),
    ("14 - evaluation", "robustness sweep", "reports/robustness_pipeline.json", ("seconds",)),
    ("14 - evaluation", "unseen-type study", "reports/unseen_type.json", ("seconds",)),
    ("14 - evaluation", "ablation matrix re-fits", "reports/ablation_matrix.json", ("seconds",)),
)

#: Runs whose duration no surviving artefact records. Listed so the total reads as a lower bound.
UNRECORDED: tuple[tuple[str, str], ...] = (
    (
        "9 - OCR",
        "the three trocr_large fine-tunes (16 epochs, 28 epochs, strong augmentation). No"
        " artefact carries their duration; the plan's S3 row records 5,497 s for the 16-epoch"
        " run in prose, and the other two ran the same schedule, so roughly 4.5 GPU-hours sit"
        " outside this table - named here rather than typed into it",
    ),
    ("9 - detection", "the component detector's own training (experiments/detect/train.json keeps no duration)"),
    ("9 - OCR", "the CRNN generation of 9.3 - four architectures, no duration kept"),
    ("5-8 - classical", "every classifier, ensemble, HMM and clustering sweep - artefacts lost with the DVC store"),
    ("11 - RL", "Q-learning, SARSA and DQN training across three seeds"),
    ("1-3 - data", "corpus rebuild, rendering and preprocessing"),
)


def _read(path: str, keys: tuple[str, ...]) -> float | None:
    file = ROOT / path
    if not file.is_file():
        return None
    value: Any = json.loads(file.read_text(encoding="utf-8"))
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return float(value) if isinstance(value, (int, float)) else None


def collect(
    rate: float = RATE_USD_PER_HOUR,
    utilisation: float = UTILISATION,
    pue: float = PUE,
    watts: float = BOARD_POWER_W,
) -> dict:
    runs = []
    for phase, label, path, keys in SOURCES:
        seconds = _read(path, keys)
        runs.append(
            {
                "phase": phase,
                "run": label,
                "source": path,
                "seconds": seconds,
                "gpu_hours": round(seconds / 3600.0, 3) if seconds else None,
                "recorded": seconds is not None,
            }
        )
    by_phase: dict[str, dict[str, Any]] = {}
    for run in runs:
        entry = by_phase.setdefault(run["phase"], {"gpu_hours": 0.0, "runs": 0, "unrecorded": 0})
        entry["runs"] += 1
        if run["recorded"]:
            entry["gpu_hours"] += run["gpu_hours"]
        else:
            entry["unrecorded"] += 1
    total_hours = sum(entry["gpu_hours"] for entry in by_phase.values())
    for entry in by_phase.values():
        entry["gpu_hours"] = round(entry["gpu_hours"], 3)
        entry["kwh"] = round(entry["gpu_hours"] * watts * utilisation * pue / 1000.0, 3)
        entry["usd"] = round(entry["gpu_hours"] * rate, 2)
    return {
        "model": {
            "board_power_w": watts,
            "utilisation": utilisation,
            "pue": pue,
            "usd_per_gpu_hour": rate,
            "energy_note": "modelled, not metered: hours x watts x utilisation x PUE",
            "cost_note": "rental equivalent; the hardware is owned and nothing was paid per hour",
        },
        "totals": {
            "gpu_hours": round(total_hours, 3),
            "kwh": round(total_hours * watts * utilisation * pue / 1000.0, 3),
            "usd": round(total_hours * rate, 2),
            "is_lower_bound": True,
            "runs_recorded": sum(1 for run in runs if run["recorded"]),
            "runs_missing_a_duration": sum(1 for run in runs if not run["recorded"]),
        },
        "by_phase": dict(sorted(by_phase.items())),
        "runs": runs,
        "unrecorded": [{"phase": phase, "what": what} for phase, what in UNRECORDED],
    }


def sensitivity(result: dict) -> list[dict]:
    hours = result["totals"]["gpu_hours"]
    watts = result["model"]["board_power_w"]
    pue = result["model"]["pue"]
    out = []
    for name, utilisation in (("idle-heavy 0.50", 0.50), ("assumed 0.75", 0.75), ("saturated 1.00", 1.0)):
        out.append(
            {
                "arm": name,
                "utilisation": utilisation,
                "kwh": round(hours * watts * utilisation * pue / 1000.0, 3),
            }
        )
    return out


def render(result: dict) -> str:
    totals = result["totals"]
    model = result["model"]
    lines = [
        "# Phase 14.11 - compute accounting",
        "",
        "Generated by `python -m src.eval.compute`. Every duration is **read from the artefact the"
        " run wrote**, never re-timed and never typed; each row names its file.",
        "",
        f"**{totals['gpu_hours']} GPU-hours, {totals['kwh']} kWh, ${totals['usd']} rental"
        f" equivalent - and this is a lower bound**, because"
        f" {totals['runs_missing_a_duration']} of the"
        f" {totals['runs_recorded'] + totals['runs_missing_a_duration']} listed runs kept no"
        " duration and whole phases kept none at all (below).",
        "",
        "| phase | GPU-hours | kWh | USD | runs | without a duration |",
        "| :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for phase, entry in result["by_phase"].items():
        lines.append(
            f"| {phase} | {entry['gpu_hours']} | {entry['kwh']} | {entry['usd']} |"
            f" {entry['runs']} | {entry['unrecorded']} |"
        )
    lines += [
        f"| **total** | **{totals['gpu_hours']}** | **{totals['kwh']}** | **${totals['usd']}** |"
        f" {totals['runs_recorded'] + totals['runs_missing_a_duration']} |"
        f" {totals['runs_missing_a_duration']} |",
        "",
        "## The model behind the energy and cost columns",
        "",
        f"* board power **{model['board_power_w']} W** (RTX 4500 Ada), utilisation"
        f" **{model['utilisation']}**, PUE **{model['pue']}** - {model['energy_note']}",
        f"* **${model['usd_per_gpu_hour']}** per GPU-hour - {model['cost_note']}",
        "",
        "| utilisation arm | kWh |",
        "| :--- | ---: |",
    ]
    for arm in sensitivity(result):
        lines.append(f"| {arm['arm']} | {arm['kwh']} |")
    lines += ["", "## Runs", "", "| phase | run | GPU-hours | source |", "| :--- | :--- | ---: | :--- |"]
    for run in result["runs"]:
        hours = run["gpu_hours"] if run["recorded"] else "*unrecorded*"
        lines.append(f"| {run['phase']} | {run['run']} | {hours} | `{run['source']}` |")
    lines += ["", "## Compute this project spent but cannot account for", ""]
    lines += [f"* **{entry['phase']}**: {entry['what']}" for entry in result["unrecorded"]]
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rate", type=float, default=RATE_USD_PER_HOUR)
    ap.add_argument("--utilisation", type=float, default=UTILISATION)
    ap.add_argument("--pue", type=float, default=PUE)
    ap.add_argument("--watts", type=float, default=BOARD_POWER_W)
    ap.add_argument("--out", type=Path, default=REPORT_MD)
    ap.add_argument("--json", dest="out_json", type=Path, default=REPORT_JSON)
    args = ap.parse_args(argv)

    result = collect(args.rate, args.utilisation, args.pue, args.watts)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(result), encoding="utf-8")
    args.out_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"totals": result["totals"], "by_phase": result["by_phase"]}, indent=2))
    print(f"-> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
