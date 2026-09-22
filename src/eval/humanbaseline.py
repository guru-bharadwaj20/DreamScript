"""Phase 14.8 - how long the same code takes a person, and what DreamScript actually saves.

    python -m src.eval.humanbaseline            # reports/human_baseline.{md,json}

The plan asks for "a speedup number for the pitch". A number like that is easy to produce and
easy to produce dishonestly, so three rules are followed here and stated in the report itself.

**No human was timed, and the report says so in its first line.** What this module builds is a
*model* of the human side - read the drawing, type the program, check it runs - with every
parameter named, defaulted to a stated value, and swept over a range. A single number from an
unnamed assumption is a slogan; the same number with its sensitivity attached is a measurement
of the model rather than of a person. The protocol for the real study is written into the report
so the missing half is a task rather than a gap.

**The machine side is measured, not modelled.** Page-to-code latency comes from 13.6's cold-cache
run over the golden pages, and the program sizes come from 12.1.6's reference programs for the
162 held-out test diagrams.

**The pipeline is not right every time, and the pitch number has to carry that.** 14.3 measured
the end-to-end functional pass rate of the emitted program at **0.1914** on the test split (and
12.3.3's 0.7037 is the model working from a *correct* IR, which is a different claim). So this
module reports two numbers: the speedup *when the output is correct*, and the expected speedup
once a wrong output costs a person the repair. The second is the honest one for a pitch, and on
this corpus the two are far apart.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.utils.config import ROOT

REPORT_JSON = ROOT / "reports" / "human_baseline.json"
REPORT_MD = ROOT / "reports" / "human_baseline.md"

CACHE_REPORT = ROOT / "reports" / "p13_cache.json"
PROPAGATION = ROOT / "reports" / "error_propagation.json"


@dataclass(frozen=True)
class Model:
    """The human side, entirely in named parameters. Nothing here was measured on a person."""

    #: seconds spent reading one drawn node before any typing starts
    seconds_per_node: float = 3.0
    #: seconds per drawn edge - following an arrow to its target on a messy page
    seconds_per_edge: float = 2.0
    #: effective characters a minute of *code* typing, including navigation and thinking pauses
    chars_per_minute: float = 200.0
    #: multiplier for running it, finding the typo, fixing it
    verification_factor: float = 1.3
    #: seconds a person spends reading a generated program before trusting it
    review_seconds: float = 45.0
    #: seconds to repair a generated program that turns out to be wrong, on top of the review
    repair_seconds: float = 240.0

    def human_seconds(self, nodes: int, edges: int, characters: int) -> float:
        reading = nodes * self.seconds_per_node + edges * self.seconds_per_edge
        typing = characters / self.chars_per_minute * 60.0
        return (reading + typing) * self.verification_factor


def corpus_shape() -> dict[str, Any]:
    """Nodes, edges and program size over the 162 held-out test diagrams."""
    from src.llm import pairs

    rows = pairs.load("test")
    nodes, edges, characters = [], [], []
    for pair in rows:
        diagram = pairs.diagram_for(pair)
        nodes.append(len(diagram.get("nodes") or []))
        edges.append(len(diagram.get("edges") or []))
        characters.append(len(pair["target_code"]))
    return {
        "diagrams": len(rows),
        "median_nodes": statistics.median(nodes),
        "median_edges": statistics.median(edges),
        "median_characters": statistics.median(characters),
        "mean_characters": round(statistics.fmean(characters), 1),
    }


def machine_seconds() -> dict[str, Any]:
    """13.6's measured page-to-code latency, cold cache, over the golden pages."""
    if not CACHE_REPORT.is_file():
        return {"note": f"{CACHE_REPORT.name} is absent"}
    payload = json.loads(CACHE_REPORT.read_text(encoding="utf-8"))
    cold = payload.get("cold", {})
    return {
        "median_s": cold.get("median_s"),
        "mean_s": cold.get("mean_s"),
        "pages": cold.get("pages"),
        "source": "reports/p13_cache.json (13.6 cold run)",
    }


def pass_rate() -> dict[str, Any]:
    """The end-to-end functional pass rate the speedup has to be discounted by."""
    if not PROPAGATION.is_file():
        return {"note": f"{PROPAGATION.name} is absent; run python -m src.eval.propagate"}
    payload = json.loads(PROPAGATION.read_text(encoding="utf-8"))
    return {
        "end_to_end": payload["pass_rate"]["predicted"],
        "from_gold_ir": payload["pass_rate"]["gold"],
        "by_source": {
            source: entry["predicted"] for source, entry in payload.get("by_source", {}).items()
        },
        "source": "reports/error_propagation.json",
    }


def compare(model: Model, shape: dict, machine: dict, rates: dict) -> dict[str, Any]:
    human = model.human_seconds(
        int(shape["median_nodes"]), int(shape["median_edges"]), int(shape["median_characters"])
    )
    latency = float(machine.get("median_s") or 0.0)
    if_correct = latency + model.review_seconds
    rate = float(rates.get("end_to_end") or 0.0)
    expected = if_correct + (1.0 - rate) * model.repair_seconds
    return {
        "human_seconds": round(human, 1),
        "machine_seconds_if_correct": round(if_correct, 1),
        "machine_seconds_expected": round(expected, 1),
        "speedup_if_correct": round(human / if_correct, 2) if if_correct else None,
        "speedup_expected": round(human / expected, 2) if expected else None,
        "pass_rate_used": rate,
        # Unclamped on purpose: a value at or below 0 means the pipeline still wins at a pass
        # rate of zero, which is a statement about repair being cheaper than writing - clamping
        # it to 0 would hide exactly that and make the row look like a near-miss.
        "break_even_pass_rate": (
            round(1.0 - (human - if_correct) / model.repair_seconds, 4)
            if model.repair_seconds
            else None
        ),
    }


def sensitivity(shape: dict, machine: dict, rates: dict) -> list[dict]:
    """The same comparison under a slower and a faster reader, typist and reviewer."""
    arms = {
        "default": Model(),
        "fast_typist_300cpm": Model(chars_per_minute=300.0),
        "slow_typist_120cpm": Model(chars_per_minute=120.0),
        "quick_reader": Model(seconds_per_node=1.5, seconds_per_edge=1.0),
        "careful_reader": Model(seconds_per_node=6.0, seconds_per_edge=4.0),
        "cheap_repair_60s": Model(repair_seconds=60.0),
        "expensive_repair_600s": Model(repair_seconds=600.0),
        # The arm that decides the row: if repairing a wrong program costs what writing one
        # costs, the pipeline's advantage is only what its correct answers buy.
        "repair_equals_rewrite": Model(repair_seconds=Model().human_seconds(15, 17, 2355)),
    }
    out = []
    for name, model in arms.items():
        out.append({"arm": name, "parameters": asdict(model), **compare(model, shape, machine, rates)})
    return out


def collect() -> dict:
    shape = corpus_shape()
    machine = machine_seconds()
    rates = pass_rate()
    arms = sensitivity(shape, machine, rates)
    return {
        "human_study": "none was run; every human-side number below comes from the stated model",
        "corpus": shape,
        "machine": machine,
        "pass_rate": rates,
        "headline": arms[0],
        "sensitivity": arms,
        "protocol_for_a_real_study": [
            "10 participants who can write Python, none of them the author",
            "8 diagrams each from the held-out test split, stratified by node count,"
            " order counterbalanced so fatigue does not land on the same pages",
            "timed from first sight of the photograph to a program that passes 12.3.3's"
            " functional test for that diagram - the same test the pipeline is scored by",
            "record think-aloud only after the timed run, so it does not slow the timing",
            "report per-participant medians and the spread, never a single mean",
        ],
    }


def render(result: dict) -> str:
    headline = result["headline"]
    shape = result["corpus"]
    machine = result["machine"]
    rates = result["pass_rate"]
    lines = [
        "# Phase 14.8 - human baseline and the speedup that survives the pass rate",
        "",
        f"**No human was timed for this row.** {result['human_study']}. The machine side is"
        " measured: 13.6's cold-cache page-to-code latency and 12.1.6's reference programs.",
        "",
        "## The comparison at the median test diagram",
        "",
        f"* the median held-out diagram has **{shape['median_nodes']} nodes,"
        f" {shape['median_edges']} edges** and a reference program of"
        f" **{shape['median_characters']} characters**"
        f" ({shape['diagrams']} diagrams)",
        f"* modelled human time: **{headline['human_seconds']} s**"
        " (read the drawing, type the program, run it and fix the typo)",
        f"* measured pipeline latency: **{machine.get('median_s')} s** median over"
        f" {machine.get('pages')} pages, cold cache",
        f"* end-to-end functional pass rate: **{rates.get('end_to_end')}**"
        f" (from a correct IR it is {rates.get('from_gold_ir')})",
        "",
        "| | seconds | speedup |",
        "| :--- | ---: | ---: |",
        f"| human, modelled | {headline['human_seconds']} | 1.00x |",
        f"| DreamScript, when the output is correct | {headline['machine_seconds_if_correct']} |"
        f" **{headline['speedup_if_correct']}x** |",
        f"| DreamScript, expected (a wrong program costs a repair) |"
        f" {headline['machine_seconds_expected']} | **{headline['speedup_expected']}x** |",
        "",
        f"The pitch number is the second one. The first assumes away the"
        f" {round((1 - float(rates.get('end_to_end') or 0)) * 100)}% of pages whose program does"
        " not pass its own functional test, and a person has to find that out by reading the"
        " program - which is why the review cost is charged in both rows and the repair cost only"
        " in the second.",
        "",
        f"**Break-even pass rate: {headline['break_even_pass_rate']}.** A value at or below zero"
        " means the pipeline still wins at a pass rate of *zero*, and that is what this corpus"
        " reports - because under these parameters repairing a wrong program costs 240 s while"
        " writing one from the drawing costs 1,017 s. So the number this row actually turns on"
        " is **the cost of a repair, not the pass rate**: the `repair_equals_rewrite` arm below"
        " charges a repair the full rewrite, and that is the arm to argue about.",
        "",
        "## Sensitivity",
        "",
        "| arm | human s | expected machine s | expected speedup |",
        "| :--- | ---: | ---: | ---: |",
    ]
    for arm in result["sensitivity"]:
        lines.append(
            f"| {arm['arm']} | {arm['human_seconds']} | {arm['machine_seconds_expected']} |"
            f" {arm['speedup_expected']}x |"
        )
    lines += ["", "## The study this row did not run", ""]
    lines += [f"{index}. {step}" for index, step in enumerate(result["protocol_for_a_real_study"], 1)]
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=REPORT_MD)
    ap.add_argument("--json", dest="out_json", type=Path, default=REPORT_JSON)
    args = ap.parse_args(argv)

    result = collect()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(result), encoding="utf-8")
    args.out_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["headline"], indent=2))
    print(f"-> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
