"""Phase 16.1.5 - what the served pipeline costs without a GPU, measured rather than estimated.

    python -m src.serve profile                      # both arms, the fixture pages
    python -m src.serve profile --arm cpu            # one arm
    python -m src.serve profile --pages 3 --repeats 2

Writes `reports/cpu_fallback.json` and `reports/cpu_fallback.md`.

## The question, and why it is not rhetorical

Phase 16's preamble decided a thin client and a heavy server, and the heavy server is where the
recogniser (`trocr-base-handwritten`, 333.9M parameters) and the detector (a YOLO pose model at
1280px) live. 13.7's budget is **< 10 s end to end on GPU**. A host that would actually run this for
free has no GPU at all, so the number that decides whether the app is deployable is one nobody has
measured: the same pipeline on CPU.

An estimate would be worthless here. "CPU is maybe 10x slower" spans a deployable service and one
where the request times out, and which of those is true is an empirical fact about two specific
models on one specific page size.

## Two arms, and a subprocess per arm because there is no other honest way

CUDA is initialised once per process and `torch.cuda.is_available()` is cached the first time it is
asked. Flipping a flag inside a running process does not un-initialise it, and a `.cpu()` call moves
a module without changing what the rest of the pipeline decides to do. So each arm is a **fresh
subprocess**, with the environment set before Python starts.

**And the value is `-1`, not `""`.** The empty string was the obvious first choice and it does not do
what it looks like: in that process torch 2.5.1+cu124 reports **`cuda.is_available() == True` with
`device_count() == 0`**. `get_device_name(0)` then raises `AssertionError: Invalid device id`, and the
detect stage does not run slowly - it **fails outright in 1.5 s** and the page stops at `detect`. A
CPU profile made that way would have measured a broken pipeline. Both values were checked directly:

    CUDA_VISIBLE_DEVICES=""    is_available: True    device_count: 0
    CUDA_VISIBLE_DEVICES="-1"  is_available: False   device_count: 0

## The cache is off, and that is the trap this row would otherwise fall into

`cache.ENV_KEYS` does **not** include `CUDA_VISIBLE_DEVICES` - correctly, because which device
answers should not change the answer. But it means a warm stage cache keyed on the *image and the
config* would hand the CPU arm the GPU arm's stored outputs, and the CPU arm would come back
instantaneous and identical. The measurement would be of a dict lookup, reported as a latency.

Both arms therefore run with `StageCache(enabled=False)`, and `cache_enabled: false` is recorded in
the JSON so the number cannot later be mistaken for a warm-path figure. 13.6's cache is a real and
useful thing; it is simply not what this row is asking about.

## The whole first pass is warm-up, not merely its first page

The obvious split - first page cold, everything after it warm - was tried and is wrong. The GPU arm's
first pass read **11.43, 3.07, 0.42, 0.13, 0.21 s** and its second read **0.29, 0.24, 0.42, 0.12,
0.21**: every page in pass one was still paying a first-use cost somewhere, and four of the five were
being counted as steady state. The p95 that fell out of it, 3.478 s, was a warm-up run wearing a
tail-latency label. So `repeat == 0` is the warm-up pass in full, and the statistics are over the
passes after it.

## What is reported, and what is deliberately not

Per arm: the first page's cold time, then per-stage median over the warm passes, the end-to-end
median, min and max, and 13.7's budget checked **twice** - against the median and against the worst
page - because on CPU those two disagree and the disagreement is the finding.

**No p95.** With five pages per pass it would be the maximum, and reporting a maximum under a
percentile's name is how a small sample gets mistaken for a distribution.

**Medians, not means**: over a handful of pages a mean is one slow page away from meaning nothing.

**And the emitter answered, not a model.** Every run reports `degraded: no` with the generate stage's
reason "no model configured; 12.1.6's emitter is the whole chain". These are therefore latencies for
the detector, the recogniser and the tracer; a deployment serving 12.2's adapter adds its cost on top.
13.4's distinction, at the one place it would be easiest to quietly drop.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from src.utils.config import ROOT

#: Where the two artefacts land. `reports/` is the row's own Definition of Done.
JSON_PATH = ROOT / "reports" / "cpu_fallback.json"
MD_PATH = ROOT / "reports" / "cpu_fallback.md"

#: The pages. Committed fixtures, so this is runnable on a clone with no dataset - the row is about
#: latency, and latency does not need held-out data.
FIXTURES = ROOT / "tests" / "fixtures"
PAGES = ("flowchart.png", "state_machine.png", "er_diagram.png", "wireframe.png", "circuit.png")

#: 13.7's budget, restated here so the report can say whether it holds rather than leaving arithmetic
#: to a reader.
LATENCY_BUDGET_S = 10.0

ARMS = ("gpu", "cpu")


# -- the child: one arm, in its own process -------------------------------------------------


def measure(pages: list[Path], repeats: int) -> dict[str, Any]:
    """Run the pipeline over `pages` `repeats` times and report every stage's timings.

    Runs in the *child* process, whose environment already decided whether CUDA exists. It never
    looks at `CUDA_VISIBLE_DEVICES` itself - it reports what torch says it can see, which is the
    thing that matters and the thing a mistake in the parent would show up in.
    """
    from src.pipeline.cache import StageCache
    from src.pipeline.core import DreamScriptPipeline

    # Off, deliberately. See the module docstring: `ENV_KEYS` does not include the device, so a warm
    # cache would hand this arm the other arm's answers and the profile would measure a dict.
    pipeline = DreamScriptPipeline(cache=StageCache(enabled=False))

    runs: list[dict[str, Any]] = []
    for repeat in range(repeats):
        for page in pages:
            started = time.perf_counter()
            result = pipeline.run(page)
            total = time.perf_counter() - started
            runs.append(
                {
                    "page": page.name,
                    "repeat": repeat,
                    # The **whole first pass** is warm-up, not merely its first page. The first
                    # version marked only `repeat == 0 and page == pages[0]` cold, and the GPU arm's
                    # first pass then read 11.447, 3.478, 1.761, 0.628, 0.412 s while its second read
                    # 0.348, 0.258, 0.478, 0.162, 0.258 - every page in pass one was still paying a
                    # first-use cost for some code path, and four of the five were being counted as
                    # steady state. The reported p95 was 3.478 s, which was a warm-up run wearing a
                    # tail-latency label.
                    "cold": repeat == 0,
                    "seconds": round(total, 3),
                    "ok": bool(result.ok),
                    "stopped_at": result.stopped_at or None,
                    "diagram_type": result.diagram_type,
                    "degraded": any(row.get("degraded") for row in result.timing_table()),
                    "stages": {row["stage"]: row["seconds"] for row in result.timing_table()},
                    # The reasons, not only the seconds. The first version of this function kept
                    # timings alone, and the CPU arm came back `stopped_at: detect` after 1.5 s with
                    # no explanation anywhere in the artefact - a profile that records *that* a stage
                    # failed and not *why* turns the interesting result into a mystery.
                    "reasons": {
                        row["stage"]: row["reason"]
                        for row in result.timing_table()
                        if row.get("reason")
                    },
                }
            )
    return {"device": device_facts(), "runs": runs}


def device_facts() -> dict[str, Any]:
    """What torch can actually see, asked rather than assumed."""
    facts: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
    }
    try:
        import torch
    except Exception as error:  # noqa: BLE001 - a profile reports, it does not raise
        facts["torch"] = f"unavailable: {type(error).__name__}: {error}"
        facts["cuda_available"] = False
        return facts

    facts["torch"] = torch.__version__
    # Each fact in its own guard rather than one `try` around all of them. The first version wrapped
    # the lot, `get_device_name(0)` raised `AssertionError: Invalid device id`, and the handler
    # overwrote `facts["torch"]` - so the report claimed torch itself was unavailable when the
    # version had been read successfully three lines earlier. One failing probe must not erase the
    # probes that worked.
    for name, probe in (
        ("cuda_available", lambda: bool(torch.cuda.is_available())),
        ("device_count", lambda: int(torch.cuda.device_count())),
        ("torch_threads", lambda: int(torch.get_num_threads())),
        ("device_name", lambda: torch.cuda.get_device_name(0)),
    ):
        try:
            facts[name] = probe()
        except Exception as error:  # noqa: BLE001 - an unanswerable probe is a reportable fact
            facts[name] = f"error: {type(error).__name__}: {error}"
    return facts


# -- the parent: both arms, one subprocess each ---------------------------------------------


def run_arm(arm: str, pages: list[Path], repeats: int, timeout_s: float) -> dict[str, Any]:
    """One arm, in a fresh interpreter with the environment that decides its device.

    `CUDA_VISIBLE_DEVICES=""` is set *before* Python starts, which is the only way to get a process
    where torch has never seen a device. Setting it after `import torch` changes nothing: CUDA
    initialisation and `is_available()` are both cached on first use.
    """
    env = dict(os.environ)
    if arm == "cpu":
        # `-1`, not `""`. The empty string was tried first and it does not do what it looks like:
        # torch 2.5.1+cu124 in that process reported **`cuda.is_available() == True`** with zero
        # usable devices, so `get_device_name(0)` raised `AssertionError: Invalid device id` and the
        # detector tried to move a model onto a device that was not there. `-1` is the documented
        # "no visible device" value and produces a process where `is_available()` is honestly False.
        env["CUDA_VISIBLE_DEVICES"] = "-1"
        env.pop("CUDA_DEVICE_ORDER", None)
    started = time.perf_counter()
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.serve.profile",
            "--child",
            "--repeats",
            str(repeats),
            "--pages",
            *[page.name for page in pages],
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env=env,
        timeout=timeout_s,
        check=False,
    )
    wall = time.perf_counter() - started
    if completed.returncode != 0:
        return {
            "arm": arm,
            "ok": False,
            "wall_seconds": round(wall, 2),
            "detail": f"the {arm} arm exited {completed.returncode}",
            # Kept, because the reason an arm could not run is the finding when an arm cannot run.
            "stderr": completed.stderr[-2000:],
        }
    try:
        payload = json.loads(_last_json_line(completed.stdout))
    except (ValueError, IndexError):
        return {
            "arm": arm,
            "ok": False,
            "wall_seconds": round(wall, 2),
            "detail": f"the {arm} arm printed no JSON",
            "stdout": completed.stdout[-2000:],
        }
    return {"arm": arm, "ok": True, "wall_seconds": round(wall, 2), **summarise(payload)}


def _last_json_line(text: str) -> str:
    """The child's payload.

    The last line that parses as JSON, not the whole of stdout: importing the pipeline drags in
    transformers, which prints a model config to stdout unbidden. Parsing everything would fail on
    somebody else's logging.
    """
    for line in reversed(text.splitlines()):
        candidate = line.strip()
        if candidate.startswith("{") and candidate.endswith("}"):
            return candidate
    raise IndexError("no JSON line in child output")


def summarise(payload: dict[str, Any]) -> dict[str, Any]:
    """Cold separated from warm, then medians per stage."""
    runs = payload["runs"]
    cold_pass = [run for run in runs if run["cold"]]
    cold = cold_pass[0] if cold_pass else None
    warm = [run for run in runs if not run["cold"]]

    stages: dict[str, dict[str, float]] = {}
    names: list[str] = []
    for run in warm:
        for stage in run["stages"]:
            if stage not in names:
                names.append(stage)
    for stage in names:
        samples = sorted(run["stages"][stage] for run in warm if stage in run["stages"])
        if not samples:
            continue
        # Median, min, max, n - and no p95, for the reason the end-to-end block gives. These had one
        # for a while and at n=10 it was equal to the maximum in every stage of both arms, which is
        # the clearest possible demonstration that it was a maximum with a percentile's name on it.
        stages[stage] = {
            "median": round(statistics.median(samples), 3),
            "min": round(min(samples), 3),
            "max": round(max(samples), 3),
            "n": len(samples),
        }

    totals = sorted(run["seconds"] for run in warm)
    return {
        "device": payload["device"],
        "cache_enabled": False,
        # The first page of the warm-up pass, and the pass total, because "the first user waits this
        # long" is the number a cold deployment is judged on.
        "cold_seconds": cold["seconds"] if cold else None,
        "cold_page": cold["page"] if cold else None,
        "warmup_pass_seconds": round(sum(run["seconds"] for run in cold_pass), 3),
        "warm": {
            "n": len(totals),
            "median": round(statistics.median(totals), 3) if totals else None,
            # No p95. With five pages per pass it would be the maximum, and reporting the maximum
            # under a percentile's name is how a small sample gets mistaken for a distribution.
            "min": round(min(totals), 3) if totals else None,
            "max": round(max(totals), 3) if totals else None,
        },
        # Both, because they answer different questions and on CPU they disagree: the median says
        # whether the service is usable and the maximum says whether a particular page times out.
        "within_budget": (
            bool(totals and statistics.median(totals) <= LATENCY_BUDGET_S) if totals else None
        ),
        "worst_within_budget": (
            bool(totals and max(totals) <= LATENCY_BUDGET_S) if totals else None
        ),
        "stages": stages,
        "pages": [
            {
                "page": run["page"],
                "seconds": run["seconds"],
                "ok": run["ok"],
                "stopped_at": run["stopped_at"],
                "degraded": run["degraded"],
                "warmup": run["cold"],
                # Kept per page, not only in aggregate: a stage that failed on one arm and not the
                # other is the most interesting thing a profile can find, and a table of seconds
                # alone cannot say why.
                "reasons": run.get("reasons") or {},
            }
            for run in runs
        ],
    }


def compare(arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The ratio, per stage and end to end - the number the deployment decision turns on."""
    gpu, cpu = arms.get("gpu"), arms.get("cpu")
    if not (gpu and cpu and gpu.get("ok") and cpu.get("ok")):
        return {"available": False, "detail": "both arms must have run to be compared"}
    ratios = {}
    for stage, cpu_stats in (cpu.get("stages") or {}).items():
        gpu_stats = (gpu.get("stages") or {}).get(stage)
        if not gpu_stats or not gpu_stats["median"]:
            # A stage the GPU arm finished in under a millisecond has no meaningful ratio; saying so
            # is better than dividing by 0.0 or by an arbitrary floor.
            ratios[stage] = None
            continue
        ratios[stage] = round(cpu_stats["median"] / gpu_stats["median"], 2)
    end_to_end = None
    if gpu["warm"]["median"]:
        end_to_end = round(cpu["warm"]["median"] / gpu["warm"]["median"], 2)
    return {
        "available": True,
        "end_to_end": end_to_end,
        "per_stage": ratios,
        "cpu_within_budget": cpu["within_budget"],
        "budget_s": LATENCY_BUDGET_S,
    }


# -- the report ------------------------------------------------------------------------------


def markdown(report: dict[str, Any]) -> str:
    """The `.md`, generated from the `.json` so the two cannot disagree."""
    lines = [
        "# Phase 16.1.5 - the CPU fallback profile",
        "",
        "Generated by `python -m src.serve profile`. Every duration below is measured in this "
        "repository on the pages named; nothing here is estimated or scaled from another machine.",
        "",
    ]
    arms = report["arms"]
    ratio = report["comparison"]

    if ratio.get("available"):
        cpu, gpu = arms["cpu"], arms["gpu"]
        lines += [
            f"**CPU is {ratio['end_to_end']}x slower end to end**: a median page takes "
            f"{cpu['warm']['median']}s against {gpu['warm']['median']}s on GPU. Against 13.7's "
            f"{ratio['budget_s']}s budget the **median holds "
            f"({'yes' if cpu['within_budget'] else 'no'}) and the worst page "
            f"{'also holds' if cpu['worst_within_budget'] else 'does not'}** "
            f"({cpu['warm']['max']}s) - which is the answer this row exists for: a CPU host is "
            f"usable and it has no headroom.",
            "",
            f"The first request is worse still and is the one a cold deployment is judged on: "
            f"**{cpu['cold_seconds']}s** on CPU against {gpu['cold_seconds']}s on GPU, because it "
            f"pays for loading the checkpoints.",
            "",
        ]
    else:
        lines += [f"**Incomplete**: {ratio.get('detail')}", ""]

    lines += [
        "## Both arms",
        "",
        "`cold` is the first page of the warm-up pass; `median`, `min` and `max` are over the passes "
        "after it. No p95 column: with five pages per pass it would be the maximum, and reporting "
        "the maximum under a percentile's name is how a small sample gets mistaken for a "
        "distribution.",
        "",
        "| arm | device | cold | warm median | min | max | median < budget | worst < budget |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | :---: | :---: |",
    ]
    for name in ARMS:
        arm = arms.get(name)
        if not arm:
            continue
        if not arm.get("ok"):
            lines.append(
                f"| {name} | *did not run* | - | - | - | - | - | {arm.get('detail', '')} |"
            )
            continue
        device = arm["device"]
        label = device.get("device_name") if device.get("cuda_available") is True else "CPU only"
        lines.append(
            f"| {name} | {label} | {arm['cold_seconds']}s | {arm['warm']['median']}s | "
            f"{arm['warm']['min']}s | {arm['warm']['max']}s | "
            f"{'yes' if arm['within_budget'] else '**no**'} | "
            f"{'yes' if arm['worst_within_budget'] else '**no**'} |"
        )
    lines.append("")

    if ratio.get("available"):
        lines += [
            "## Per stage",
            "",
            "Medians over the warm runs. The ratio is the number a deployment decision turns on: "
            "a stage that is 40x slower on CPU is a stage that has to move or be replaced.",
            "",
            "| stage | GPU median | CPU median | CPU / GPU |",
            "| :--- | ---: | ---: | ---: |",
        ]
        for stage, factor in ratio["per_stage"].items():
            gpu_median = arms["gpu"]["stages"].get(stage, {}).get("median", "-")
            cpu_median = arms["cpu"]["stages"].get(stage, {}).get("median", "-")
            shown = "*n/a*" if factor is None else f"{factor}x"
            lines.append(f"| {stage} | {gpu_median}s | {cpu_median}s | {shown} |")
        lines.append("")

    lines += [
        "## How this was measured, and what it is not",
        "",
        "* **A subprocess per arm.** CUDA initialises once per process and `torch.cuda.is_available()"
        "` is cached on first use, so the CPU arm is a fresh interpreter whose environment was set "
        "before Python started. Each arm reports what torch says it can see rather than what the "
        "parent intended - which is how the next point was found.",
        '* **`CUDA_VISIBLE_DEVICES=-1`, not `""`.** The empty string was tried first and does not '
        "do what it looks like: in that process torch 2.5.1+cu124 reported "
        "**`cuda.is_available() == True` with `device_count() == 0`**, so `get_device_name(0)` raised "
        "`AssertionError: Invalid device id` and the detect stage failed outright in 1.5 s instead of "
        "running slowly. `-1` gives a process where `is_available()` is honestly `False`. Both values "
        "were checked directly, not inferred.",
        "* **The stage cache is off in both arms.** `cache.ENV_KEYS` does not include the device - "
        "correctly, since which device answers must not change the answer - so a warm cache would "
        "have handed the CPU arm the GPU arm's stored outputs and the CPU arm would have looked "
        "instantaneous. That number would have been a dict lookup reported as a latency.",
        "* **The whole first pass is warm-up, not merely its first page.** The first version marked "
        "only the first page cold, and the GPU arm's first pass then read 11.447, 3.478, 1.761, 0.628 "
        "and 0.412 s while its second read 0.348, 0.258, 0.478, 0.162 and 0.258 - every page in pass "
        "one was still paying a first-use cost somewhere, and four of the five were being counted as "
        "steady state. The p95 that came out of it, 3.478 s, was a warm-up run wearing a "
        "tail-latency label.",
        "* **Medians, not means.** Over a handful of pages a mean is one slow page away from meaning "
        "nothing.",
        "* **The emitter answered, not a model.** Every run reports `degraded: no` and the generate "
        "stage's reason is \"no model configured; 12.1.6's emitter is the whole chain\" - so these "
        "are latencies for the detector, the recogniser and the tracer, and a deployment that serves "
        "12.2's adapter would add its cost on top. 13.4's distinction, at the one place it would be "
        "easiest to quietly drop.",
        "",
        "## Pages",
        "",
        "| arm | pass | page | seconds | ok | stopped at | degraded |",
        "| :--- | :--- | :--- | ---: | :---: | :--- | :---: |",
    ]
    for name in ARMS:
        arm = arms.get(name)
        if not arm or not arm.get("ok"):
            continue
        for page in arm["pages"]:
            lines.append(
                f"| {name} | {'warm-up' if page['warmup'] else 'warm'} | {page['page']} | "
                f"{page['seconds']}s | {'yes' if page['ok'] else 'no'} | "
                f"{page['stopped_at'] or '-'} | {'yes' if page['degraded'] else 'no'} |"
            )
    lines.append("")

    lines += ["## Environment", ""]
    for name in ARMS:
        arm = arms.get(name)
        if not arm or not arm.get("ok"):
            continue
        device = arm["device"]
        lines.append(
            f"* **{name}**: torch {device.get('torch')}, cuda_available "
            f"{device.get('cuda_available')}, devices {device.get('device_count')}, "
            f"`CUDA_VISIBLE_DEVICES={device.get('cuda_visible_devices')}`, "
            f"{device.get('cpu_count')} CPUs, torch threads {device.get('torch_threads')}"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Phase 16.1.5 - GPU and CPU latency for the served pipeline"
    )
    ap.add_argument("--arm", choices=ARMS, action="append", help="only this arm; repeatable")
    ap.add_argument("--pages", nargs="*", default=list(PAGES), help="fixture file names")
    ap.add_argument("--repeats", type=int, default=2, help="passes over the page set")
    ap.add_argument("--timeout", type=float, default=3600.0, help="seconds per arm")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    pages = [FIXTURES / name for name in args.pages]
    missing = [page.name for page in pages if not page.is_file()]
    if missing:
        print(f"no such fixture: {', '.join(missing)}", file=sys.stderr)
        return 2

    if args.child:
        # One line of JSON on stdout, last. transformers prints a model config unbidden, which is
        # why the parent reads the last JSON line rather than the whole stream.
        print(json.dumps(measure(pages, args.repeats)))
        return 0

    wanted = args.arm or list(ARMS)
    arms: dict[str, dict[str, Any]] = {}
    for arm in wanted:
        print(f"> {arm} arm: {len(pages)} pages x {args.repeats}", file=sys.stderr)
        arms[arm] = run_arm(arm, pages, args.repeats, args.timeout)
        state = "ok" if arms[arm].get("ok") else f"failed - {arms[arm].get('detail')}"
        print(f"  {state} in {arms[arm]['wall_seconds']}s", file=sys.stderr)

    report = {
        "row": "16.1.5",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "budget_s": LATENCY_BUDGET_S,
        "pages": [page.name for page in pages],
        "repeats": args.repeats,
        "cache_enabled": False,
        "arms": arms,
        "comparison": compare(arms),
    }
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    # No trailing newline added: `markdown` already ends its last line, and appending one produced a
    # blank final line that pre-commit's `end-of-file-fixer` stripped on every commit - leaving the
    # committed artefact different from what the code writes, which is the one thing a generated file
    # must never be.
    MD_PATH.write_text(markdown(report), encoding="utf-8")
    print(f"wrote {JSON_PATH.relative_to(ROOT)} and {MD_PATH.relative_to(ROOT)}", file=sys.stderr)
    return 0 if all(arm.get("ok") for arm in arms.values()) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
