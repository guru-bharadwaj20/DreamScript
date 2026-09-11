"""Phase 12.2.3 / 12.2.6 - the reduced-budget sweeps, chained, selected on validation loss.

python -m src.llm.sweep --wait-for experiments/llm/benchmark/dscoder67b_validation.json

Every arm is one `src.llm.run` process (so no state leaks between arms), run sequentially on
the one GPU, with the same token budget and the same seed - hence the same packed rows in the
same order - and each is scored by its best validation loss over the 176 validation pairs
(hdbpmn + fa_bresler). Test is never touched.

Stages, each fixing what the previous one chose (coordinate search, not a grid - a full grid
of r x targets x LR x warmup x batch x mix at ~25 min per arm is ~2 days of GPU):

    lora     r in {8, 16, 32, 64} (alpha = 2r) on all seven projections, plus r=16 on the four
             attention projections only                                         12.2.3
    lr       LR in {1e-4, 5e-5} against stage 1's winner at 2e-4                 12.2.6
    extra    warmup 0.10 vs 0.03; 65,536 vs 32,768 tokens per optimizer step;
             synthetic tokens capped at 1x real tokens vs all synthetic          12.2.6
    full     the chosen configuration for one epoch over the whole training set 12.2.5

`results.json` is rewritten after every arm so a crash loses at most one arm, and an arm whose
`summary.json` already exists is not re-run.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "experiments" / "llm" / "runs"
OUT = ROOT / "experiments" / "llm" / "sweep"
BUDGET = 1_200_000
ALL = "[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]"
ATTN = "[q_proj,k_proj,v_proj,o_proj]"


def summary_for(name: str) -> dict | None:
    for d in sorted(RUNS.glob(f"*_{name}"), reverse=True):
        path = d / "summary.json"
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            data["run_dir"] = str(d.relative_to(ROOT))
            return data
    return None


def run_arm(name: str, overrides: list[str], log) -> dict:
    done = summary_for(name)
    if done is not None:
        return done

    cmd = [
        sys.executable,
        "-m",
        "src.llm.run",
        f"logging.run_name={name}",
        *overrides,
    ]

    log.write(f"{time.strftime('%H:%M:%S')} start {name}: {' '.join(overrides)}\n")
    log.flush()

    began = time.perf_counter()

    with (OUT / f"{name}.log").open("w", encoding="utf-8") as out:
        code = subprocess.run(
            cmd,
            cwd=ROOT,
            stdout=out,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode

    elapsed = time.perf_counter() - began

    log.write(f"{time.strftime('%H:%M:%S')} end {name} " f"rc={code} {elapsed:.0f}s\n")
    log.flush()

    result = summary_for(name)
    if result is None:
        # A failed arm is a bug to fix, not a data point: stop rather than select around it.
        log.write(f"aborting sweep: {name} produced no summary\n")
        log.close()
        raise SystemExit(f"arm {name} failed (rc={code}); see {OUT / (name + '.log')}")

    return result


def best(results: dict[str, dict], names: list[str]) -> str:
    ok = [name for name in names if not results[name].get("failed")]
    return min(ok, key=lambda name: results[name]["best_val_loss"])


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - GPU orchestration
    parser = argparse.ArgumentParser(description="Phase 12.2.3/12.2.6 sweeps")
    parser.add_argument("--wait-for", type=Path, default=None)
    parser.add_argument("--budget", type=int, default=BUDGET)
    parser.add_argument("--no-full", action="store_true")
    args = parser.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)

    if args.wait_for is not None:
        while not args.wait_for.exists():
            time.sleep(30)

    log = (OUT / "sweep.log").open("a", encoding="utf-8")
    common = [f"train.token_budget={args.budget}", "train.eval_every=12"]
    results: dict[str, dict] = {}

    def record() -> None:
        (OUT / "results.json").write_text(
            json.dumps(results, indent=2),
            encoding="utf-8",
        )

    stage1 = {}
    for r in (8, 16, 32, 64):
        stage1[f"p12-qwen7b-lora{r}-s42"] = [
            f"lora.r={r}",
            f"lora.alpha={2 * r}",
        ]

    stage1["p12-qwen7b-lora16attn-s42"] = [
        "lora.r=16",
        "lora.alpha=32",
        f"lora.targets={ATTN}",
    ]

    for name, overrides in stage1.items():
        results[name] = run_arm(name, common + overrides, log)
        record()

    win = best(results, list(stage1))
    lora = stage1[win]
    results["_selected_lora"] = {"arm": win}
    record()

    stage2 = {win: lora}

    for lr in ("1e-4", "5e-5"):
        stage2[f"p12-qwen7b-lr{lr}-s42"] = lora + [f"train.lr={lr}"]

    for name, overrides in stage2.items():
        results[name] = run_arm(name, common + overrides, log)
        record()

    win2 = best(results, list(stage2))
    base = stage2[win2]
    results["_selected_lr"] = {"arm": win2}
    record()

    stage3 = {win2: base}

    stage3["p12-qwen7b-warmup0.1-s42"] = base + ["train.warmup_frac=0.1"]

    stage3["p12-qwen7b-tok65k-s42"] = base + ["train.tokens_per_step=65536"]

    stage3["p12-qwen7b-syn1x-s42"] = base + ["data.synthetic_ratio=1.0"]

    for name, overrides in stage3.items():
        results[name] = run_arm(name, common + overrides, log)
        record()

    win3 = best(results, list(stage3))
    results["_selected_final"] = {
        "arm": win3,
        "overrides": stage3[win3],
    }
    record()

    if not args.no_full:
        full = [
            override for override in stage3[win3] if not override.startswith("train.token_budget")
        ]

        results["p12-qwen7b-full-s42"] = run_arm(
            "p12-qwen7b-full-s42",
            full + ["train.epochs=1.0", "train.eval_every=50"],
            log,
        )
        record()

    log.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
