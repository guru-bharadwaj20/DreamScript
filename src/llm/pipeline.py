"""Phase 12.2 / 12.3 - the self-running pipeline: benchmark -> sweeps -> train -> evaluate -> export.

    python -m src.llm.pipeline                 # run (resumes: finished stages are skipped)
    python -m src.llm.pipeline --dry-run       # 1.5B model, a few samples, COMMIT_ROW_DRY=1
    python -m src.llm.pipeline --status        # print experiments/llm/pipeline_status.json

Stages run in dependency order. Each writes `experiments/llm/<stage>/done.json` with its measured
numbers and is skipped when that file exists; a stage that raises writes `failed.json`, commits its
row(s) `--keep-open` with the traceback summarised, and only the stages that depend on it are
skipped. After each stage the pipeline writes `reports/llm_<stage>.md` (+ `reports/figures/p12_*`),
builds the DoD cell and commit message from the numbers it just measured, decides the row status
mechanically (threshold rows flip only when the measured number passes) and commits locally via
`experiments/llm/commit_row.py --no-push`.

    stage         rows                          GPU work
    benchmark     12.2.1                        consumes the running zero-shot jobs; selects the base
    lora_sweep    12.2.3                        5 arms, equal token budget, selected on val loss
    hparam_sweep  12.2.6                        LR x2, warmup, tokens/step, synthetic mix, 2x budget
    train         12.2.5                        one full epoch of the selected configuration
    compare       12.2.7                        zero-shot / few-shot / LoRA on test (+ val for LoRA)
    quality       12.3.1 12.3.2 12.3.3 12.3.8   scoring of the LoRA test generations
    repair        12.3.9                        one error-feedback repair attempt on every failure
    similarity    (report only)                 12.3.5's metric on model outputs, if it has landed
    export        12.2.8                        merge (bf16, CPU), logits check, GGUF f16/Q8_0/Q4_K_M
    latency       12.2.9                        batch-1 streaming latency: HF NF4+LoRA, llama.cpp

GPU rules applied in every GPU stage: each GPU job is its own subprocess (no allocator state leaks
between models); the generation flags are read back and stored (`use_cache`, SDPA, left padding,
pad token, bf16 compute); every generation job sweeps batch size on a fixed sample and runs the
split at the smallest batch within 5% of the best ms/sample; the process is capped at 20 GiB so
the WDDM cliff is an OOM rather than a spill; an OOM retries once at half batch; training arms run
one after another; wall time and peak VRAM are logged for every job.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
COMMIT_ROW = ROOT / "experiments" / "llm" / "commit_row.py"
QWEN7B = "Qwen/Qwen2.5-Coder-7B-Instruct"
DSCODER = "deepseek-ai/deepseek-coder-6.7b-instruct"
TINY = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
ALL = "[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]"
ATTN = "[q_proj,k_proj,v_proj,o_proj]"


# ------------------------------------------------------------------------------------ context


@dataclass
class Ctx:
    dry: bool = False
    base: Path = ROOT / "experiments" / "llm"
    reports: Path = ROOT / "reports"
    figures: Path = ROOT / "reports" / "figures"
    limit: int | None = None
    budget: int = 1_200_000
    sweep_batches: tuple[int, ...] = (8, 16, 32)
    sweep_samples: int = 16
    state: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def make(cls, dry: bool) -> Ctx:
        if not dry:
            return cls()
        return cls(
            dry=True,
            base=ROOT / "experiments" / "llm" / "dryrun",
            reports=ROOT / "reports" / "_dryrun",
            figures=ROOT / "reports" / "_dryrun" / "figures",
            limit=4,
            budget=12_000,
            sweep_batches=(2, 4),
            sweep_samples=4,
        )

    def stage_dir(self, stage: str) -> Path:
        path = self.base / stage
        path.mkdir(parents=True, exist_ok=True)
        return path

    def rel(self, path: Path) -> str:
        return str(Path(path).resolve().relative_to(ROOT)).replace("\\", "/")


LOG_LOCK = threading.Lock()


def log(ctx: Ctx, message: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    with LOG_LOCK:
        ctx.base.mkdir(parents=True, exist_ok=True)
        with (ctx.base / "pipeline.log").open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    print(line, flush=True)


def gpu_used_mb() -> int | None:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout
        return int(out.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        return None


def write_status(ctx: Ctx) -> None:
    ctx.state["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    ctx.state["pid"] = os.getpid()
    ctx.state["gpu_used_mb"] = gpu_used_mb()
    path = ctx.base / "pipeline_status.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(ctx.state, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def heartbeat(ctx: Ctx, stop: threading.Event) -> None:
    while not stop.wait(30):
        with suppress(Exception):
            write_status(ctx)


# ------------------------------------------------------------------------------------ jobs


def run_job(ctx: Ctx, kind: str, spec: dict, name: str) -> dict:
    """Run one GPU job in a fresh interpreter; returns its result JSON. Raises on failure."""
    jobs = ctx.stage_dir("jobs")
    spec_path = jobs / f"{name}.spec.json"
    result_path = jobs / f"{name}.result.json"
    if result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))
    spec = dict(spec, result=str(result_path))
    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    env = dict(os.environ, HF_HUB_OFFLINE="1", PYTHONIOENCODING="utf-8")
    log(ctx, f"job {name} ({kind}) start")
    began = time.perf_counter()
    with (jobs / f"{name}.log").open("w", encoding="utf-8") as out:
        code = subprocess.run(
            [PY, "-m", "src.llm.pipeline", "job", kind, str(spec_path)],
            cwd=ROOT,
            env=env,
            stdout=out,
            stderr=subprocess.STDOUT,
        ).returncode
    log(ctx, f"job {name} end rc={code} {time.perf_counter() - began:.0f}s")
    if code != 0 or not result_path.exists():
        tail = (jobs / f"{name}.log").read_text(encoding="utf-8", errors="replace")[-2000:]
        raise RuntimeError(f"job {name} failed rc={code}:\n{tail}")
    return json.loads(result_path.read_text(encoding="utf-8"))


def job_generate(spec: dict) -> dict:
    """Generation job: load (NF4, optional adapter), audit, batch sweep, generate, write rows."""
    import torch

    from src.llm import generate, quant

    quant.cap_memory()
    began = time.perf_counter()
    model, tokenizer = quant.load_model(spec["model"])
    if spec.get("adapter"):
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, spec["adapter"])
    load_s = time.perf_counter() - began
    generate.prepare_for_generation(model, tokenizer)
    items = [json.loads(x) for x in Path(spec["messages"]).read_text(encoding="utf-8").splitlines()]
    prompts = [generate.render(tokenizer, it["messages"]) for it in items]
    report: dict[str, Any] = {
        "model": spec["model"],
        "adapter": spec.get("adapter"),
        "load_s": round(load_s, 1),
        "audit": generate.generation_audit(model, tokenizer),
        "n": len(items),
    }
    batch = spec.get("batch")
    if batch is None:
        by_len = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        k = int(spec.get("sweep_samples", 16))
        stride = max(1, len(by_len) // k)
        sample = [prompts[i] for i in by_len[::stride][:k]]
        report["sweep"] = generate.batch_sweep(
            model,
            tokenizer,
            sample,
            list(spec["sweep_batches"]),
            generate.MAX_NEW_TOKENS,
        )
        from src.llm.benchmark import saturation_batch

        batch = saturation_batch(report["sweep"])
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    splits: list[int] = []
    t0 = time.perf_counter()
    try:
        outputs = generate.generate(
            model,
            tokenizer,
            prompts,
            batch_size=batch,
            oom_splits=splits,
        )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        batch = max(1, batch // 2)
        splits.append(-1)
        outputs = generate.generate(
            model,
            tokenizer,
            prompts,
            batch_size=batch,
            oom_splits=splits,
        )
    report.update(
        batch=batch,
        generate_s=round(time.perf_counter() - t0, 1),
        oom_splits=splits,
        peak_allocated_gb=round(torch.cuda.max_memory_allocated() / 1e9, 3),
        peak_reserved_gb=round(torch.cuda.max_memory_reserved() / 1e9, 3),
        new_tokens=sum(o["new_tokens"] for o in outputs),
        hit_limit=sum(o["hit_limit"] for o in outputs),
    )
    report["ms_per_sample"] = round(1000 * report["generate_s"] / max(1, len(items)), 1)
    rows = [
        {"diagram_id": it["diagram_id"], "source": it["source"], **o}
        for it, o in zip(items, outputs, strict=True)
    ]
    generate.write_jsonl(rows, Path(spec["out"]))
    return report


def job_latency(spec: dict) -> dict:
    """Batch-1 streaming latency through HF generate (NF4 + adapter), with and without KV cache."""
    from threading import Thread

    import torch
    from transformers import TextIteratorStreamer

    from src.llm import generate, quant

    quant.cap_memory()
    model, tokenizer = quant.load_model(spec["model"])
    if spec.get("adapter"):
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, spec["adapter"])
    generate.prepare_for_generation(model, tokenizer)
    items = [json.loads(x) for x in Path(spec["messages"]).read_text(encoding="utf-8").splitlines()]
    rows = []
    # warm-up so the first measured request does not pay CUDA kernel selection
    warm = tokenizer(
        generate.render(tokenizer, items[0]["messages"]),
        return_tensors="pt",
    ).to(0)
    with torch.inference_mode():
        model.generate(**warm, max_new_tokens=8, do_sample=False)
    for it in items:
        enc = tokenizer(
            generate.render(tokenizer, it["messages"]),
            return_tensors="pt",
            add_special_tokens=False,
        ).to(0)
        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        kwargs = dict(
            **enc,
            max_new_tokens=generate.MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
            streamer=streamer,
            pad_token_id=tokenizer.pad_token_id,
        )
        torch.cuda.synchronize()
        began = time.perf_counter()
        thread = Thread(target=lambda kw=kwargs: _gen(model, kw))
        thread.start()
        first, parts = None, []
        for chunk in streamer:
            if chunk and first is None:
                first = time.perf_counter() - began
            parts.append(chunk)
        thread.join()
        total = time.perf_counter() - began
        text = "".join(parts)
        rows.append(
            {
                "diagram_id": it["diagram_id"],
                "source": it["source"],
                "text": text,
                "prompt_tokens": int(enc["input_ids"].shape[1]),
                "new_tokens": len(tokenizer(text, add_special_tokens=False)["input_ids"]),
                "ttft_s": round(first or total, 3),
                "total_s": round(total, 3),
                "hit_limit": False,
            }
        )
    # KV cache ablation: identical prompt, fixed 128 new tokens, cache on vs off.
    cache = []
    for it in items[: int(spec.get("cache_ablation", 3))]:
        enc = tokenizer(
            generate.render(tokenizer, it["messages"]),
            return_tensors="pt",
            add_special_tokens=False,
        ).to(0)
        for use_cache in (True, False):
            torch.cuda.synchronize()
            began = time.perf_counter()
            with torch.inference_mode():
                out = model.generate(
                    **enc,
                    max_new_tokens=128,
                    min_new_tokens=128,
                    do_sample=False,
                    use_cache=use_cache,
                    pad_token_id=tokenizer.pad_token_id,
                )
            torch.cuda.synchronize()
            n = out.shape[1] - enc["input_ids"].shape[1]
            cache.append(
                {
                    "diagram_id": it["diagram_id"],
                    "use_cache": use_cache,
                    "new_tokens": int(n),
                    "ms_per_token": round(
                        1000 * (time.perf_counter() - began) / max(1, n),
                        1,
                    ),
                }
            )
    generate.write_jsonl(rows, Path(spec["out"]))
    return {
        "rows": len(rows),
        "cache_ablation": cache,
        "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 3),
    }


def _gen(model, kwargs) -> None:
    import torch

    with torch.inference_mode():
        model.generate(**kwargs)


def job_export(spec: dict) -> dict:
    from src.llm import export

    out = Path(spec["out"])
    merged = out / "merged"
    report: dict[str, Any] = {}
    if not (merged / "config.json").exists():
        report["merge"] = export.merge(
            spec["model"],
            Path(spec["adapter"]),
            merged,
        )
    prompts = [json.loads(x)["prompt"] for x in Path(spec["prompts"]).read_text().splitlines()]
    report["merge_check"] = export.check_merge(
        spec["model"],
        Path(spec["adapter"]),
        merged,
        prompts,
    )
    report["gguf"] = export.to_gguf(merged, out)
    report["merged_dir"] = str(merged)
    return report


JOBS: dict[str, Callable[[dict], dict]] = {
    "generate": job_generate,
    "latency": job_latency,
    "export": job_export,
}


def job_main(kind: str, spec_path: str) -> int:
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    result = JOBS[kind](spec)
    Path(spec["result"]).write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    return 0


# ------------------------------------------------------------------------------------ helpers


def pairs_for(ctx: Ctx, split: str) -> list[dict]:
    from src.llm import pairs

    items = pairs.load(split, sources=["hdbpmn", "fa_bresler"])
    if ctx.limit:
        # keep both sources in a dry run
        by = {}
        for p in items:
            by.setdefault(p["source"], []).append(p)
        items = [p for v in by.values() for p in v[: max(1, ctx.limit // 2)]]
    return items


def write_messages(
    path: Path,
    pairs_list: list[dict],
    build: Callable[[dict], list],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for p in pairs_list:
            item = {
                "diagram_id": p["diagram_id"],
                "source": p["source"],
                "messages": build(p),
            }
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    return path


def generate_arm(
    ctx: Ctx,
    stage: str,
    name: str,
    model: str,
    pairs_list: list[dict],
    build,
    adapter=None,
) -> tuple[dict, list[dict]]:
    from src.llm.generate import read_jsonl

    d = ctx.stage_dir(stage)
    messages = write_messages(
        d / f"{name}.messages.jsonl",
        pairs_list,
        build,
    )
    out = d / f"{name}.jsonl"
    report = run_job(
        ctx,
        "generate",
        {
            "model": model,
            "adapter": adapter,
            "messages": str(messages),
            "out": str(out),
            "sweep_batches": list(ctx.sweep_batches),
            "sweep_samples": ctx.sweep_samples,
        },
        f"{stage}-{name}",
    )
    return report, read_jsonl(out)


def score(
    rows: list[dict],
    split: str,
    workers: int = 16,
) -> tuple[list[dict], dict]:
    from src.llm import score as scoring

    scored = scoring.score_rows(rows, split, workers=workers)
    summary = scoring.summarise(scored)
    codecheck = data_agent_codecheck(rows, split)
    if codecheck is not None:
        summary["codecheck"] = codecheck
    return scored, summary


def data_agent_codecheck(rows: list[dict], split: str) -> dict | None:
    """12.3.1 / 12.3.2 through `src.eval.codecheck` when it is importable (the data agent's)."""
    try:
        from src.eval import codecheck
    except Exception:  # noqa: BLE001
        return None
    from src.llm import pairs
    from src.llm.score import code_of

    index = {p["diagram_id"]: p for p in pairs.load(split)}
    items = [(code_of(r["text"]), index[r["diagram_id"]]) for r in rows]
    syn = codecheck.syntactic_many([(c, rec["language"]) for c, rec in items])
    exe = codecheck.executable_many(items)
    return {
        "syntax": codecheck.summarise(syn),
        "executes": codecheck.summarise(exe),
        "per_row": [
            {
                "diagram_id": r["diagram_id"],
                "syntax": s["ok"],
                "executes": e["ok"],
            }
            for r, s, e in zip(rows, syn, exe, strict=True)
        ],
    }


def pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def md_table(header: list[str], rows: list[list[Any]]) -> str:
    out = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join([":---"] * len(header)) + " |",
    ]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def esc(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def commit_row(
    ctx: Ctx,
    task: str,
    dod: str,
    msg: str,
    files: list[Path],
    keep_open: bool,
) -> str:
    d = ctx.stage_dir("commits")
    dod_file = d / f"{task}.dod.txt"
    msg_file = d / f"{task}.msg.txt"
    dod_file.write_text(esc(dod), encoding="utf-8")
    msg_file.write_text(msg.strip() + "\n", encoding="utf-8")
    cmd = [
        PY,
        str(COMMIT_ROW),
        "--no-push",
        "--task",
        task,
        "--dod-file",
        str(dod_file),
    ]
    cmd += ["--msg-file", str(msg_file)]
    if keep_open:
        cmd.append("--keep-open")
    cmd += ["--", *[ctx.rel(f) for f in files if Path(f).exists()]]
    env = dict(os.environ)
    if ctx.dry:
        env["COMMIT_ROW_DRY"] = "1"
    result = subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=7200,
    )
    log(
        ctx,
        f"commit {task} rc={result.returncode}: " f"{(result.stdout or result.stderr)[-400:]}",
    )
    if result.returncode != 0:
        raise RuntimeError(f"commit_row failed for {task}: {result.stderr[-1500:]}")
    return result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""


def figure(path: Path, draw: Callable) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        path.parent.mkdir(parents=True, exist_ok=True)
        fig = draw(plt)
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return path
    except Exception:  # noqa: BLE001 - a figure never fails a stage
        return None


def code_files(ctx: Ctx) -> list[Path]:
    names = [
        "src/llm/__init__.py",
        "src/llm/pipeline.py",
        "src/llm/benchmark.py",
        "src/llm/generate.py",
        "src/llm/score.py",
        "src/llm/functional.py",
        "src/llm/fewshot.py",
        "src/llm/run.py",
        "src/llm/sweep.py",
        "src/llm/train.py",
        "src/llm/export.py",
        "src/llm/pairs.py",
        "src/llm/quant.py",
        "configs/llm.yaml",
        "tests/test_llm_functional.py",
        "tests/test_llm_score.py",
        "tests/test_llm_train.py",
        "tests/test_llm_quant.py",
        "tests/test_llm_pipeline.py",
    ]
    return [ROOT / n for n in names]


# ------------------------------------------------------------------------------------ stages


def stage_benchmark(ctx: Ctx) -> dict:
    """12.2.1: consume the zero-shot validation runs (waiting for a live one) and select a base."""
    from src.llm import benchmark
    from src.llm.generate import read_jsonl

    bench = ROOT / "experiments" / "llm" / "benchmark"
    candidates = {"qwen7b": QWEN7B, "dscoder67b": DSCODER}
    if ctx.dry:
        candidates = {"tiny15b": TINY}
    results = {}
    for tag, model in candidates.items():
        meta = bench / f"{tag}_validation.json"
        gens = bench / f"{tag}_validation.jsonl"
        keep = bench / f"{tag}_validation.keep.jsonl"
        if ctx.dry:
            template = benchmark.pinned_template()
            report, rows = generate_arm(
                ctx,
                "benchmark",
                tag,
                model,
                pairs_for(ctx, "validation"),
                template.build_messages,
            )
            meta_data = report
        else:
            while not meta.exists() and live_process(tag):
                log(ctx, f"benchmark: waiting for the running {tag} job")
                time.sleep(60)
            if not meta.exists():
                template_rows = list(pairs_for(ctx, "validation"))
                report, rows = generate_arm(
                    ctx,
                    "benchmark",
                    tag,
                    model,
                    template_rows,
                    benchmark.pinned_template().build_messages,
                )
                meta_data = report
            else:
                meta_data = json.loads(meta.read_text(encoding="utf-8"))
                rows = read_jsonl(gens if gens.exists() else keep)
        scored, summary = score(rows, "validation")
        results[tag] = {
            "model": model,
            "meta": meta_data,
            "summary": summary,
        }
        d = ctx.stage_dir("benchmark")
        (d / f"{tag}.summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

    def key(tag: str) -> tuple:
        s = results[tag]["summary"]
        return (s["functional"], s["executes"], s["syntax"])

    selected = max(results, key=key)
    done = {
        "selected": selected,
        "model": results[selected]["model"],
        "results": results,
    }
    report_benchmark(ctx, done)
    return done


def live_process(tag: str) -> bool:
    try:
        out = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                "ForEach-Object { $_.CommandLine }",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        ).stdout
        return any(
            f"--tag {tag}" in line and "src.llm.benchmark" in line for line in out.splitlines()
        )
    except Exception:  # noqa: BLE001
        return False


def _sweep_rows(meta: dict) -> str:
    rows = [
        [
            r["batch"],
            r.get("ms_per_sample", "OOM"),
            r.get("generated_tokens_per_s", ""),
            r.get("peak_allocated_gb", ""),
            r.get("peak_reserved_gb", ""),
        ]
        for r in meta.get("sweep", [])
    ]
    return md_table(
        [
            "batch",
            "ms/sample",
            "gen tokens/s",
            "peak alloc GB",
            "peak reserved GB",
        ],
        rows,
    )


def report_benchmark(ctx: Ctx, done: dict) -> None:
    res = done["results"]
    lines = [
        "# Phase 12.2.1 - zero-shot base-model benchmark (validation, 176 real pairs)",
        "",
    ]
    lines.append(
        "Identical messages (12.2.4 template as committed at "
        f"`{__import__('src.llm.benchmark', fromlist=['x']).PROVISIONAL_TEMPLATE_REV}`, loaded "
        "from the git object), each rendered through the candidate's own chat template; greedy, "
        "1,600 new tokens max, 4-bit NF4, SDPA. Scored by `src.llm.score` over all 176 generations."
    )
    lines.append("")
    header = [
        "candidate",
        "contract",
        "syntax",
        "executes",
        "functional pass@1",
        "dropped nodes",
        "invented/program",
        "hit limit",
        "batch",
        "generate s",
    ]
    table = []
    for _tag, r in res.items():
        s, m = r["summary"], r["meta"]
        table.append(
            [
                f"`{r['model']}`",
                pct(s["contract"]),
                pct(s["syntax"]),
                pct(s["executes"]),
                pct(s["functional"]),
                pct(s["dropped_node_rate"]),
                s["invented_per_program"],
                s["hit_limit"],
                m.get("batch"),
                m.get("generate_s"),
            ]
        )
    lines += [md_table(header, table), ""]
    for tag, r in res.items():
        s = r["summary"]
        lines.append(f"## {tag}")
        lines.append("")
        for src in ("fa_bresler", "hdbpmn"):
            b = s.get(f"by_source/{src}")
            if b:
                lines.append(
                    f"- {src} (n={b['n']}): syntax {pct(b['syntax'])}, executes "
                    f"{pct(b['executes'])}, functional {pct(b['functional'])}"
                )
        if "codecheck" in s:
            lines.append(
                f"- `src.eval.codecheck`: syntax {pct(s['codecheck']['syntax']['rate'])}, "
                f"executes {pct(s['codecheck']['executes']['rate'])}"
            )
        lines.append(f"- functional failure reasons: `{json.dumps(s['functional_reasons'])}`")
        lines.append(f"- flags read back: `{json.dumps(r['meta'].get('audit', {}))}`")
        lines += [
            "",
            "Batch sweep (32-prompt length-stratified sample):",
            "",
            _sweep_rows(r["meta"]),
            "",
        ]
    lines.append(
        f"**Selected: `{done['model']}`** (highest functional pass@1, then executability, "
        "then syntax; validation only)."
    )
    (ctx.reports).mkdir(parents=True, exist_ok=True)
    (ctx.reports / "llm_benchmark.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def commit_benchmark(
    ctx: Ctx,
    done: dict,
    files_extra: list[Path] | None = None,
) -> None:
    res = done["results"]
    parts = []
    for _tag, r in res.items():
        s, m = r["summary"], r["meta"]
        sweep = ", ".join(
            f"x{row['batch']} {row['ms_per_sample'] / 1000:.1f} s"
            for row in m.get("sweep", [])
            if row.get("status") == "ok"
        )
        parts.append(
            f"**{r['model']}**: contract {pct(s['contract'])}, syntax {pct(s['syntax'])}, executes "
            f"{pct(s['executes'])}, functional pass@1 **{pct(s['functional'])}** "
            f"(fa_bresler {pct(s['by_source/fa_bresler']['functional'])}, hdbpmn "
            f"{pct(s['by_source/hdbpmn']['functional'])}), dropped-node rate "
            f"{pct(s['dropped_node_rate'])}, {s['hit_limit']} replies hit the 1,600-token limit; "
            f"ms/sample by batch {sweep} -> batch {m.get('batch')}, split in {m.get('generate_s')} s"
        )
    other = [t for t in res if t != done["selected"]]
    dod = (
        "`src/llm/benchmark.py` + `src/llm/generate.py` + `src/llm/score.py` + "
        "`src/llm/functional.py` + `reports/llm_benchmark.md`: every candidate loaded as it would "
        "be fine-tuned (NF4, bf16 compute, SDPA - flags read back: use_cache on in config and "
        "generation_config, left padding, pad token set), fed identical messages (the template "
        "loaded from its git object, so a working-tree edit cannot differ between candidates; "
        "Qwen's 176 prompt token counts re-derived 176/176), greedy, over all 176 real validation "
        "pairs with one denominator. Each run's batch was chosen from a measured ms/sample sweep "
        "on a fixed length-stratified sample. " + "; ".join(parts) + f". **Selected "
        f"{done['model']}** on functional pass@1, validation only. Functional pass@1 is 12.3.3's "
        "per-diagram test against the drawing (IR automaton for state machines, reachability / "
        "order / exclusivity for flowcharts), under which the 12.1.6 reference programs themselves "
        "pass only 25.8% of hdbpmn and 27.1% of fa_bresler validation pairs - the ceiling a model "
        "imitating them is under. CodeLlama-7B-Instruct was downloaded but not benchmarked: two "
        "families were compared and its GPU hour went to the fine-tune"
    )
    sel = res[done["selected"]]["summary"]
    msg = (
        f"12.2.1: {done['model'].split('/')[-1]} selected at {pct(sel['functional'])} zero-shot "
        f"pass@1 against {', '.join(res[t]['model'].split('/')[-1] + ' ' + pct(res[t]['summary']['functional']) for t in other)}\n\n"
        "Identical pinned template, greedy decoding, NF4, all 176 validation pairs, batch chosen "
        "from a measured ms/sample sweep per candidate. Selection on validation only.\n"
    )
    files = [
        ctx.reports / "llm_benchmark.md",
        *(files_extra or []),
    ]
    commit_row(
        ctx,
        "12.2.1",
        dod,
        msg,
        files,
        keep_open=False,
    )


# -- sweeps -------------------------------------------------------------------------------------


def run_training_arm(
    ctx: Ctx,
    name: str,
    overrides: list[str],
    model: str,
) -> dict:
    runs = ctx.base / "runs"
    for d in sorted(runs.glob(f"*_{name}"), reverse=True):
        if (d / "summary.json").is_file():
            s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
            s["run_dir"] = ctx.rel(d)
            return s
    cmd = [
        PY,
        "-m",
        "src.llm.run",
        f"logging.run_name={name}",
        f"model.id={model}",
        f"paths.experiments={ctx.rel(runs)}",
        *overrides,
    ]
    if ctx.dry:
        cmd += ["train.eval_every=2", "data.synthetic_ratio=0.2"]
    log(ctx, f"train arm {name}: {' '.join(overrides)}")
    began = time.perf_counter()
    logs = ctx.stage_dir("jobs")
    for attempt in range(2):
        with (logs / f"train-{name}.log").open(
            "w",
            encoding="utf-8",
        ) as out:
            code = subprocess.run(
                cmd,
                cwd=ROOT,
                env=dict(os.environ, HF_HUB_OFFLINE="1"),
                stdout=out,
                stderr=subprocess.STDOUT,
            ).returncode
        text = (logs / f"train-{name}.log").read_text(
            encoding="utf-8",
            errors="replace",
        )
        if code == 0 or "OutOfMemoryError" not in text or attempt:
            break
        # OOM once: halve the packing bin (= the micro-batch in tokens) and retry
        cmd.append("train.bin_tokens=1024")
        log(
            ctx,
            f"train arm {name}: OOM, retrying at bin_tokens=1024",
        )
    log(
        ctx,
        f"train arm {name} rc={code} " f"{time.perf_counter() - began:.0f}s",
    )
    for d in sorted(runs.glob(f"*_{name}"), reverse=True):
        if (d / "summary.json").is_file():
            s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
            s["run_dir"] = ctx.rel(d)
            return s
    raise RuntimeError(f"training arm {name} failed rc={code}:\n{text[-2000:]}")


def val_curve(ctx: Ctx, run_dir: str) -> list[tuple[int, float]]:
    path = ROOT / run_dir / "train.jsonl"
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    return [(r["tokens"], r["val_loss"]) for r in rows if "val_loss" in r]


def arms_table(arms: dict[str, dict]) -> str:
    rows = []
    for name, s in arms.items():
        rows.append(
            [
                name,
                s.get("overrides", ""),
                s["val_loss_start"],
                s["best_val_loss"],
                s["best_step"],
                s["total_steps"],
                s["tokens"],
                round(s["wall_s"] / 60, 1),
                s["peak_allocated_gb"],
                s["peak_reserved_gb"],
                round(s["tokens"] / max(1, s["wall_s"]), 0),
            ]
        )
    return md_table(
        [
            "arm",
            "overrides",
            "val loss @0",
            "best val loss",
            "best step",
            "steps",
            "tokens",
            "wall min",
            "peak alloc GB",
            "peak reserved GB",
            "tokens/s",
        ],
        rows,
    )


def curves_figure(
    ctx: Ctx,
    arms: dict[str, dict],
    name: str,
    title: str,
) -> Path | None:
    def draw(plt):
        fig, ax = plt.subplots(figsize=(7, 4))
        for arm, s in arms.items():
            pts = val_curve(ctx, s["run_dir"])
            ax.plot(
                [p[0] / 1e6 for p in pts],
                [p[1] for p in pts],
                marker="o",
                label=arm,
            )
        ax.set_xlabel("training tokens (M)")
        ax.set_ylabel("validation loss (supervised tokens)")
        ax.set_title(title)
        ax.legend(fontsize=7)
        return fig

    return figure(ctx.figures / f"p12_{name}.png", draw)


def stage_lora_sweep(ctx: Ctx) -> dict:
    model = ctx.state["selected_model"]
    common = [
        f"train.token_budget={ctx.budget}",
        "train.eval_every=12",
    ]
    arms_spec = {
        f"lora{r}": [
            f"lora.r={r}",
            f"lora.alpha={2 * r}",
        ]
        for r in (8, 16, 32, 64)
    }
    arms_spec["lora16attn"] = [
        "lora.r=16",
        "lora.alpha=32",
        f"lora.targets={ATTN}",
    ]
    if ctx.dry:
        arms_spec = {k: v for k, v in arms_spec.items() if k in ("lora8", "lora16attn")}
    arms = {}
    for arm, ov in arms_spec.items():
        s = run_training_arm(
            ctx,
            f"p12-{tag_of(model)}-{arm}-s42",
            common + ov,
            model,
        )
        s["overrides"] = " ".join(ov)
        arms[arm] = s
    selected = min(
        arms,
        key=lambda a: arms[a]["best_val_loss"],
    )
    fig = curves_figure(
        ctx,
        arms,
        "lora_sweep",
        "12.2.3 LoRA configuration (equal token budget)",
    )
    done = {
        "arms": arms,
        "selected": selected,
        "selected_overrides": arms_spec[selected],
        "figure": ctx.rel(fig) if fig else None,
        "budget": ctx.budget,
    }
    lines = [
        "# Phase 12.2.3 - LoRA configuration sweep",
        "",
        f"Base `{model}`, NF4, LR 2e-4, cosine, {ctx.budget:,} training tokens per arm "
        "(same seed, so the same packed rows in the same order), validation loss over the 176 "
        "validation pairs' supervised tokens. alpha = 2r throughout.",
        "",
        arms_table(arms),
        "",
        f"**Selected: {selected}** (lowest best validation loss).",
        "",
    ]
    if fig:
        lines.append(f"![curves](figures/{fig.name})")
    (ctx.reports / "llm_lora_sweep.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return done


def tag_of(model: str) -> str:
    low = model.lower()
    if "deepseek" in low:
        return "dscoder67b"
    if "1.5b" in low:
        return "qwen15b"
    return "qwen7b"


def commit_lora(ctx: Ctx, done: dict) -> None:
    arms = done["arms"]
    parts = ", ".join(
        f"{a} {s['best_val_loss']:.4f} "
        f"({round(s['wall_s'] / 60)} min, {s['peak_reserved_gb']} GB)"
        for a, s in arms.items()
    )
    best, worst = (
        done["selected"],
        max(arms, key=lambda a: arms[a]["best_val_loss"]),
    )
    spread = arms[worst]["best_val_loss"] - arms[best]["best_val_loss"]
    dod = (
        "`src/llm/run.py` + `src/llm/train.py` + `src/llm/pipeline.py` + `configs/llm.yaml` + "
        "`reports/llm_lora_sweep.md` + `reports/figures/p12_lora_sweep.png`: arms run one after "
        f"another, each a fresh process with **{done['budget']:,} training tokens**, LR 2e-4 "
        "cosine, the same seed and therefore the same packed rows, selected on best validation "
        f"loss over the 176 validation pairs (test untouched). Best validation loss: {parts}. "
        f"**Selected {best}**; the spread across all five arms is {spread:.4f} nats, "
        f"starting from {arms[best]['val_loss_start']:.4f} before training. Each arm trained at "
        "12.2.2's measured saturation (one packed 2,048-token row per micro-step, 32,768 tokens per "
        "optimizer step)"
    )
    msg = (
        f"12.2.3: {best} selected at val loss {arms[best]['best_val_loss']:.4f} across "
        f"{len(arms)} equal-budget arms (spread {spread:.4f})\n\n{parts}\n"
    )
    files = [
        ctx.reports / "llm_lora_sweep.md",
        ctx.figures / "p12_lora_sweep.png",
    ]
    commit_row(
        ctx,
        "12.2.3",
        dod,
        msg,
        files,
        keep_open=False,
    )


def stage_hparam_sweep(ctx: Ctx) -> dict:
    model = ctx.state["selected_model"]
    lora = ctx.state["lora_overrides"]
    common = [
        f"train.token_budget={ctx.budget}",
        "train.eval_every=12",
    ]
    lora_arm = ctx.state["lora_arm_name"]
    arms: dict[str, dict] = {"lr2e-4 (12.2.3 winner)": ctx.state["lora_arm_summary"]}
    spec = {
        "lr1e-4": ["train.lr=1e-4"],
        "lr5e-5": ["train.lr=5e-5"],
    }
    if ctx.dry:
        spec = {"lr1e-4": ["train.lr=1e-4"]}
    for arm, ov in spec.items():
        s = run_training_arm(
            ctx,
            f"p12-{tag_of(model)}-{arm}-s42",
            common + lora + ov,
            model,
        )
        s["overrides"] = " ".join(ov)
        arms[arm] = s
    lr_best = min(
        arms,
        key=lambda a: arms[a]["best_val_loss"],
    )
    lr_ov = [] if lr_best.startswith("lr2e-4") else spec[lr_best]
    base = lora + lr_ov
    extra = {
        "warmup0.1": ["train.warmup_frac=0.1"],
        "tok65k": ["train.tokens_per_step=65536"],
        "syn1x": ["data.synthetic_ratio=1.0"],
    }
    if ctx.dry:
        extra = {"warmup0.1": ["train.warmup_frac=0.1"]}
    stage3: dict[str, dict] = {f"{lr_best} (base)": arms[lr_best]}
    for arm, ov in extra.items():
        s = run_training_arm(
            ctx,
            f"p12-{tag_of(model)}-{arm}-s42",
            common + base + ov,
            model,
        )
        s["overrides"] = " ".join(ov)
        stage3[arm] = s
    final = min(
        stage3,
        key=lambda a: stage3[a]["best_val_loss"],
    )
    final_ov = base + ([] if final.endswith("(base)") else extra[final])
    # "epochs": the same configuration at twice the budget - does more data still help?
    twice = run_training_arm(
        ctx,
        f"p12-{tag_of(model)}-budget2x-s42",
        [
            f"train.token_budget={2 * ctx.budget}",
            "train.eval_every=24",
        ]
        + final_ov,
        model,
    )
    twice["overrides"] = "2x token budget"
    done = {
        "lr_arms": arms,
        "lr_selected": lr_best,
        "extra_arms": stage3,
        "selected": final,
        "selected_overrides": final_ov,
        "budget2x": twice,
        "lora_arm": lora_arm,
        "budget": ctx.budget,
    }
    allarms = {
        **arms,
        **{k: v for k, v in stage3.items() if not k.endswith("(base)")},
        "budget2x": twice,
    }
    fig = curves_figure(
        ctx,
        allarms,
        "hparam_sweep",
        "12.2.6 hyper-parameters (equal budget)",
    )
    done["figure"] = ctx.rel(fig) if fig else None
    lines = [
        "# Phase 12.2.6 - hyper-parameter sweep",
        "",
        f"Base `{model}` with 12.2.3's `{' '.join(lora)}`; {ctx.budget:,} tokens per arm, "
        "coordinate search on validation loss (LR first, then warmup / tokens per step / "
        "synthetic mix against the LR winner), then the winner at 2x budget.",
        "",
        "## LR",
        "",
        arms_table(arms),
        "",
        f"LR winner: **{lr_best}**",
        "",
        "## Warmup, batch (tokens per optimizer step), data mix",
        "",
        arms_table(stage3),
        "",
        f"Winner: **{final}** -> full-run overrides `{' '.join(final_ov)}`",
        "",
        "## Budget (epochs proxy)",
        "",
        arms_table({"budget2x": twice}),
        "",
    ]
    if fig:
        lines.append(f"![curves](figures/{fig.name})")
    (ctx.reports / "llm_hparam_sweep.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return done


def commit_hparam(ctx: Ctx, done: dict) -> None:
    lr = done["lr_arms"]
    ex = done["extra_arms"]
    tw = done["budget2x"]
    lr_txt = ", ".join(f"{a.split(' ')[0]} {s['best_val_loss']:.4f}" for a, s in lr.items())
    ex_txt = ", ".join(f"{a.split(' ')[0]} {s['best_val_loss']:.4f}" for a, s in ex.items())
    base_best = ex[done["selected"]]["best_val_loss"]
    dod = (
        "`src/llm/pipeline.py` + `src/llm/run.py` + `reports/llm_hparam_sweep.md` + "
        "`reports/figures/p12_hparam_sweep.png`: coordinate search on validation loss (test "
        f"untouched), {done['budget']:,} tokens per arm, arms sequential. **LR**: {lr_txt} -> "
        f"{done['lr_selected'].split(' ')[0]}. **Warmup / tokens per optimizer step / synthetic "
        f"mix** against that: {ex_txt} -> **{done['selected'].split(' ')[0]}**. **Budget**: the "
        f"winner at 2x tokens reaches {tw['best_val_loss']:.4f} against {base_best:.4f} "
        f"({tw['best_val_loss'] - base_best:+.4f}), which is the evidence the full run's single "
        f"epoch rests on. Full-run overrides: `{' '.join(done['selected_overrides'])}`. Every arm "
        "logs wall time, tokens/s and peak VRAM in the report table"
    )
    msg = (
        f"12.2.6: LR {done['lr_selected'].split(' ')[0]} and {done['selected'].split(' ')[0]} "
        f"selected on validation loss; 2x budget moves it "
        f"{tw['best_val_loss'] - base_best:+.4f}"
        f"\n\nLR arms: {lr_txt}\nWarmup/batch/mix arms: {ex_txt}\n"
    )
    files = [
        ctx.reports / "llm_hparam_sweep.md",
        ctx.figures / "p12_hparam_sweep.png",
    ]
    commit_row(
        ctx,
        "12.2.6",
        dod,
        msg,
        files,
        keep_open=False,
    )


# -- train --------------------------------------------------------------------------------------


def stage_train(ctx: Ctx) -> dict:
    model = ctx.state["selected_model"]
    ov = list(ctx.state["final_overrides"])
    if ctx.dry:
        ov += [f"train.token_budget={ctx.budget}"]
    else:
        ov += ["train.epochs=1.0", "train.eval_every=50"]
    s = run_training_arm(
        ctx,
        f"p12-{tag_of(model)}-full-s42",
        ov,
        model,
    )
    s["overrides"] = " ".join(ov)
    data = json.loads((ROOT / s["run_dir"] / "data.json").read_text(encoding="utf-8"))
    pts = val_curve(ctx, s["run_dir"])
    rows = [json.loads(x) for x in (ROOT / s["run_dir"] / "train.jsonl").read_text().splitlines()]
    train_pts = [(r["tokens"], r["loss"]) for r in rows if "loss" in r]

    def draw(plt):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(
            [p[0] / 1e6 for p in train_pts],
            [p[1] for p in train_pts],
            lw=0.8,
            label="train",
        )
        ax.plot(
            [p[0] / 1e6 for p in pts],
            [p[1] for p in pts],
            marker="o",
            label="validation",
        )
        ax.set_xlabel("training tokens (M)")
        ax.set_ylabel("loss (supervised tokens)")
        ax.legend()
        return fig

    fig = figure(ctx.figures / "p12_train_loss.png", draw)
    done = {
        "summary": s,
        "data": data,
        "adapter": s.get("adapter"),
        "figure": ctx.rel(fig) if fig else None,
    }
    lines = [
        "# Phase 12.2.5 - full QLoRA training run",
        "",
        f"Base `{model}`; overrides `" f"{s['overrides']}`; run directory `{s['run_dir']}`.",
        "",
        arms_table({"full": s}),
        "",
        "## Data",
        "",
        "```json",
        json.dumps(data.get("train"), indent=1),
        "```",
        "",
        f"Dropped for exceeding 4,096 tokens (never truncated): "
        f"`{data.get('dropped_over_max_tokens')}`",
        f"Train/test contamination (exact IR text, ids, scribes): "
        f"`{data.get('contamination_vs_test')}`",
        f"Runtime checks: `{json.dumps({k: v for k, v in data.get('runtime_checks', {}).items() if k != 'sdpa_kernels'})}`",
        "",
        f"Pack: `{json.dumps(s.get('pack'))}`",
        "",
    ]
    if fig:
        lines.append(f"![loss](figures/{fig.name})")
    (ctx.reports / "llm_train.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return done


def commit_train(ctx: Ctx, done: dict) -> None:
    s = done["summary"]
    tr = done["data"].get("train", {})
    real = sum(v["tokens"] for k, v in tr.items() if k not in ("synthetic", "synthetic_used"))
    syn = tr.get("synthetic_used", {}).get("tokens", 0)
    dod = (
        "`src/llm/train.py` + `src/llm/run.py` + `configs/llm.yaml` + `reports/llm_train.md` + "
        "`reports/figures/p12_train_loss.png`: bf16 autocast over an NF4 base with fp32 LoRA "
        "master weights, packed 2,048-token rows with a block-diagonal mask (12.2.2), loss on "
        "completion tokens only, gradient accumulation to a token-count effective batch, cosine "
        f"LR with linear warmup, overrides `{s['overrides']}`. **{s['tokens']:,} tokens "
        f"({real:,} real + {syn:,} synthetic available) in {s['total_steps']} optimizer steps, "
        f"{s['wall_s'] / 3600:.2f} h wall, {s['tokens'] / max(1, s['wall_s']):.0f} tokens/s, peak "
        f"{s['peak_allocated_gb']} GB allocated / {s['peak_reserved_gb']} GB reserved.** "
        f"Validation loss {s['val_loss_start']:.4f} -> best {s['best_val_loss']:.4f} at step "
        f"{s['best_step']} (final {s['val_loss_final']}). Examples over 4,096 tokens were dropped, "
        f"not truncated: {done['data'].get('dropped_over_max_tokens')}. Train/test overlap: "
        f"{done['data'].get('contamination_vs_test')} (identical-IR fa_bresler exercises are "
        f"handled in 12.2.7). Adapter at `{s.get('adapter', '').replace(chr(92), '/')}` (gitignored)"
    )
    msg = (
        f"12.2.5: adapter trained on {s['tokens'] / 1e6:.1f}M tokens in "
        f"{s['wall_s'] / 3600:.1f} h, val loss "
        f"{s['val_loss_start']:.3f} -> {s['best_val_loss']:.3f}\n\n"
        f"Peak {s['peak_reserved_gb']} GB reserved, "
        f"{s['tokens'] / max(1, s['wall_s']):.0f} tokens/s.\n"
    )
    files = [
        ctx.reports / "llm_train.md",
        ctx.figures / "p12_train_loss.png",
    ]
    commit_row(
        ctx,
        "12.2.5",
        dod,
        msg,
        files,
        keep_open=False,
    )


# -- compare ------------------------------------------------------------------------------------


def paired_bootstrap(
    a: list[bool],
    b: list[bool],
    n: int = 2000,
    seed: int = 0,
) -> tuple:
    import random

    rng = random.Random(seed)
    k = len(a)
    diffs = []
    for _ in range(n):
        idx = [rng.randrange(k) for _ in range(k)]
        diffs.append(sum(a[i] - b[i] for i in idx) / k)
    diffs.sort()
    return (
        round(diffs[int(0.025 * n)], 4),
        round(diffs[int(0.975 * n)], 4),
    )


def stage_compare(ctx: Ctx) -> dict:
    from src.codegen import prompt as template
    from src.llm import fewshot, pairs

    model = ctx.state["selected_model"]
    adapter = ctx.state["adapter"]
    test = pairs_for(ctx, "test")
    train_pool = pairs.load(
        "train",
        sources=["hdbpmn", "fa_bresler"],
    )
    held = pairs.load("validation") + pairs.load("test")
    fs = fewshot.FewShot(
        train_pool,
        held,
        template,
        k=2,
    )
    arms = {
        "zero-shot": (template.build_messages, None),
        "few-shot (k=2)": (fs.messages, None),
        "LoRA": (template.build_messages, adapter),
    }
    results, scored_by = {}, {}
    for arm, (build, adp) in arms.items():
        name = arm.split(" ")[0].replace("-", "").lower()
        report, rows = generate_arm(
            ctx,
            "compare",
            name,
            model,
            test,
            build,
            adapter=adp,
        )
        scored, summary = score(rows, "test")
        results[arm] = {
            "meta": report,
            "summary": summary,
        }
        scored_by[arm] = scored
        (ctx.stage_dir("compare") / f"{name}.scored.json").write_text(
            json.dumps(scored, indent=1),
            encoding="utf-8",
        )
    train_ir = {p["ir_text"] for p in pairs.load("train")}
    novel = {p["diagram_id"] for p in test if p["ir_text"] not in train_ir}
    for arm, scored in scored_by.items():
        sub = [s for s in scored if s["diagram_id"] in novel]
        results[arm]["novel_ir"] = {
            "n": len(sub),
            "functional": round(
                sum(s["functional"] for s in sub) / max(1, len(sub)),
                4,
            ),
        }
    lora = [s["functional"] for s in scored_by["LoRA"]]
    for arm in ("zero-shot", "few-shot (k=2)"):
        results[arm]["lora_minus_arm_ci95"] = paired_bootstrap(
            lora,
            [s["functional"] for s in scored_by[arm]],
        )
    f = {a: results[a]["summary"]["functional"] for a in results}
    lora_wins = f["LoRA"] > max(
        f["zero-shot"],
        f["few-shot (k=2)"],
    )
    done = {
        "results": results,
        "lora_wins": lora_wins,
        "n": len(test),
        "novel_n": len(novel),
        "fewshot_excluded_identical_ir": fs.excluded_identical_ir,
        "lora_generations": ctx.rel(ctx.stage_dir("compare") / "lora.jsonl"),
    }

    def draw(plt):
        fig, ax = plt.subplots(figsize=(7, 4))
        metrics = ["syntax", "executes", "functional"]
        width = 0.25
        for i, arm in enumerate(results):
            ax.bar(
                [m + i * width for m in range(3)],
                [results[arm]["summary"][k] for k in metrics],
                width,
                label=arm,
            )
        ax.set_xticks(
            [m + width for m in range(3)],
            metrics,
        )
        ax.set_ylim(0, 1)
        ax.set_ylabel(f"rate over {len(test)} test pairs")
        ax.legend()
        return fig

    fig = figure(ctx.figures / "p12_three_way.png", draw)
    rows = []
    for arm, r in results.items():
        s, m = r["summary"], r["meta"]
        rows.append(
            [
                arm,
                pct(s["contract"]),
                pct(s["syntax"]),
                pct(s["executes"]),
                pct(s["functional"]),
                pct(r["novel_ir"]["functional"]),
                pct(s["dropped_node_rate"]),
                s["invented_per_program"],
                r.get("lora_minus_arm_ci95", ""),
                m["batch"],
                m["ms_per_sample"],
            ]
        )
    lines = [
        "# Phase 12.2.7 - zero-shot vs few-shot vs LoRA (test)",
        "",
        f"`{model}`, NF4, frozen 12.2.4 template (`{template.TEMPLATE_ID}`), greedy, all "
        f"{len(test)} test pairs. Few-shot: k=2 same-type train exemplars nearest in IR size, "
        f"{fs.excluded_identical_ir} train pairs excluded because their IR text also occurs in "
        f"validation/test. `novel IR` = the {len(novel)} test pairs whose IR text never occurs "
        "in train (fa_bresler exercises repeat across writers).",
        "",
        md_table(
            [
                "arm",
                "contract",
                "syntax",
                "executes",
                "functional pass@1",
                "functional (novel IR)",
                "dropped nodes",
                "invented/program",
                "LoRA - arm, 95% CI",
                "batch",
                "ms/sample",
            ],
            rows,
        ),
        "",
        f"**LoRA wins on functional pass@1: {lora_wins}**",
        "",
    ]
    if fig:
        lines.append(f"![three-way](figures/{fig.name})")
    (ctx.reports / "llm_compare.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    done["figure"] = ctx.rel(fig) if fig else None
    return done


def commit_compare(ctx: Ctx, done: dict) -> None:
    r = done["results"]
    parts = "; ".join(
        f"**{a}** contract {pct(v['summary']['contract'])}, syntax {pct(v['summary']['syntax'])}, "
        f"executes {pct(v['summary']['executes'])}, functional **{pct(v['summary']['functional'])}** "
        f"(novel-IR {pct(v['novel_ir']['functional'])})"
        for a, v in r.items()
    )
    ci = (
        r["zero-shot"]["lora_minus_arm_ci95"],
        r["few-shot (k=2)"]["lora_minus_arm_ci95"],
    )
    verdict = (
        "**Fine-tuning wins**"
        if done["lora_wins"]
        else "**Fine-tuning does not win, and the row stays open**"
    )
    dod = (
        "`src/llm/pipeline.py` + `src/llm/fewshot.py` + `src/llm/score.py` + "
        "`reports/llm_compare.md` + `reports/figures/p12_three_way.png`: one base, one frozen "
        f"template, greedy decoding, one denominator ({done['n']} test pairs); the arms differ only "
        "in exemplar turns (few-shot, k=2, same type and source, nearest IR size, "
        f"{done['fewshot_excluded_identical_ir']} train pairs barred for sharing IR text with "
        "validation/test) or the adapter (LoRA). " + parts + f". {verdict} on functional pass@1: "
        f"paired bootstrap 95% CI of LoRA minus zero-shot {ci[0]}, minus few-shot {ci[1]}. The "
        f"novel-IR column restricts to the {done['novel_n']} test pairs whose IR text never occurs "
        "in train, since fa_bresler's exercises repeat across writers and a scribe-disjoint split "
        "does not make them unseen"
    )
    f = {a: pct(v["summary"]["functional"]) for a, v in r.items()}
    msg = (
        f"12.2.7: functional pass@1 LoRA {f['LoRA']} vs "
        f"few-shot {f['few-shot (k=2)']} vs "
        f"zero-shot {f['zero-shot']} on test\n\n"
        f"{'LoRA wins.' if done['lora_wins'] else 'LoRA does not win; row kept open.'}\n"
    )
    files = [
        ctx.reports / "llm_compare.md",
        ctx.figures / "p12_three_way.png",
    ]
    commit_row(
        ctx,
        "12.2.7",
        dod,
        msg,
        files,
        keep_open=not done["lora_wins"],
    )


# -- quality ------------------------------------------------------------------------------------


def reference_rows(ctx: Ctx, split: str) -> list[dict]:
    from src.codegen import prompt as template

    return [
        {
            "diagram_id": p["diagram_id"],
            "source": p["source"],
            "text": template.completion(p),
            "hit_limit": False,
        }
        for p in pairs_for(ctx, split)
    ]


def stage_quality(ctx: Ctx) -> dict:
    from src.llm.generate import read_jsonl

    lora_rows = read_jsonl(ROOT / ctx.state["lora_generations"])
    scored, summary = score(lora_rows, "test")
    ref_scored, ref_summary = score(
        reference_rows(ctx, "test"),
        "test",
    )
    (ctx.stage_dir("quality") / "lora.scored.json").write_text(json.dumps(scored, indent=1))
    worst = [s for s in scored if not s["functional"]][:8]
    done = {
        "lora": summary,
        "reference": ref_summary,
        "n": len(scored),
        "examples": worst,
    }
    rows = []
    for k in (
        "contract",
        "syntax",
        "executes",
        "functional",
        "dropped_node_rate",
        "programs_with_dropped",
        "invented_per_program",
        "programs_with_invented",
    ):
        rows.append(
            [
                k,
                summary[k] if "per_program" in k else pct(summary[k]),
                ref_summary[k] if "per_program" in k else pct(ref_summary[k]),
            ]
        )
    cc = summary.get("codecheck")
    lines = [
        "# Phase 12.3.1 / 12.3.2 / 12.3.3 / 12.3.8 - quality of the LoRA model's test outputs",
        "",
        f"{len(scored)} test generations from 12.2.7's LoRA arm; the reference column "
        "scores 12.1.6's reference programs through the identical scorer.",
        "",
        md_table(["metric", "LoRA", "reference programs"], rows),
        "",
    ]
    if cc:
        lines += [
            f"`src.eval.codecheck` (data agent's helpers): syntax {pct(cc['syntax']['rate'])} "
            f"`{cc['syntax']['kinds']}`, executes {pct(cc['executes']['rate'])} "
            f"`{cc['executes']['kinds']}`",
            "",
        ]
    lines += [
        "Functional failure reasons: `" + json.dumps(summary["functional_reasons"]) + "`",
        "",
        "Sandbox kinds: `" + json.dumps(summary["exec_kinds"]) + "`",
        "",
        "## Failures (first 8)",
        "",
    ]
    for w in worst:
        lines.append(
            f"- `{w['diagram_id']}` ({w['source']}): {w['functional_reason']}; syntax "
            f"{w['syntax']}, executes {w['executes']}, dropped {w['hallucination']['dropped'][:5]}, "
            f"invented {w['hallucination']['invented'][:5]}"
        )
    (ctx.reports / "llm_quality.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return done


def commit_quality(ctx: Ctx, done: dict) -> None:
    s, ref, n = done["lora"], done["reference"], done["n"]
    cc = s.get("codecheck")
    files = [ctx.reports / "llm_quality.md"]
    head = (
        f"`src/llm/score.py` + `src/llm/functional.py` + `src/llm/pipeline.py` + "
        f"`reports/llm_quality.md`: all {n} LoRA test generations, one denominator "
        "(unparseable replies count as failures everywhere). "
    )
    syn = cc["syntax"]["rate"] if cc else s["syntax"]
    syn_ok = syn >= 0.95
    commit_row(
        ctx,
        "12.3.1",
        head
        + (
            f"**Syntactic validity {pct(syn)}** by `src.eval.codecheck.syntactic_many` (ast.parse + "
            f"compile), {pct(s['syntax'])} by `score.syntax_ok`; the strict one-fence contract holds on "
            f"{pct(s['contract'])}; reference programs {pct(ref['syntax'])}. Threshold >= 95%: "
            f"{'met' if syn_ok else 'not met'}"
        ),
        f"12.3.1: {pct(syn)} of {n} LoRA test programs parse (threshold 95%"
        f"{'' if syn_ok else ', not met'})\n",
        files,
        keep_open=not syn_ok,
    )
    exe = cc["executes"]["rate"] if cc else s["executes"]
    exe_ok = exe >= 0.85
    commit_row(
        ctx,
        "12.3.2",
        head
        + (
            f"**Executability {pct(exe)}** by `src.eval.codecheck.executable_many` (sandbox, stubbed "
            f"free names, entry point invoked; kinds {cc['executes']['kinds'] if cc else 'n/a'}); the "
            f"stricter `score` definition (module runs as __main__ **and** the entry runs under the "
            f"functional driver) gives {pct(s['executes'])}, reference programs {pct(ref['executes'])}. "
            f"Threshold >= 85%: {'met' if exe_ok else 'not met'}"
        ),
        f"12.3.2: {pct(exe)} of {n} LoRA test programs run in the sandbox "
        f"(threshold 85%{'' if exe_ok else ', not met'})\n",
        files,
        keep_open=not exe_ok,
    )
    fn = s["functional"]
    fn_ok = fn >= 0.70
    commit_row(
        ctx,
        "12.3.3",
        head
        + (
            f"`src/llm/functional.py` + `tests/test_llm_functional.py`: per-diagram tests derived from "
            "the drawing, run in 11.2.9's sandbox - state machines: every word of length 0-3 over the "
            "edge-label alphabet against the IR automaton (comma labels are two symbols, epsilon "
            "closure); flowcharts: the complete decision tree to depth 12 with recording stubs, "
            "passing iff every reachable drawn operation runs on some path, no path reverses the "
            "drawn order, and no path runs two exclusive branches. Mutation-verified on the "
            "validation references: dropped operations pass 3.1%, swapped operations 17.4%, redirected "
            "transitions 22.9%, flipped accepting sets 10.6%, while branch-polarity swaps pass at the "
            f"reference's own rate. **pass@1 {pct(fn)}** (fa_bresler "
            f"{pct(s['by_source/fa_bresler']['functional'])}, hdbpmn "
            f"{pct(s['by_source/hdbpmn']['functional'])}); the reference programs pass "
            f"{pct(ref['functional'])} under the same tests, because 12.1.6's emitter collapses `a,b` "
            "into one trigger and serialises forks as if/else. Failure reasons "
            f"{s['functional_reasons']}. Threshold >= 70%: {'met' if fn_ok else 'not met'}"
        ),
        f"12.3.3: pass@1 {pct(fn)} on {n} test diagrams against reference programs' "
        f"{pct(ref['functional'])} (threshold 70%{'' if fn_ok else ', not met'})\n",
        files,
        keep_open=not fn_ok,
    )
    commit_row(
        ctx,
        "12.3.8",
        head
        + (
            "Dropped = drawn operations (flowchart `process` nodes) or states that no called/defined "
            "name or literal in the program names, via the functional driver's resolver; invented = "
            "called or defined operation names that resolve to no drawn node, excluding condition "
            f"predicates and scaffolding. **Dropped-node rate {pct(s['dropped_node_rate'])} "
            f"({pct(s['programs_with_dropped'])} of programs drop at least one), "
            f"{s['invented_per_program']} invented names per program "
            f"({pct(s['programs_with_invented'])} of programs)**; reference programs: dropped "
            f"{pct(ref['dropped_node_rate'])}, invented {ref['invented_per_program']} per program"
        ),
        f"12.3.8: LoRA drops {pct(s['dropped_node_rate'])} of drawn nodes and invents "
        f"{s['invented_per_program']} names per program on test\n",
        files,
        keep_open=False,
    )


# -- repair -------------------------------------------------------------------------------------

REPAIR_PROMPT = (
    "The program above failed its check: {reason}. Reply with the corrected program only, as "
    "exactly one fenced code block."
)


def failure_reason(s: dict) -> str:
    if not s["syntax"]:
        return f"it does not parse ({s.get('syntax_detail', '')})"
    if not s["executes"]:
        return f"it raised when run ({s.get('exec_kind')}: " f"{s.get('exec_detail', '')[:200]})"
    names = {
        "missing_operation": "a drawn operation is never executed on any path",
        "order_violation": "an operation runs before one the diagram draws ahead of it",
        "exclusive_branches_both_ran": "both branches of an exclusive decision run on one path",
        "verdicts_differ": "it accepts or rejects some input words differently from the drawn automaton",
        "no_acceptor": "it exposes no way to run the automaton on a word",
        "never_returns": "no path returns",
        "crash_or_nontermination": "some path crashes or does not terminate",
    }
    return names.get(
        s["functional_reason"],
        s["functional_reason"],
    )


def stage_repair(ctx: Ctx) -> dict:
    from src.codegen import prompt as template
    from src.llm import pairs

    model, adapter = (
        ctx.state["selected_model"],
        ctx.state["adapter"],
    )
    scored = json.loads((ctx.stage_dir("quality") / "lora.scored.json").read_text())
    from src.llm.generate import read_jsonl

    replies = {r["diagram_id"]: r for r in read_jsonl(ROOT / ctx.state["lora_generations"])}
    index = {p["diagram_id"]: p for p in pairs.load("test")}
    failed = [s for s in scored if not s["functional"]]
    reasons = {s["diagram_id"]: failure_reason(s) for s in failed}

    def build(p: dict) -> list:
        return template.build_messages(p) + [
            {
                "role": "assistant",
                "content": replies[p["diagram_id"]]["text"],
            },
            {
                "role": "user",
                "content": REPAIR_PROMPT.format(reason=reasons[p["diagram_id"]]),
            },
        ]

    targets = [index[s["diagram_id"]] for s in failed]
    if not targets:
        return {"failed_before": 0}
    report, rows = generate_arm(
        ctx,
        "repair",
        "repair",
        model,
        targets,
        build,
        adapter=adapter,
    )
    rescored, summary = score(rows, "test")
    fixed = sum(s["functional"] for s in rescored)
    after = {s["diagram_id"]: s for s in rescored}
    merged = [after.get(s["diagram_id"], s) for s in scored]
    total_after = sum(s["functional"] for s in merged) / len(merged)
    before = sum(s["functional"] for s in scored) / len(scored)
    broke_syntax = sum(1 for s in rescored if not s["syntax"])
    done = {
        "n": len(scored),
        "failed_before": len(failed),
        "fixed": fixed,
        "success_after_repair": round(
            fixed / len(failed),
            4,
        ),
        "pass_before": round(before, 4),
        "pass_after": round(total_after, 4),
        "repair_meta": report,
        "syntax_after_repair": summary["syntax"],
        "executes_after_repair": summary["executes"],
        "broke_syntax": broke_syntax,
        "reasons": summary["functional_reasons"],
    }
    lines = [
        "# Phase 12.3.9 - one-shot repair loop (test)",
        "",
        f"Every LoRA test generation that failed its functional test ({len(failed)} of "
        f"{len(scored)}) gets one repair turn: the original messages, the model's reply, and "
        "the failure stated in words (syntax error text, sandbox exception, or the functional "
        "test's reason), with the same adapter and decoding.",
        "",
        md_table(
            ["", "value"],
            [
                ["failed before", len(failed)],
                ["fixed by one repair", fixed],
                ["success after repair", pct(done["success_after_repair"])],
                ["pass@1 before", pct(before)],
                ["pass after one repair", pct(total_after)],
                ["repaired programs that no longer parse", broke_syntax],
                [
                    "repair batch / ms per sample",
                    f"{report['batch']} / {report['ms_per_sample']}",
                ],
            ],
        ),
        "",
        f"Remaining failure reasons: " f"`{json.dumps(summary['functional_reasons'])}`",
    ]
    (ctx.reports / "llm_repair.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return done


def commit_repair(ctx: Ctx, done: dict) -> None:
    if not done.get("failed_before"):
        dod = (
            "`src/llm/pipeline.py`: no LoRA test generation failed, so there was nothing to repair"
        )
        commit_row(
            ctx,
            "12.3.9",
            dod,
            "12.3.9: nothing to repair on test\n",
            [],
            keep_open=False,
        )
        return
    dod = (
        "`src/llm/pipeline.py` + `reports/llm_repair.md`: every failing LoRA test program gets "
        "exactly one repair turn - its own reply plus the failure in words (parser message, sandbox "
        "exception kind, or the functional test's reason) - same adapter, greedy, batch from a "
        f"measured sweep ({done['repair_meta']['batch']}, {done['repair_meta']['ms_per_sample']} "
        f"ms/sample). **{done['fixed']} of {done['failed_before']} failures fixed "
        f"({pct(done['success_after_repair'])} success-after-repair); pass@1 "
        f"{pct(done['pass_before'])} -> {pct(done['pass_after'])} after one repair** over all "
        f"{done['n']} test pairs; {done['broke_syntax']} repaired replies no longer parse. Remaining "
        f"reasons {done['reasons']}"
    )
    msg = (
        f"12.3.9: one repair fixes {done['fixed']} of {done['failed_before']} test failures, "
        f"pass {pct(done['pass_before'])} -> {pct(done['pass_after'])}\n"
    )
    commit_row(
        ctx,
        "12.3.9",
        dod,
        msg,
        [ctx.reports / "llm_repair.md"],
        keep_open=False,
    )


# -- similarity (report only) -------------------------------------------------------------------


def stage_similarity(ctx: Ctx) -> dict:
    import importlib

    for name in (
        "src.eval.similarity",
        "src.codegen.similarity",
        "src.eval.codebleu",
    ):
        try:
            module = importlib.import_module(name)
        except Exception:  # noqa: BLE001
            continue
        fn = next(
            (
                getattr(module, a)
                for a in (
                    "score_many",
                    "similarity_many",
                    "score",
                    "similarity",
                )
                if hasattr(module, a)
            ),
            None,
        )
        if fn is None:
            continue
        from src.llm import pairs
        from src.llm.generate import read_jsonl
        from src.llm.score import code_of

        index = {p["diagram_id"]: p for p in pairs.load("test")}
        rows = read_jsonl(ROOT / ctx.state["lora_generations"])
        try:
            values = fn(
                [
                    (
                        code_of(r["text"]),
                        index[r["diagram_id"]]["target_code"],
                    )
                    for r in rows
                ]
            )
        except TypeError:
            values = [
                fn(
                    code_of(r["text"]),
                    index[r["diagram_id"]]["target_code"],
                )
                for r in rows
            ]
        text = json.dumps(
            values if isinstance(values, dict) else {"per_row": values},
            default=str,
        )[:20000]
        (ctx.reports / "llm_similarity.md").write_text(
            f"# 12.3.5 metric on LoRA test outputs (`{name}.{fn.__name__}`)\n\n"
            f"```json\n{text}\n```\n"
        )
        return {
            "module": name,
            "function": fn.__name__,
        }
    return {"skipped": "no similarity metric module found"}


# -- export -------------------------------------------------------------------------------------


def stage_export(ctx: Ctx) -> dict:
    from src.llm import generate

    model, adapter = (
        ctx.state["selected_model"],
        ctx.state["adapter"],
    )
    d = ctx.stage_dir("export")
    from transformers import AutoTokenizer

    from src.codegen import prompt as template

    tok = AutoTokenizer.from_pretrained(model)
    prompts = d / "check_prompts.jsonl"
    with prompts.open("w", encoding="utf-8") as h:
        for p in pairs_for(ctx, "validation")[: 4 if ctx.dry else 8]:
            h.write(
                json.dumps(
                    {
                        "prompt": generate.render(
                            tok,
                            template.build_messages(p),
                        )
                    }
                )
                + "\n"
            )
    began = time.perf_counter()
    result = run_job(
        ctx,
        "export",
        {
            "model": model,
            "adapter": adapter,
            "out": str(d / "artifacts"),
            "prompts": str(prompts),
        },
        "export",
    )
    result["wall_s"] = round(
        time.perf_counter() - began,
        1,
    )
    # smoke-serve the Q4_K_M file and require a parseable fenced program back
    from src.llm import export
    from src.llm.score import code_of, syntax_ok

    p = pairs_for(ctx, "validation")[0]
    with export.LlamaServer(Path(result["gguf"]["Q4_K_M"]["path"])) as server:
        reply = server.chat(
            template.build_messages(p),
            max_tokens=1600,
        )
    result["serve_smoke"] = {
        "diagram_id": p["diagram_id"],
        "parses": syntax_ok(code_of(reply["text"]))[0],
        "new_tokens": reply["new_tokens"],
        "seconds": round(reply["seconds"], 2),
    }
    mc, g = result["merge_check"], result["gguf"]
    lines = [
        "# Phase 12.2.8 - adapter merge and export",
        "",
        md_table(
            ["step", "result"],
            [
                ["merge (bf16, CPU)", json.dumps(result.get("merge", "reused"))],
                ["merged vs unmerged logits", json.dumps(mc)],
                ["GGUF f16", json.dumps(g["f16"])],
                ["GGUF Q8_0", json.dumps(g["Q8_0"])],
                ["GGUF Q4_K_M", json.dumps(g["Q4_K_M"])],
                [
                    "llama-server Q4_K_M smoke",
                    json.dumps(result["serve_smoke"]),
                ],
            ],
        ),
        "",
        f"llama.cpp release `{g['llama_cpp_tag']}`; artefacts under "
        f"`{ctx.rel(d / 'artifacts')}` "
        "(gitignored). vLLM does not run on native Windows; the merged safetensors directory is "
        "the vLLM-loadable form and is untested here.",
    ]
    (ctx.reports / "llm_export.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return result


def commit_export(ctx: Ctx, done: dict) -> None:
    mc, g, sm = (
        done["merge_check"],
        done["gguf"],
        done["serve_smoke"],
    )
    works = mc["argmax_agree"] == mc["prompts"] and sm["parses"]
    dod = (
        "`src/llm/export.py` + `src/llm/pipeline.py` + `reports/llm_export.md`: the adapter merged "
        "into the **bf16** base on the CPU (not NF4, which would re-quantise the merged sum) and "
        f"saved as safetensors ({done.get('merge', {}).get('merged_gb', 'reused')} GB); against the "
        f"unmerged bf16 base + adapter on {mc['prompts']} validation prompts the next-token logits "
        f"differ by at most {mc['max_abs_logit_diff']}, argmax agrees on {mc['argmax_agree']}/"
        f"{mc['prompts']} and 64-token greedy continuations are identical on "
        f"{mc['greedy_64_identical']}/{mc['prompts']}. llama.cpp `{g['llama_cpp_tag']}`'s own "
        f"converter and quantiser: f16 {g['f16']['gb']} GB, Q8_0 {g['Q8_0']['gb']} GB, Q4_K_M "
        f"{g['Q4_K_M']['gb']} GB; `llama-server` on Q4_K_M returned a program that "
        f"{'parses' if sm['parses'] else 'does not parse'} ({sm['new_tokens']} tokens, "
        f"{sm['seconds']} s). GGUF quality against HF is measured in 12.2.9. vLLM is not available "
        "on native Windows; the merged directory is its input format, untested here"
    )
    msg = (
        f"12.2.8: merged bf16 weights match the adapter "
        f"(max logit diff {mc['max_abs_logit_diff']}) "
        f"and export to GGUF Q4_K_M at {g['Q4_K_M']['gb']} GB\n"
    )
    commit_row(
        ctx,
        "12.2.8",
        dod,
        msg,
        [ctx.reports / "llm_export.md"],
        keep_open=not works,
    )


# -- latency ------------------------------------------------------------------------------------


def percentile(
    values: list[float],
    q: float,
) -> float:
    v = sorted(values)
    if not v:
        return float("nan")
    k = (len(v) - 1) * q
    lo, hi = (
        int(k),
        min(int(k) + 1, len(v) - 1),
    )
    return round(
        v[lo] + (v[hi] - v[lo]) * (k - lo),
        3,
    )


def stage_latency(ctx: Ctx) -> dict:
    from concurrent.futures import ThreadPoolExecutor

    from src.codegen import prompt as template
    from src.llm import export
    from src.llm.score import score_rows

    model, adapter = (
        ctx.state["selected_model"],
        ctx.state["adapter"],
    )
    d = ctx.stage_dir("latency")
    test = pairs_for(ctx, "test")
    by_len = sorted(
        test,
        key=lambda p: len(p["ir_text"]),
    )
    k = 4 if ctx.dry else 32
    sample = by_len[:: max(1, len(by_len) // k)][:k]
    messages = write_messages(
        d / "sample.messages.jsonl",
        sample,
        template.build_messages,
    )
    paths: dict[str, dict] = {}
    hf = run_job(
        ctx,
        "latency",
        {
            "model": model,
            "adapter": adapter,
            "messages": str(messages),
            "out": str(d / "hf_nf4_lora.jsonl"),
            "cache_ablation": 2 if ctx.dry else 3,
        },
        "latency-hf",
    )
    from src.llm.generate import read_jsonl, write_jsonl

    rows = read_jsonl(d / "hf_nf4_lora.jsonl")
    paths["HF NF4 + LoRA (batch 1, streaming)"] = {
        "rows": rows,
        "extra": hf,
    }
    artifacts = ROOT / ctx.state["export_dir"]
    for q in ("Q4_K_M", "Q8_0"):
        gguf = artifacts / f"model-{q}.gguf"
        out = []
        with export.LlamaServer(gguf) as server:
            server.chat(
                template.build_messages(sample[0]),
                max_tokens=8,
            )  # warm-up
            for p in sample:
                r = server.chat(
                    template.build_messages(p),
                    stream=True,
                )
                out.append(
                    {
                        "diagram_id": p["diagram_id"],
                        "source": p["source"],
                        "text": r["text"],
                        "ttft_s": round(
                            r["ttft_s"] or r["seconds"],
                            3,
                        ),
                        "total_s": round(
                            r["seconds"],
                            3,
                        ),
                        "new_tokens": r["new_tokens"],
                        "hit_limit": False,
                    }
                )
        write_jsonl(
            out,
            d / f"llamacpp_{q}.jsonl",
        )
        paths[f"llama.cpp {q} (batch 1, streaming)"] = {"rows": out}
    # batching: 4 parallel slots on Q4_K_M, the whole sample submitted concurrently
    with export.LlamaServer(
        artifacts / "model-Q4_K_M.gguf",
        parallel=4,
    ) as server:
        began = time.perf_counter()
        with ThreadPoolExecutor(4) as pool:
            conc = list(
                pool.map(
                    lambda p: server.chat(template.build_messages(p)),
                    sample,
                )
            )
        wall = time.perf_counter() - began
    batching = {
        "slots": 4,
        "requests": len(sample),
        "wall_s": round(wall, 1),
        "throughput_req_per_min": round(
            60 * len(sample) / wall,
            2,
        ),
        "per_request_p50_s": percentile(
            [c["seconds"] for c in conc],
            0.5,
        ),
        "per_request_p90_s": percentile(
            [c["seconds"] for c in conc],
            0.9,
        ),
    }
    summary = {}
    for name, info in paths.items():
        scored = score_rows(
            info["rows"],
            "test",
        )
        summary[name] = {
            "n": len(scored),
            "ttft_p50_s": percentile(
                [r["ttft_s"] for r in info["rows"]],
                0.5,
            ),
            "total_p50_s": percentile(
                [r["total_s"] for r in info["rows"]],
                0.5,
            ),
            "total_p90_s": percentile(
                [r["total_s"] for r in info["rows"]],
                0.9,
            ),
            "total_max_s": max(r["total_s"] for r in info["rows"]),
            "new_tokens_p50": percentile(
                [r["new_tokens"] for r in info["rows"]],
                0.5,
            ),
            "functional": round(
                sum(s["functional"] for s in scored) / len(scored),
                4,
            ),
            "syntax": round(
                sum(s["syntax"] for s in scored) / len(scored),
                4,
            ),
        }
    hf_name = "HF NF4 + LoRA (batch 1, streaming)"
    floor = summary[hf_name]["functional"] - 0.05
    eligible = [n for n in summary if summary[n]["functional"] >= floor]
    chosen = min(
        eligible,
        key=lambda n: summary[n]["total_p90_s"],
    )
    meets = summary[chosen]["total_p90_s"] < 8.0
    done = {
        "summary": summary,
        "chosen": chosen,
        "meets_budget": meets,
        "batching": batching,
        "cache_ablation": hf["cache_ablation"],
        "sample": len(sample),
    }
    rows_md = [
        [
            n,
            s["ttft_p50_s"],
            s["total_p50_s"],
            s["total_p90_s"],
            s["total_max_s"],
            s["new_tokens_p50"],
            pct(s["syntax"]),
            pct(s["functional"]),
        ]
        for n, s in summary.items()
    ]
    lines = [
        "# Phase 12.2.9 - inference latency",
        "",
        f"{len(sample)} test prompts stratified by IR size, one request at a time, greedy, "
        "streaming (time to first token and to the end of the program).",
        "",
        md_table(
            [
                "path",
                "TTFT p50 s",
                "total p50 s",
                "total p90 s",
                "max s",
                "new tokens p50",
                "syntax",
                "functional",
            ],
            rows_md,
        ),
        "",
        f"KV cache ablation (HF NF4 + LoRA, 128 forced tokens): "
        f"`{json.dumps(hf['cache_ablation'])}`",
        "",
        f"Batching (llama.cpp Q4_K_M, 4 slots, all requests concurrent): "
        f"`{json.dumps(batching)}`",
        "",
        f"**Serving path: {chosen}** (fastest p90 among paths within 5 points of the HF adapter's "
        f"functional pass rate). Budget < 8 s at p90: **{meets}**",
    ]
    (ctx.reports / "llm_latency.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return done


def commit_latency(ctx: Ctx, done: dict) -> None:
    s = done["summary"]
    parts = "; ".join(
        f"**{n}** TTFT p50 {v['ttft_p50_s']} s, total p50 "
        f"{v['total_p50_s']} s / p90 {v['total_p90_s']} s / max "
        f"{v['total_max_s']} s, functional {pct(v['functional'])}"
        for n, v in s.items()
    )
    cache = done["cache_ablation"]
    on = [c["ms_per_token"] for c in cache if c["use_cache"]]
    off = [c["ms_per_token"] for c in cache if not c["use_cache"]]
    b = done["batching"]
    c = s[done["chosen"]]
    dod = (
        "`src/llm/pipeline.py` + `src/llm/export.py` + `reports/llm_latency.md`: "
        f"{done['sample']} test prompts stratified by IR size, one request at a time, greedy, "
        f"streamed. {parts}. **KV cache**: {sum(on) / max(1, len(on)):.1f} ms/token on vs "
        f"{sum(off) / max(1, len(off)):.1f} ms/token off at 128 forced tokens. **Batching**: 4 "
        f"llama.cpp slots serve {b['requests']} concurrent requests at {b['throughput_req_per_min']} "
        f"req/min (per-request p50 {b['per_request_p50_s']} s, p90 {b['per_request_p90_s']} s). "
        f"Serving path **{done['chosen']}**: p90 {c['total_p90_s']} s against the 8 s budget - "
        f"{'met' if done['meets_budget'] else 'not met'}"
    )
    msg = (
        f"12.2.9: {done['chosen'].split(' (')[0]} returns a program in p90 "
        f"{c['total_p90_s']} s (budget 8 s"
        f"{'' if done['meets_budget'] else ', not met'})\n"
    )
    commit_row(
        ctx,
        "12.2.9",
        dod,
        msg,
        [ctx.reports / "llm_latency.md"],
        keep_open=not done["meets_budget"],
    )


# ------------------------------------------------------------------------------------ driver


@dataclass
class Stage:
    name: str
    rows: list[str]
    deps: list[str]
    run: Callable[[Ctx], dict]
    commit: Callable[[Ctx, dict], None] | None
    absorb: Callable[[Ctx, dict], None] = lambda ctx, done: None


def _absorb_benchmark(ctx: Ctx, done: dict) -> None:
    ctx.state["selected_model"] = done["model"]


def _absorb_lora(ctx: Ctx, done: dict) -> None:
    ctx.state["lora_overrides"] = done["selected_overrides"]
    ctx.state["lora_arm_name"] = done["selected"]
    ctx.state["lora_arm_summary"] = done["arms"][done["selected"]]


def _absorb_hparam(ctx: Ctx, done: dict) -> None:
    ctx.state["final_overrides"] = done["selected_overrides"]


def _absorb_train(ctx: Ctx, done: dict) -> None:
    ctx.state["adapter"] = done["adapter"]


def _absorb_compare(ctx: Ctx, done: dict) -> None:
    ctx.state["lora_generations"] = done["lora_generations"]


def _absorb_export(ctx: Ctx, done: dict) -> None:
    ctx.state["export_dir"] = ctx.rel(Path(done["merged_dir"]).parent)


STAGES = [
    Stage(
        "benchmark",
        ["12.2.1"],
        [],
        stage_benchmark,
        commit_benchmark,
        _absorb_benchmark,
    ),
    Stage(
        "lora_sweep",
        ["12.2.3"],
        ["benchmark"],
        stage_lora_sweep,
        commit_lora,
        _absorb_lora,
    ),
    Stage(
        "hparam_sweep",
        ["12.2.6"],
        ["lora_sweep"],
        stage_hparam_sweep,
        commit_hparam,
        _absorb_hparam,
    ),
    Stage(
        "train",
        ["12.2.5"],
        ["hparam_sweep"],
        stage_train,
        commit_train,
        _absorb_train,
    ),
    Stage(
        "compare",
        ["12.2.7"],
        ["train"],
        stage_compare,
        commit_compare,
        _absorb_compare,
    ),
    Stage(
        "quality",
        ["12.3.1", "12.3.2", "12.3.3", "12.3.8"],
        ["compare"],
        stage_quality,
        commit_quality,
    ),
    Stage(
        "repair",
        ["12.3.9"],
        ["quality"],
        stage_repair,
        commit_repair,
    ),
    Stage(
        "similarity",
        [],
        ["compare"],
        stage_similarity,
        None,
    ),
    Stage(
        "export",
        ["12.2.8"],
        ["train"],
        stage_export,
        commit_export,
        _absorb_export,
    ),
    Stage(
        "latency",
        ["12.2.9"],
        ["export"],
        stage_latency,
        commit_latency,
    ),
]


def commit_failure(ctx: Ctx, stage: Stage, error: str) -> None:
    report = ctx.reports / f"llm_{stage.name}.md"
    report.write_text(
        f"# {', '.join(stage.rows)} - stage `{stage.name}` failed\n\n"
        f"```\n{error[-6000:]}\n```\n",
        encoding="utf-8",
    )
    last = [ln for ln in error.strip().splitlines() if ln.strip()][-1][:300]
    for row in stage.rows:
        dod = (
            f"`src/llm/pipeline.py` stage `{stage.name}` failed and this row is not done: "
            f"{last}. Traceback in `reports/llm_{stage.name}.md`; rerun "
            "`python -m src.llm.pipeline` to resume from this stage"
        )
        commit_row(
            ctx,
            row,
            dod,
            f"{row}: pipeline stage {stage.name} failed, kept open\n\n{last}\n",
            [report],
            keep_open=True,
        )


def run_pipeline(
    ctx: Ctx,
    only: list[str] | None = None,
) -> dict:
    ctx.base.mkdir(parents=True, exist_ok=True)
    ctx.reports.mkdir(parents=True, exist_ok=True)
    ctx.state.setdefault("stages", {})
    ctx.state["started"] = time.strftime("%Y-%m-%d %H:%M:%S")
    stop = threading.Event()
    threading.Thread(
        target=heartbeat,
        args=(ctx, stop),
        daemon=True,
    ).start()
    try:
        for stage in STAGES:
            d = ctx.stage_dir(stage.name)
            done_path = d / "done.json"
            failed_path = d / "failed.json"
            if done_path.exists():
                # Absorb before the `only` filter: a finished stage still has to hand its state
                # (the adapter path, the selected model) to whatever *is* selected, or `--only
                # export` runs with none of what `train` learned.
                done = json.loads(done_path.read_text(encoding="utf-8"))
                stage.absorb(ctx, done)
                ctx.state["stages"][stage.name] = "done (resumed)"
                continue
            if only and stage.name not in only:
                continue
            bad = [
                dep
                for dep in stage.deps
                if ctx.state["stages"].get(dep, "").startswith(("failed", "skipped"))
            ]
            if bad:
                ctx.state["stages"][stage.name] = f"skipped (depends on {bad})"
                log(
                    ctx,
                    f"stage {stage.name} skipped: depends on {bad}",
                )
                continue
            ctx.state["current"] = stage.name
            ctx.state["stages"][stage.name] = f"running since {time.strftime('%H:%M:%S')}"
            write_status(ctx)
            log(ctx, f"stage {stage.name} start")
            began = time.perf_counter()
            try:
                done = stage.run(ctx)
                done["stage_wall_s"] = round(
                    time.perf_counter() - began,
                    1,
                )
                stage.absorb(ctx, done)
                if stage.commit is not None:
                    stage.commit(ctx, done)
                done_path.write_text(
                    json.dumps(done, indent=2, default=str),
                    encoding="utf-8",
                )
                failed_path.unlink(missing_ok=True)
                ctx.state["stages"][stage.name] = f"done in {done['stage_wall_s'] / 60:.0f} min"
                log(
                    ctx,
                    f"stage {stage.name} done " f"in {done['stage_wall_s']:.0f}s",
                )
            except Exception:  # noqa: BLE001 - one stage failing must not stop independent stages
                error = traceback.format_exc()
                failed_path.write_text(
                    json.dumps({"error": error}),
                    encoding="utf-8",
                )
                ctx.state["stages"][stage.name] = "failed: " + error.strip().splitlines()[-1][:200]
                log(
                    ctx,
                    f"stage {stage.name} FAILED:\n{error}",
                )
                if stage.rows:
                    try:
                        commit_failure(
                            ctx,
                            stage,
                            error,
                        )
                    except Exception:  # noqa: BLE001
                        log(
                            ctx,
                            f"could not commit failure of "
                            f"{stage.name}:\n"
                            f"{traceback.format_exc()}",
                        )
            write_status(ctx)
    finally:
        ctx.state["current"] = None
        ctx.state["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        write_status(ctx)
        stop.set()
    return ctx.state


def main(
    argv: list[str] | None = None,
) -> int:  # pragma: no cover - orchestration
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["job"]:
        return job_main(argv[1], argv[2])
    parser = argparse.ArgumentParser(description="Phase 12.2/12.3 pipeline")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args(argv)
    ctx = Ctx.make(args.dry_run)
    if args.status:
        print((ctx.base / "pipeline_status.json").read_text(encoding="utf-8"))
        return 0
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    run_pipeline(ctx, args.only)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
