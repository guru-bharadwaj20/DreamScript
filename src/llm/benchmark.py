"""Phase 12.2.1 - zero-shot benchmark of candidate base models on the real validation pairs.

    python -m src.llm.benchmark generate --model Qwen/Qwen2.5-Coder-7B-Instruct --tag qwen7b
    python -m src.llm.benchmark score --tags qwen7b dscoder67b

Every candidate is loaded the way it would be fine-tuned (4-bit NF4, bf16 compute, SDPA), sees
the identical messages (12.2.4's template rendered through the candidate's *own* chat template),
and decodes identically (greedy, `MAX_NEW_TOKENS`). Before the split is generated, a batch sweep
on a fixed length-stratified sample of 32 prompts finds the batch past which ms/sample stops
falling, and the split is generated at that batch - so no candidate is slowed or advantaged by a
batch size chosen for another model.

Selection is on validation only. Test is never generated here.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

OUT = Path("experiments/llm/benchmark")

#: The committed revision of `src/codegen/prompt.py` every 12.2.1 arm renders with. 12.2.4 was
#: still open while this row ran and the file was being edited in the working tree, so the
#: template is loaded from the git object, not the file - otherwise two candidates generated an
#: hour apart could silently see different words.
PROVISIONAL_TEMPLATE_REV = "a7d945b"


def pinned_template(rev: str = PROVISIONAL_TEMPLATE_REV):
    """`src/codegen/prompt.py` as committed at `rev`, imported from the git object."""
    import subprocess
    import types

    root = Path(__file__).resolve().parents[2]
    source = subprocess.run(
        ["git", "show", f"{rev}:src/codegen/prompt.py"],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    ).stdout
    module = types.ModuleType(f"prompt_{rev}")
    exec(compile(source, f"prompt@{rev}", "exec"), module.__dict__)
    return module


def saturation_batch(rows: list[dict[str, Any]], tolerance: float = 0.05) -> int:
    """Smallest batch whose ms/sample is within `tolerance` of the best measured."""
    ok = [r for r in rows if r.get("status") == "ok"]
    best = min(r["ms_per_sample"] for r in ok)
    return min(r["batch"] for r in ok if r["ms_per_sample"] <= (1 + tolerance) * best)


def generate_split(
    model_id: str,
    tag: str,
    split: str = "validation",
    sweep: tuple[int, ...] = (1, 4, 8, 16, 32),
    batch: int | None = None,
    messages_fn=None,
    adapter: str | None = None,
    out_dir: Path = OUT,
    limit: int | None = None,
) -> dict[str, Any]:
    from src.llm import generate, pairs, quant

    template = pinned_template()

    quant.cap_memory()
    started = time.perf_counter()
    model, tokenizer = quant.load_model(model_id)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    load_s = time.perf_counter() - started
    generate.prepare_for_generation(model, tokenizer)
    chosen = pairs.by_split(pairs.real_pairs(), split)
    if limit:
        chosen = chosen[:limit]
    build = messages_fn or template.build_messages
    messages = [build(p) for p in chosen]
    prompts = [generate.render(tokenizer, m) for m in messages]
    report: dict[str, Any] = {
        "model": model_id,
        "adapter": adapter,
        "tag": tag,
        "split": split,
        "template_id": getattr(template, "TEMPLATE_ID", None),
        "template_rev": PROVISIONAL_TEMPLATE_REV,
        "load_s": round(load_s, 1),
        "audit": generate.generation_audit(model, tokenizer),
    }
    if batch is None:
        by_len = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        stride = max(1, len(by_len) // 32)
        sample = [prompts[i] for i in by_len[::stride][:32]]
        report["sweep_sample_prompt_chars"] = sorted(len(p) for p in sample)
        report["sweep"] = generate.batch_sweep(
            model, tokenizer, sample, sweep, generate.MAX_NEW_TOKENS
        )
        batch = saturation_batch(report["sweep"])
    report["batch"] = batch
    began = time.perf_counter()
    oom_splits: list[int] = []
    outputs = generate.generate(
        model,
        tokenizer,
        prompts,
        batch_size=batch,
        oom_splits=oom_splits,
        progress=lambda d, n: print(f"{tag}: {d}/{n}", flush=True),
    )
    report["generate_s"] = round(time.perf_counter() - began, 1)
    report["oom_batch_splits"] = oom_splits
    rows = [
        {"diagram_id": p["diagram_id"], "source": p["source"], **o}
        for p, o in zip(chosen, outputs, strict=False)
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    generate.write_jsonl(rows, out_dir / f"{tag}_{split}.jsonl")
    report["new_tokens"] = sum(r["new_tokens"] for r in rows)
    report["hit_limit"] = sum(r["hit_limit"] for r in rows)
    (out_dir / f"{tag}_{split}.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "sweep"}, indent=2), flush=True)
    return report


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - GPU entry point
    parser = argparse.ArgumentParser(description="Phase 12.2.1 - zero-shot base benchmark")
    sub = parser.add_subparsers(dest="cmd", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("--model", required=True)
    gen.add_argument("--tag", required=True)
    gen.add_argument("--split", default="validation")
    gen.add_argument("--batch", type=int, default=None)
    gen.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)
    if args.cmd == "generate":
        generate_split(args.model, args.tag, args.split, batch=args.batch, limit=args.limit)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
