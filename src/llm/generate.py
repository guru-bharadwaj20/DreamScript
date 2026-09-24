"""Phase 12.2.1 / 12.2.7 / 12.2.9 - batched greedy generation, and the batch-size sweep behind it.

    python -m src.llm.generate --model Qwen/Qwen2.5-Coder-7B-Instruct --split validation \
        --out experiments/llm/zeroshot_qwen7b_val.jsonl

One generation path serves every arm (zero-shot, few-shot, LoRA) so an arm can differ only in
its messages or its weights, never in decoding: greedy, `max_new_tokens` fixed, the model's own
chat template, left padding, KV cache on. `generation_audit` reads back the flags that silently
cost throughput when wrong - `use_cache` in both `config` and `generation_config`, the attention
implementation, the padding side, the pad token and the compute dtype - and the result is stored
next to every run.

Prompts are bucketed by length (longest first) so a batch pads to its own longest prompt rather
than to the split's, and because output length tracks diagram size here, sorting by prompt length
also stops one 1,400-token program holding a batch of 300-token ones hostage.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

MAX_NEW_TOKENS = 1600


def render(tokenizer, messages: list[dict[str, str]]) -> str:
    """The chat-templated prompt string, ending where the assistant turn starts."""
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:  # noqa: BLE001 - a template that rejects the system role
        merged = [dict(m) for m in messages if m["role"] != "system"]
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        if merged and system:
            merged[0]["content"] = f"{system}\n\n{merged[0]['content']}"
        return tokenizer.apply_chat_template(merged, tokenize=False, add_generation_prompt=True)


def prepare_for_generation(model, tokenizer) -> None:
    """Set the flags batched decoding depends on. Idempotent."""
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = True
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.use_cache = True
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.eval()


def generation_audit(model, tokenizer) -> dict[str, Any]:
    gen = getattr(model, "generation_config", None)
    quant = getattr(model.config, "quantization_config", None)
    compute = None
    if quant is not None:
        compute = str(
            getattr(quant, "bnb_4bit_compute_dtype", None) or quant.get("bnb_4bit_compute_dtype")
        )
    return {
        "config_use_cache": bool(getattr(model.config, "use_cache", False)),
        "generation_config_use_cache": bool(getattr(gen, "use_cache", False)),
        "attn_implementation": getattr(model.config, "_attn_implementation", None),
        "padding_side": tokenizer.padding_side,
        "pad_token_set": tokenizer.pad_token_id is not None,
        "eos_token_id": getattr(gen, "eos_token_id", None),
        "torch_dtype": str(getattr(model, "dtype", None)),
        "bnb_compute_dtype": compute,
        "training_mode": bool(model.training),
    }


def generate(
    model,
    tokenizer,
    prompts: Sequence[str],
    *,
    batch_size: int = 8,
    max_new_tokens: int = MAX_NEW_TOKENS,
    progress: Callable[[int, int], None] | None = None,
    split_on_oom: bool = True,
    oom_splits: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Greedy completions for `prompts`, returned in input order.

    Each result carries `text`, `prompt_tokens`, `new_tokens`, `hit_limit` (the output was cut
    by `max_new_tokens`) and `batch_seconds` / `batch_size` of the batch it rode in.
    """
    import torch

    prepare_for_generation(model, tokenizer)
    lengths = [len(tokenizer(p, add_special_tokens=False)["input_ids"]) for p in prompts]
    order = sorted(range(len(prompts)), key=lambda i: -lengths[i])
    results: list[dict[str, Any] | None] = [None] * len(prompts)
    eos = set(_as_list(model.generation_config.eos_token_id)) | {tokenizer.eos_token_id}
    done = 0
    oom_splits = [] if oom_splits is None else oom_splits
    queue = [order[i : i + batch_size] for i in range(0, len(order), batch_size)]
    while queue:
        idx = queue.pop(0)
        enc = tokenizer(
            [prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False
        ).to(model.device)
        torch.cuda.synchronize()
        began = time.perf_counter()
        try:
            with torch.inference_mode():
                out = model.generate(
                    **enc,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
        except torch.cuda.OutOfMemoryError:
            # Only the longest prompts' batches can hit this; split rather than abort the split.
            if len(idx) == 1 or not split_on_oom:
                raise
            del enc
            torch.cuda.empty_cache()
            half = len(idx) // 2
            queue[:0] = [idx[:half], idx[half:]]
            oom_splits.append(len(idx))
            continue
        torch.cuda.synchronize()
        seconds = time.perf_counter() - began
        new = out[:, enc["input_ids"].shape[1] :]
        for row, i in enumerate(idx):
            ids = new[row].tolist()
            n = len(ids)
            for k, tok in enumerate(ids):
                if tok in eos:
                    n = k
                    break
            results[i] = {
                "text": tokenizer.decode(ids[:n], skip_special_tokens=True),
                "prompt_tokens": lengths[i],
                "new_tokens": n,
                "hit_limit": n >= max_new_tokens,
                "batch_seconds": round(seconds, 3),
                "batch_size": len(idx),
            }
        done += len(idx)
        if progress:
            progress(done, len(prompts))
    return [r for r in results if r is not None]


def _as_list(value) -> list[int]:
    if value is None:
        return []
    return list(value) if isinstance(value, list | tuple) else [int(value)]


def batch_sweep(
    model, tokenizer, prompts: Sequence[str], batches: Sequence[int], max_new_tokens: int
) -> list[dict[str, Any]]:
    """ms per sample and generated tokens/s at each batch size, on the same prompts.

    The prompts are one fixed sample, so every batch size decodes identical work; a batch that
    raises OOM is recorded and ends the sweep.
    """
    import torch

    rows = []
    for batch in batches:
        n = max(batch, len(prompts) - len(prompts) % batch) if len(prompts) >= batch else batch
        sample = list(prompts[:n]) if len(prompts) >= n else list(prompts) * (n // len(prompts) + 1)
        sample = sample[:n]
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        began = time.perf_counter()
        try:
            out = generate(
                model,
                tokenizer,
                sample,
                batch_size=batch,
                max_new_tokens=max_new_tokens,
                split_on_oom=False,
            )
        except torch.cuda.OutOfMemoryError:
            rows.append({"batch": batch, "status": "oom"})
            torch.cuda.empty_cache()
            break
        seconds = time.perf_counter() - began
        tokens = sum(r["new_tokens"] for r in out)
        row = {
            "batch": batch,
            "status": "ok",
            "samples": len(out),
            "ms_per_sample": round(1000 * seconds / len(out), 1),
            "generated_tokens_per_s": round(tokens / seconds, 1),
            "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
            "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 3),
        }
        print(json.dumps(row), flush=True)
        rows.append(row)
    return rows


def write_jsonl(rows: Sequence[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - GPU entry point
    parser = argparse.ArgumentParser(description="Phase 12.2 - batched greedy generation")
    parser.add_argument("--model", required=True)
    parser.add_argument("--sweep", type=int, nargs="*", default=None)
    parser.add_argument("--sweep-samples", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    from src.codegen import prompt as template
    from src.llm import pairs, quant

    quant.cap_memory()
    model, tokenizer = quant.load_model(args.model)
    prepare_for_generation(model, tokenizer)
    val = pairs.by_split(pairs.real_pairs(), "validation")
    prompts = [render(tokenizer, template.build_messages(p)) for p in val]
    report = {"model": args.model, "audit": generation_audit(model, tokenizer)}
    if args.sweep:
        # A spread of lengths, not the longest N: the sweep must see the split's real mix.
        step = max(1, len(prompts) // args.sweep_samples)
        sample = prompts[::step][: args.sweep_samples]
        report["sweep"] = batch_sweep(model, tokenizer, sample, args.sweep, args.max_new_tokens)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
