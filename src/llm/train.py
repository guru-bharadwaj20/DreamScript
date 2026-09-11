"""Phase 12.2.5 - the QLoRA training loop: packed bf16 SFT with cosine LR, logging VRAM and wall time.

    python -m src.llm.train --config configs/llm.yaml train.lr=2e-4 lora.r=16 ...

A hand-written loop rather than `trl.SFTTrainer`, for three reasons that were each measured in
12.2.2 rather than assumed:

1. **Loss only where it is paid for.** Qwen2.5's vocabulary is 152,064, so full-sequence logits
   are 152,064 x L floats, upcast to float32 inside the loss: at L = 4096 that is ~2.5 GB for the
   logits alone and it dominated peak memory in the fit grid. Supervision is on completion
   tokens only, so `lm_head` is applied only at those positions (`selected_loss`) - the prompt
   positions' logits are never materialised.
2. **Packing without cross-contamination.** Examples are packed first-fit-decreasing into
   `max_len` rows. `flash-attn` is not available on this Windows build (and torch's own flash
   SDPA kernel is compiled out), so the varlen path does not exist; instead every packed row
   carries a block-diagonal causal 4-D mask and per-segment `position_ids`, so a token attends
   only within its own example and positions restart at 0 exactly as they would unpacked. SDPA
   runs that mask on the memory-efficient kernel.
3. **No silent fp32.** `peft.prepare_model_for_kbit_training` casts every non-4-bit parameter to
   float32; 12.2.2 measured that at 2.9x slower. The loop keeps the base in bf16, trains fp32
   LoRA master weights under bf16 autocast, and asserts the model is in train mode so gradient
   checkpointing is actually applied.
"""

from __future__ import annotations

import json
import math
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

IGNORE = -100

#: Longest single row ever built. Real pairs top out at 3,696 tokens (Qwen tokenizer).
HARD_MAX = 4096


@dataclass
class Example:
    """One tokenised pair: prompt ids (unsupervised) and completion ids (supervised)."""

    prompt: list[int]
    completion: list[int]

    def __len__(self) -> int:
        return len(self.prompt) + len(self.completion)


def tokenise(tokenizer, messages: list[dict[str, str]], completion: str, eos: str) -> Example:
    """Prompt through the chat template; completion + the turn terminator as the target."""
    from src.llm.generate import render

    prompt_text = render(tokenizer, messages)
    prompt = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    target = tokenizer(completion + eos, add_special_tokens=False)["input_ids"]
    return Example(prompt, target)


def pack(examples: Sequence[Example], max_len: int) -> list[list[int]]:
    """First-fit-decreasing bins of example indices, each bin's total length <= `max_len`.

    An example longer than `max_len` gets a bin of its own at its own length (up to `HARD_MAX`)
    rather than being truncated: 12.2.2 measured 2,048-token rows as the throughput optimum, and
    ~12% of real pairs are longer than that - truncating them would cut exactly the largest
    diagrams' programs. Only an example over `HARD_MAX` is cut, and `pack_stats` counts it.
    """
    order = sorted(range(len(examples)), key=lambda i: -len(examples[i]))
    bins: list[list[int]] = []
    room: list[int] = []
    for i in order:
        size = len(examples[i])
        if size >= max_len:
            bins.append([i])
            room.append(0)
            continue
        for b, free in enumerate(room):
            if size <= free:
                bins[b].append(i)
                room[b] -= size
                break
        else:
            bins.append([i])
            room.append(max_len - size)
    return bins


def pack_stats(examples: Sequence[Example], bins: Sequence[Sequence[int]], max_len: int) -> dict:
    tokens = sum(min(len(e), HARD_MAX) for e in examples)
    capacity = sum(max(max_len, sum(len(examples[i]) for i in b)) for b in bins)
    return {
        "examples": len(examples),
        "rows": len(bins),
        "tokens": tokens,
        "supervised_tokens": sum(len(e.completion) for e in examples),
        "fill": round(tokens / max(1, capacity), 4),
        "tokens_per_row": round(tokens / max(1, len(bins)), 1),
        "own_row": sum(1 for b in bins if len(b) == 1 and len(examples[b[0]]) >= max_len),
        "truncated": sum(1 for e in examples if len(e) > HARD_MAX),
    }


def collate(examples: Sequence[Example], max_len: int = HARD_MAX) -> dict[str, Any]:
    """One packed row: ids, labels, position ids and the block-diagonal causal mask."""
    import torch

    ids: list[int] = []
    labels: list[int] = []
    positions: list[int] = []
    segments: list[int] = []
    for seg, ex in enumerate(examples):
        prompt, completion = ex.prompt, ex.completion
        overflow = len(prompt) + len(completion) - max_len
        if overflow > 0:
            prompt = prompt[overflow:] if overflow < len(prompt) else []
            completion = completion[: max_len - len(prompt)]
        ids += prompt + completion
        labels += [IGNORE] * len(prompt) + completion
        positions += list(range(len(prompt) + len(completion)))
        segments += [seg] * (len(prompt) + len(completion))
    seg = torch.tensor(segments)
    n = len(ids)
    causal = torch.tril(torch.ones(n, n, dtype=torch.bool))
    allowed = causal & (seg[:, None] == seg[None, :])
    return {
        "input_ids": torch.tensor([ids]),
        "labels": torch.tensor([labels]),
        "position_ids": torch.tensor([positions]),
        "allowed": allowed[None, None],
    }


def additive_mask(allowed, dtype):
    """bool (1,1,L,L) -> the additive float mask transformers expects for a 4-D mask."""
    import torch

    mask = torch.zeros(allowed.shape, dtype=dtype, device=allowed.device)
    return mask.masked_fill(~allowed, torch.finfo(dtype).min)


def decoder_and_head(model):
    """(decoder stack, lm_head) under a peft wrapper or bare."""
    inner = model.get_base_model() if hasattr(model, "get_base_model") else model
    return inner.model, inner.lm_head


def selected_loss(model, batch: dict[str, Any], chunk: int = 2048) -> tuple[Any, int]:
    """Mean next-token CE over supervised positions only; `lm_head` runs on those alone."""
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint

    decoder, head = decoder_and_head(model)
    device = next(model.parameters()).device
    ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    on_cuda = device.type == "cuda"
    mask = additive_mask(
        batch["allowed"].to(device), torch.bfloat16 if on_cuda else next(model.parameters()).dtype
    )
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=on_cuda):
        hidden = decoder(
            input_ids=ids,
            attention_mask=mask,
            position_ids=batch["position_ids"].to(device),
            use_cache=False,
        ).last_hidden_state
    # position t predicts token t+1
    target = labels[:, 1:]
    keep = target != IGNORE
    states = hidden[:, :-1][keep]
    gold = target[keep]
    total = states.new_zeros((), dtype=torch.float32)

    def _chunk(h, y):
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=on_cuda):
            logits = head(h)
        return F.cross_entropy(logits.float(), y, reduction="sum")

    for start in range(0, gold.numel(), chunk):
        total = total + checkpoint(
            _chunk, states[start : start + chunk], gold[start : start + chunk], use_reentrant=False
        )
    count = int(gold.numel())
    return total / max(1, count), count


def cosine_lr(step: int, total: int, warmup: int, peak: float, floor: float = 0.1) -> float:
    """Linear warmup to `peak`, cosine decay to `floor * peak` at `total`."""
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return peak * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress)))


def evaluate_loss(model, rows: Sequence[dict[str, Any]]) -> float:
    """Token-weighted mean CE over supervised tokens of packed validation rows."""
    import torch

    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for row in rows:
            loss, n = selected_loss(model, row)
            total += float(loss) * n
            count += n
    model.train(was_training)
    return total / max(1, count)


def train(
    model,
    train_examples: Sequence[Example],
    val_examples: Sequence[Example],
    *,
    run_dir: Path,
    lr: float = 2e-4,
    epochs: float = 1.0,
    max_len: int = 4096,
    tokens_per_step: int = 32768,
    warmup_frac: float = 0.03,
    weight_decay: float = 0.0,
    max_grad_norm: float = 1.0,
    eval_every: int = 50,
    token_budget: int | None = None,
    seed: int = 42,
    log_every: int = 5,
) -> dict[str, Any]:
    """Train the LoRA parameters of `model` in place. Returns the run summary.

    `tokens_per_step` sets gradient accumulation: rows are accumulated until at least that
    many packed tokens have been seen, so the effective batch is a token count and does not
    change when the pack fill changes. `token_budget` caps total training tokens for the
    reduced-step sweep arms (every arm of a sweep gets the same budget).
    """
    import random

    import torch

    run_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    torch.manual_seed(seed)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, fused=True)

    bins = pack(train_examples, max_len)
    stats = pack_stats(train_examples, bins, max_len)
    val_bins = pack(val_examples, max_len)
    val_rows = [collate([val_examples[i] for i in b]) for b in val_bins]
    rows_per_step = max(1, round(tokens_per_step / stats["tokens_per_row"]))
    epoch_tokens = stats["tokens"]
    budget = token_budget or int(epoch_tokens * epochs)
    rows_needed = math.ceil(budget / stats["tokens_per_row"])
    total_steps = max(1, math.ceil(rows_needed / rows_per_step))
    warmup = max(1, round(warmup_frac * total_steps))

    schedule: list[list[int]] = []
    while len(schedule) < rows_needed:
        epoch = list(bins)
        rng.shuffle(epoch)
        schedule.extend(epoch)
    schedule = schedule[:rows_needed]

    log = (run_dir / "train.jsonl").open("w", encoding="utf-8")
    summary: dict[str, Any] = {
        "pack": stats,
        "rows_per_step": rows_per_step,
        "total_steps": total_steps,
        "warmup_steps": warmup,
        "token_budget": budget,
        "lr": lr,
        "max_len": max_len,
    }
    torch.cuda.reset_peak_memory_stats()
    began = time.perf_counter()
    seen_tokens = 0
    history = []
    best = (float("inf"), -1)
    val0 = evaluate_loss(model, val_rows)
    history.append({"step": 0, "val_loss": round(val0, 5), "tokens": 0, "wall_s": 0.0})
    log.write(json.dumps(history[-1]) + "\n")
    print(json.dumps(history[-1]), flush=True)

    for step in range(total_steps):
        for group in optimizer.param_groups:
            group["lr"] = cosine_lr(step, total_steps, warmup, lr)
        rows = schedule[step * rows_per_step : (step + 1) * rows_per_step]
        if not rows:
            break
        batches = [collate([train_examples[i] for i in b]) for b in rows]
        supervised = sum(int((b["labels"] != IGNORE).sum()) for b in batches)
        step_loss = 0.0
        for b in batches:
            loss, n = selected_loss(model, b)
            # token-weighted across the accumulated rows, not row-weighted
            (loss * (n / max(1, supervised))).backward()
            step_loss += float(loss) * n / max(1, supervised)
            seen_tokens += int(b["input_ids"].numel())
        norm = torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        record = {
            "step": step + 1,
            "loss": round(step_loss, 5),
            "lr": optimizer.param_groups[0]["lr"],
            "grad_norm": round(float(norm), 4),
            "tokens": seen_tokens,
            "wall_s": round(time.perf_counter() - began, 1),
            "tokens_per_s": round(seen_tokens / (time.perf_counter() - began), 1),
            "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        }
        if (step + 1) % eval_every == 0 or step + 1 == total_steps:
            record["val_loss"] = round(evaluate_loss(model, val_rows), 5)
            if record["val_loss"] < best[0]:
                best = (record["val_loss"], step + 1)
        history.append(record)
        log.write(json.dumps(record) + "\n")
        log.flush()
        if (step + 1) % log_every == 0 or "val_loss" in record:
            print(json.dumps(record), flush=True)

    log.close()
    summary.update(
        wall_s=round(time.perf_counter() - began, 1),
        tokens=seen_tokens,
        val_loss_start=round(val0, 5),
        val_loss_final=history[-1].get("val_loss"),
        best_val_loss=best[0],
        best_step=best[1],
        peak_allocated_gb=round(torch.cuda.max_memory_allocated() / 1e9, 3),
        peak_reserved_gb=round(torch.cuda.max_memory_reserved() / 1e9, 3),
    )
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def iter_records(paths: Iterable[Path]) -> Iterable[dict]:
    for path in paths:
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


if __name__ == "__main__":  # pragma: no cover
    sys.exit("use src.llm.run (configs/llm.yaml) to launch training")
