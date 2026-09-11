"""Phase 12.2.2 - 4-bit NF4 QLoRA on a 24 GB RTX 4500 Ada: does it fit, and where it saturates.

    python -m src.llm.quant --model Qwen/Qwen2.5-Coder-7B-Instruct --out experiments/llm/quant.json

`load_model` is the one loader every other Phase 12.2 row uses, so the precision that was
profiled here is the precision that is trained, evaluated and served. `profile` runs real
training steps (forward, backward, optimiser step on the LoRA parameters) at a grid of sequence
length x micro-batch x gradient checkpointing and records, per cell, peak allocated and peak
reserved VRAM and training throughput in tokens per second. A cell that raises CUDA OOM is
recorded as `oom`, not skipped, because the OOM boundary is one of the two numbers the row
exists to produce; the other is the micro-batch past which tokens per second stops rising.

Everything that could be silently disabled is checked and recorded instead of assumed
(`runtime_checks`): the attention implementation the model actually resolved to, which SDPA
kernels this torch build can run, whether gradient checkpointing is really on (the flag the
model reports, not the one that was passed), that the base weights are really 4-bit
(`Linear4bit` count and quant type), and the trainable-parameter count.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

#: Every linear projection in a Qwen2 / Llama decoder block - 12.2.3's full target set.
ALL_LINEAR = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


#: Hard ceiling for this process. The card has 24.0 GiB; under WDDM an allocation past dedicated
#: VRAM does not raise OOM, it spills into shared host memory and throughput collapses, so the
#: cap turns the cliff into a clean `OutOfMemoryError`. 20 GiB leaves the desktop (~0.8 GiB) and
#: a concurrent 3 GiB job room.
VRAM_CAP_GIB = 20.0


def cap_memory(cap_gib: float = VRAM_CAP_GIB) -> float:
    """Apply the per-process VRAM ceiling; returns the fraction set."""
    import torch

    total = torch.cuda.get_device_properties(0).total_memory
    fraction = min(1.0, cap_gib * (1 << 30) / total)
    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    return fraction


def bnb_config(double_quant: bool = True) -> Any:
    """NF4 weights, bf16 compute. Double quantisation saves ~0.4 bit/param on the scales."""
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=double_quant,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.uint8,
    )


def load_model(
    model_id: str,
    *,
    quantize: bool = True,
    attn_implementation: str = "sdpa",
    for_training: bool = False,
    gradient_checkpointing: bool = True,
    upcast: bool = False,
):
    """(model, tokenizer). 4-bit NF4 unless `quantize=False` (bf16)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "attn_implementation": attn_implementation,
        "device_map": {"": 0},
    }
    if quantize:
        kwargs["quantization_config"] = bnb_config()
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if for_training:
        from peft import prepare_model_for_kbit_training

        if quantize and upcast:
            # peft's helper casts every non-4-bit parameter to float32 - on Qwen2.5-7B that is
            # the 1.09B-parameter embedding and lm_head. Kept selectable so 12.2.2 can price it.
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=gradient_checkpointing,
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
        else:
            for param in model.parameters():
                param.requires_grad_(False)
            if gradient_checkpointing:
                model.gradient_checkpointing_enable({"use_reentrant": False})
                model.enable_input_require_grads()
        if not gradient_checkpointing:
            model.gradient_checkpointing_disable()
        model.config.use_cache = False
    return model, tokenizer


def lora_config(
    r: int = 16,
    alpha: int | None = None,
    dropout: float = 0.05,
    targets: tuple[str, ...] = ALL_LINEAR,
) -> Any:
    from peft import LoraConfig

    return LoraConfig(
        r=r,
        lora_alpha=alpha if alpha is not None else 2 * r,
        lora_dropout=dropout,
        target_modules=list(targets),
        bias="none",
        task_type="CAUSAL_LM",
    )


def sdpa_kernels() -> dict[str, bool]:
    """Which scaled-dot-product-attention kernels actually execute on this build and GPU."""
    import warnings

    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    q = torch.randn(1, 4, 256, 128, device="cuda", dtype=torch.bfloat16)
    out = {}
    for name in ("FLASH_ATTENTION", "EFFICIENT_ATTENTION", "CUDNN_ATTENTION", "MATH"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with sdpa_kernel(getattr(SDPBackend, name)):
                    F.scaled_dot_product_attention(q, q, q, is_causal=True)
            out[name.lower()] = True
        except RuntimeError:
            out[name.lower()] = False
    return out


def runtime_checks(model) -> dict[str, Any]:
    """What the loaded model is really doing, read back from the objects rather than the args."""
    import bitsandbytes as bnb
    import torch

    linear4 = [m for m in model.modules() if isinstance(m, bnb.nn.Linear4bit)]
    quant_types = sorted({str(getattr(m.weight, "quant_type", "?")) for m in linear4})
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    base = getattr(model, "base_model", model)
    inner = getattr(base, "model", base)
    return {
        "attn_implementation": getattr(model.config, "_attn_implementation", None),
        "gradient_checkpointing_flag": bool(getattr(inner, "is_gradient_checkpointing", False)),
        # HF applies checkpointing only `if self.gradient_checkpointing and self.training`, and
        # `from_pretrained` returns a model in eval mode: the flag alone proves nothing.
        "training_mode": bool(model.training),
        "gradient_checkpointing_effective": bool(
            getattr(inner, "is_gradient_checkpointing", False) and model.training
        ),
        "linear4bit_modules": len(linear4),
        "quant_types": quant_types,
        "trainable_params": int(trainable),
        "stored_params": int(total),
        "weights_footprint_gb": round(model.get_memory_footprint() / 1e9, 3),
        "sdpa_kernels": sdpa_kernels(),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
    }


def _step(model, optimizer, batch: int, seq_len: int, vocab: int, steps: int) -> dict[str, Any]:
    import torch

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gen = torch.Generator(device="cuda").manual_seed(0)
    ids = torch.randint(0, vocab, (batch, seq_len), device="cuda", generator=gen)
    times = []
    for _ in range(steps):
        torch.cuda.synchronize()
        start = time.perf_counter()
        loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    # The first step pays allocator warm-up and kernel selection; it is not steady state.
    steady = times[1:] if len(times) > 1 else times
    step_s = sum(steady) / len(steady)
    free, total = torch.cuda.mem_get_info()
    return {
        "status": "ok",
        "step_s": round(step_s, 3),
        "tokens_per_s": round(batch * seq_len / step_s, 1),
        "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 3),
        "device_used_gb": round((total - free) / 1e9, 3),
    }


def profile(
    model_id: str,
    seq_lens: tuple[int, ...] = (1024, 2048, 4096),
    batches: tuple[int, ...] = (1, 2, 4, 8),
    checkpointing: tuple[bool, ...] = (True, False),
    r: int = 16,
    steps: int = 4,
    upcast: tuple[bool, ...] = (False,),
    arms: tuple[str, ...] | None = None,
    memory_cap_gib: float = VRAM_CAP_GIB,
    train_mode: bool = True,
) -> dict[str, Any]:
    """The fit grid. Each (upcast, checkpointing) arm gets a fresh model; no flag is inherited."""
    import torch
    from peft import get_peft_model

    cap_fraction = cap_memory(memory_cap_gib)
    report: dict[str, Any] = {
        "model": model_id,
        "lora_r": r,
        "memory_cap_gib": memory_cap_gib,
        "memory_cap_fraction": round(cap_fraction, 4),
        "cells": [],
    }
    if arms is None:
        pairs = [(up, ckpt) for up in upcast for ckpt in checkpointing]
    else:
        pairs = [(a.startswith("upcast"), not a.endswith("no_ckpt")) for a in arms]
    for up, ckpt in pairs:
        started = time.perf_counter()
        model, _ = load_model(model_id, for_training=True, gradient_checkpointing=ckpt, upcast=up)
        load_s = time.perf_counter() - started
        after_load = torch.cuda.memory_allocated() / 1e9
        model = get_peft_model(model, lora_config(r=r))
        model.train(train_mode)
        checks = runtime_checks(model)
        checks.update(load_s=round(load_s, 1), allocated_after_load_gb=round(after_load, 3))
        arm = f"{'upcast' if up else 'bf16'}_{'ckpt' if ckpt else 'no_ckpt'}" + (
            "" if train_mode else "_evalmode"
        )
        report.setdefault("runtime", {})[arm] = checks
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-5, fused=True
        )
        vocab = model.config.vocab_size
        for seq_len in seq_lens:
            for batch in batches:
                cell = {"arm": arm, "seq_len": seq_len, "batch": batch}
                try:
                    cell.update(_step(model, optimizer, batch, seq_len, vocab, steps))
                except torch.cuda.OutOfMemoryError:
                    cell.update(status="oom", peak_allocated_gb=None)
                    optimizer.zero_grad(set_to_none=True)
                    gc.collect()
                    torch.cuda.empty_cache()
                print(json.dumps(cell), flush=True)
                report["cells"].append(cell)
                if cell["status"] == "oom":
                    break  # a larger micro-batch at this length can only OOM harder
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()
    return report


def packed_probe(
    model_id: str, bins: tuple[int, ...] = (1536, 2048, 3072), rows: int = 12, quantize: bool = True
) -> dict[str, Any]:
    """The configuration 12.2.5 actually trains with, on real pairs rather than random ids.

    Two measurements. **Isolation**: the three shortest train examples packed into one row must
    give the same loss as the three scored separately, and a control row with a plain causal mask
    (no segment isolation) must not. **Throughput**: `rows` shuffled packed rows per bin size
    through `train.selected_loss` + backward + AdamW step, steady state (first two rows dropped).
    """
    import random

    import torch
    from peft import get_peft_model

    from src.codegen import prompt as template
    from src.llm import pairs, train

    cap_memory()
    model, tok = load_model(model_id, quantize=quantize, for_training=True)
    model = get_peft_model(model, lora_config(r=16))
    model.train()
    examples = [
        train.tokenise(tok, template.build_messages(p), template.completion(p), tok.eos_token)
        for p in pairs.load("train", sources=("hdbpmn", "fa_bresler"))
    ]
    small = sorted(examples, key=len)[:3]
    model.eval()
    with torch.no_grad():
        packed, _ = train.selected_loss(model, train.collate(small))
        parts = [train.selected_loss(model, train.collate([e])) for e in small]
        control = train.collate(small)
        control["allowed"] = torch.tril(torch.ones_like(control["allowed"]))
        leaked, _ = train.selected_loss(model, control)
    model.train()
    report: dict[str, Any] = {
        "isolation": {
            "packed_loss": float(packed),
            "unpacked_loss": sum(float(v) * n for v, n in parts) / sum(n for _, n in parts),
            "plain_causal_control_loss": float(leaked),
        },
        "bins": [],
    }
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], fused=True)
    for size in bins:
        packing = train.pack(examples, size)
        random.Random(0).shuffle(packing)
        batch = [train.collate([examples[i] for i in b]) for b in packing[:rows]]
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        times, tokens = [], []
        for row in batch:
            torch.cuda.synchronize()
            start = time.perf_counter()
            loss, _ = train.selected_loss(model, row)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
            tokens.append(row["input_ids"].numel())
        cell = {
            "bin": size,
            **train.pack_stats(examples, packing, size),
            "tokens_per_s": round(sum(tokens[2:]) / sum(times[2:]), 1),
            "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
            "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 3),
        }
        print(json.dumps(cell), flush=True)
        report["bins"].append(cell)
    return report


def saturation(cells: list[dict], tolerance: float = 0.05) -> dict[int, int]:
    """Per sequence length, the smallest micro-batch within `tolerance` of the best tokens/s.

    Past that point a larger micro-batch buys VRAM use and nothing else, so it is the batch to
    train at (with gradient accumulation making up the effective batch).
    """
    best: dict[int, int] = {}
    for seq_len in sorted({c["seq_len"] for c in cells}):
        ok = [c for c in cells if c["seq_len"] == seq_len and c["status"] == "ok"]
        if not ok:
            continue
        top = max(c["tokens_per_s"] for c in ok)
        best[seq_len] = min(c["batch"] for c in ok if c["tokens_per_s"] >= (1 - tolerance) * top)
    return best


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 12.2.2 - NF4 QLoRA fit profile")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--out", type=Path, default=Path("experiments/llm/quant.json"))
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[1024, 2048, 4096])
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--arms", nargs="+", default=["bf16_ckpt", "bf16_no_ckpt", "upcast_ckpt"])
    parser.add_argument("--cap-gib", type=float, default=VRAM_CAP_GIB)
    parser.add_argument("--eval-mode", action="store_true", help="reproduce the eval-mode trap")
    args = parser.parse_args(argv)
    report = profile(
        args.model,
        tuple(args.seq_lens),
        tuple(args.batches),
        steps=args.steps,
        arms=tuple(args.arms),
        memory_cap_gib=args.cap_gib,
        train_mode=not args.eval_mode,
    )
    for arm in report["runtime"]:
        cells = [c for c in report["cells"] if c["arm"] == arm]
        report.setdefault("saturation", {})[arm] = saturation(cells)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "cells"}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
