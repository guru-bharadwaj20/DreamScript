"""Phase 0.1.5 acceptance check — a 7B model loads in 4-bit and generates.

This is the exact loading path Phase 12 uses for QLoRA fine-tuning: NF4 quantization with
double quantization and a bf16 compute dtype. Passing here means the RTX 4500 Ada can host
the base model with room left for LoRA adapters, optimizer state and activations.

    .venv/Scripts/python.exe scripts/check_4bit_load.py
    .venv/Scripts/python.exe scripts/check_4bit_load.py --model Qwen/Qwen2.5-Coder-7B-Instruct
"""

from __future__ import annotations

import argparse
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Pre-quantized NF4 build of Qwen2.5-Coder-7B: same weights Phase 12 fine-tunes, but a
# ~5 GB download instead of ~15 GB of bf16 shards.
DEFAULT_MODEL = "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit"

PROMPT = "# Python function that returns True when age is at least 18\ndef is_adult(age):"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("RESULT: FAIL - no CUDA device")
        return 1

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print(f"model            : {args.model}")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=quant, device_map={"": 0}, torch_dtype=torch.bfloat16
    )
    load_s = time.time() - t0

    n_params = sum(p.numel() for p in model.parameters())
    vram_gb = torch.cuda.memory_allocated() / 1024**3
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"load time        : {load_s:.1f}s")
    print(f"parameters       : {n_params / 1e9:.2f}B (4-bit packed)")
    print(f"vram allocated   : {vram_gb:.2f} GB of {total_gb:.1f} GB")

    n_4bit = sum(1 for m in model.modules() if type(m).__name__ == "Linear4bit")
    print(f"Linear4bit layers: {n_4bit}")

    t0 = time.time()
    inputs = tok(PROMPT, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    gen_s = time.time() - t0
    text = tok.decode(out[0], skip_special_tokens=True)
    print(f"generate time    : {gen_s:.1f}s for {args.max_new_tokens} tokens")
    print("-" * 60)
    print(text)
    print("-" * 60)

    checks = {
        "is_7b_scale": n_params > 3.0e9,  # 4-bit packing halves the reported element count
        "uses_4bit_layers": n_4bit > 100,
        "fits_with_headroom": vram_gb < 12.0,  # leaves >12 GB for LoRA training state
        "generates_text": len(text) > len(PROMPT),
    }
    for name, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    ok = all(checks.values())
    print("RESULT:", "4-bit 7B load OK" if ok else "4-bit 7B load FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
