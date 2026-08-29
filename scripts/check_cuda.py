"""Phase 0.1.2 acceptance check — CUDA + PyTorch on the RTX 4500 Ada.

Asserts the three things every later phase depends on:
  1. torch.cuda.is_available()
  2. bf16 is supported (QLoRA in Phase 12 trains in bf16)
  3. a real matmul executes on the GPU and matches the CPU result

Exit code 0 = the environment is ready for GPU work.
"""

from __future__ import annotations

import sys

import torch

REQUIRED_VRAM_GB = 20.0  # Phase 12 QLoRA needs ~18 GB; RTX 4500 Ada ships 24 GB


def main() -> int:
    print(f"torch            : {torch.__version__}")
    print(f"cuda (built)     : {torch.version.cuda}")
    print(f"cudnn            : {torch.backends.cudnn.version()}")

    available = torch.cuda.is_available()
    print(f"cuda available   : {available}")
    if not available:
        print("RESULT: FAIL - no CUDA device visible to torch")
        return 1

    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    vram_gb = props.total_memory / 1024**3
    cap = f"{props.major}.{props.minor}"
    print(f"device           : {props.name}")
    print(f"compute capability: sm_{props.major}{props.minor} ({cap})")
    print(f"vram             : {vram_gb:.1f} GB")

    bf16 = torch.cuda.is_bf16_supported()
    print(f"bf16 supported   : {bf16}")

    tf32 = torch.backends.cuda.matmul.allow_tf32
    print(f"tf32 matmul flag : {tf32}")

    # End-to-end GPU sanity: bf16 matmul on device, compared against fp32 on CPU.
    torch.manual_seed(0)
    a = torch.randn(512, 512)
    b = torch.randn(512, 512)
    ref = a @ b
    got = (a.cuda().bfloat16() @ b.cuda().bfloat16()).float().cpu()
    err = (got - ref).abs().max().item()
    print(f"bf16 matmul max abs err vs fp32 cpu: {err:.4f}")

    checks = {
        "cuda_available": available,
        "bf16_supported": bf16,
        "vram_sufficient": vram_gb >= REQUIRED_VRAM_GB,
        "matmul_sane": err < 5.0,  # bf16 has ~3 decimal digits; 512-dim sums drift
    }
    print("-" * 60)
    for name, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    ok = all(checks.values())
    print("RESULT:", "GPU environment ready" if ok else "GPU environment NOT ready")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
