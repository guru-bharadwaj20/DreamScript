# Phase 0.1 — Environment Acceptance Report

Recorded on the target machine after completing tasks 0.1.1 – 0.1.7. Every number below came
from running the listed command, not from a specification sheet.

## Machine

| Property | Value |
| :--- | :--- |
| GPU | NVIDIA RTX 4500 Ada Generation, 24 GB, sm_89 |
| Driver | 595.95 |
| OS | Windows 11 Pro 26200 |
| Python | 3.11.15 (`.venv`, created by `uv`) |
| PyTorch | 2.5.1+cu124 (cuDNN 9.1.0) |

## Acceptance checks

| Task | Command | Result |
| :--- | :--- | :--- |
| 0.1.1 | `.venv/Scripts/python.exe scripts/verify_env.py --only python` | PASS — 7/7 core packages |
| 0.1.2 | `.venv/Scripts/python.exe scripts/check_cuda.py` | PASS — CUDA available, bf16 supported, 24.0 GB, bf16 matmul err 0.362 vs fp32 |
| 0.1.3 | `.venv/Scripts/python.exe scripts/smoke_cv.py` | PASS — adaptive/Otsu/Sauvola binarization, ink fractions 0.139 / 0.046 / 0.046, connected structure 345 px wide |
| 0.1.4 | `.venv/Scripts/python.exe -m pytest tests/test_stack_imports.py` | PASS — 5/5 (sklearn CV, XGBoost, LightGBM, hmmlearn Viterbi, SMOTE) |
| 0.1.5 | `.venv/Scripts/python.exe scripts/check_4bit_load.py` | PASS — 4-bit 7B loaded in **5.22 GB** of 24 GB, 196 `Linear4bit` layers, generated valid Python in 4.8 s |
| 0.1.6 | `.venv/Scripts/python.exe scripts/determinism_check.py --compare` | PASS — two runs byte-identical (CPU loss 0.0024653908, CUDA loss 0.0023451271) |
| 0.1.6 | `.venv/Scripts/python.exe -m pytest tests/test_determinism.py` | PASS — 4/4 |
| 0.1.7 | `docs/hardware.md` | Written — per-phase VRAM budget and QLoRA breakdown |

Full stack verification (`scripts/verify_env.py`, no arguments): **28/28 PASS**.

## Notes carried forward

- **`pip` is not present in the venv.** `uv venv` omits it by design; install with
  `uv pip install -r <file>` after setting `VIRTUAL_ENV`, or `uv pip install --python .venv/Scripts/python.exe`.
- **PyTorch must come from the CUDA index.** The default PyPI wheel is CPU-only:
  `uv pip install -r requirements/torch.txt --index-url https://download.pytorch.org/whl/cu124`.
- **`PYTHONHASHSEED` needs a re-exec.** It is read at interpreter start, so
  `src.utils.seed.ensure_hashseed()` is called first in any entry point whose results must
  reproduce; `set_seed()` alone does not fix string hashing.
- **4-bit base model choice.** `unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit` is pre-quantized
  NF4 (~5 GB download) rather than the ~15 GB bf16 upstream, and loads with the same
  `BitsAndBytesConfig` Phase 12 will use. First load took 732 s including the download;
  cached loads are seconds.
- **The GPU is shared.** Another process on this machine held ~20 GB during setup. Phase 9
  and Phase 12 runs must check free VRAM before starting, not just total VRAM.
- **`pdf2image` exposes no `__version__`** and needs the poppler binary on PATH for real PDF
  input; only the import is checked here since PDF ingest is a Phase 1 concern.
- **TF32 is off**, consistent with the determinism policy in `docs/hardware.md`.

## Reproducing

```
uv venv --python 3.11 .venv
export VIRTUAL_ENV=$PWD/.venv
uv pip install -r requirements/base.txt
uv pip install -r requirements/torch.txt --index-url https://download.pytorch.org/whl/cu124
uv pip install -r requirements/cv.txt -r requirements/classical.txt -r requirements/genai.txt
.venv/Scripts/python.exe scripts/verify_env.py
```

Conda users: `conda env create -f env.yml` installs the same pinned set.
