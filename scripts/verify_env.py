"""Phase 0.1 environment verification.

Runs every Phase 0.1 acceptance check that can be checked programmatically and prints a
pass/fail table. Usage:

    .venv/Scripts/python.exe scripts/verify_env.py           # all available checks
    .venv/Scripts/python.exe scripts/verify_env.py --only python cv
"""

from __future__ import annotations

import argparse
import importlib
import platform
import sys

CHECKS: dict[str, list[str]] = {
    # 0.1.1 interpreter + core runtime
    "python": ["numpy", "pandas", "pyarrow", "yaml", "tqdm", "matplotlib", "pytest"],
    # 0.1.2 torch + cuda
    "torch": ["torch"],
    # 0.1.3 computer vision
    "cv": ["cv2", "skimage", "PIL", "pdf2image"],
    # 0.1.4 classical ML
    "classical": ["sklearn", "xgboost", "lightgbm", "hmmlearn", "imblearn"],
    # 0.1.5 deep learning / generative
    "genai": ["transformers", "peft", "trl", "bitsandbytes", "accelerate", "datasets"],
}


def check_group(name: str) -> list[tuple[str, bool, str]]:
    rows: list[tuple[str, bool, str]] = []
    for mod in CHECKS[name]:
        try:
            m = importlib.import_module(mod)
            rows.append((mod, True, getattr(m, "__version__", "-")))
        except Exception as exc:  # noqa: BLE001 - report, never raise
            rows.append((mod, False, f"{type(exc).__name__}: {exc}"))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=sorted(CHECKS), default=sorted(CHECKS))
    args = ap.parse_args()

    print(f"python   : {platform.python_version()} ({sys.executable})")
    print(f"platform : {platform.platform()}")
    print("-" * 68)

    ok = True
    for group in args.only:
        print(f"[{group}]")
        for mod, passed, info in check_group(group):
            mark = "PASS" if passed else "FAIL"
            ok &= passed
            print(f"  {mark}  {mod:<14} {info}")
    print("-" * 68)
    print("RESULT:", "all checks passed" if ok else "one or more checks FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
