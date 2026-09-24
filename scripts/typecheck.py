"""Phase 15 - the type-checking gate, as a ratchet.

    python scripts/typecheck.py            # fail if mypy is worse than the baseline
    python scripts/typecheck.py --update   # record today's count as the new baseline
    python scripts/typecheck.py --report   # per-error-code breakdown, nothing enforced

mypy was installed - there is a `.mypy_cache/` in the tree - reported 661 errors across 193
files, and was in **no gate**: not `make lint`, not `.pre-commit-config.yaml`, not CI. A checker
nobody runs is a checker whose findings cannot be told apart from noise.

661 is too many to fix in one pass and too many to ignore honestly, and the usual answers are
both wrong here. Turning mypy on as a pass/fail gate makes `main` red on day one and it stays
red. Leaving it out is where this started.

So: a **baseline that may fall and may not rise**. The count is recorded per error code, which
makes a regression legible - "3 new `arg-type`" is a review comment, "344 errors" is not - and
`--update` is the only way the numbers go up, which makes raising one a deliberate line in a
diff rather than a silent drift.

`pyproject.toml`'s `[tool.mypy]` sets `ignore_missing_imports`, which removed 318 of the
original 661: they were `Library stubs not installed for "cv2"`-shaped reports about
third-party packages and say nothing about this repo's code.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "reports" / "mypy_baseline.json"

_CODE = re.compile(r"\[([a-z][a-z-]*)\]\s*$")


def run() -> tuple[Counter[str], int, str]:
    """`(errors by code, files with errors, raw output)`."""
    result = subprocess.run(
        [sys.executable, "-m", "mypy"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    output = result.stdout + result.stderr
    codes: Counter[str] = Counter()
    files: set[str] = set()
    for line in output.splitlines():
        match = _CODE.search(line)
        if match and ": error:" in line:
            codes[match.group(1)] += 1
            files.add(line.split(":", 1)[0])
    return codes, len(files), output


def load() -> dict:
    if not BASELINE.is_file():
        return {"total": 0, "files": 0, "by_code": {}}
    return json.loads(BASELINE.read_text(encoding="utf-8"))


def save(codes: Counter[str], files: int) -> None:
    BASELINE.parent.mkdir(parents=True, exist_ok=True)
    BASELINE.write_text(
        json.dumps(
            {"total": sum(codes.values()), "files": files, "by_code": dict(sorted(codes.items()))},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def compare(codes: Counter[str], baseline: dict) -> list[str]:
    """Every code that is worse than it was, most-regressed first."""
    before = baseline.get("by_code", {})
    worse = [
        f"  {code:<16} {before.get(code, 0)} -> {count}  (+{count - before.get(code, 0)})"
        for code, count in codes.items()
        if count > before.get(code, 0)
    ]
    return sorted(worse, key=lambda row: -int(row.rsplit("+", 1)[1].rstrip(")")))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--update", action="store_true", help="record today's counts as the baseline")
    ap.add_argument("--report", action="store_true", help="print the breakdown, enforce nothing")
    args = ap.parse_args(argv)

    codes, files, output = run()
    total = sum(codes.values())

    if args.report:
        for code, count in codes.most_common():
            print(f"{count:>5}  {code}")
        print(f"{total:>5}  TOTAL across {files} files")
        return 0

    if args.update:
        save(codes, files)
        print(f"baseline updated: {total} errors across {files} files")
        return 0

    baseline = load()
    regressions = compare(codes, baseline)
    if regressions:
        print(output.strip()[-4000:], file=sys.stderr)
        print("\nmypy regressed against reports/mypy_baseline.json:", file=sys.stderr)
        print("\n".join(regressions), file=sys.stderr)
        print(
            "\nFix them, or run `python scripts/typecheck.py --update` to raise the baseline "
            "deliberately - which is a line in the diff rather than a silent drift.",
            file=sys.stderr,
        )
        return 1

    improved = baseline.get("total", 0) - total
    print(f"mypy: {total} errors across {files} files (baseline {baseline.get('total', 0)})")
    if improved > 0:
        print(f"{improved} fewer than the baseline - run --update to lock it in.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
