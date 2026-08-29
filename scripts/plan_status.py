"""Flip a plan.md task row to done and keep the progress table consistent.

    python scripts/plan_status.py 2.1.1 [--dod "new definition-of-done text"]

Editing plan.md by hand across 302 rows is how the progress table drifts away from the rows
it summarises. This does both edits at once, so they cannot disagree.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "plan.md"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("tasks", nargs="+", help="task ids, e.g. 2.1.1")
    ap.add_argument("--dod", help="replace the Definition of Done cell (single task only)")
    args = ap.parse_args(argv)

    lines = PLAN.read_text(encoding="utf-8").split("\n")
    if args.dod and len(args.tasks) != 1:
        ap.error("--dod applies to exactly one task")

    for task in args.tasks:
        for i, line in enumerate(lines):
            if not line.startswith(f"| {task} |"):
                continue
            cells = line.split(" | ")
            if args.dod:
                cells[-2] = args.dod
            cells[-1] = "✅ |"
            lines[i] = " | ".join(cells)
            break
        else:
            print(f"no row for {task}", file=sys.stderr)
            return 1

    # Recount from the rows themselves rather than trusting the summary table.
    phase = None
    done: dict[str, int] = {}
    total: dict[str, int] = {}
    for line in lines:
        m = re.match(r"^# Phase (\d+)", line)
        if m:
            phase = m.group(1)
        if line.startswith("## Cross-Phase"):
            break  # everything past here is summary tables, not task rows
        # Phases 8, 13-15 and 17 number tasks <phase>.<n>; the rest <phase>.<n>.<n>.
        cell = re.match(r"^\| (\d+)\.\d+(?:\.\d+)? \|", line)
        if cell and phase is not None and cell.group(1) == phase:
            total[phase] = total.get(phase, 0) + 1
            if line.rstrip().endswith("✅ |"):
                done[phase] = done.get(phase, 0) + 1

    for i, line in enumerate(lines):
        m = re.match(r"^\| (\d+) — (.+?) \| (\d+) \| (\d+) \| (✅|❌) \|$", line)
        if not m:
            continue
        p = m.group(1)
        d, t = done.get(p, 0), total.get(p, int(m.group(3)))
        mark = "✅" if d == t else "❌"
        lines[i] = f"| {p} — {m.group(2)} | {t} | {d} | {mark} |"
    grand_d, grand_t = sum(done.values()), sum(total.values())
    for i, line in enumerate(lines):
        if line.startswith("| **Total** |"):
            mark = "✅" if grand_d == grand_t else "❌"
            lines[i] = f"| **Total** | **{grand_t}** | **{grand_d}** | {mark} |"

    PLAN.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"{' '.join(args.tasks)} -> done   ({grand_d}/{grand_t})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
