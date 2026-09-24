"""Phase 12.1.7 / 12.3 - SPICE netlists checked by a real simulator, ngspice 47, in batch mode.

    from src.eval.spice import check, check_many
    check(netlist) -> {ok, kind, detail}

`kind` is one of:

    spice.ok               ngspice parsed the deck and the `.op` operating point solved
    spice.parse_error      ngspice reported an `Error` (unknown device, missing model, bad
                           value) or exited non-zero
    spice.no_convergence   parsed, but the operating point did not solve: a singular matrix or
                           failed gmin/source stepping - e.g. two ideal inductors in parallel
                           form a zero-resistance DC loop, and a floating node has no DC path
    spice.no_analysis      parsed, but there was nothing to simulate
    spice.timeout          the simulator ran past `timeout_s`
    spice.unavailable      no ngspice binary; never folded into a pass

Why a simulator and not only 12.1.7's structural rules: the structural check reads cards and
counts nets; it cannot know that `L3 3 0 1m` and `L4 3 0 1m` short each other at DC. The deck
is well-formed and physically unsolvable, and a target like that teaches a model to draw it.

The binary lives in `data/processed/codegen/tools/Spice64` (gitignored). `ensure_toolchain()`
downloads `ngspice-47_64.7z` from the project's SourceForge release and unpacks it with `py7zr`
when that module is importable; otherwise it reports unavailable rather than guessing.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.utils.config import ROOT

TOOL_DIR = ROOT / "data" / "processed" / "codegen" / "tools"
BINARY = TOOL_DIR / "Spice64" / "bin" / ("ngspice_con.exe" if os.name == "nt" else "ngspice")
URL = "https://downloads.sourceforge.net/project/ngspice/ng-spice-rework/47/ngspice-47_64.7z"

_ERROR = re.compile(r"^\s*Error\b|Simulation interrupted due to error", re.M | re.I)
_NO_CONVERGE = re.compile(
    r"singular matrix|gmin stepping failed|source stepping failed|timestep too small", re.I
)
_NO_ANALYSIS = re.compile(r"no simulations run", re.I)


def available() -> bool:
    return BINARY.is_file()


def ensure_toolchain() -> bool:
    """Download and unpack ngspice if it is missing. Returns availability."""
    if available():
        return True
    try:
        import py7zr
    except ImportError:
        return False
    TOOL_DIR.mkdir(parents=True, exist_ok=True)
    archive = TOOL_DIR / "ngspice.7z"
    urllib.request.urlretrieve(URL, archive)  # fixed https release URL
    with py7zr.SevenZipFile(archive) as handle:
        handle.extractall(TOOL_DIR)
    return available()


def classify(returncode: int, output: str) -> tuple[bool, str, str]:
    """(ok, kind, detail) from ngspice's exit code and combined output."""
    first = next((ln.strip() for ln in output.splitlines() if _ERROR.search(ln)), "")
    if returncode != 0 or _ERROR.search(output):
        lines = output.splitlines()
        index = next((i for i, ln in enumerate(lines) if _ERROR.search(ln)), 0)
        detail = " ".join(ln.strip() for ln in lines[index : index + 3]) or first
        return False, "spice.parse_error", detail[:300] or f"exit {returncode}"
    match = _NO_CONVERGE.search(output)
    if match:
        line = next(ln.strip() for ln in output.splitlines() if _NO_CONVERGE.search(ln))
        return False, "spice.no_convergence", line[:300]
    if _NO_ANALYSIS.search(output):
        return False, "spice.no_analysis", "no simulations run"
    return True, "spice.ok", "parsed and .op solved"


def check(netlist: str, timeout_s: float = 20.0) -> dict:
    """Run one deck through `ngspice -b` in a private temp directory."""
    if not available():
        return {"ok": False, "kind": "spice.unavailable", "detail": f"no {BINARY}"}
    with tempfile.TemporaryDirectory(prefix="ds_spice_") as tmp:
        deck = Path(tmp) / "deck.cir"
        deck.write_text(netlist if netlist.endswith("\n") else netlist + "\n", encoding="utf-8")
        try:
            proc = subprocess.run(
                [str(BINARY), "-b", str(deck)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=tmp,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "kind": "spice.timeout", "detail": f">{timeout_s}s"}
    ok, kind, detail = classify(proc.returncode, (proc.stdout or "") + (proc.stderr or ""))
    return {"ok": ok, "kind": kind, "detail": detail}


def check_many(netlists: list[str], workers: int | None = None) -> list[dict]:
    """`check` over many decks on a thread pool - each call is its own ngspice process."""
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(check, netlists))
