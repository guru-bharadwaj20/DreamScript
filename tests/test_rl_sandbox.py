"""Adversarial tests for the Phase 11.2.9 sandbox.

Every test here is a way the RL loop could hang, leak or lie. The happy path is three tests; the
rest are the ones that matter.
"""

from __future__ import annotations

import contextlib
import glob
import os
import tempfile
import time
from pathlib import Path

import psutil
import pytest

from src.rl.sandbox import DEFAULT_DENY_IMPORTS, MIN_MEMORY_LIMIT_MB, run

REPO = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- happy path


def test_ok_returns_stdout_and_zero_exit():
    r = run("print(6 * 7)", timeout_s=10)
    assert r["ok"] is True
    assert r["kind"] == "ok"
    assert r["exit_code"] == 0
    assert r["stdout"].strip() == "42"
    assert r["timed_out"] is False


def test_result_has_the_full_contract():
    r = run("pass", timeout_s=10)
    for key in (
        "ok",
        "kind",
        "exit_code",
        "stdout",
        "stderr",
        "stdout_bytes",
        "stderr_bytes",
        "truncated",
        "wall_time_s",
        "peak_memory_bytes",
        "timed_out",
        "detail",
    ):
        assert key in r, key
    assert r["wall_time_s"] > 0


def test_unicode_survives_the_round_trip():
    r = run("print('é→中')", timeout_s=10)
    assert r["ok"] is True
    assert "中" in r["stdout"]


# --------------------------------------------------------------------------- failure kinds


def test_syntax_error_is_reported_as_such_not_as_a_crash():
    r = run("def f(:\n    pass", timeout_s=10)
    assert r["kind"] == "syntax_error"
    assert r["ok"] is False
    assert "SyntaxError" in r["stderr"]
    assert r["timed_out"] is False


def test_exception_is_reported_with_a_traceback():
    r = run("raise ValueError('boom')", timeout_s=10)
    assert r["kind"] == "exception"
    assert "ValueError: boom" in r["stderr"]
    assert "boom" in r["detail"]


def test_explicit_nonzero_exit_is_its_own_kind():
    r = run("import sys; sys.exit(3)", timeout_s=10)
    assert r["kind"] == "nonzero_exit"
    assert r["exit_code"] == 3


def test_sys_exit_zero_is_success():
    r = run("import sys; sys.exit(0)", timeout_s=10)
    assert r["ok"] is True


def test_unbounded_recursion_is_contained_not_a_harness_crash():
    r = run(
        "import sys\nsys.setrecursionlimit(10**7)\ndef f(n):\n    return f(n + 1)\nf(0)",
        timeout_s=20,
        memory_limit_mb=128,
    )
    assert r["kind"] in {"exception", "crash", "memory"}
    assert r["ok"] is False


# --------------------------------------------------------------------------- timeout


def test_infinite_loop_hits_the_timeout():
    t = time.perf_counter()
    r = run("while True:\n    pass", timeout_s=1.0)
    elapsed = time.perf_counter() - t
    assert r["kind"] == "timeout"
    assert r["timed_out"] is True
    assert 1.0 <= elapsed < 3.0, elapsed  # measured kill latency is ~14 ms


def test_a_process_that_ignores_signals_still_dies():
    # SIGINT is swallowed; only TerminateJobObject ends this one.
    code = (
        "import signal, time\n"
        "try:\n"
        "    signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
        "except Exception:\n"
        "    pass\n"
        "while True:\n"
        "    time.sleep(0.01)\n"
    )
    r = run(code, timeout_s=1.0)
    assert r["kind"] == "timeout"


def test_a_child_that_spawns_a_grandchild_is_still_cleaned_up(tmp_path):
    marker = tmp_path / "pid.txt"
    inner = (
        "import os, time\n"
        "open(os.environ['MARKER'], 'w').write(str(os.getpid()))\n"
        "time.sleep(600)\n"
    )
    outer = (
        "import subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, '-c', {inner!r}])\n"
        "time.sleep(600)\n"
    )
    r = run(outer, timeout_s=2.0, deny_imports=(), env_extra={"MARKER": str(marker)})
    assert r["kind"] == "timeout"

    for _ in range(50):  # give the grandchild a moment to have written its pid
        if marker.exists() and marker.read_text().strip():
            break
        time.sleep(0.05)
    pid = int(marker.read_text().strip())

    gone = False
    for _ in range(40):
        if not psutil.pid_exists(pid):
            gone = True
            break
        time.sleep(0.05)
    if not gone:  # never leave a 600 s sleeper behind, even on failure
        with contextlib.suppress(psutil.Error):
            psutil.Process(pid).kill()
    assert gone, f"grandchild {pid} survived the timeout kill"


# --------------------------------------------------------------------------- memory


def test_runaway_allocation_is_contained():
    code = "b = []\nfor _ in range(400):\n    b.append(bytearray(1024 * 1024))\nprint('survived')"
    r = run(code, timeout_s=30, memory_limit_mb=128)
    assert r["kind"] == "memory"
    assert "survived" not in r["stdout"]
    assert r["peak_memory_bytes"] is not None
    assert r["peak_memory_bytes"] < 160 * 1024 * 1024, r["peak_memory_bytes"]


def test_one_giant_allocation_is_refused_rather_than_swapped():
    r = run("b = bytearray(1024 * 1024 * 1024)\nprint(len(b))", timeout_s=30, memory_limit_mb=128)
    assert r["kind"] == "memory"
    assert r["stdout"].strip() == ""


def test_memory_limit_is_clamped_to_a_workable_floor():
    # A 1 MB cap would kill the interpreter before it read a byte of code; the floor prevents a
    # cap that silently turns every run into a failure.
    r = run("print('alive')", timeout_s=20, memory_limit_mb=1)
    assert r["ok"] is True
    assert MIN_MEMORY_LIMIT_MB >= 64


def test_a_run_under_the_cap_reports_a_plausible_peak():
    r = run("x = bytearray(30 * 1024 * 1024)\nprint(len(x))", timeout_s=30, memory_limit_mb=256)
    assert r["ok"] is True
    if r["peak_memory_bytes"] is not None:  # POSIX has no job counters
        assert r["peak_memory_bytes"] > 25 * 1024 * 1024


def test_process_spawning_is_capped():
    code = (
        "import subprocess, sys\n"
        "n = 0\n"
        "for _ in range(40):\n"
        "    try:\n"
        "        subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
        "    except Exception:\n"
        "        break\n"
        "    n += 1\n"
        "print('spawned', n)\n"
    )
    r = run(code, timeout_s=30, deny_imports=(), active_process_limit=4)
    assert r["kind"] in {"ok", "timeout", "exception"}
    if r["stdout"].strip():
        assert int(r["stdout"].split()[-1]) < 10


# --------------------------------------------------------------------------- output flooding


def test_megabytes_to_stdout_do_not_deadlock_or_blow_memory():
    code = "import sys\nblock = 'x' * 65536\nfor _ in range(1024):\n    sys.stdout.write(block)\n"
    t = time.perf_counter()
    r = run(code, timeout_s=30, memory_limit_mb=128, max_output_bytes=1 << 20)
    elapsed = time.perf_counter() - t
    assert r["ok"] is True, r["kind"]
    assert r["stdout_bytes"] == 64 * 1024 * 1024
    assert len(r["stdout"]) <= 1 << 20
    assert r["truncated"] is True
    assert elapsed < 20, elapsed


def test_megabytes_to_stderr_also_survive():
    code = "import sys\nblock = 'e' * 65536\nfor _ in range(256):\n    sys.stderr.write(block)\n"
    r = run(code, timeout_s=30, max_output_bytes=1 << 16)
    assert r["ok"] is True
    assert r["stderr_bytes"] == 16 * 1024 * 1024
    assert len(r["stderr"]) <= 1 << 16
    assert r["truncated"] is True


def test_flooding_plus_timeout_still_terminates():
    code = "import sys\nwhile True:\n    sys.stdout.write('y' * 4096)\n"
    t = time.perf_counter()
    r = run(code, timeout_s=1.0, max_output_bytes=1 << 16)
    assert r["kind"] == "timeout"
    assert time.perf_counter() - t < 5


# --------------------------------------------------------------------------- isolation


def test_the_child_runs_in_a_temp_directory_not_the_repo():
    r = run("import os\nprint(os.getcwd())", timeout_s=10)
    cwd = Path(r["stdout"].strip())
    assert r["ok"] is True
    assert "dsbox_" in str(cwd)
    assert REPO not in cwd.parents and cwd != REPO


def test_files_written_by_generated_code_never_reach_the_repo():
    before = {p.name for p in REPO.iterdir()}
    r = run("open('artifact.txt', 'w').write('x' * 100)\nprint('wrote')", timeout_s=10)
    assert r["ok"] is True
    assert not (REPO / "artifact.txt").exists()
    assert {p.name for p in REPO.iterdir()} == before


def test_the_temp_directory_is_cleaned_up():
    r = run("import os\nprint(os.getcwd())", timeout_s=10)
    cwd = Path(r["stdout"].strip())
    assert not cwd.exists()
    assert not cwd.parent.exists()


def test_repeated_runs_leave_no_temp_dirs_behind():
    pattern = os.path.join(tempfile.gettempdir(), "dsbox_*")
    before = set(glob.glob(pattern))
    for _ in range(5):
        run("print(1)", timeout_s=10)
    leaked = set(glob.glob(pattern)) - before
    assert not leaked, leaked


def test_seeded_files_are_visible_to_the_generated_code():
    r = run(
        "print(open('data.txt').read().strip())",
        timeout_s=10,
        files={"data.txt": "seeded"},
    )
    assert r["stdout"].strip() == "seeded"


def test_each_run_gets_a_fresh_interpreter():
    a = run("import os\nprint(os.getpid())", timeout_s=10)
    b = run("import os\nprint(os.getpid())", timeout_s=10)
    assert a["stdout"].strip() != b["stdout"].strip()


def test_state_does_not_leak_between_runs():
    run("import builtins\nbuiltins.LEAKED = 1", timeout_s=10)
    r = run("import builtins\nprint(hasattr(builtins, 'LEAKED'))", timeout_s=10)
    assert r["stdout"].strip() == "False"


# --------------------------------------------------------------------------- static pre-scan


@pytest.mark.parametrize("mod", ["socket", "subprocess", "shutil", "urllib.request"])
def test_denied_imports_are_refused_without_spawning(mod):
    r = run(f"import {mod}", timeout_s=10)
    assert r["kind"] == "refused"
    assert r["exit_code"] is None
    assert r["wall_time_s"] < 0.05  # refusal must be cheaper than a spawn (~70 ms)


def test_from_imports_are_refused_too():
    r = run("from socket import socket", timeout_s=10)
    assert r["kind"] == "refused"


def test_the_scan_can_be_disabled():
    r = run("import socket\nprint('ok')", timeout_s=10, deny_imports=())
    assert r["ok"] is True


def test_the_scan_is_advisory_and_says_so_by_being_bypassable():
    # Documented limitation, pinned as a test so nobody mistakes the tripwire for a boundary.
    r = run("m = __import__('socket')\nprint(type(m).__name__)", timeout_s=10)
    assert r["ok"] is True
    assert "socket" in DEFAULT_DENY_IMPORTS


def test_a_syntax_error_is_not_swallowed_by_the_scan():
    r = run("import socket(", timeout_s=10)
    assert r["kind"] == "syntax_error"


# --------------------------------------------------------------------------- cost


def test_per_call_overhead_stays_within_budget():
    times = []
    for _ in range(10):
        t = time.perf_counter()
        run("pass", timeout_s=10)
        times.append(time.perf_counter() - t)
    median = sorted(times)[len(times) // 2]
    assert median < 0.30, median  # measured median 71 ms; the budget is deliberately loose
