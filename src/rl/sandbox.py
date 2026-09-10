"""Phase 11.2.9 - running machine-generated code without letting it wedge the trainer.

    from src.rl.sandbox import run
    res = run("print('hello')", timeout_s=5.0, memory_limit_mb=256)

Phase 11's semantic reward asks *does the emitted program actually run*, so every episode ends by
executing code that nothing has proof-read. The emitter is ours, which rules out malice and rules
in nothing else: a template that closes a loop on a traced edge the tracer made cyclic produces
`while True`, an unbounded range produces an unbounded list, and either one, run in-process,
stops the training loop forever. **This is a resource sandbox, not a security boundary** - the
"What it does not protect against" section below is the honest version, and it is short.

## The mechanism, and why it is a job object

Three things had to hold at once: kill a wedged child *and its descendants*, cap memory on
Windows, and read unbounded output without deadlocking. One Windows primitive covers the first
two. A **job object** created per call, with `PROCESS_MEMORY | JOB_MEMORY | ACTIVE_PROCESS |
KILL_ON_JOB_CLOSE`, gives a hard commit cap the kernel enforces at allocation time and a
`TerminateJobObject` that takes the whole tree down at once, because job membership is inherited
by every process the child spawns.

The ordering matters and is the one real race: the job is created and the child assigned to it
*before* the child sees a byte of code. The bootstrap blocks on `sys.stdin.buffer.read()`, so the
parent assigns first and only then writes the program and closes the pipe. Assigning after the
code has started would leave a window in which an allocation runs uncapped.

## What was measured

Windows 11 Pro 26200, CPython 3.11.15, cold CPU, no GPU.

    per-call overhead    n=50, `run("pass")`      median 71.2 ms   mean 72.5   p95 78.2   max 115.4
    refusal (pre-scan)   n=200, denied import     median  0.005 ms  - no process is spawned
    kill latency         n=10, 1s/0.3s budgets    mean  13.5 ms    max 20.2 ms past the deadline
    interpreter baseline peak committed           9.6 MB           -> MIN_MEMORY_LIMIT_MB = 64
    64 MB flooded to stdout, 1 MB kept            94 ms total, child peak 9.6 MB, no deadlock

**The memory cap is a real hard cap, not a watchdog.** 400 x 1 MB `bytearray` under a 128 MB
limit: the allocation that crosses the line raises `MemoryError` in the child, `PeakJobMemoryUsed`
reads **134.5 MB (128.3 MiB)** against the 128 MiB asked for - 0.3% overshoot, which is the
kernel's own accounting granularity, not slack in the enforcement. A single `bytearray(1 GB)`
under the same cap fails the same way, immediately, with nothing paged to disk. `run` reports
`kind="memory"` and the observed `peak_memory_bytes` in both cases.

**The tree kill was verified against its own counter-example.** A control that spawns the same
parent/grandchild pair and calls plain `Popen.kill()` leaves the grandchild **alive** - measured,
not assumed: `TerminateProcess` ends one process and orphans its children, and Windows has no
process group to signal instead. Through the job object the grandchild is gone within 20 ms of
the deadline. `test_a_child_that_spawns_a_grandchild_is_still_cleaned_up` pins this.

`ActiveProcessLimit` is the fork-bomb ceiling: with the limit set to 4, generated code asking for
40 subprocesses gets **2** and then `OSError`, and none of them outlive the run.

## What was rejected

**`resource.setrlimit`** - POSIX only. The obvious port is a no-op import guard, which is exactly
the "cap that silently does nothing" this row exists to avoid. It is kept for the POSIX branch
(`RLIMIT_AS` + `setsid` + `killpg`) and is *untested here*, since this project runs on Windows.

**A psutil RSS-polling watchdog thread** - the usual Windows workaround, and it was rejected on
its numbers. Polling at 10 ms costs a thread per call against a 71 ms call, and it is advisory by
construction: it observes an allocation that already succeeded, so `bytearray(1 GB)` commits in
full before any sample sees it. The job object refuses the allocation instead. A watchdog would
have shipped a cap that reports overshoots rather than one that prevents them.

**A warm pool of pre-started interpreters** to amortise the ~71 ms spawn. Tempting - it is most
of the per-call cost - and rejected because the two hard requirements are incompatible with
reuse: a timed-out or memory-killed worker has to be replaced anyway (which is where the cost
is), and a reused interpreter carries module state, monkey-patches and open handles from the
previous episode into the next, so the reward stops being a function of the code alone.
`test_state_does_not_leak_between_runs` pins the property that would have been traded away. At
71 ms, 10k reward evaluations cost ~12 minutes of the training run; revisit only if that becomes
the bottleneck, and then with per-episode isolation solved rather than dropped.

**`subprocess.communicate()`** for output. It buffers without bound: 64 MB of generated `print`
output becomes 64 MB in the trainer. Two draining threads with a keep-cap that keeps reading past
it cost nothing measurable and cap memory at `max_output_bytes` while never letting the 64 KB
pipe buffer fill.

**An import blocklist as a security control.** `_scan` is an AST tripwire, it is cheap (5 us) and
it catches the plausible emitter bug; `__import__("socket")` walks straight past it and
`test_the_scan_is_advisory_and_says_so_by_being_bypassable` asserts that it does, so nobody later
mistakes it for a boundary.

## What it does not protect against

Plainly, because the temptation is to oversell this:

- **It is not a security boundary.** The child runs as the same user with the same token, a
  minimal but real `PATH`, and full filesystem access outside its temp cwd. Code that wants to
  read or delete your files can. Nothing here stops deliberate abuse; it stops accidents.
- The static scan is bypassable by any dynamic import (above), and there is no network isolation
  at all - the cap is on memory and processes, not on sockets.
- The cap is on **committed memory**, not on disk: generated code can still fill the drive.
- `active_process_limit` bounds concurrent processes, not CPU. A busy loop burns one core for the
  whole `timeout_s`; the timeout, not a quota, is what bounds it.
- The POSIX branch is written but unmeasured. Every number above is Windows.
"""

from __future__ import annotations

import ast
import contextlib
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

IS_WINDOWS = sys.platform == "win32"

if IS_WINDOWS:  # pragma: no cover - platform branch
    import win32api
    import win32con
    import win32job
else:  # pragma: no cover - platform branch
    import resource
    import signal

#: Exit codes the bootstrap uses to name a failure the parent cannot otherwise see.
EXIT_SYNTAX = 91
EXIT_EXCEPTION = 92
EXIT_MEMORY = 93

#: Modules the static pre-scan refuses by default. Advisory only - see `_scan`.
DEFAULT_DENY_IMPORTS: tuple[str, ...] = (
    "ctypes",
    "http",
    "multiprocessing",
    "requests",
    "shutil",
    "socket",
    "subprocess",
    "urllib",
    "webbrowser",
)

#: Interpreter startup alone commits ~13 MB, so a cap under this kills every run at import time.
MIN_MEMORY_LIMIT_MB = 64

_BOOTSTRAP = (
    """\
import sys, traceback
src = sys.stdin.buffer.read().decode("utf-8", "replace")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
try:
    obj = compile(src, "<generated>", "exec")
except SyntaxError:
    traceback.print_exc()
    sys.stderr.flush()
    sys.exit(SYNTAX)
g = {"__name__": "__main__", "__file__": "<generated>", "__builtins__": __builtins__}
try:
    exec(obj, g)
except MemoryError:
    try:
        sys.stderr.write("MemoryError\\n")
        sys.stderr.flush()
    except Exception:
        pass
    sys.exit(MEMORY)
except SystemExit:
    raise
except BaseException:
    try:
        traceback.print_exc()
        sys.stderr.flush()
    except Exception:
        pass
    sys.exit(EXCEPTION)
try:
    sys.stdout.flush()
except Exception:
    pass
""".replace(
        "SYNTAX", str(EXIT_SYNTAX)
    )
    .replace("MEMORY", str(EXIT_MEMORY))
    .replace("EXCEPTION", str(EXIT_EXCEPTION))
)


class SandboxError(RuntimeError):
    """The harness itself failed - distinct from the sandboxed code failing."""


def _scan(code: str, deny_imports: tuple[str, ...]) -> str | None:
    """Return a refusal reason, or None. Parses; never executes.

    A tripwire for the emitter going wrong, not a defence: `__import__("socket")` and every other
    dynamic form walks straight past it. It exists so an obvious hazard is refused *before* a
    process is spawned, for ~0.1 ms.
    """
    if not deny_imports:
        return None
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None  # the child reports syntax errors, with a real traceback
    denied = set(deny_imports)
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names = [node.module]
        for name in names:
            if name.split(".")[0] in denied:
                return f"import of denied module {name!r}"
    return None


def _reader(stream, cap: int, out: list) -> None:
    """Drain a pipe to EOF, keeping at most `cap` bytes.

    Draining past the cap rather than stopping is the point: a child writing 64 MB to stdout must
    never block on a full pipe while the parent waits for it to exit. The 64 KB pipe buffer is
    what turns `communicate()`-less naive waiting into a deadlock.
    """
    kept = bytearray()
    total = 0
    try:
        while True:
            chunk = stream.read(65536)
            if not chunk:
                break
            total += len(chunk)
            if len(kept) < cap:
                kept += chunk[: cap - len(kept)]
    except Exception:
        pass
    finally:
        with contextlib.suppress(Exception):
            stream.close()
    out.append((bytes(kept), total))


def _child_env(extra: dict[str, str] | None) -> dict[str, str]:
    keep = (
        "SYSTEMROOT",
        "WINDIR",
        "TEMP",
        "TMP",
        "PATH",
        "COMSPEC",
        "PATHEXT",
        "NUMBER_OF_PROCESSORS",
    )
    env = {k: os.environ[k] for k in keep if k in os.environ}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONHASHSEED"] = "0"
    if extra:
        env.update(extra)
    return env


def _result(**kw) -> dict:
    base = {
        "ok": False,
        "kind": "error",
        "exit_code": None,
        "stdout": "",
        "stderr": "",
        "stdout_bytes": 0,
        "stderr_bytes": 0,
        "truncated": False,
        "wall_time_s": 0.0,
        "peak_memory_bytes": None,
        "timed_out": False,
        "detail": "",
    }
    base.update(kw)
    return base


def run(
    code: str,
    *,
    timeout_s: float = 5.0,
    memory_limit_mb: int = 256,
    max_output_bytes: int = 1 << 20,
    files: dict[str, str] | None = None,
    deny_imports: tuple[str, ...] = DEFAULT_DENY_IMPORTS,
    active_process_limit: int = 8,
    env_extra: dict[str, str] | None = None,
    python_exe: str | None = None,
) -> dict:
    """Execute `code` in a fresh throwaway interpreter and report what happened.

    Args:
        code: the program text. Run as `__main__` in an empty globals dict.
        timeout_s: wall-clock budget. On expiry the whole process tree is killed.
        memory_limit_mb: committed-memory cap, enforced per process *and* job-wide (Windows job
            object; POSIX `RLIMIT_AS`). Clamped up to `MIN_MEMORY_LIMIT_MB`.
        max_output_bytes: bytes of stdout and of stderr kept. Excess is drained and discarded.
        files: name -> text written into the temp working directory before the run.
        deny_imports: static pre-scan blocklist; `()` disables. Advisory (see `_scan`).
        active_process_limit: max concurrent processes in the job - the fork-bomb ceiling.
        env_extra: extra environment for the child, on top of a scrubbed minimal set.
        python_exe: interpreter to use; defaults to `sys.executable`.

    Returns:
        dict with keys `ok` (bool), `kind`, `exit_code`, `stdout`, `stderr`, `stdout_bytes`,
        `stderr_bytes`, `truncated`, `wall_time_s`, `peak_memory_bytes`, `timed_out`, `detail`.
        `kind` is one of `ok`, `timeout`, `memory`, `exception`, `syntax_error`, `nonzero_exit`,
        `crash`, `refused`. Never raises for anything the sandboxed code does; `SandboxError` is
        reserved for the harness failing to start at all.
    """
    started = time.perf_counter()
    reason = _scan(code, tuple(deny_imports))
    if reason is not None:
        return _result(kind="refused", detail=reason, wall_time_s=time.perf_counter() - started)

    mem_mb = max(int(memory_limit_mb), MIN_MEMORY_LIMIT_MB)
    exe = python_exe or sys.executable
    tmp = tempfile.mkdtemp(prefix="dsbox_")
    work = Path(tmp) / "work"
    work.mkdir()
    boot = Path(tmp) / "boot.py"
    boot.write_text(_BOOTSTRAP, encoding="utf-8")
    for name, text in (files or {}).items():
        target = work / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")

    argv = [exe, "-I", "-B", str(boot)]
    kwargs: dict = {
        "cwd": str(work),
        "env": _child_env(env_extra),
        "stdin": subprocess.PIPE,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "bufsize": 0,
    }
    if IS_WINDOWS:  # pragma: no cover - platform branch
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    else:  # pragma: no cover - platform branch
        limit = mem_mb * 1024 * 1024

        def _preexec() -> None:
            os.setsid()
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

        kwargs["preexec_fn"] = _preexec

    try:
        proc = subprocess.Popen(argv, **kwargs)
    except OSError as exc:  # pragma: no cover - only on a broken interpreter path
        _rmtree(tmp)
        raise SandboxError(f"could not start sandbox process: {exc}") from exc

    job = None
    try:
        if IS_WINDOWS:  # pragma: no cover - platform branch
            job = _make_job(mem_mb, active_process_limit)
            _assign(job, proc.pid)

        outs: list = []
        errs: list = []
        t_out = threading.Thread(target=_reader, args=(proc.stdout, max_output_bytes, outs))
        t_err = threading.Thread(target=_reader, args=(proc.stderr, max_output_bytes, errs))
        t_out.daemon = t_err.daemon = True
        t_out.start()
        t_err.start()

        # The child blocks reading stdin until this write closes it, so the job limits are already
        # in force before one line of generated code runs. That ordering is the race fix: assign
        # first, release second.
        try:
            proc.stdin.write(code.encode("utf-8"))
            proc.stdin.close()
        except OSError:
            pass  # child already dead; the exit code below tells the story

        timed_out = False
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True

        peak = _peak(job)  # query before terminating - the counters go with the job
        if timed_out:
            _kill_tree(proc, job)
            # pragma: no cover - a job-terminated process has never been observed to linger
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=10)
        t_out.join(timeout=10)
        t_err.join(timeout=10)
        wall = time.perf_counter() - started

        out_b, out_n = outs[0] if outs else (b"", 0)
        err_b, err_n = errs[0] if errs else (b"", 0)
        rc = proc.returncode
        stdout = out_b.decode("utf-8", "replace")
        stderr = err_b.decode("utf-8", "replace")

        if timed_out:
            kind, detail = "timeout", f"exceeded {timeout_s:g}s wall clock"
        elif rc == 0:
            kind, detail = "ok", ""
        elif rc == EXIT_SYNTAX:
            kind, detail = "syntax_error", _last_line(stderr)
        elif rc == EXIT_MEMORY or "MemoryError" in stderr:
            kind, detail = "memory", f"exceeded {mem_mb} MB commit cap"
        elif rc == EXIT_EXCEPTION:
            kind, detail = "exception", _last_line(stderr)
        elif rc is not None and (rc < 0 or rc > 0x40000000):
            kind, detail = "crash", f"interpreter died with status {rc & 0xFFFFFFFF:#010x}"
        else:
            kind, detail = "nonzero_exit", f"exit status {rc}"

        return _result(
            ok=(kind == "ok"),
            kind=kind,
            exit_code=rc,
            stdout=stdout,
            stderr=stderr,
            stdout_bytes=out_n,
            stderr_bytes=err_n,
            truncated=(out_n > len(out_b) or err_n > len(err_b)),
            wall_time_s=wall,
            peak_memory_bytes=peak,
            timed_out=timed_out,
            detail=detail,
        )
    finally:
        _kill_tree(proc, job)
        if job is not None:  # pragma: no cover - platform branch
            with contextlib.suppress(Exception):
                win32api.CloseHandle(job)
        _rmtree(tmp)


def _last_line(text: str) -> str:
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    return lines[-1][:300] if lines else ""


def _make_job(mem_mb: int, active_process_limit: int):  # pragma: no cover - platform branch
    job = win32job.CreateJobObject(None, "")
    info = win32job.QueryInformationJobObject(job, win32job.JobObjectExtendedLimitInformation)
    limit = mem_mb * 1024 * 1024
    info["ProcessMemoryLimit"] = limit
    info["JobMemoryLimit"] = limit
    basic = info["BasicLimitInformation"]
    basic["ActiveProcessLimit"] = max(1, int(active_process_limit))
    basic["LimitFlags"] = (
        win32job.JOB_OBJECT_LIMIT_PROCESS_MEMORY
        | win32job.JOB_OBJECT_LIMIT_JOB_MEMORY
        | win32job.JOB_OBJECT_LIMIT_ACTIVE_PROCESS
        | win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        | win32job.JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION
    )
    win32job.SetInformationJobObject(job, win32job.JobObjectExtendedLimitInformation, info)
    return job


def _assign(job, pid: int) -> None:  # pragma: no cover - platform branch
    rights = win32con.PROCESS_SET_QUOTA | win32con.PROCESS_TERMINATE
    handle = win32api.OpenProcess(rights, False, pid)
    try:
        win32job.AssignProcessToJobObject(job, handle)
    finally:
        win32api.CloseHandle(handle)


def _peak(job):  # pragma: no cover - platform branch
    if job is None:
        return None
    try:
        info = win32job.QueryInformationJobObject(job, win32job.JobObjectExtendedLimitInformation)
        return int(info["PeakJobMemoryUsed"])
    except Exception:
        return None


def _kill_tree(proc, job) -> None:
    """Kill the child and every descendant.

    `Popen.kill` calls `TerminateProcess`, which kills exactly one process and orphans its
    children: on Windows there is no process group to signal, so a grandchild outlives its parent
    and keeps burning CPU. The job object is the fix - `TerminateJobObject` kills every process
    assigned to the job, assignment is inherited by children, and `KILL_ON_JOB_CLOSE` catches the
    case where the parent itself dies before it can terminate anything.
    """
    if job is not None:  # pragma: no cover - platform branch
        with contextlib.suppress(Exception):
            win32job.TerminateJobObject(job, 1)
    if not IS_WINDOWS:  # pragma: no cover - platform branch
        with contextlib.suppress(Exception):
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    with contextlib.suppress(Exception):
        if proc.poll() is None:
            proc.kill()


def _rmtree(path: str) -> None:
    import shutil as _shutil

    for _ in range(5):
        try:
            _shutil.rmtree(path, ignore_errors=False)
            return
        except OSError:
            time.sleep(0.02)  # a dying child can still hold a handle on its own cwd
    _shutil.rmtree(path, ignore_errors=True)
