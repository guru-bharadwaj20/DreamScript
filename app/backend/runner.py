"""Phase 16.1.3 - running the generated code, and the honest limits of doing so.

    from app.backend.runner import run_stored
    verdict = run_stored(record, timeout_s=5.0)
    verdict["kind"], verdict["stdout"]

This is the payoff of the whole project on a phone: photograph a diagram, get code, **press play**.
It reuses 11.2.9's sandbox for Python and 12.3.2's per-language checkers for the rest, and it adds no
second implementation of any of them.

## What "hardened" means here, and what it does not

11.2.9 says it plainly about itself: **it is a resource sandbox, not a security boundary.** A job
object gives a hard memory cap, a whole-tree kill and a process ceiling; it does not stop a program
reading a file the service user can read. Repeating that here matters, because this row moves the
sandbox from an offline evaluator to a socket, and the honest question changes from "can a runaway
loop wedge the trainer" to "whose program is this".

**Whose program is it? Ours - and that was checked rather than assumed.** The emitter is 12.1.6's
templates, and a diagram label is the only thing a stranger controls. Every one of the five targets
was probed with labels engineered to break out (`x'); import os; os.system('calc`, an embedded
newline plus `import os`, `__import__('os').system(...)`, a U+2028 line separator):

    flowchart, state_machine -> python   labels are slugified to identifiers. The worst case is a
                                        function called `import___os_system_calc`, which is a name,
                                        not a statement
    er -> sql                            identifiers likewise; the payloads did not survive at all
    wireframe -> react                   labels land inside JSX as JSON-escaped string literals,
                                        so the text appears in the output and none of it is code
    circuit -> spice                     labels are not emitted into the deck

So on the emitter rung the sandbox is guarding exactly what 11.2.9 built it to guard: **our own
output going wrong** - a `while True` from a trace the tracer made cyclic, an unbounded range.

**The rung above is different, and it is the reason for every tightening below.** 13.4's chain is
`model -> emitter`, and when `DREAMSCRIPT_MODEL_URL` is set the program is written by 12.2's
fine-tuned adapter from an IR the photograph produced. That is a language model's output, not a
template's, and a label is prompt content. Nothing in the sandbox distinguishes the two, so this
module assumes the worse rung is the one that answered.

## The tightenings, each with its reason

    only stored code        A program is never accepted in a request body. The runner takes an id
                            and executes what the pipeline generated for it. This is the whole
                            difference between a sandbox runner and arbitrary-code-execution as a
                            service, and it is one `if` that cannot be argued away later
    active_process_limit=2  12.3.2 uses 8 because it batches. 2 is the floor and not 1: on this
                            machine `sys.executable` is a uv trampoline that execs the real CPython,
                            so a limit of 1 refuses the *interpreter's* spawn - see the constant,
                            which records what that looked like
    memory_limit_mb=128     Twice the 64 MB floor the interpreter itself needs, and small
    deny_imports extended   The templates import `collections`. Everything a generated program could
                            plausibly want is arithmetic, and the whole filesystem, network and
                            process surface is refused by name. Advisory, as 11.2.9 says - a
                            pre-scan, not a loader hook - so it is the outer of several layers
    timeout clamped         1 to 15 s. A caller cannot ask for a minute
    output capped           64 KiB each way, not 11.2.9's 1 MiB default: this is going to a phone
    one run at a time       A semaphore. Ten phones pressing play must not become ten interpreters,
                            and 16.1.4's rate limit is about requests where this is about processes

## What is unavailable is said, never folded into a pass

`react` needs node and esbuild, `spice` needs ngspice, and a hobby host has neither. 12.3.2's
checkers already answer `react.unavailable` / `spice.unavailable` rather than failing, and that
distinction is carried through to the client verbatim: a wireframe whose code was never run must not
look like a wireframe whose code ran and failed.
"""

from __future__ import annotations

import threading
import time
from typing import Any

#: Wall-clock budget for one run, and the range a caller may ask for.
DEFAULT_TIMEOUT_S = 5.0
MIN_TIMEOUT_S = 1.0
MAX_TIMEOUT_S = 15.0

#: Committed-memory cap. 11.2.9's floor is 64 MB (the interpreter alone commits ~13 MB), so this is
#: twice the floor and a long way below anything a diagram's program should need.
MEMORY_LIMIT_MB = 128

#: 64 KiB each of stdout and stderr, against 11.2.9's 1 MiB default. The evaluator can afford a
#: megabyte per program; a phone on a mobile link is a different consumer with a different budget.
MAX_OUTPUT_BYTES = 64 * 1024

#: How many programs may be executing at once, across every caller. A semaphore rather than a queue
#: depth: the honest answer to the fifth simultaneous run is "busy", not an unbounded wait.
MAX_CONCURRENT_RUNS = 4

#: Processes the job object permits, and **2 rather than 1, because 1 does not work**.
#:
#: This was written as 1 with the comment "a diagram's program has no business spawning anything at
#: all", which is true about the program and false about the interpreter. On this machine
#: `sys.executable` is `.venv/Scripts/python.exe`, a **uv trampoline** that execs the real CPython as
#: a second process, and inside a job with `ActiveProcessLimit = 1` that spawn is refused: every run
#: came back `exec.nonzero_exit` in 20 ms with `uv trampoline failed to spawn Python child process /
#: uncategorized error (os error 1816)` - `ERROR_NOT_ENOUGH_QUOTA`. Not a sandbox catching anything;
#: a sandbox that could not start, reported as the program failing.
#:
#: It is worth knowing that this was invisible in the verdict: `nonzero_exit` is a perfectly
#: plausible thing for a generated program to do, and the run took 20 ms, which looks like a fast
#: program rather than one that never began. Only reading `stderr` showed it. `test_a_trivial_program
#: _actually_runs` is the regression test, and it exists because every other assertion in this file
#: passed while nothing executed.
#:
#: 2 is still tighter than 12.3.2's 8 and is the floor: both slots are the interpreter's own, so a
#: generated program that reaches `subprocess` finds nothing left to spawn into.
ACTIVE_PROCESS_LIMIT = 2

#: Added to 11.2.9's `DEFAULT_DENY_IMPORTS`. The generated templates import `collections` and
#: nothing else, so this is not a list of what is dangerous - it is the filesystem, the network, the
#: process table and the import machinery, refused by name because none of them has any business in
#: a program derived from a drawing.
EXTRA_DENY_IMPORTS: tuple[str, ...] = (
    "asyncio",
    "builtins",
    "cgi",
    "code",
    "codeop",
    "compileall",
    "ftplib",
    "glob",
    "imaplib",
    "importlib",
    "mmap",
    "nntplib",
    "os",
    "pathlib",
    "pdb",
    "pickle",
    "pickletools",
    "platform",
    "poplib",
    "pty",
    "runpy",
    "select",
    "selectors",
    "signal",
    "site",
    "smtplib",
    "sqlite3",
    "ssl",
    "sysconfig",
    "telnetlib",
    "tempfile",
    "threading",
    "xmlrpc",
    "zipimport",
)

#: The verdict kinds this module answers with, beyond the sandbox's own `exec.<kind>`.
KIND_NO_CODE = "run.no_code"
KIND_UNSUPPORTED = "run.unsupported"
KIND_NO_ENTRY = "run.no_entry"
KIND_BUSY = "run.busy"

_GATE = threading.BoundedSemaphore(MAX_CONCURRENT_RUNS)


def deny_imports() -> tuple[str, ...]:
    """11.2.9's default blocklist, plus this row's. Sorted and deduplicated so the list a run was
    refused by is reportable rather than incidental."""
    from src.rl.sandbox import DEFAULT_DENY_IMPORTS

    return tuple(sorted(set(DEFAULT_DENY_IMPORTS) | set(EXTRA_DENY_IMPORTS)))


def clamp_timeout(asked: float | None) -> float:
    """A caller may ask, within reason. A minute is not within reason."""
    if asked is None:
        return DEFAULT_TIMEOUT_S
    try:
        value = float(asked)
    except (TypeError, ValueError):
        return DEFAULT_TIMEOUT_S
    return max(MIN_TIMEOUT_S, min(MAX_TIMEOUT_S, value))


def run_stored(record: dict[str, Any], *, timeout_s: float | None = None) -> dict[str, Any]:
    """Execute the code the pipeline generated for one stored prediction.

    Takes the whole record rather than a code string, and that signature is the hardening: there is
    no parameter here through which a caller could supply a program. The code, the language and the
    diagram type all come from what the pipeline produced.
    """
    result = record.get("result") or {}
    code = result.get("code")
    language = result.get("language") or ""
    budget = clamp_timeout(timeout_s)
    head: dict[str, Any] = {
        "id": record.get("id"),
        "language": language or None,
        "timeout_s": budget,
    }

    if not isinstance(code, str) or not code.strip():
        return {
            **head,
            "ok": False,
            "kind": KIND_NO_CODE,
            "detail": "this page produced no code to run"
            + (f"; it stopped at {result['stopped_at']}" if result.get("stopped_at") else ""),
            "stdout": "",
            "stderr": "",
            "seconds": 0.0,
        }

    if not _GATE.acquire(blocking=False):
        # The honest answer to the eleventh simultaneous run. A queue here would hold a phone's
        # request open for as long as the queue is deep, which is a timeout wearing a progress bar.
        return {
            **head,
            "ok": False,
            "kind": KIND_BUSY,
            "detail": f"{MAX_CONCURRENT_RUNS} programs are already running; try again",
            "stdout": "",
            "stderr": "",
            "seconds": 0.0,
        }
    try:
        started = time.perf_counter()
        if language == "python":
            verdict = _python(code, str(result.get("diagram_type") or ""), budget)
        elif language in ("sql", "react", "spice"):
            verdict = _delegated(code, language)
        else:
            verdict = {
                "ok": False,
                "kind": KIND_UNSUPPORTED,
                "detail": f"nothing here runs {language or 'an unlabelled language'}",
                "stdout": "",
                "stderr": "",
            }
        verdict["seconds"] = round(time.perf_counter() - started, 3)
        return {**head, **verdict}
    finally:
        _GATE.release()


def _python(code: str, diagram_type: str, timeout_s: float) -> dict[str, Any]:
    """One Python program, in 11.2.9's sandbox, with 12.3.2's harness around it.

    The harness is reused rather than rewritten, and it is doing real work: a generated flowchart
    calls functions the diagram only *names* (`validate_order(ctx)`), so every free name is stubbed -
    one called in an `if`/`while` test returns True twice and then False, so a correct loop
    terminates and a `while True:` with no exit times out and is reported as a timeout instead of
    hanging the request.
    """
    from src.eval.codecheck import python_program
    from src.rl import sandbox

    program = python_program(code, diagram_type)
    if program is None:
        return {
            "ok": False,
            "kind": KIND_NO_ENTRY,
            "detail": "the generated program has no parseable entry point to call",
            "stdout": "",
            "stderr": "",
        }

    outcome = sandbox.run(
        program,
        timeout_s=timeout_s,
        memory_limit_mb=MEMORY_LIMIT_MB,
        max_output_bytes=MAX_OUTPUT_BYTES,
        deny_imports=deny_imports(),
        # Two: the uv trampoline and the interpreter it execs. See ACTIVE_PROCESS_LIMIT - 1 was
        # tried, and it refused the interpreter's own spawn rather than the program's.
        active_process_limit=ACTIVE_PROCESS_LIMIT,
    )
    # `__ran__` is 12.3.2's marker that the entry point returned rather than merely imported. The
    # harness prints it last, so it is stripped from what the user is shown - it is our bookkeeping,
    # not their program's output.
    stdout = outcome.get("stdout") or ""
    ran = bool(outcome["ok"]) and "__ran__" in stdout
    return {
        "ok": ran,
        "kind": "exec.ok" if ran else f"exec.{outcome['kind']}",
        "detail": (outcome.get("detail") or "")[:500],
        "stdout": _strip_marker(stdout),
        "stderr": (outcome.get("stderr") or "")[:MAX_OUTPUT_BYTES],
        "truncated": bool(outcome.get("truncated")),
        "peak_memory_bytes": outcome.get("peak_memory_bytes"),
        "sandbox": {
            "memory_limit_mb": MEMORY_LIMIT_MB,
            "active_process_limit": ACTIVE_PROCESS_LIMIT,
            "denied_imports": len(deny_imports()),
        },
    }


def _delegated(code: str, language: str) -> dict[str, Any]:
    """SQL, React and SPICE, through 12.3.2's own checkers.

    Not reimplemented here, and not routed through `executable_many` either: that function is built
    to score a batch and it discards stdout, which is the one thing a person pressing play wants to
    see. The per-language modules underneath it are called directly.

    `react` and `spice` answer `*.unavailable` on a host with no node or no ngspice, and that answer
    is passed through rather than folded into a failure. A wireframe whose code was never run must
    not look like one whose code ran and failed.
    """
    from src.eval import react, spice, sql

    if language == "sql":
        verdict = sql.check(code)
        tables = verdict.get("tables") or []
        return {
            "ok": bool(verdict["ok"]),
            "kind": verdict["kind"],
            "detail": str(verdict.get("detail") or "")[:500],
            # The schema SQLite actually built, which is this language's equivalent of stdout: what
            # a person wants back from an ER diagram is the tables, not a bare "ok".
            "stdout": "\n".join(
                str(name) for name in (tables if isinstance(tables, list) else sorted(tables))
            ),
            "stderr": "",
            "relationships": verdict.get("relationships"),
        }

    checker = react.check_many if language == "react" else spice.check_many
    verdict = checker([code])[0]
    return {
        "ok": bool(verdict.get("ok")),
        "kind": str(verdict.get("kind") or f"{language}.unknown"),
        "detail": str(verdict.get("detail") or "")[:500],
        "stdout": str(verdict.get("stdout") or "")[:MAX_OUTPUT_BYTES],
        "stderr": str(verdict.get("stderr") or "")[:MAX_OUTPUT_BYTES],
        "available": not str(verdict.get("kind", "")).endswith(".unavailable"),
    }


def _strip_marker(stdout: str) -> str:
    """Drop 12.3.2's `__ran__` line. It is the harness's receipt, not the program's output."""
    lines = [line for line in stdout.splitlines() if line.strip() != "__ran__"]
    text = "\n".join(lines)
    return text[:MAX_OUTPUT_BYTES]
