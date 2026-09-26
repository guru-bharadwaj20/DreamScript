"""Phase 16.1.3 - the sandbox runner, and each tightening against its counter-example.

The sandbox itself is 11.2.9's and is tested there, exhaustively: the memory cap against
`PeakJobMemoryUsed`, the tree kill against a control that leaks a grandchild. What is tested here is
what *this* row added on top of it, and the first test in the file is the one that matters most.

**`test_a_trivial_program_actually_runs` exists because every other assertion in this file once
passed while nothing executed.** The first version set `active_process_limit=1`, reasoning that a
program derived from a drawing has no business spawning anything. True of the program, false of the
interpreter: `sys.executable` here is a uv trampoline that execs the real CPython, and inside a job
limited to one process that spawn is refused. Every run came back `exec.nonzero_exit` in 20 ms -
which is exactly what a fast, genuinely failing program looks like - and the reason was only in
`stderr`: `uv trampoline failed to spawn Python child process / os error 1816`
(`ERROR_NOT_ENOUGH_QUOTA`). A sandbox that cannot start, reported as the program failing.

So: prove something ran, then prove the limits bite.
"""

from __future__ import annotations

import threading
import time

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from app.backend import runner  # noqa: E402
from app.backend.main import build_app  # noqa: E402
from app.backend.runner import (  # noqa: E402
    ACTIVE_PROCESS_LIMIT,
    DEFAULT_TIMEOUT_S,
    MAX_OUTPUT_BYTES,
    MAX_TIMEOUT_S,
    MEMORY_LIMIT_MB,
    MIN_TIMEOUT_S,
    clamp_timeout,
    deny_imports,
    run_stored,
)
from app.backend.store import Store  # noqa: E402
from src.codegen.targets import for_type  # noqa: E402

pytestmark = pytest.mark.skipif(
    __import__("sys").platform != "win32",
    reason="Windows-only: 11.2.9's sandbox enforces its caps with a Win32 job object",
)


def record(code, language="python", diagram_type="flowchart", **result):
    """A prediction record shaped as `/predict` stores one."""
    return {
        "id": "0" * 16,
        "result": {
            "code": code,
            "language": language,
            "diagram_type": diagram_type,
            "ok": True,
            **result,
        },
    }


FLOWCHART = {
    "nodes": [
        {"id": "a", "text": "Start", "type": "terminator"},
        {"id": "b", "text": "Validate order", "type": "decision"},
        {"id": "c", "text": "Ship it", "type": "process"},
        {"id": "d", "text": "End", "type": "terminator"},
    ],
    "edges": [
        {"source": "a", "target": "b"},
        {"source": "b", "target": "c", "text": "yes"},
        {"source": "c", "target": "d"},
    ],
}


@pytest.fixture(scope="module")
def emitted() -> str:
    """Real emitter output, not a hand-written program. What the runner actually receives."""
    return for_type("flowchart")(FLOWCHART, ["a", "b", "c", "d"])


# == first: something ran =====================================================================


def test_a_trivial_program_actually_runs():
    """The regression test for the `active_process_limit=1` failure. Read this file's docstring.

    `ok` is asserted *and* `stderr` is asserted empty, because the whole point of the original defect
    was that a plausible-looking failure verdict hid a sandbox that never started.
    """
    verdict = run_stored(record("def go(ctx):\n    print('hello')\n    return ctx\n"))
    assert verdict["ok"], verdict
    assert verdict["kind"] == "exec.ok"
    assert verdict["stderr"] == "", f"the sandbox itself failed: {verdict['stderr']!r}"
    assert "hello" in verdict["stdout"]


def test_the_process_limit_is_the_floor_the_interpreter_needs_and_not_below_it():
    assert ACTIVE_PROCESS_LIMIT == 2, "1 refuses the uv trampoline's own spawn; see the constant"


def test_real_emitter_output_runs(emitted):
    """The program under test is the one the pipeline produces, not one written for the test."""
    verdict = run_stored(record(emitted))
    assert verdict["ok"], verdict
    assert verdict["peak_memory_bytes"] and verdict["peak_memory_bytes"] > 0


def test_the_harness_marker_is_not_shown_to_the_user(emitted):
    """`__ran__` is 12.3.2's receipt that the entry point returned. It is our bookkeeping."""
    assert "__ran__" not in run_stored(record(emitted))["stdout"]


# == the limits bite ==========================================================================


def test_a_runaway_loop_is_a_timeout_and_not_a_hung_request():
    """The failure 11.2.9 was built for: a trace the tracer made cyclic becomes `while True`."""
    started = time.perf_counter()
    verdict = run_stored(record("def go(ctx):\n    while True:\n        pass\n"), timeout_s=2)
    elapsed = time.perf_counter() - started
    assert verdict["kind"] == "exec.timeout"
    assert verdict["ok"] is False
    assert elapsed < 6, f"the deadline was 2 s and the call took {elapsed:.2f} s"


def test_an_unbounded_allocation_is_a_memory_kill_at_the_declared_cap():
    code = "def go(ctx):\n    x = []\n    while True:\n        x.append(bytearray(1 << 20))\n"
    verdict = run_stored(record(code), timeout_s=10)
    assert verdict["kind"] == "exec.memory"
    # The kernel's own accounting granularity puts the peak just over the limit, which 11.2.9
    # measured at 0.3%. Asserted as a band rather than a number.
    peak_mb = (verdict["peak_memory_bytes"] or 0) / (1024 * 1024)
    assert MEMORY_LIMIT_MB <= peak_mb < MEMORY_LIMIT_MB * 1.2, peak_mb


@pytest.mark.parametrize("module", ["os", "socket", "subprocess", "shutil", "urllib", "pathlib"])
def test_a_denied_import_is_refused_before_any_process_is_spawned(module):
    """The pre-scan is advisory and it is the outer layer, not the only one - but it is cheap and
    it is where `subprocess` stops, which is the thing a resource sandbox does not stop."""
    started = time.perf_counter()
    verdict = run_stored(record(f"import {module}\n\ndef go(ctx):\n    return 1\n"))
    assert verdict["kind"] == "exec.refused"
    assert module in verdict["detail"]
    assert (time.perf_counter() - started) < 1.0, "a refusal should not cost a process"


def test_the_deny_list_extends_the_sandboxs_own_rather_than_replacing_it():
    from src.rl.sandbox import DEFAULT_DENY_IMPORTS

    denied = deny_imports()
    assert set(DEFAULT_DENY_IMPORTS) <= set(denied)
    assert "os" in denied and "pathlib" in denied
    assert denied == tuple(sorted(set(denied))), "reportable, so sorted and deduplicated"


@pytest.mark.parametrize(
    ("asked", "expected"),
    [
        (None, DEFAULT_TIMEOUT_S),
        (0.01, MIN_TIMEOUT_S),
        (900, MAX_TIMEOUT_S),
        ("nonsense", DEFAULT_TIMEOUT_S),
    ],
)
def test_a_caller_may_ask_for_a_budget_within_reason(asked, expected):
    assert clamp_timeout(asked) == expected


def test_output_is_capped_for_a_phone_rather_than_for_an_evaluator():
    code = "def go(ctx):\n    print('x' * 200000)\n    return ctx\n"
    verdict = run_stored(record(code))
    assert len(verdict["stdout"]) <= MAX_OUTPUT_BYTES


def test_the_declared_sandbox_settings_are_the_ones_used(emitted):
    """Reported back to the client, so what it was run under is not a claim in a docstring."""
    verdict = run_stored(record(emitted))
    assert verdict["sandbox"] == {
        "memory_limit_mb": MEMORY_LIMIT_MB,
        "active_process_limit": ACTIVE_PROCESS_LIMIT,
        "denied_imports": len(deny_imports()),
    }


# == the shape of the answer ==================================================================


def test_a_page_that_produced_no_code_says_where_it_stopped():
    verdict = run_stored({"id": "0" * 16, "result": {"code": None, "stopped_at": "assemble"}})
    assert verdict["kind"] == "run.no_code"
    assert "assemble" in verdict["detail"]


def test_a_program_with_no_entry_point_is_its_own_verdict():
    """Not a crash and not a pass: there is nothing to call."""
    assert run_stored(record("x = 1\n"))["kind"] == "run.no_entry"


def test_an_unlabelled_language_is_refused_rather_than_guessed():
    assert run_stored(record("x", language=""))["kind"] == "run.unsupported"


def test_sql_returns_the_schema_it_built_and_not_a_bare_ok():
    """What a person wants back from an ER diagram is the tables."""
    ddl = "CREATE TABLE customer (id INTEGER PRIMARY KEY, name TEXT);"
    verdict = run_stored(record(ddl, language="sql", diagram_type="er"))
    assert verdict["ok"] is True
    assert "customer" in verdict["stdout"]


def test_broken_sql_is_a_failure_with_sqlites_own_reason():
    verdict = run_stored(record("CREATE TABLE (;", language="sql", diagram_type="er"))
    assert verdict["ok"] is False
    assert verdict["detail"]


@pytest.mark.parametrize("language", ["react", "spice"])
def test_a_toolchain_that_is_absent_is_reported_as_absent_not_as_a_failure(language, monkeypatch):
    """A wireframe whose code was never run must not look like one whose code ran and failed."""
    from src.eval import react, spice

    module = react if language == "react" else spice
    monkeypatch.setattr(
        module,
        "check_many",
        lambda codes, **kw: [
            {"ok": False, "kind": f"{language}.unavailable", "detail": "toolchain missing"}
        ],
    )
    verdict = run_stored(record("x", language=language, diagram_type=language))
    assert verdict["kind"].endswith(".unavailable")
    assert verdict["available"] is False


def test_every_verdict_carries_the_same_head_fields(emitted):
    """A client renders one panel for every language, so the envelope cannot vary by language."""
    cases = [
        record(emitted),
        record("CREATE TABLE t (id INTEGER PRIMARY KEY);", language="sql", diagram_type="er"),
        record("x = 1\n"),
        record(None),
    ]
    for case in cases:
        verdict = run_stored(case)
        assert {
            "id",
            "language",
            "timeout_s",
            "ok",
            "kind",
            "detail",
            "stdout",
            "stderr",
            "seconds",
        } <= set(verdict), verdict["kind"]


# == the concurrency gate =====================================================================


def test_the_fifth_simultaneous_run_is_refused_and_the_gate_releases(monkeypatch):
    """A semaphore rather than a queue: a queue holds a phone's request open for as long as the
    queue is deep, which is a timeout wearing a progress bar."""
    holding = record("def go(ctx):\n    import time\n    time.sleep(2)\n    return ctx\n")
    threads = [
        threading.Thread(target=lambda: run_stored(holding, timeout_s=5))
        for _ in range(runner.MAX_CONCURRENT_RUNS)
    ]
    for thread in threads:
        thread.start()
    try:
        # Long enough for every holder to have taken its permit, short of the 2 s sleep.
        time.sleep(1.0)
        refused = run_stored(record("def go(ctx):\n    return ctx\n"))
        assert refused["kind"] == "run.busy"
        assert str(runner.MAX_CONCURRENT_RUNS) in refused["detail"]
    finally:
        for thread in threads:
            thread.join()
    assert run_stored(record("def go(ctx):\n    return ctx\n"))["ok"], "the gate did not release"


# == the route ================================================================================


@pytest.fixture()
def client(tmp_path):
    from app.backend.upstream import Upstream

    store = Store(tmp_path)
    return TestClient(build_app(store=store, upstream=Upstream("http://model:8000"))), store


def test_the_route_runs_the_stored_code(client, emitted):
    api, store = client
    rid = store.put({"code": emitted, "language": "python", "diagram_type": "flowchart"})["id"]
    body = api.post(f"/run/{rid}").json()
    assert body["ok"] is True
    assert body["id"] == rid


def test_the_route_takes_an_id_and_nothing_a_caller_could_put_a_program_in(client):
    """The single most important line of this row: there is no `code` parameter.

    A body offering one is ignored - the runner reads the store - and the proof is that a request
    carrying a program the store does not have still runs the stored code. This is what separates a
    sandbox runner from arbitrary-code-execution as a service.
    """
    api, store = client
    rid = store.put({"code": "def go(ctx):\n    print('stored')\n", "language": "python"})["id"]
    body = api.post(f"/run/{rid}", json={"code": "print('injected')"}).json()
    assert "stored" in body["stdout"]
    assert "injected" not in body["stdout"]

    schema = api.get("/openapi.json").json()
    parameters = schema["paths"]["/run/{record_id}"]["post"].get("parameters", [])
    names = {p["name"] for p in parameters}
    assert names <= {
        "record_id",
        "timeout_s",
    }, f"the route accepts {names - {'record_id', 'timeout_s'}}"
    assert "requestBody" not in schema["paths"]["/run/{record_id}"]["post"]


def test_the_route_clamps_the_budget_at_the_schema(client, emitted):
    api, store = client
    rid = store.put({"code": emitted, "language": "python", "diagram_type": "flowchart"})["id"]
    assert api.post(f"/run/{rid}?timeout_s=600").status_code == 422
    assert api.post(f"/run/{rid}?timeout_s=0.001").status_code == 422
    assert api.post(f"/run/{rid}?timeout_s={MAX_TIMEOUT_S}").status_code == 200


def test_running_an_unknown_id_is_404(client):
    api, _ = client
    assert api.post("/run/deadbeefdeadbeef").status_code == 404
    assert api.post("/run/../../secrets").status_code in (404, 422)


def test_a_saturated_service_is_503_with_a_retry_after_not_a_200_carrying_a_failure(
    client, monkeypatch
):
    """Nothing is wrong with the page or the program; the client should retry the same request."""
    api, store = client
    rid = store.put({"code": "def go(ctx):\n    return ctx\n", "language": "python"})["id"]
    monkeypatch.setattr(
        "app.backend.main.run_stored",
        lambda record, timeout_s=None: {
            "kind": "run.busy",
            "detail": "4 programs are already running; try again",
        },
    )
    response = api.post(f"/run/{rid}")
    assert response.status_code == 503
    assert response.headers["retry-after"] == "5"
