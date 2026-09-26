"""Phase 16.1.4 - rate and size limits, and the three places they are honestly weak.

A limiter is easy to test into looking stronger than it is, so the weaknesses are tested *as*
weaknesses, with the reason each is accepted:

  * it is **per process**, so two workers double the budget. Asserted by giving two apps their own
    counters and watching the same client spend both,
  * it is a **fixed window**, so a client can burst twice the rate across a boundary. Asserted by
    doing exactly that, because a limit whose failure mode is untested is a limit whose failure mode
    is unknown,
  * `X-Forwarded-For` is **not trusted by default**, because a limiter keyed on a header the client
    sets has a bypass. Asserted from both sides: spoofing it does nothing until the deployment says
    how many proxies are real, and then the *rightmost* entries are the ones believed.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from app.backend.limits import (  # noqa: E402
    BUDGETS,
    EXEMPT_PATHS,
    MAX_BODY_BYTES,
    TRUSTED_PROXY_HOPS_ENV,
    Limits,
    classify,
    client_key,
    trusted_proxy_hops,
)
from app.backend.main import build_app  # noqa: E402
from app.backend.store import Store  # noqa: E402
from app.backend.upstream import Reply, Upstream  # noqa: E402

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64

GOOD = {
    "ok": True,
    "degraded": False,
    "stopped_at": None,
    "needs_confirmation": False,
    "diagram_type": "flowchart",
    "language": "python",
    "code": "print(1)\n",
    "ir": {"nodes": [], "edges": []},
    "traversal": [],
    "stages": [],
    "seconds": 0.1,
}


class FakeUpstream(Upstream):
    def __init__(self) -> None:
        super().__init__("http://model:8000")

    def predict(self, body, filename, content_type):
        return Reply(200, dict(GOOD), 0.1)

    def health(self):
        return {"reachable": True, "url": self.url, "status": 200}


@pytest.fixture()
def wired(tmp_path):
    """A client with generous read budget and a tiny predict budget, so a test is three requests."""
    limits = Limits(budgets={"predict": 2, "run": 2, "write": 2, "read": 4})
    store = Store(tmp_path)
    app = build_app(store=store, upstream=FakeUpstream(), limits=limits)
    return TestClient(app), store, limits


# == which budget a request draws on ==========================================================


@pytest.mark.parametrize(
    ("method", "path", "expected"),
    [
        ("POST", "/predict", "predict"),
        ("POST", "/predict/stream", "predict"),
        # A read of a stored prediction, not an upload. This line said "predict" until the client
        # was built and six page loads in a row answered 429 - the test had encoded the bug.
        ("GET", "/predict/abc", "read"),
        ("POST", "/run/abc", "run"),
        ("POST", "/feedback", "write"),
        ("GET", "/ir/abc", "read"),
        ("GET", "/code/abc", "read"),
        ("GET", "/feedback/abc", "read"),
    ],
)
def test_a_request_is_classified_by_what_it_costs_not_by_its_verb(method, path, expected):
    """`POST /feedback` is a write and cheap; `POST /predict` is seconds of GPU. Keying on the
    method alone would put them in one bucket."""
    assert classify(method, path) == expected


def test_reopening_a_result_does_not_spend_the_upload_budget():
    """The defect the client found. A result screen issues `GET /predict/{id}` on every open, and
    six opens in a minute is ordinary use - a person comparing two captures does it without
    noticing. Charging that to the six-per-minute GPU budget protected nothing, because the
    expensive thing had already happened."""
    limits = Limits(budgets={"predict": 6, "read": 120})
    for _ in range(20):
        assert limits.check("c", classify("GET", "/predict/abc"), now=0.0)["allowed"]


def test_the_expensive_routes_get_the_smallest_budgets():
    """A phone makes three reads per capture. Throttling those to protect the GPU limits the wrong
    thing."""
    assert BUDGETS["predict"] < BUDGETS["run"] < BUDGETS["write"] < BUDGETS["read"]


# == the counters =============================================================================


def test_a_budget_is_spent_and_then_refused():
    limits = Limits(budgets={"read": 3})
    verdicts = [limits.check("1.2.3.4", "read", now=0.0) for _ in range(4)]
    assert [v["allowed"] for v in verdicts] == [True, True, True, False]
    assert [v["remaining"] for v in verdicts] == [2, 1, 0, 0]


def test_the_window_resets():
    limits = Limits(budgets={"read": 1}, window_s=60.0)
    assert limits.check("c", "read", now=0.0)["allowed"]
    assert not limits.check("c", "read", now=30.0)["allowed"]
    assert limits.check("c", "read", now=60.0)["allowed"]


def test_two_clients_do_not_share_a_bucket():
    limits = Limits(budgets={"read": 1})
    assert limits.check("a", "read", now=0.0)["allowed"]
    assert limits.check("b", "read", now=0.0)["allowed"]


def test_two_classes_do_not_share_a_bucket():
    limits = Limits(budgets={"read": 1, "predict": 1})
    assert limits.check("a", "read", now=0.0)["allowed"]
    assert limits.check("a", "predict", now=0.0)["allowed"]


def test_expired_windows_are_pruned_so_the_dict_is_not_unbounded():
    """Without this it grows one entry per (client, class) forever - an unbounded allocation keyed
    by whatever the internet sends."""
    limits = Limits(window_s=60.0)
    for i in range(50):
        limits.check(f"10.0.0.{i}", "read", now=0.0)
    assert limits.tracked == 50
    assert limits.prune(now=61.0) == 50
    assert limits.tracked == 0


def test_pruning_does_not_drop_a_live_window():
    limits = Limits(window_s=60.0)
    limits.check("old", "read", now=0.0)
    limits.check("new", "read", now=59.0)
    limits.prune(now=61.0)
    assert limits.tracked == 1
    assert not limits.check("new", "read", now=61.0)["allowed"] or limits.tracked == 1


# == the accepted weaknesses, tested as such ==================================================


def test_the_fixed_window_permits_twice_the_rate_across_a_boundary():
    """The classic burst, and it is accepted rather than unnoticed.

    A sliding window costs a deque per client and a token bucket costs two floats; either would be
    defensible. The fixed window is kept because the failure it permits is twice the intended rate
    *for one second*, and the budgets are low enough that twice them is still servable.
    """
    limits = Limits(budgets={"predict": 6}, window_s=60.0)
    # The window opens on the first request, so it has to be opened at t=0 for 59.9 to be near its
    # end. Spending it all at 59.9 and again at 60.1 is twelve requests in 0.2 s.
    assert limits.check("c", "predict", now=0.0)["allowed"]
    allowed = 1 + sum(limits.check("c", "predict", now=59.9)["allowed"] for _ in range(5))
    allowed += sum(limits.check("c", "predict", now=60.1)["allowed"] for _ in range(6))
    assert allowed == 12, "twelve in 0.2 s against a budget of six per minute"


def test_the_limit_is_per_process_and_two_processes_double_it(tmp_path):
    """The counters are a dict in memory. `X-RateLimit-Scope` says `process` for this reason."""
    one = Limits(budgets={"read": 1})
    two = Limits(budgets={"read": 1})
    assert one.check("c", "read", now=0.0)["allowed"]
    assert not one.check("c", "read", now=0.0)["allowed"]
    assert two.check("c", "read", now=0.0)["allowed"], "a second worker is a second budget"


def test_the_scope_header_does_not_claim_more_than_the_limiter_holds(wired):
    client, _, _ = wired
    response = client.get("/ir/deadbeefdeadbeef")
    assert response.headers["x-ratelimit-scope"].startswith("process:")


# == the key ==================================================================================


def test_the_key_is_the_socket_peer_by_default(monkeypatch):
    monkeypatch.delenv(TRUSTED_PROXY_HOPS_ENV, raising=False)
    scope = {
        "client": ("203.0.113.9", 4001),
        "headers": [(b"x-forwarded-for", b"1.1.1.1, 2.2.2.2")],
    }
    assert client_key(scope) == "203.0.113.9"


def test_spoofing_x_forwarded_for_does_not_move_a_client_to_a_fresh_bucket(wired, monkeypatch):
    """A limiter keyed on a header the client sets is a limiter with a bypass: one bucket per
    request, for the cost of a random string."""
    monkeypatch.delenv(TRUSTED_PROXY_HOPS_ENV, raising=False)
    client, _, limits = wired
    statuses = [
        client.post(
            "/predict",
            files={"image": ("p.png", PNG, "image/png")},
            headers={"X-Forwarded-For": f"9.9.9.{i}"},
        ).status_code
        for i in range(4)
    ]
    assert statuses.count(429) >= 1, "every request bought itself a new bucket"


def test_a_declared_proxy_hop_believes_what_the_proxy_appended_not_what_the_client_wrote(
    monkeypatch,
):
    """The arithmetic, spelled out, because getting it backwards is the usual mistake.

    Client C reaches us through one proxy P. P appends the address it received from - `nginx`'s
    `$proxy_add_x_forwarded_for` appends `$remote_addr` - so with one trusted hop the **last** entry
    is C, and anything to the left of it is whatever C chose to send. Our socket peer is P.

    Here C sent `X-Forwarded-For: evil-spoof` and P appended `198.51.100.7`, C's real address. One
    hop, so `len(chain) - 1` picks the appended entry and the spoof is ignored.
    """
    monkeypatch.setenv(TRUSTED_PROXY_HOPS_ENV, "1")
    scope = {
        "client": ("10.0.0.1", 1),  # the proxy
        "headers": [(b"x-forwarded-for", b"evil-spoof, 198.51.100.7")],
    }
    assert client_key(scope) == "198.51.100.7"


def test_two_declared_hops_reach_past_both_of_them(monkeypatch):
    """C -> P1 -> P2 -> us. P1 sets `C`, P2 appends P1, so with two hops the client is `chain[0]`."""
    monkeypatch.setenv(TRUSTED_PROXY_HOPS_ENV, "2")
    scope = {
        "client": ("10.0.0.2", 1),
        "headers": [(b"x-forwarded-for", b"198.51.100.7, 10.0.0.1")],
    }
    assert client_key(scope) == "198.51.100.7"


def test_a_chain_shorter_than_declared_falls_back_rather_than_indexing_off_the_end(monkeypatch):
    monkeypatch.setenv(TRUSTED_PROXY_HOPS_ENV, "5")
    scope = {"client": ("10.0.0.1", 1), "headers": [(b"x-forwarded-for", b"198.51.100.7")]}
    assert client_key(scope) == "198.51.100.7"


def test_hops_default_to_zero_because_the_safe_failure_is_a_shared_bucket(monkeypatch):
    """Not an unlimited one. A misconfiguration must make the limiter stricter, never absent."""
    monkeypatch.delenv(TRUSTED_PROXY_HOPS_ENV, raising=False)
    assert trusted_proxy_hops() == 0
    monkeypatch.setenv(TRUSTED_PROXY_HOPS_ENV, "not a number")
    assert trusted_proxy_hops() == 0
    monkeypatch.setenv(TRUSTED_PROXY_HOPS_ENV, "-3")
    assert trusted_proxy_hops() == 0


def test_a_client_with_no_socket_still_gets_a_key():
    assert client_key({"client": None, "headers": []}) == "unknown"


# == the middleware ===========================================================================


def test_the_limit_is_enforced_over_http_with_a_retry_after(wired):
    client, _, _ = wired
    last = None
    for _ in range(4):
        last = client.post("/predict", files={"image": ("p.png", PNG, "image/png")})
    assert last.status_code == 429
    assert int(last.headers["retry-after"]) >= 1
    assert "per 60s" in last.json()["detail"]


def test_every_response_carries_the_budget_not_only_the_refusal(wired):
    """A client told its budget can pace itself instead of discovering the limit by hitting it."""
    client, _, _ = wired
    response = client.post("/predict", files={"image": ("p.png", PNG, "image/png")})
    assert response.status_code == 200
    assert response.headers["x-ratelimit-limit"] == "2"
    assert response.headers["x-ratelimit-remaining"] == "1"
    assert float(response.headers["x-ratelimit-reset"]) > 0


def test_spending_the_predict_budget_does_not_stop_a_client_reading(wired):
    """Three reads per capture is the client's normal behaviour; they must not share the GPU's
    budget."""
    client, store, _ = wired
    rid = store.put(dict(GOOD))["id"]
    for _ in range(3):
        client.post("/predict", files={"image": ("p.png", PNG, "image/png")})
    assert client.get(f"/ir/{rid}").status_code == 200


@pytest.mark.parametrize("path", EXEMPT_PATHS)
def test_an_exempt_path_is_never_rate_limited(wired, path):
    """A 429 to a liveness probe gets the container restarted. 15.11's lesson, third outing."""
    client, _, limits = wired
    statuses = {client.get(path).status_code for _ in range(limits.budgets["read"] + 5)}
    assert 429 not in statuses


def test_health_reports_the_limiter_rather_than_hiding_it(wired):
    client, _, limits = wired
    body = client.get("/health").json()["limits"]
    assert body["budgets"] == limits.budgets
    assert body["scope"] == "process"
    assert body["max_body_bytes"] == limits.max_body_bytes


# == size =====================================================================================


def test_a_content_length_over_the_cap_is_refused_before_the_body_is_read(wired):
    """25 MB is not sent here. The point is that declaring it is enough to be refused."""
    client, _, _ = wired
    response = client.post(
        "/predict",
        content=b"x",
        headers={"Content-Length": str(MAX_BODY_BYTES + 1), "Content-Type": "image/png"},
    )
    assert response.status_code == 413
    assert "photograph" in response.json()["detail"]


def test_a_malformed_content_length_is_a_400_not_a_traceback(wired):
    client, _, _ = wired
    response = client.request(
        "POST",
        "/predict",
        content=b"x",
        headers={"Content-Length": "banana", "Content-Type": "image/png"},
    )
    assert response.status_code in (400, 422)


def test_the_body_cap_matches_the_routes_own_so_neither_accepts_what_the_other_refuses():
    from app.backend.main import MAX_UPLOAD_BYTES

    assert MAX_BODY_BYTES == MAX_UPLOAD_BYTES


def test_the_route_still_refuses_on_the_overflow_byte_because_a_client_can_lie(wired):
    """The `Content-Length` check is an optimisation. `_accept` is the check that holds."""
    client, _, _ = wired
    body = b"\x89PNG\r\n\x1a\n" + b"\x00" * MAX_BODY_BYTES
    # Sent without a declared length the middleware would catch, by chunking it.
    response = client.post(
        "/predict", files={"image": ("p.png", body, "image/png")}, headers={"Expect": ""}
    )
    assert response.status_code in (413,), response.status_code
