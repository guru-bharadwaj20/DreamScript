"""Phase 15.11 - the inference service.

The model itself is exercised by Phase 13's tests. What matters at this boundary is the contract:
that a caller can tell a trustworthy answer from a degraded one, that a liveness probe does not
drag 1.3 GB of weights in behind it, and that bad input is refused with the right status rather
than a stack trace.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from src.serve import api  # noqa: E402


@pytest.fixture()
def client():
    return TestClient(api.app)


def test_health_does_not_load_the_model(client, monkeypatch):
    """A liveness probe that loads the weights is one that times out and gets the pod killed."""
    monkeypatch.setattr(api, "_PIPELINE", None)

    def explode():
        raise AssertionError("/health must not construct the pipeline")

    monkeypatch.setattr(api, "pipeline", explode)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is False


def test_openapi_publishes_the_endpoint_contract(client):
    """The row's criterion is a documented /predict; FastAPI publishes it from the code itself."""
    schema = client.get("/openapi.json")
    assert schema.status_code == 200
    paths = schema.json()["paths"]
    assert {"/predict", "/health", "/version"} <= set(paths)
    assert "post" in paths["/predict"]


def test_version_reports_the_registry_rather_than_a_constant(client):
    response = client.get("/version")
    assert response.status_code == 200
    body = response.json()
    assert "registry" in body
    # Whatever 15.3 holds, nothing below Staging may be presented as what this service answers with.
    for row in body["registry"]:
        assert row["stage"] in ("Production", "Staging")


def test_a_registry_that_cannot_be_read_does_not_stop_the_service(monkeypatch, client):
    """A service that will not start because a report is missing is worse than one that says so."""
    monkeypatch.setattr(
        "src.mlops.registry.listing",
        lambda *_, **__: (_ for _ in ()).throw(RuntimeError("no store")),
    )
    assert client.get("/version").status_code == 200


def test_a_non_image_is_refused_with_415(client):
    response = client.post("/predict", files={"image": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 415
    assert "unsupported file type" in response.json()["detail"]


def test_a_file_with_no_suffix_is_refused(client):
    response = client.post("/predict", files={"image": ("page", b"\x89PNG", "image/png")})
    assert response.status_code == 415


def test_an_empty_upload_is_refused_with_400(client):
    response = client.post("/predict", files={"image": ("page.png", b"", "image/png")})
    assert response.status_code == 400


def test_an_oversized_upload_is_refused_with_413(client, monkeypatch):
    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 16)
    response = client.post("/predict", files={"image": ("page.png", b"x" * 64, "image/png")})
    assert response.status_code == 413


def test_a_pipeline_crash_is_502_not_500(client, monkeypatch):
    """The pipeline is the upstream here; the distinction tells a caller what to retry."""

    class Broken:
        def run(self, _path):
            raise RuntimeError("the card fell out")

    monkeypatch.setattr(api, "pipeline", lambda: Broken())
    response = client.post("/predict", files={"image": ("page.png", b"\x89PNG\r\n", "image/png")})
    assert response.status_code == 502
    assert "the card fell out" in response.json()["detail"]


def test_a_page_that_does_not_reach_code_is_200_with_ok_false(client, monkeypatch):
    """A page the pipeline honestly could not read is a successful answer to the question asked."""

    class Stopped:
        def run(self, _path):
            from src.pipeline.contracts import Result, StageReport

            result = Result(source="x")
            result.stages.append(StageReport("detect", ok=False, seconds=0.1, reason="no boxes"))
            result.stopped_at = "detect"
            return result

    monkeypatch.setattr(api, "pipeline", lambda: Stopped())
    response = client.post("/predict", files={"image": ("page.png", b"\x89PNG\r\n", "image/png")})
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is False
    assert body["stopped_at"] == "detect"


def test_degradation_is_reported_rather_than_hidden():
    """13.4's rule at the API boundary: an emitter answer must never read as a model answer."""
    from src.pipeline.contracts import Result, StageReport

    result = Result(source="x", code="print(1)", language="python")
    result.stages.append(StageReport("generate", ok=True, seconds=0.1, degraded=True))
    payload = api.result_payload(result, 0.1)
    assert payload["degraded"] is True

    clean = Result(source="x", code="print(1)", language="python")
    clean.stages.append(StageReport("generate", ok=True, seconds=0.1))
    assert api.result_payload(clean, 0.1)["degraded"] is False


def test_the_payload_carries_everything_needed_to_judge_the_answer():
    from src.pipeline.contracts import Result, StageReport

    result = Result(source="x", code="print(1)", language="python")
    result.stages.append(StageReport("generate", ok=True, seconds=0.1))
    payload = api.result_payload(result, 0.1)
    for key in (
        "ok",
        "degraded",
        "stopped_at",
        "needs_confirmation",
        "diagram_type",
        "language",
        "code",
        "ir",
        "stages",
        "seconds",
    ):
        assert key in payload, key


def test_self_test_covers_every_route_that_needs_no_model():
    result = api.self_test()
    assert result["ok"] is True
    assert set(result["checks"]) >= {"health", "version", "openapi", "rejects_wrong_type"}
