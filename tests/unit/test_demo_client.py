from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

from next_poi.contracts import RecommendationRequest
from next_poi.demo.api_client import (
    ApiClient,
    ApiProtocolError,
    ApiUnavailableError,
    ApiValidationError,
)
from next_poi.demo.app import build_request, parse_history


def _request() -> RecommendationRequest:
    return RecommendationRequest(
        dataset="synthetic",
        history=(
            {
                "poi_id": "unknown-poi",
                "category_name": "unknown-category",
                "timestamp": "2026-01-01T00:00:00+00:00",
            },
        ),
        target_time="2026-01-01T01:00:00+00:00",
        top_k=5,
    )


def _success_payload(request_id: str = "smoke-request") -> dict[str, object]:
    return {
        "recommendations": [],
        "macro": [],
        "versions": {"release": "smoke-v1", "data": "data-v1", "model": "model-v1"},
        "latency": {"total_ms": 1.0, "candidate_ms": 0.5, "model_ms": 0.0},
        "request_id": request_id,
    }


def test_client_uses_only_http_and_canonical_request_payload() -> None:
    observed: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        observed["method"] = request.method
        observed["path"] = request.url.path
        observed["json"] = json.loads(request.content)
        return httpx.Response(200, json=_success_payload(), request=request)

    with ApiClient("http://api:8000", transport=httpx.MockTransport(handler)) as client:
        response = client.recommend(_request())

    assert response.request_id == "smoke-request"
    assert observed["method"] == "POST"
    assert observed["path"] == "/recommend"
    payload = observed["json"]
    assert set(payload) == {"dataset", "history", "target_time", "top_k", "profile"}
    assert not {"target", "target_poi", "target_category", "label", "result"} & set(payload)


def test_client_maps_422_503_timeout_and_invalid_json_to_typed_errors() -> None:
    def validation(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            422,
            json={"error": {"code": "request_validation_error", "message": "invalid"}},
            request=request,
        )

    with ApiClient("http://api:8000", transport=httpx.MockTransport(validation)) as client:
        with pytest.raises(ApiValidationError) as validation_error:
            client.recommend(_request())
    assert validation_error.value.code == "request_validation_error"
    assert validation_error.value.status_code == 422

    def not_ready(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            503,
            json={"error": {"code": "bundle_not_ready", "message": "not ready"}},
            request=request,
        )

    with ApiClient("http://api:8000", transport=httpx.MockTransport(not_ready)) as client:
        with pytest.raises(ApiUnavailableError) as unavailable_error:
            client.ready()
    assert unavailable_error.value.status_code == 503

    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    with ApiClient("http://api:8000", transport=httpx.MockTransport(timeout)) as client:
        with pytest.raises(ApiUnavailableError, match="timed out"):
            client.health()

    def invalid_json(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not-json", request=request)

    with ApiClient("http://api:8000", transport=httpx.MockTransport(invalid_json)) as client:
        with pytest.raises(ApiProtocolError, match="invalid JSON"):
            client.health()


def test_client_validates_status_version_and_metrics_payloads() -> None:
    responses = {
        "/health": {"status": "ok"},
        "/ready": {
            "status": "ready",
            "profile": "smoke",
            "bundle_manifest_sha256": "a" * 64,
            "model": "smoke-b3-v1",
        },
        "/version": {"release": "smoke-v1", "data": "data-v1", "model": "model-v1"},
        "/metrics": {
            "requests_total": 3,
            "requests_by_endpoint": {
                "health": 1,
                "ready": 1,
                "version": 1,
                "recommend": 0,
                "metrics": 0,
                "other": 0,
            },
            "responses_by_status_class": {"2xx": 3, "4xx": 0, "5xx": 0},
            "latency_ms": {"count": 3, "sum": 1.0, "average": 0.3, "max": 0.5},
        },
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=responses[request.url.path], request=request)

    with ApiClient("http://api:8000/", transport=httpx.MockTransport(handler)) as client:
        assert client.health() == {"status": "ok"}
        assert client.ready()["profile"] == "smoke"
        assert client.version().release == "smoke-v1"
        assert client.metrics()["requests_total"] == 3


def test_client_rejects_redirects_and_malformed_nested_metrics() -> None:
    def redirect(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, json={"status": "ok"}, request=request)

    with ApiClient(
        "http://api:8000", transport=httpx.MockTransport(redirect)
    ) as client:
        with pytest.raises(ApiProtocolError, match="unexpected HTTP status"):
            client.health()

    def malformed_metrics(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "requests_total": 1,
                "requests_by_endpoint": {
                    "health": 1,
                    "ready": 0,
                    "version": 0,
                    "recommend": 0,
                    "metrics": 0,
                    "other": 0,
                },
                "responses_by_status_class": {"2xx": 1, "4xx": 0, "5xx": 0},
                "latency_ms": {
                    "count": 1,
                    "sum": -1.0,
                    "average": -1.0,
                    "max": -1.0,
                },
            },
            request=request,
        )

    with ApiClient(
        "http://api:8000", transport=httpx.MockTransport(malformed_metrics)
    ) as client:
        with pytest.raises(ApiProtocolError, match="metrics response"):
            client.metrics()


def test_history_and_request_builders_validate_timezone_and_target_order() -> None:
    history_text = "poi-a | cafe | 2026-01-01T00:00:00+00:00"
    events = parse_history(history_text)
    assert len(events) == 1
    assert events[0].poi_id == "poi-a"

    request = build_request(
        dataset="synthetic",
        history_text=history_text,
        target_time=datetime(2026, 1, 1, 1, tzinfo=timezone.utc),
        top_k=5,
    )
    assert request.profile == "smoke"

    with pytest.raises(ValueError, match="历史轨迹不能为空"):
        parse_history("\n")
    with pytest.raises(ValueError, match="第 1 行格式无效"):
        parse_history("poi-a | cafe | 2026-01-01T00:00:00")


def test_demo_package_has_no_model_data_or_bundle_imports() -> None:
    package_root = Path(__file__).parents[2] / "src" / "next_poi" / "demo"
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(package_root.glob("*.py"))
    )
    assert "from next_poi.models" not in source
    assert "import next_poi.models" not in source
    assert "from next_poi.data" not in source
    assert "import next_poi.data" not in source
