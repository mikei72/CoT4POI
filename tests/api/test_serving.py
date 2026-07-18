from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from next_poi.contracts import RecommendationRequest
from next_poi.data import read_synthetic_splits
from next_poi.models import CandidateIndex, SmokePredictor, save_smoke_bundle
from next_poi.serving import create_app

FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "synthetic"


def _predictor_and_request() -> tuple[SmokePredictor, RecommendationRequest]:
    splits = read_synthetic_splits(
        {split: FIXTURE_ROOT / f"{split}.csv" for split in ("train", "validation", "test")}
    )
    taxonomy = {event.category for events in splits.values() for event in events}
    predictor = SmokePredictor(CandidateIndex.fit(splits["train"], taxonomy=taxonomy))
    history = splits["validation"][0]
    target = splits["validation"][1]
    request = RecommendationRequest(
        dataset="synthetic",
        history=(
            {
                "poi_id": history.raw_poi_id,
                "model_poi_id": history.model_poi_id,
                "category_name": history.category,
                "timestamp": history.timestamp_utc,
            },
        ),
        target_time=target.timestamp_utc,
        top_k=5,
    )
    return predictor, request


@pytest.fixture
def bundle(tmp_path: Path) -> tuple[Path, SmokePredictor, RecommendationRequest]:
    predictor, request = _predictor_and_request()
    bundle_path = tmp_path / "bundle"
    save_smoke_bundle(bundle_path, predictor)
    return bundle_path, predictor, request


def test_five_endpoints_load_verified_bundle_and_share_predictor(bundle) -> None:
    bundle_path, predictor, request = bundle
    expected = predictor.predict(request)

    with TestClient(create_app(bundle_path)) as client:
        assert client.get("/health").json() == {"status": "ok"}

        ready = client.get("/ready")
        assert ready.status_code == 200
        assert ready.json()["status"] == "ready"
        assert len(ready.json()["bundle_manifest_sha256"]) == 64

        version = client.get("/version")
        assert version.status_code == 200
        assert version.json() == predictor.versions.model_dump(mode="json")

        recommendation = client.post(
            "/recommend", json=request.model_dump(mode="json")
        )
        assert recommendation.status_code == 200
        actual = recommendation.json()
        assert actual["request_id"] == expected.request_id
        assert [item["poi_id"] for item in actual["recommendations"]] == [
            item.poi_id for item in expected.recommendations
        ]
        assert [item["score"] for item in actual["recommendations"]] == [
            item.score for item in expected.recommendations
        ]

        metrics = client.get("/metrics")
        assert metrics.status_code == 200
        payload = metrics.json()
        assert payload["requests_total"] == 4
        assert payload["requests_by_endpoint"] == {
            "health": 1,
            "ready": 1,
            "version": 1,
            "recommend": 1,
            "metrics": 0,
            "other": 0,
        }
        assert payload["responses_by_status_class"] == {"2xx": 4, "4xx": 0, "5xx": 0}
        assert payload["latency_ms"]["count"] == 4


def test_missing_or_invalid_bundle_is_live_but_not_ready(tmp_path: Path) -> None:
    missing_path = tmp_path / "private" / "missing-bundle"
    with TestClient(create_app(missing_path)) as client:
        assert client.get("/health").status_code == 200
        first = client.get("/ready")
        second = client.get("/ready")
        assert first.status_code == second.status_code == 503
        assert first.json() == second.json() == {
            "error": {
                "code": "bundle_validation_failed",
                "message": "validated smoke bundle is not ready",
            }
        }
        assert str(missing_path) not in first.text
        assert client.get("/version").status_code == 503


def test_tampered_bundle_hash_is_live_but_not_ready(bundle) -> None:
    bundle_path, _predictor, _request = bundle
    config_path = bundle_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["catalog_size"] += 1
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with TestClient(create_app(bundle_path)) as client:
        assert client.get("/health").status_code == 200
        ready = client.get("/ready")

    assert ready.status_code == 503
    assert ready.json()["error"]["code"] == "bundle_validation_failed"
    assert str(bundle_path) not in ready.text


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.update(history=[]),
        lambda payload: payload.update(target="synthetic-poi-secret"),
        lambda payload: payload.update({"/Users/private/local-path": "hidden"}),
        lambda payload: payload.update(target_time="2026-01-01T12:00:00"),
        lambda payload: payload.update(top_k=101),
        lambda payload: payload.update(target_time="2020-01-01T00:00:00+00:00"),
    ],
)
def test_contract_validation_returns_deterministic_sanitized_422(bundle, mutation) -> None:
    bundle_path, _predictor, request = bundle
    payload = request.model_dump(mode="json")
    mutation(payload)

    with TestClient(create_app(bundle_path)) as client:
        first = client.post("/recommend", json=payload)
        second = client.post("/recommend", json=payload)

    assert first.status_code == second.status_code == 422
    assert first.json() == second.json()
    body = first.json()
    assert body["error"]["code"] == "request_validation_error"
    serialized = json.dumps(body, sort_keys=True)
    assert "synthetic-poi-secret" not in serialized
    assert "/Users/private/local-path" not in serialized
    assert str(bundle_path) not in serialized
    assert "traceback" not in serialized.lower()


def test_dataset_mismatch_is_a_stable_4xx(bundle) -> None:
    bundle_path, _predictor, request = bundle
    payload = request.model_dump(mode="json")
    payload["dataset"] = "nyc"

    with TestClient(create_app(bundle_path)) as client:
        first = client.post("/recommend", json=payload)
        second = client.post("/recommend", json=payload)

    assert first.status_code == second.status_code == 422
    assert first.json() == second.json() == {
        "error": {
            "code": "unsupported_dataset",
            "message": "dataset is not supported by the loaded bundle",
        }
    }


def test_unknown_history_uses_global_fallback_and_metrics_are_private(bundle) -> None:
    bundle_path, predictor, request = bundle
    payload = request.model_dump(mode="json")
    payload["history"] = [
        {
            "poi_id": "private-unknown-poi",
            "category_name": "private-unknown-category",
            "timestamp": payload["history"][0]["timestamp"],
        }
    ]

    with TestClient(create_app(bundle_path)) as client:
        recommendation = client.post("/recommend", json=payload)
        metrics = client.get("/metrics")

    assert recommendation.status_code == 200
    items = recommendation.json()["recommendations"]
    assert items
    assert len({item["poi_id"] for item in items}) == len(items)
    assert {item["poi_id"] for item in items} <= set(predictor.index.global_counts)
    assert all("global_popularity" in item["candidate_sources"] for item in items)

    serialized_metrics = json.dumps(metrics.json(), sort_keys=True)
    assert "private-unknown-poi" not in serialized_metrics
    assert "private-unknown-category" not in serialized_metrics
    assert str(bundle_path) not in serialized_metrics
