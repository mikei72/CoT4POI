from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from next_poi.data import read_synthetic_splits
from next_poi.models import CandidateIndex, SmokePredictor, save_smoke_bundle
from next_poi.monitoring import PRIVACY_ALLOWLIST, JsonlEventStore, category_bucket
from next_poi.serving import create_app

FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "synthetic"


def test_api_writes_only_privacy_safe_aggregate_event(tmp_path: Path) -> None:
    splits = read_synthetic_splits(
        {
            split: FIXTURE_ROOT / f"{split}.csv"
            for split in ("train", "validation", "test")
        }
    )
    taxonomy = {event.category for events in splits.values() for event in events}
    predictor = SmokePredictor(CandidateIndex.fit(splits["train"], taxonomy=taxonomy))
    bundle_path = tmp_path / "bundle"
    save_smoke_bundle(bundle_path, predictor)
    event_path = tmp_path / "monitoring" / "events.jsonl"
    payload = {
        "dataset": "synthetic",
        "history": [
            {
                "poi_id": "private-user-history-poi",
                "category_name": "private-user-category",
                "timestamp": "2026-01-01T08:00:00Z",
                "latitude": 40.7128,
                "longitude": -74.006,
            }
        ],
        "target_time": "2026-01-01T09:00:00Z",
        "top_k": 5,
    }

    with TestClient(create_app(bundle_path, monitoring_path=event_path)) as client:
        response = client.post("/recommend", json=payload)

    assert response.status_code == 200
    events = JsonlEventStore(event_path).read()
    assert len(events) == 1
    event = events[0]
    assert set(event.model_dump()) == PRIVACY_ALLOWLIST
    assert event.unknown_count == 1
    assert event.candidate_count == len(predictor.index.global_counts)
    assert set(event.category_histogram) == {
        category_bucket(item["category"])
        for item in response.json()["recommendations"]
    }
    serialized = event_path.read_text(encoding="utf-8")
    for forbidden in (
        "private-user-history-poi",
        "private-user-category",
        "40.7128",
        "-74.006",
        "target_time",
        "target_poi",
        "accuracy",
    ):
        assert forbidden not in serialized


def test_monitoring_write_failure_does_not_fail_recommendation(tmp_path: Path) -> None:
    splits = read_synthetic_splits(
        {split: FIXTURE_ROOT / f"{split}.csv" for split in ("train", "validation", "test")}
    )
    taxonomy = {event.category for events in splits.values() for event in events}
    predictor = SmokePredictor(CandidateIndex.fit(splits["train"], taxonomy=taxonomy))
    bundle_path = tmp_path / "bundle"
    save_smoke_bundle(bundle_path, predictor)
    unwritable_event_path = tmp_path / "events.jsonl"
    unwritable_event_path.mkdir()
    payload = {
        "dataset": "synthetic",
        "history": [
            {
                "poi_id": "unknown-poi",
                "category_name": "unknown-category",
                "timestamp": "2026-01-01T08:00:00Z",
            }
        ],
        "target_time": "2026-01-01T09:00:00Z",
        "top_k": 5,
    }

    with TestClient(
        create_app(bundle_path, monitoring_path=unwritable_event_path)
    ) as client:
        response = client.post("/recommend", json=payload)

    assert response.status_code == 200
    assert response.json()["recommendations"]
