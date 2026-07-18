from __future__ import annotations

import pytest
from pydantic import ValidationError

from next_poi.contracts import VersionInfo
from next_poi.monitoring.events import (
    PRIVACY_ALLOWLIST,
    MonitoringEvent,
    category_bucket,
    history_length_bucket,
)


def event_payload() -> dict[str, object]:
    return {
        "request_id": "req-001",
        "versions": {"release": "r1", "data": "d1", "model": "m1"},
        "history_length_bucket": "2-5",
        "unknown_count": 0,
        "candidate_count": 3,
        "candidate_source_histogram": {"global_popularity": 3},
        "category_histogram": {
            category_bucket("Cafe"): 2,
            category_bucket("Park"): 1,
        },
        "score_entropy": 0.8,
        "stage_latency_ms": {"candidate": 1.2, "model": 0.7},
        "total_latency_ms": 2.1,
        "status": "ok",
        "error_code": None,
    }


def test_monitoring_event_has_exact_privacy_allowlist() -> None:
    assert PRIVACY_ALLOWLIST == {
        "request_id",
        "versions",
        "history_length_bucket",
        "unknown_count",
        "candidate_count",
        "candidate_source_histogram",
        "category_histogram",
        "score_entropy",
        "stage_latency_ms",
        "total_latency_ms",
        "status",
        "error_code",
    }


@pytest.mark.parametrize(
    "forbidden_field",
    [
        "user_id",
        "history",
        "latitude",
        "longitude",
        "target",
        "target_poi_id",
        "target_category",
        "label",
        "accuracy",
        "error_message",
    ],
)
def test_monitoring_event_rejects_private_or_label_fields(forbidden_field: str) -> None:
    payload = event_payload()
    payload[forbidden_field] = "private"

    with pytest.raises(ValidationError):
        MonitoringEvent.model_validate(payload)


def test_monitoring_event_enforces_stable_status_contract() -> None:
    payload = event_payload()
    payload.update(status="error", error_code="BUNDLE_NOT_READY")
    event = MonitoringEvent.model_validate(payload)

    assert event.versions == VersionInfo(release="r1", data="d1", model="m1")

    payload["error_code"] = "/Users/private/model.bin"
    with pytest.raises(ValidationError):
        MonitoringEvent.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "unsafe_key"),
    [
        ("candidate_source_histogram", "user_id"),
        ("candidate_source_histogram", "/Users/private/source"),
        ("stage_latency_ms", "user_lookup"),
        ("stage_latency_ms", "/Users/private/stage"),
        ("category_histogram", "/Users/private/category"),
        ("category_histogram", "Cafe\\private"),
        ("category_histogram", "Cafe\nprivate"),
        ("category_histogram", "user:secret"),
        ("category_histogram", "user_id"),
        ("category_histogram", "target"),
        ("category_histogram", "private-user-id"),
    ],
)
def test_monitoring_event_rejects_nested_key_leakage(field: str, unsafe_key: str) -> None:
    payload = event_payload()
    payload[field] = {unsafe_key: 1}

    with pytest.raises(ValidationError):
        MonitoringEvent.model_validate(payload)


def test_history_length_is_bucketed_without_retaining_history() -> None:
    assert [history_length_bucket(value) for value in (0, 1, 2, 6, 11, 21, 51, 101)] == [
        "0",
        "1",
        "2-5",
        "6-10",
        "11-20",
        "21-50",
        "51-100",
        "101+",
    ]


def test_category_bucket_is_deterministic_and_opaque() -> None:
    assert category_bucket("Cafe") == category_bucket("Cafe")
    assert category_bucket("Cafe").startswith("category_")
    assert "Cafe" not in category_bucket("Cafe")
