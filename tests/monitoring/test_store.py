from __future__ import annotations

import json
from pathlib import Path

import pytest

from next_poi.monitoring.events import MonitoringEvent, category_bucket
from next_poi.monitoring.store import JsonlEventStore, MonitoringStoreError


def make_event(request_id: str) -> MonitoringEvent:
    return MonitoringEvent(
        request_id=request_id,
        versions={"release": "r1", "data": "d1", "model": "m1"},
        history_length_bucket="2-5",
        unknown_count=0,
        candidate_count=2,
        candidate_source_histogram={"global_popularity": 2},
        category_histogram={
            category_bucket("Cafe"): 1,
            category_bucket("Park"): 1,
        },
        score_entropy=0.5,
        stage_latency_ms={"candidate": 1.0},
        total_latency_ms=1.5,
        status="ok",
    )


def test_jsonl_store_appends_stable_records_and_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    store = JsonlEventStore(path)
    expected = (make_event("req-1"), make_event("req-2"))

    for event in expected:
        store.append(event)

    assert store.read() == expected
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert lines[0] == json.dumps(
        expected[0].model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def test_jsonl_store_rejects_unallowlisted_record_without_exposing_path(tmp_path: Path) -> None:
    path = tmp_path / "private" / "events.jsonl"
    path.parent.mkdir()
    path.write_text('{"user_id":"secret"}\n', encoding="utf-8")

    with pytest.raises(MonitoringStoreError) as caught:
        JsonlEventStore(path).read()

    assert str(path) not in str(caught.value)
    assert str(caught.value) == "invalid monitoring event at JSONL line 1"


def test_missing_store_reads_as_empty_window(tmp_path: Path) -> None:
    assert JsonlEventStore(tmp_path / "missing.jsonl").read() == ()


@pytest.mark.parametrize(
    "unsafe_key",
    ["/Users/private/category", "user_id", "target", "private-user-id"],
)
def test_append_revalidates_mutated_nested_histogram(
    tmp_path: Path, unsafe_key: str
) -> None:
    event = make_event("req-mutated")
    event.category_histogram[unsafe_key] = 1

    with pytest.raises(MonitoringStoreError, match="invalid monitoring event for append"):
        JsonlEventStore(tmp_path / "events.jsonl").append(event)
