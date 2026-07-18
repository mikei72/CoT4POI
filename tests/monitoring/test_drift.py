from __future__ import annotations

import json

import pytest

from next_poi.monitoring.drift import (
    build_drift_report,
    js_divergence,
    population_stability_index,
)
from next_poi.monitoring.events import MonitoringEvent, category_bucket
from next_poi.monitoring.replay import inject_category_history_drift


def make_event(index: int) -> MonitoringEvent:
    return MonitoringEvent(
        request_id=f"req-{index}",
        versions={"release": "r1", "data": "d1", "model": "m1"},
        history_length_bucket="2-5",
        unknown_count=index % 2,
        candidate_count=10,
        candidate_source_histogram={"global_popularity": 10},
        category_histogram={
            category_bucket("Cafe"): 6,
            category_bucket("Park"): 4,
        },
        score_entropy=0.8,
        stage_latency_ms={"candidate": 1.0, "model": 2.0},
        total_latency_ms=3.5,
        status="ok",
    )


def test_js_divergence_and_psi_are_deterministic() -> None:
    reference = {"a": 9, "b": 1}
    current = {"a": 1, "b": 9}

    assert js_divergence(reference, current) == pytest.approx(
        js_divergence(reference, current)
    )
    assert population_stability_index(reference, current) == pytest.approx(
        population_stability_index(reference, current)
    )
    assert js_divergence(reference, current) > 0
    assert population_stability_index(reference, current) > 0


def test_normal_replay_has_no_alert_and_no_label_metric() -> None:
    reference = tuple(make_event(index) for index in range(20))
    report = build_drift_report(reference, reference)

    assert report.overall_alert is False
    assert all(not item.alert for item in report.distributions.values())
    serialized = json.dumps(report.model_dump(mode="json"), sort_keys=True)
    assert "accuracy" not in serialized.lower()
    assert "label" not in serialized.lower()


def test_drift_revalidates_shallow_mutable_histograms() -> None:
    event = make_event(0)
    event.candidate_source_histogram["user-secret"] = 1  # type: ignore[index]

    with pytest.raises(ValueError):
        build_drift_report((event,), (make_event(1),))


@pytest.mark.parametrize("unsafe_key", ["user_id", "target", "private-user-id"])
def test_drift_rejects_mutated_category_key_leakage(unsafe_key: str) -> None:
    event = make_event(0)
    event.category_histogram[unsafe_key] = 1

    with pytest.raises(ValueError):
        build_drift_report((event,), (make_event(1),))


def test_injected_category_and_history_drift_alerts_reproducibly() -> None:
    reference = tuple(make_event(index) for index in range(20))
    injected_once = inject_category_history_drift(reference)
    injected_twice = inject_category_history_drift(reference)

    assert injected_once == injected_twice
    report = build_drift_report(reference, injected_once)
    assert report.overall_alert is True
    assert report.distributions["history_length"].alert is True
    assert report.distributions["category"].alert is True


@pytest.mark.parametrize("empty_side", ["reference", "current"])
def test_replay_rejects_empty_windows(empty_side: str) -> None:
    events = (make_event(1),)
    reference = () if empty_side == "reference" else events
    current = () if empty_side == "current" else events

    with pytest.raises(ValueError, match="window must contain"):
        build_drift_report(reference, current)
