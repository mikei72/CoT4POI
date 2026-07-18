from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path

import pytest

from next_poi.contracts import RecommendationRequest
from next_poi.data import read_synthetic_splits
from next_poi.models import (
    VARIANT_SOURCES,
    CandidateIndex,
    SmokePredictor,
    rank_candidates,
    time_bucket,
)

FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "synthetic"


def fixture_splits():
    return read_synthetic_splits(
        {split: FIXTURE_ROOT / f"{split}.csv" for split in ("train", "validation", "test")}
    )


def fitted_index() -> CandidateIndex:
    splits = fixture_splits()
    taxonomy = {event.category for events in splits.values() for event in events}
    return CandidateIndex.fit(splits["train"], taxonomy=taxonomy)


def request() -> RecommendationRequest:
    splits = fixture_splits()
    event = splits["validation"][0]
    return RecommendationRequest(
        dataset="synthetic",
        history=(
            {
                "poi_id": event.raw_poi_id,
                "category_name": event.category,
                "timestamp": event.timestamp_utc,
            },
        ),
        target_time=splits["validation"][1].timestamp_utc,
        top_k=5,
    )


def test_fit_is_train_only_and_input_order_invariant() -> None:
    splits = fixture_splits()
    taxonomy = {event.category for events in splits.values() for event in events}
    first = CandidateIndex.fit(splits["train"], taxonomy=taxonomy)
    second = CandidateIndex.fit(reversed(splits["train"]), taxonomy=reversed(sorted(taxonomy)))
    assert first == second

    with pytest.raises(ValueError, match="train events only"):
        CandidateIndex.fit(splits["train"] + splits["validation"], taxonomy=taxonomy)


def test_time_bucket_is_defined_in_normalized_utc() -> None:
    utc_time = datetime(2026, 1, 3, 23, tzinfo=timezone.utc)
    same_instant = datetime.fromisoformat("2026-01-04T08:00:00+09:00")
    assert time_bucket(utc_time) == "weekend:night"
    assert time_bucket(same_instant) == time_bucket(utc_time)


def test_all_frozen_variants_are_deterministic_and_source_ordered() -> None:
    index = fitted_index()
    online_request = request()
    canonical = tuple(VARIANT_SOURCES["b3"])
    for variant, active_sources in VARIANT_SOURCES.items():
        first = rank_candidates(index, online_request, variant=variant)
        second = rank_candidates(index, online_request, variant=variant)
        assert first == second
        assert {item.poi_id for item in first} == set(index.global_counts)
        for item in first:
            assert item.candidate_sources == tuple(
                source for source in canonical if source in item.candidate_sources
            )
            assert set(item.candidate_sources) <= set(active_sources)


def test_b3_uses_the_frozen_formula_exactly() -> None:
    index = CandidateIndex(
        schema_version="1",
        dataset="synthetic",
        train_data_sha256="a" * 64,
        taxonomy=("Synthetic Cafe", "Synthetic Park"),
        poi_categories={"poi-a": "Synthetic Cafe", "poi-b": "Synthetic Park"},
        model_poi_ids={"poi-a": 0, "poi-b": 1},
        global_counts={"poi-a": 5, "poi-b": 8},
        time_counts={
            "weekday:morning": {"poi-a": 2},
            "weekday:afternoon": {"poi-a": 3},
            "weekend:night": {"poi-b": 8},
        },
        transition_counts={"poi-a": {"poi-b": 4}},
        category_counts={
            "Synthetic Cafe": {"poi-a": 5},
            "Synthetic Park": {"poi-b": 8},
        },
    )
    online_request = RecommendationRequest(
        dataset="synthetic",
        history=(
            {
                "poi_id": "poi-a",
                "category_name": "Synthetic Park",
                "timestamp": "2026-01-05T08:00:00Z",
            },
        ),
        target_time="2026-01-05T09:00:00Z",
        top_k=2,
    )

    signals = {
        "global_a": math.log1p(5),
        "global_b": math.log1p(8),
        "time_a": 1.25 * math.log1p(2),
        "transition_b": 2.0 * math.log1p(4),
        "revisit_a": 1.5,
        "category_b": 0.75 * math.log1p(8),
    }
    expected = {
        "b0": (signals["global_a"], signals["global_b"]),
        "b1": (
            signals["global_a"] + signals["time_a"],
            signals["global_b"],
        ),
        "b2": (
            signals["global_a"] + signals["time_a"],
            signals["global_b"] + signals["transition_b"],
        ),
        "b3": (
            signals["global_a"] + signals["time_a"] + signals["revisit_a"],
            signals["global_b"] + signals["transition_b"] + signals["category_b"],
        ),
        "b3_no_time": (
            signals["global_a"] + signals["revisit_a"],
            signals["global_b"] + signals["transition_b"] + signals["category_b"],
        ),
        "b3_no_transition": (
            signals["global_a"] + signals["time_a"] + signals["revisit_a"],
            signals["global_b"] + signals["category_b"],
        ),
        "b3_no_history": (
            signals["global_a"] + signals["time_a"],
            signals["global_b"] + signals["transition_b"],
        ),
    }
    for variant, (expected_a, expected_b) in expected.items():
        by_id = {
            item.poi_id: item
            for item in rank_candidates(index, online_request, variant=variant)
        }
        assert by_id["poi-a"].score == pytest.approx(expected_a)
        assert by_id["poi-b"].score == pytest.approx(expected_b)
    b3_by_id = {
        item.poi_id: item for item in rank_candidates(index, online_request, variant="b3")
    }
    assert b3_by_id["poi-b"].candidate_sources == (
        "global_popularity",
        "transition",
        "history_category",
    )


def test_predictor_accepts_request_only_and_returns_stable_unique_top_k() -> None:
    predictor = SmokePredictor(fitted_index(), variant="b3")
    online_request = request()
    first = predictor.predict_with_trace(online_request)
    second = predictor.predict_with_trace(online_request)

    assert [item.poi_id for item in first.response.recommendations] == [
        item.poi_id for item in second.response.recommendations
    ]
    assert first.response.request_id == second.response.request_id
    assert len(first.response.recommendations) == online_request.top_k
    assert len({item.poi_id for item in first.response.recommendations}) == online_request.top_k
    assert len(first.candidates) == len(predictor.index.global_counts)
