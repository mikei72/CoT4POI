from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from next_poi.contracts import NormalizedEvent, RecommendationRequest
from next_poi.models import CandidateIndex, SmokePredictor, rank_candidates


def test_vectorized_ranker_matches_frozen_score_formula() -> None:
    index = CandidateIndex(
        schema_version="1",
        dataset="synthetic",
        train_data_sha256="a" * 64,
        taxonomy=("c1", "c2"),
        poi_categories={"A": "c1", "B": "c2"},
        model_poi_ids={"A": 0, "B": 1},
        global_counts={"A": 5, "B": 8},
        time_counts={
            "weekday:morning": {"A": 2},
            "weekday:afternoon": {"A": 3},
            "weekend:morning": {"B": 8},
        },
        transition_counts={"A": {"B": 4}},
        category_counts={"c1": {"A": 5}, "c2": {"B": 8}},
    )
    request = RecommendationRequest(
        dataset="synthetic",
        history=(
            {
                "poi_id": "A",
                "category_name": "c1",
                "timestamp": "2026-01-05T07:00:00Z",
            },
        ),
        target_time="2026-01-05T08:00:00Z",
    )

    by_poi = {item.poi_id: item for item in rank_candidates(index, request)}

    assert by_poi["A"].score == pytest.approx(
        math.log1p(5) + 1.25 * math.log1p(2) + 1.5 + 0.75 * math.log1p(5)
    )
    assert by_poi["B"].score == pytest.approx(math.log1p(8) + 2.0 * math.log1p(4))
    assert by_poi["A"].candidate_sources == (
        "global_popularity",
        "time_popularity",
        "recent_revisit",
        "history_category",
    )


def test_predictor_materializes_top_100_but_reports_full_candidate_count() -> None:
    started = datetime(2026, 1, 1, tzinfo=timezone.utc)
    events = tuple(
        NormalizedEvent(
            dataset="synthetic",
            split="train",
            raw_user_id="user",
            session_id="session",
            timestamp_utc=started + timedelta(minutes=position),
            raw_poi_id=f"poi-{position:03d}",
            model_poi_id=position,
            category="category",
        )
        for position in range(120)
    )
    index = CandidateIndex.fit(events, taxonomy=("category",))
    request = RecommendationRequest(
        dataset="synthetic",
        history=(
            {
                "poi_id": events[-1].raw_poi_id,
                "model_poi_id": events[-1].model_poi_id,
                "category_name": events[-1].category,
                "timestamp": events[-1].timestamp_utc,
            },
        ),
        target_time=events[-1].timestamp_utc + timedelta(minutes=1),
        top_k=10,
    )

    full = rank_candidates(index, request)
    limited = rank_candidates(index, request, limit=50)
    trace = SmokePredictor(index).predict_with_trace(request)

    assert len(full) == 120
    assert limited == full[:50]
    assert len(trace.candidates) == 100
    assert trace.candidate_count == 120
    assert tuple(item.poi_id for item in trace.candidates) == tuple(
        item.poi_id for item in full[:100]
    )
