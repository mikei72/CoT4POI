"""Privacy-safe conversion from Predictor traces to aggregate events."""

from __future__ import annotations

import math
from collections import Counter

from next_poi.contracts import RecommendationRequest
from next_poi.models import PredictionTrace, SmokePredictor
from next_poi.monitoring import MonitoringEvent, category_bucket, history_length_bucket


def build_success_event(
    request: RecommendationRequest,
    trace: PredictionTrace,
    predictor: SmokePredictor,
) -> MonitoringEvent:
    """Build one event without retaining identities, history, coordinates, or labels."""

    source_counts: Counter[str] = Counter()
    for candidate in trace.candidates:
        source_counts.update(candidate.candidate_sources)
    category_counts = Counter(
        category_bucket(item.category) for item in trace.response.recommendations
    )
    unknown_count = sum(
        event.poi_id not in predictor.index.global_counts for event in request.history
    )
    latency = trace.response.latency
    return MonitoringEvent(
        request_id=trace.response.request_id,
        versions=trace.response.versions,
        history_length_bucket=history_length_bucket(len(request.history)),
        unknown_count=unknown_count,
        candidate_count=trace.candidate_count,
        candidate_source_histogram=dict(source_counts),
        category_histogram=dict(category_counts),
        score_entropy=_score_entropy(
            tuple(item.score for item in trace.response.recommendations)
        ),
        stage_latency_ms={
            "candidate": latency.candidate_ms,
            "model": latency.model_ms,
        },
        total_latency_ms=latency.total_ms,
        status="ok",
    )


def _score_entropy(scores: tuple[float, ...]) -> float:
    if len(scores) < 2:
        return 0.0
    maximum = max(scores)
    weights = [math.exp(score - maximum) for score in scores]
    total = sum(weights)
    probabilities = [weight / total for weight in weights]
    return -sum(value * math.log(value) for value in probabilities if value > 0)
