"""Shared batch/API predictor for the deterministic smoke backend."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from time import perf_counter

from next_poi.contracts import (
    CategoryScore,
    LatencyBreakdown,
    Recommendation,
    RecommendationRequest,
    RecommendationResponse,
    VersionInfo,
)
from next_poi.models.index import CandidateIndex
from next_poi.models.ranker import CompiledRanker, RankedCandidate

TRACE_CANDIDATE_LIMIT = 100


@dataclass(frozen=True)
class PredictionTrace:
    response: RecommendationResponse
    candidates: tuple[RankedCandidate, ...]
    candidate_count: int


class SmokePredictor:
    def __init__(
        self,
        index: CandidateIndex,
        *,
        variant: str = "b3",
        versions: VersionInfo | None = None,
    ) -> None:
        self.index = index
        self.variant = variant
        self.versions = versions or VersionInfo(
            release="smoke-v1",
            data=f"{index.dataset}-data-v1",
            model=f"smoke-{variant}-v1",
        )
        self._ranker = CompiledRanker(index)

    def predict(self, request: RecommendationRequest) -> RecommendationResponse:
        return self.predict_with_trace(request).response

    def predict_with_trace(self, request: RecommendationRequest) -> PredictionTrace:
        started = perf_counter()
        candidate_started = perf_counter()
        ranking = self._ranker.rank(
            request,
            variant=self.variant,
            limit=max(TRACE_CANDIDATE_LIMIT, request.top_k),
        )
        candidates = ranking.candidates
        candidate_ms = (perf_counter() - candidate_started) * 1000.0
        recommendations = tuple(
            Recommendation(
                rank=rank,
                poi_id=item.poi_id,
                model_poi_id=item.model_poi_id,
                category=item.category,
                score=item.score,
                candidate_sources=item.candidate_sources,
            )
            for rank, item in enumerate(candidates[: request.top_k], start=1)
        )
        macro = tuple(
            CategoryScore(category=category, score=score)
            for category, score in ranking.macro_scores
        )
        total_ms = (perf_counter() - started) * 1000.0
        response = RecommendationResponse(
            recommendations=recommendations,
            macro=macro,
            versions=self.versions,
            latency=LatencyBreakdown(total_ms=total_ms, candidate_ms=candidate_ms),
            request_id=_request_id(request, self.variant),
        )
        return PredictionTrace(
            response=response,
            candidates=candidates,
            candidate_count=ranking.candidate_count,
        )


def _request_id(request: RecommendationRequest, variant: str) -> str:
    payload = {
        "request": request.model_dump(mode="json"),
        "variant": variant,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "smoke-" + hashlib.sha256(encoded).hexdigest()[:24]
