"""Frozen deterministic B0-B3 scoring and ablation variants."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from next_poi.contracts import RecommendationRequest
from next_poi.models.index import CandidateIndex, time_bucket

VariantName = str

VARIANT_SOURCES: dict[str, tuple[str, ...]] = {
    "b0": ("global_popularity",),
    "b1": ("global_popularity", "time_popularity"),
    "b2": ("global_popularity", "time_popularity", "transition"),
    "b3": (
        "global_popularity",
        "time_popularity",
        "transition",
        "recent_revisit",
        "history_category",
    ),
    "b3_no_time": (
        "global_popularity",
        "transition",
        "recent_revisit",
        "history_category",
    ),
    "b3_no_transition": (
        "global_popularity",
        "time_popularity",
        "recent_revisit",
        "history_category",
    ),
    "b3_no_history": ("global_popularity", "time_popularity", "transition"),
}

SOURCE_WEIGHTS = {
    "global_popularity": 1.0,
    "time_popularity": 1.25,
    "transition": 2.0,
    "recent_revisit": 1.5,
    "history_category": 0.75,
}


@dataclass(frozen=True)
class RankedCandidate:
    poi_id: str
    model_poi_id: int | None
    category: str
    score: float
    candidate_sources: tuple[str, ...]


@dataclass(frozen=True)
class RankingResult:
    candidates: tuple[RankedCandidate, ...]
    candidate_count: int
    macro_scores: tuple[tuple[str, float], ...]


def _ordered_history(request: RecommendationRequest):
    return sorted(
        request.history,
        key=lambda event: (event.timestamp, event.poi_id, event.category_name),
    )


def rank_candidates(
    index: CandidateIndex,
    request: RecommendationRequest,
    *,
    variant: VariantName = "b3",
    limit: int | None = None,
) -> tuple[RankedCandidate, ...]:
    """Rank candidates, optionally materializing only the requested prefix."""

    return CompiledRanker(index).rank(request, variant=variant, limit=limit).candidates


class CompiledRanker:
    """Vectorized immutable view of a train-only candidate index."""

    def __init__(self, index: CandidateIndex) -> None:
        self.index = index
        self._poi_ids = tuple(sorted(index.global_counts))
        self._poi_id_array = np.asarray(self._poi_ids)
        self._positions = {
            poi_id: position for position, poi_id in enumerate(self._poi_ids)
        }
        self._categories = tuple(
            index.poi_categories[poi_id] for poi_id in self._poi_ids
        )
        self._model_poi_ids = tuple(
            index.model_poi_ids[poi_id] for poi_id in self._poi_ids
        )
        self._global_counts = np.asarray(
            [index.global_counts[poi_id] for poi_id in self._poi_ids], dtype=np.int64
        )
        self._global_scores = np.log1p(self._global_counts.astype(np.float64))
        self._time_counts = {
            bucket: self._count_vector(counts)
            for bucket, counts in index.time_counts.items()
        }
        self._category_counts = {
            category: self._count_vector(counts)
            for category, counts in index.category_counts.items()
        }
        self._source_cache: dict[
            tuple[str, bool, bool, bool, bool], tuple[str, ...]
        ] = {}

    def _count_vector(self, counts: dict[str, int]) -> np.ndarray:
        values = np.zeros(len(self._poi_ids), dtype=np.int64)
        for poi_id, count in counts.items():
            values[self._positions[poi_id]] = count
        return values

    def rank(
        self,
        request: RecommendationRequest,
        *,
        variant: VariantName = "b3",
        limit: int | None = None,
    ) -> RankingResult:
        if request.dataset != self.index.dataset:
            raise ValueError("request dataset does not match candidate index")
        if variant not in VARIANT_SOURCES:
            raise ValueError(f"unsupported smoke variant: {variant}")
        if limit is not None and limit < 1:
            raise ValueError("candidate limit must be at least one")

        active_sources = VARIANT_SOURCES[variant]
        ordered_history = _ordered_history(request)
        latest_poi = ordered_history[-1].poi_id
        empty_counts = np.zeros(len(self._poi_ids), dtype=np.int64)
        bucket_counts = self._time_counts.get(
            time_bucket(request.target_time), empty_counts
        )
        transition_counts = self._count_vector(
            self.index.transition_counts.get(latest_poi, {})
        )

        revisit_strength = np.zeros(len(self._poi_ids), dtype=np.float64)
        for position, event in enumerate(reversed(ordered_history), start=1):
            candidate_position = self._positions.get(event.poi_id)
            if candidate_position is not None:
                revisit_strength[candidate_position] = max(
                    revisit_strength[candidate_position], 1.0 / position
                )

        category_signal = np.zeros(len(self._poi_ids), dtype=np.int64)
        recent_categories = {event.category_name for event in ordered_history[-10:]}
        for category in sorted(recent_categories):
            counts = self._category_counts.get(category)
            if counts is not None:
                category_signal += counts

        scores = self._global_scores.copy()
        if "time_popularity" in active_sources:
            scores += SOURCE_WEIGHTS["time_popularity"] * np.log1p(bucket_counts)
        if "transition" in active_sources:
            scores += SOURCE_WEIGHTS["transition"] * np.log1p(transition_counts)
        if "recent_revisit" in active_sources:
            scores += SOURCE_WEIGHTS["recent_revisit"] * revisit_strength
        if "history_category" in active_sources:
            scores += SOURCE_WEIGHTS["history_category"] * np.log1p(category_signal)

        order = np.lexsort((self._poi_id_array, -scores))
        materialized_count = len(order) if limit is None else min(limit, len(order))
        candidates = tuple(
            RankedCandidate(
                poi_id=self._poi_ids[position],
                model_poi_id=self._model_poi_ids[position],
                category=self._categories[position],
                score=float(scores[position]),
                candidate_sources=self._sources(
                    variant,
                    time_count=int(bucket_counts[position]),
                    transition_count=int(transition_counts[position]),
                    revisit_strength=float(revisit_strength[position]),
                    category_count=int(category_signal[position]),
                ),
            )
            for position in order[:materialized_count]
        )

        macro_scores: list[tuple[str, float]] = []
        seen_categories: set[str] = set()
        for position in order:
            category = self._categories[position]
            if category in seen_categories:
                continue
            seen_categories.add(category)
            macro_scores.append((category, float(scores[position])))
            if len(macro_scores) == 3:
                break
        return RankingResult(
            candidates=candidates,
            candidate_count=len(self._poi_ids),
            macro_scores=tuple(macro_scores),
        )

    def _sources(
        self,
        variant: str,
        *,
        time_count: int,
        transition_count: int,
        revisit_strength: float,
        category_count: int,
    ) -> tuple[str, ...]:
        key = (
            variant,
            time_count > 0,
            transition_count > 0,
            revisit_strength > 0,
            category_count > 0,
        )
        cached = self._source_cache.get(key)
        if cached is not None:
            return cached
        values = {
            "global_popularity": True,
            "time_popularity": time_count > 0,
            "transition": transition_count > 0,
            "recent_revisit": revisit_strength > 0,
            "history_category": category_count > 0,
        }
        sources = tuple(
            source for source in VARIANT_SOURCES[variant] if values[source]
        )
        self._source_cache[key] = sources
        return sources
