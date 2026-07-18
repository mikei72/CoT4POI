"""Pure deterministic ranking, candidate, coverage, and validity metrics."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class RankingObservation:
    """Target-aware record created only after a target-blind prediction completes."""

    sample_id: str
    target_poi_id: str | None
    predictions: tuple[str, ...]
    candidates: tuple[str, ...]
    target_category: str | None = None
    macro_predictions: tuple[str, ...] = ()


def hit_at_k(predictions: Sequence[str], target: str | None, k: int) -> float:
    _validate_k(k)
    if not target:
        return 0.0
    return float(target in predictions[:k])


def reciprocal_rank(predictions: Sequence[str], target: str | None) -> float:
    if not target:
        return 0.0
    for rank, poi_id in enumerate(predictions, start=1):
        if poi_id == target:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(predictions: Sequence[str], target: str | None, k: int) -> float:
    """Binary single-relevant-item NDCG used by the next-POI task."""

    _validate_k(k)
    if not target:
        return 0.0
    for rank, poi_id in enumerate(predictions[:k], start=1):
        if poi_id == target:
            return 1.0 / math.log2(rank + 1)
    return 0.0


def candidate_recall_at_k(candidates: Sequence[str], target: str | None, k: int) -> float:
    return hit_at_k(candidates, target, k)


def aggregate_ranking_metrics(
    observations: Iterable[RankingObservation],
    *,
    catalog: Iterable[str],
) -> dict[str, float]:
    """Aggregate the frozen Gate 2 metric matrix with explicit empty semantics."""

    rows = tuple(observations)
    catalog_ids = frozenset(catalog)
    sample_count = len(rows)
    predicted_items = [poi_id for row in rows for poi_id in row.predictions]
    invalid_count = sum(poi_id not in catalog_ids for poi_id in predicted_items)
    duplicate_count = sum(
        len(row.predictions) - len(set(row.predictions)) for row in rows
    )
    empty_count = sum(not row.predictions for row in rows)
    valid_sample_count = sum(
        bool(row.predictions)
        and len(row.predictions) == len(set(row.predictions))
        and all(poi_id in catalog_ids for poi_id in row.predictions)
        for row in rows
    )
    covered = {poi_id for poi_id in predicted_items if poi_id in catalog_ids}

    metrics = {
        "hit_at_1": _mean(hit_at_k(row.predictions, row.target_poi_id, 1) for row in rows),
        "hit_at_5": _mean(hit_at_k(row.predictions, row.target_poi_id, 5) for row in rows),
        "hit_at_10": _mean(
            hit_at_k(row.predictions, row.target_poi_id, 10) for row in rows
        ),
        "mrr": _mean(reciprocal_rank(row.predictions, row.target_poi_id) for row in rows),
        "ndcg_at_5": _mean(ndcg_at_k(row.predictions, row.target_poi_id, 5) for row in rows),
        "ndcg_at_10": _mean(
            ndcg_at_k(row.predictions, row.target_poi_id, 10) for row in rows
        ),
        "candidate_recall_at_50": _mean(
            candidate_recall_at_k(row.candidates, row.target_poi_id, 50) for row in rows
        ),
        "candidate_recall_at_100": _mean(
            candidate_recall_at_k(row.candidates, row.target_poi_id, 100) for row in rows
        ),
        "macro_hit_at_1": _mean(
            hit_at_k(row.macro_predictions, row.target_category, 1) for row in rows
        ),
        "macro_hit_at_3": _mean(
            hit_at_k(row.macro_predictions, row.target_category, 3) for row in rows
        ),
        "macro_mrr": _mean(
            reciprocal_rank(row.macro_predictions, row.target_category) for row in rows
        ),
        "coverage": len(covered) / len(catalog_ids) if catalog_ids else 0.0,
        "validity_rate": valid_sample_count / sample_count if sample_count else 0.0,
        "invalid_rate": invalid_count / len(predicted_items) if predicted_items else 0.0,
        "duplicate_rate": duplicate_count / len(predicted_items) if predicted_items else 0.0,
        "empty_rate": empty_count / sample_count if sample_count else 0.0,
    }
    return {name: float(value) for name, value in metrics.items()}


def _validate_k(k: int) -> None:
    if k < 1:
        raise ValueError("k must be at least one")


def _mean(values: Iterable[float]) -> float:
    items = tuple(values)
    return sum(items) / len(items) if items else 0.0
