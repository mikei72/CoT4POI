"""Leakage-safe batch evaluation for the shared deterministic Predictor."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from next_poi.contracts import LabeledExample, RecommendationRequest
from next_poi.evaluation.metrics import RankingObservation, aggregate_ranking_metrics
from next_poi.evaluation.report import EvaluationReport, build_report
from next_poi.models import SmokePredictor, time_bucket


@dataclass(frozen=True)
class EvaluationResult:
    report: EvaluationReport
    observations: tuple[RankingObservation, ...]
    prediction_count: int


@dataclass(frozen=True)
class _PredictionInput:
    sample_id: str
    request: RecommendationRequest


@dataclass(frozen=True)
class _PredictionSnapshot:
    predictions: tuple[str, ...]
    candidates: tuple[str, ...]
    candidate_sources: tuple[tuple[str, ...], ...]
    macro_predictions: tuple[str, ...]
    latency_ms: float


def evaluate_examples(
    predictor: SmokePredictor,
    examples: tuple[LabeledExample, ...],
    *,
    data_fingerprint: str,
) -> EvaluationResult:
    """Predict every target-blind request before accessing any label field."""

    ordered_examples = tuple(sorted(examples, key=lambda item: item.sample_id))
    sample_ids = tuple(example.sample_id for example in ordered_examples)
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("evaluation sample_id values must be unique")
    if any(example.request.dataset != predictor.index.dataset for example in ordered_examples):
        raise ValueError("evaluation dataset does not match predictor index")

    prediction_inputs = tuple(
        _PredictionInput(sample_id=example.sample_id, request=example.request)
        for example in ordered_examples
    )
    started = perf_counter()
    traces = _predict_without_labels(predictor, prediction_inputs)

    # This is the first target-aware stage. Prediction above receives only
    # RecommendationRequest objects and completes for the whole batch first.
    observations = tuple(
        _attach_label(example, traces[example.sample_id]) for example in ordered_examples
    )
    catalog = tuple(predictor.index.global_counts)
    metrics = aggregate_ranking_metrics(observations, catalog=catalog)
    slices = _build_slices(ordered_examples, observations, predictor, catalog)
    failure_cases = _failure_cases(ordered_examples, observations, traces, catalog)
    elapsed_ms = (perf_counter() - started) * 1000.0
    latencies = tuple(
        traces[sample_id].latency_ms for sample_id in sample_ids
    )
    runtime = {
        "evaluation_duration_ms": elapsed_ms,
        "prediction_latency_ms": _latency_summary(latencies),
    }
    report = build_report(
        dataset=predictor.index.dataset,
        variant=predictor.variant,
        model_version=predictor.versions.model,
        data_fingerprint=data_fingerprint,
        sample_ids=sample_ids,
        metrics=metrics,
        slices=slices,
        failure_cases=failure_cases,
        runtime=runtime,
    )
    return EvaluationResult(
        report=report,
        observations=observations,
        prediction_count=len(traces),
    )


def _predict_without_labels(
    predictor: SmokePredictor,
    items: tuple[_PredictionInput, ...],
) -> dict[str, _PredictionSnapshot]:
    snapshots: dict[str, _PredictionSnapshot] = {}
    for item in items:
        trace = predictor.predict_with_trace(item.request)
        snapshots[item.sample_id] = _PredictionSnapshot(
            predictions=tuple(
                recommendation.poi_id
                for recommendation in trace.response.recommendations
            ),
            candidates=tuple(candidate.poi_id for candidate in trace.candidates),
            candidate_sources=tuple(
                candidate.candidate_sources for candidate in trace.candidates
            ),
            macro_predictions=tuple(
                category.category for category in trace.response.macro
            ),
            latency_ms=trace.response.latency.total_ms,
        )
    return snapshots


def _attach_label(
    example: LabeledExample,
    trace: _PredictionSnapshot,
) -> RankingObservation:
    return RankingObservation(
        sample_id=example.sample_id,
        target_poi_id=example.target_poi_id,
        predictions=trace.predictions,
        candidates=trace.candidates,
        target_category=example.target_category,
        macro_predictions=trace.macro_predictions,
    )


def _build_slices(
    examples: tuple[LabeledExample, ...],
    observations: tuple[RankingObservation, ...],
    predictor: SmokePredictor,
    catalog: tuple[str, ...],
) -> dict[str, object]:
    dimensions: dict[str, dict[str, list[RankingObservation]]] = {
        "history_length": {},
        "target_rarity": {},
        "day_type": {},
    }
    for example, observation in zip(examples, observations, strict=True):
        keys = {
            "history_length": _history_length_bucket(len(example.request.history)),
            "target_rarity": _rarity_bucket(
                predictor.index.global_counts.get(example.target_poi_id, 0)
            ),
            "day_type": time_bucket(example.request.target_time).split(":", 1)[0],
        }
        for dimension, key in keys.items():
            dimensions[dimension].setdefault(key, []).append(observation)

    return {
        dimension: {
            key: {
                "count": len(rows),
                "metrics": aggregate_ranking_metrics(rows, catalog=catalog),
            }
            for key, rows in sorted(groups.items())
        }
        for dimension, groups in dimensions.items()
    }


def _failure_cases(
    examples: tuple[LabeledExample, ...],
    observations: tuple[RankingObservation, ...],
    traces: dict[str, _PredictionSnapshot],
    catalog: tuple[str, ...],
) -> tuple[dict[str, object], ...]:
    catalog_set = set(catalog)
    failures: list[dict[str, object]] = []
    for example, observation in zip(examples, observations, strict=True):
        reasons: list[str] = []
        if not observation.predictions:
            reasons.append("empty_predictions")
        if len(observation.predictions) != len(set(observation.predictions)):
            reasons.append("duplicate_predictions")
        if any(poi_id not in catalog_set for poi_id in observation.predictions):
            reasons.append("invalid_predictions")
        if observation.target_poi_id not in observation.predictions[:10]:
            reasons.append("target_miss_at_10")
        if not reasons:
            continue
        trace = traces[example.sample_id]
        target_sources = next(
            (
                sources
                for poi_id, sources in zip(
                    trace.candidates, trace.candidate_sources, strict=True
                )
                if poi_id == observation.target_poi_id
            ),
            (),
        )
        failures.append(
            {
                "sample_id": example.sample_id,
                "reasons": reasons,
                "target_poi_id": example.target_poi_id,
                "target_category": example.target_category,
                "predictions": list(observation.predictions[:10]),
                "target_candidate_sources": list(target_sources),
            }
        )
    return tuple(sorted(failures, key=lambda item: str(item["sample_id"])))


def _history_length_bucket(length: int) -> str:
    if length == 1:
        return "1"
    if length <= 4:
        return "2-4"
    if length <= 9:
        return "5-9"
    return "10+"


def _rarity_bucket(train_count: int) -> str:
    if train_count == 0:
        return "unseen"
    if train_count == 1:
        return "rare"
    return "frequent"


def _latency_summary(values: tuple[float, ...]) -> dict[str, float]:
    if not values:
        return {"count": 0.0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    ordered = tuple(sorted(values))
    return {
        "count": float(len(ordered)),
        "mean": sum(ordered) / len(ordered),
        "p50": _percentile(ordered, 0.50),
        "p95": _percentile(ordered, 0.95),
        "max": ordered[-1],
    }


def _percentile(values: tuple[float, ...], quantile: float) -> float:
    position = (len(values) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction
