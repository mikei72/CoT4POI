"""Deterministic aggregate distribution drift for unlabeled monitoring windows."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from next_poi.monitoring.events import MonitoringEvent

_EPSILON = 1e-6


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DriftThresholds(_StrictFrozenModel):
    js_divergence: float = Field(default=0.10, ge=0, allow_inf_nan=False)
    psi: float = Field(default=0.20, ge=0, allow_inf_nan=False)


class DistributionComparison(_StrictFrozenModel):
    reference: dict[str, float]
    current: dict[str, float]
    js_divergence: float = Field(ge=0, allow_inf_nan=False)
    psi: float = Field(ge=0, allow_inf_nan=False)
    alert: bool


class NumericSummary(_StrictFrozenModel):
    count: int = Field(ge=0)
    mean: float | None = Field(default=None, allow_inf_nan=False)
    minimum: float | None = Field(default=None, allow_inf_nan=False)
    maximum: float | None = Field(default=None, allow_inf_nan=False)
    p50: float | None = Field(default=None, allow_inf_nan=False)
    p95: float | None = Field(default=None, allow_inf_nan=False)


class NumericComparison(_StrictFrozenModel):
    reference: NumericSummary
    current: NumericSummary


class DriftReport(_StrictFrozenModel):
    """Unlabeled system/input/output drift report; intentionally has no accuracy."""

    schema_version: str = "monitoring-drift-v1"
    reference_event_count: int = Field(ge=1)
    current_event_count: int = Field(ge=1)
    thresholds: DriftThresholds
    distributions: dict[str, DistributionComparison]
    numeric_summaries: dict[str, NumericComparison]
    overall_alert: bool


def _validated_counts(counts: Mapping[str, int | float]) -> dict[str, float]:
    validated: dict[str, float] = {}
    for key, raw_value in counts.items():
        value = float(raw_value)
        if not key or not math.isfinite(value) or value < 0:
            raise ValueError(
                "distribution counts must use non-empty keys and finite non-negative values"
            )
        validated[key] = value
    return validated


def normalize_distribution(counts: Mapping[str, int | float]) -> dict[str, float]:
    """Normalize counts in sorted-key order for stable reports."""

    values = _validated_counts(counts)
    total = sum(values.values())
    return {
        key: (values[key] / total if total else 0.0)
        for key in sorted(values)
    }


def js_divergence(
    reference: Mapping[str, int | float], current: Mapping[str, int | float]
) -> float:
    """Return natural-log Jensen-Shannon divergence on the union support."""

    reference_probability = normalize_distribution(reference)
    current_probability = normalize_distribution(current)
    keys = sorted(set(reference_probability) | set(current_probability))
    divergence = 0.0
    for key in keys:
        ref = reference_probability.get(key, 0.0)
        cur = current_probability.get(key, 0.0)
        midpoint = 0.5 * (ref + cur)
        if ref > 0:
            divergence += 0.5 * ref * math.log(ref / midpoint)
        if cur > 0:
            divergence += 0.5 * cur * math.log(cur / midpoint)
    return max(0.0, divergence)


def population_stability_index(
    reference: Mapping[str, int | float],
    current: Mapping[str, int | float],
    *,
    epsilon: float = _EPSILON,
) -> float:
    """Return deterministic PSI with explicit smoothing for missing bins."""

    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be a finite positive value")
    reference_probability = normalize_distribution(reference)
    current_probability = normalize_distribution(current)
    keys = sorted(set(reference_probability) | set(current_probability))
    result = 0.0
    for key in keys:
        ref = max(reference_probability.get(key, 0.0), epsilon)
        cur = max(current_probability.get(key, 0.0), epsilon)
        result += (cur - ref) * math.log(cur / ref)
    return max(0.0, result)


def _small_count_bucket(value: int) -> str:
    if value == 0:
        return "0"
    if value == 1:
        return "1"
    if value <= 5:
        return "2-5"
    return "6+"


def _candidate_count_bucket(value: int) -> str:
    if value == 0:
        return "0"
    if value <= 10:
        return "1-10"
    if value <= 25:
        return "11-25"
    if value <= 50:
        return "26-50"
    if value <= 100:
        return "51-100"
    return "101+"


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def _numeric_summary(values: Iterable[float]) -> NumericSummary:
    ordered = sorted(float(value) for value in values)
    if any(not math.isfinite(value) for value in ordered):
        raise ValueError("numeric summary values must be finite")
    if not ordered:
        return NumericSummary(count=0)
    return NumericSummary(
        count=len(ordered),
        mean=sum(ordered) / len(ordered),
        minimum=ordered[0],
        maximum=ordered[-1],
        p50=_quantile(ordered, 0.50),
        p95=_quantile(ordered, 0.95),
    )


def _aggregate_distributions(events: Sequence[MonitoringEvent]) -> dict[str, Counter[str]]:
    distributions = {
        "history_length": Counter(event.history_length_bucket for event in events),
        "unknown_count": Counter(_small_count_bucket(event.unknown_count) for event in events),
        "candidate_count": Counter(
            _candidate_count_bucket(event.candidate_count) for event in events
        ),
        "candidate_source": Counter(),
        "category": Counter(),
        "status": Counter(event.status for event in events),
    }
    for event in events:
        distributions["candidate_source"].update(event.candidate_source_histogram)
        distributions["category"].update(event.category_histogram)
    return distributions


def _aggregate_numeric(events: Sequence[MonitoringEvent]) -> dict[str, NumericSummary]:
    values: dict[str, list[float]] = {
        "unknown_count": [float(event.unknown_count) for event in events],
        "candidate_count": [float(event.candidate_count) for event in events],
        "score_entropy": [event.score_entropy for event in events],
        "total_latency_ms": [event.total_latency_ms for event in events],
    }
    for event in events:
        for stage, latency in event.stage_latency_ms.items():
            values.setdefault(f"stage_latency_ms.{stage}", []).append(latency)
    return {name: _numeric_summary(items) for name, items in sorted(values.items())}


def build_drift_report(
    reference: Sequence[MonitoringEvent],
    current: Sequence[MonitoringEvent],
    thresholds: DriftThresholds | None = None,
) -> DriftReport:
    """Aggregate two unlabeled windows and compare their safe distributions."""

    if not reference:
        raise ValueError("reference monitoring window must contain at least one event")
    if not current:
        raise ValueError("current monitoring window must contain at least one event")
    reference = tuple(
        MonitoringEvent.model_validate(event.model_dump(mode="json")) for event in reference
    )
    current = tuple(
        MonitoringEvent.model_validate(event.model_dump(mode="json")) for event in current
    )
    applied_thresholds = thresholds or DriftThresholds()
    reference_distributions = _aggregate_distributions(reference)
    current_distributions = _aggregate_distributions(current)

    comparisons: dict[str, DistributionComparison] = {}
    for name in sorted(reference_distributions):
        reference_counts = reference_distributions[name]
        current_counts = current_distributions[name]
        js_value = js_divergence(reference_counts, current_counts)
        psi_value = population_stability_index(reference_counts, current_counts)
        comparisons[name] = DistributionComparison(
            reference=normalize_distribution(reference_counts),
            current=normalize_distribution(current_counts),
            js_divergence=js_value,
            psi=psi_value,
            alert=(
                js_value > applied_thresholds.js_divergence
                or psi_value > applied_thresholds.psi
            ),
        )

    reference_numeric = _aggregate_numeric(reference)
    current_numeric = _aggregate_numeric(current)
    numeric_names = sorted(set(reference_numeric) | set(current_numeric))
    empty_summary = NumericSummary(count=0)
    numeric_comparisons = {
        name: NumericComparison(
            reference=reference_numeric.get(name, empty_summary),
            current=current_numeric.get(name, empty_summary),
        )
        for name in numeric_names
    }
    return DriftReport(
        reference_event_count=len(reference),
        current_event_count=len(current),
        thresholds=applied_thresholds,
        distributions=comparisons,
        numeric_summaries=numeric_comparisons,
        overall_alert=any(comparison.alert for comparison in comparisons.values()),
    )
