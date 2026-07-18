"""Train-only candidate statistics for the deterministic CPU smoke backend."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from next_poi.contracts import DatasetName, NormalizedEvent
from next_poi.data import hash_events

INDEX_SCHEMA_VERSION = "1"
TIME_BUCKET_VERSION = "utc-weekpart-four-period-v1"
TIME_BUCKETS = frozenset(
    f"{day_type}:{period}"
    for day_type in ("weekday", "weekend")
    for period in ("morning", "afternoon", "evening", "night")
)


def time_bucket(timestamp: datetime) -> str:
    """Return the frozen weekday/weekend and four-period bucket."""

    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("time bucket requires a timezone-aware timestamp")
    timestamp = timestamp.astimezone(timezone.utc)
    day_type = "weekend" if timestamp.weekday() >= 5 else "weekday"
    hour = timestamp.hour
    if 5 <= hour < 12:
        period = "morning"
    elif 12 <= hour < 17:
        period = "afternoon"
    elif 17 <= hour < 22:
        period = "evening"
    else:
        period = "night"
    return f"{day_type}:{period}"


def _stable_nested_counts(
    values: Mapping[str, Mapping[str, int]],
) -> dict[str, dict[str, int]]:
    return {
        outer: {inner: int(count) for inner, count in sorted(inner_values.items())}
        for outer, inner_values in sorted(values.items())
    }


@dataclass(frozen=True)
class CandidateIndex:
    schema_version: str
    dataset: DatasetName
    train_data_sha256: str
    taxonomy: tuple[str, ...]
    poi_categories: dict[str, str]
    model_poi_ids: dict[str, int | None]
    global_counts: dict[str, int]
    time_counts: dict[str, dict[str, int]]
    transition_counts: dict[str, dict[str, int]]
    category_counts: dict[str, dict[str, int]]

    def __post_init__(self) -> None:
        _validate_candidate_index(self)

    @classmethod
    def fit(
        cls,
        events: Iterable[NormalizedEvent],
        *,
        taxonomy: Iterable[str],
    ) -> CandidateIndex:
        train_events = tuple(events)
        if not train_events:
            raise ValueError("candidate index requires at least one train event")
        if any(event.split != "train" for event in train_events):
            raise ValueError("candidate index fit accepts train events only")
        datasets = {event.dataset for event in train_events}
        if len(datasets) != 1:
            raise ValueError("candidate index fit requires exactly one dataset")

        known_taxonomy = tuple(sorted(set(taxonomy)))
        if not known_taxonomy or any(not category for category in known_taxonomy):
            raise ValueError("taxonomy must contain non-empty categories")
        unknown_categories = sorted(
            {event.category for event in train_events} - set(known_taxonomy)
        )
        if unknown_categories:
            raise ValueError(f"train categories are outside taxonomy: {unknown_categories}")

        global_counts: Counter[str] = Counter()
        time_counts: dict[str, Counter[str]] = defaultdict(Counter)
        category_counts: dict[str, Counter[str]] = defaultdict(Counter)
        poi_category_votes: dict[str, Counter[str]] = defaultdict(Counter)
        poi_model_votes: dict[str, Counter[int]] = defaultdict(Counter)
        sessions: dict[tuple[str, str], list[NormalizedEvent]] = defaultdict(list)

        for event in train_events:
            global_counts[event.raw_poi_id] += 1
            time_counts[time_bucket(event.timestamp_utc)][event.raw_poi_id] += 1
            category_counts[event.category][event.raw_poi_id] += 1
            poi_category_votes[event.raw_poi_id][event.category] += 1
            if event.model_poi_id is not None:
                poi_model_votes[event.raw_poi_id][event.model_poi_id] += 1
            sessions[(event.raw_user_id, event.session_id)].append(event)

        transition_counts: dict[str, Counter[str]] = defaultdict(Counter)
        for session_events in sessions.values():
            ordered = sorted(
                session_events,
                key=lambda event: (
                    event.timestamp_utc,
                    event.raw_poi_id,
                    event.category,
                ),
            )
            for previous, current in zip(ordered, ordered[1:], strict=False):
                transition_counts[previous.raw_poi_id][current.raw_poi_id] += 1

        def voted_value(counter: Counter[Any]) -> Any:
            return min(counter, key=lambda value: (-counter[value], value))

        poi_categories = {
            poi_id: voted_value(poi_category_votes[poi_id])
            for poi_id in sorted(global_counts)
        }
        model_poi_ids = {
            poi_id: (
                voted_value(poi_model_votes[poi_id]) if poi_model_votes[poi_id] else None
            )
            for poi_id in sorted(global_counts)
        }
        return cls(
            schema_version=INDEX_SCHEMA_VERSION,
            dataset=next(iter(datasets)),
            train_data_sha256=hash_events(train_events),
            taxonomy=known_taxonomy,
            poi_categories=poi_categories,
            model_poi_ids=model_poi_ids,
            global_counts=dict(sorted(global_counts.items())),
            time_counts=_stable_nested_counts(time_counts),
            transition_counts=_stable_nested_counts(transition_counts),
            category_counts=_stable_nested_counts(category_counts),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "dataset": self.dataset,
            "train_data_sha256": self.train_data_sha256,
            "taxonomy": list(self.taxonomy),
            "poi_categories": dict(sorted(self.poi_categories.items())),
            "model_poi_ids": dict(sorted(self.model_poi_ids.items())),
            "global_counts": dict(sorted(self.global_counts.items())),
            "time_counts": _stable_nested_counts(self.time_counts),
            "transition_counts": _stable_nested_counts(self.transition_counts),
            "category_counts": _stable_nested_counts(self.category_counts),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> CandidateIndex:
        required_fields = {
            "schema_version",
            "dataset",
            "train_data_sha256",
            "taxonomy",
            "poi_categories",
            "model_poi_ids",
            "global_counts",
            "time_counts",
            "transition_counts",
            "category_counts",
        }
        if set(payload) != required_fields:
            raise ValueError("candidate index fields are invalid")
        taxonomy = payload["taxonomy"]
        if not isinstance(taxonomy, list):
            raise ValueError("candidate index taxonomy must be a string list")
        mapping_fields = {
            name: payload[name]
            for name in (
                "poi_categories",
                "model_poi_ids",
                "global_counts",
                "time_counts",
                "transition_counts",
                "category_counts",
            )
        }
        if any(not isinstance(value, dict) for value in mapping_fields.values()):
            raise ValueError("candidate index mappings must be JSON objects")
        for name in ("time_counts", "transition_counts", "category_counts"):
            if any(not isinstance(value, dict) for value in mapping_fields[name].values()):
                raise ValueError(f"candidate index {name} values must be JSON objects")
        return cls(
            schema_version=payload["schema_version"],
            dataset=payload["dataset"],
            train_data_sha256=payload["train_data_sha256"],
            taxonomy=tuple(taxonomy),
            poi_categories=dict(mapping_fields["poi_categories"]),
            model_poi_ids=dict(mapping_fields["model_poi_ids"]),
            global_counts=dict(mapping_fields["global_counts"]),
            time_counts={
                outer: dict(inner)
                for outer, inner in mapping_fields["time_counts"].items()
            },
            transition_counts={
                outer: dict(inner)
                for outer, inner in mapping_fields["transition_counts"].items()
            },
            category_counts={
                outer: dict(inner)
                for outer, inner in mapping_fields["category_counts"].items()
            },
        )


def _validate_candidate_index(index: CandidateIndex) -> None:
    if index.schema_version != INDEX_SCHEMA_VERSION:
        raise ValueError("unsupported candidate index schema_version")
    if index.dataset not in {"nyc", "tky", "ca", "synthetic"}:
        raise ValueError("candidate index contains an unsupported dataset")
    if not _is_sha256(index.train_data_sha256):
        raise ValueError("candidate index train_data_sha256 is invalid")
    if (
        not index.taxonomy
        or any(not isinstance(category, str) or not category for category in index.taxonomy)
        or tuple(sorted(set(index.taxonomy))) != index.taxonomy
    ):
        raise ValueError("candidate index taxonomy must be sorted, unique, and non-empty")

    _validate_flat_counts(index.global_counts, "global_counts")
    catalog = set(index.global_counts)
    if not catalog:
        raise ValueError("candidate index catalog must be non-empty")
    if set(index.poi_categories) != catalog or set(index.model_poi_ids) != catalog:
        raise ValueError("candidate index catalog mappings must have identical POI keys")
    taxonomy = set(index.taxonomy)
    if any(
        not isinstance(category, str) or not category or category not in taxonomy
        for category in index.poi_categories.values()
    ):
        raise ValueError("candidate index POI categories must belong to taxonomy")
    model_ids = tuple(value for value in index.model_poi_ids.values() if value is not None)
    if any(not _is_nonnegative_int(value) for value in model_ids):
        raise ValueError("candidate index model POI IDs must be non-negative integers")
    if len(model_ids) != len(set(model_ids)):
        raise ValueError("candidate index model POI IDs must be unique")

    _validate_nested_counts(index.time_counts, "time_counts", catalog)
    _validate_nested_counts(index.transition_counts, "transition_counts", catalog)
    _validate_nested_counts(index.category_counts, "category_counts", catalog)
    if not set(index.time_counts).issubset(TIME_BUCKETS):
        raise ValueError("candidate index contains an unsupported time bucket")
    if not set(index.transition_counts).issubset(catalog):
        raise ValueError("candidate index transition sources must belong to the catalog")
    if not set(index.category_counts).issubset(taxonomy):
        raise ValueError("candidate index category counts must belong to taxonomy")

    time_totals = _sum_nested_by_poi(index.time_counts, catalog)
    category_totals = _sum_nested_by_poi(index.category_counts, catalog)
    if time_totals != index.global_counts:
        raise ValueError("candidate index time counts must match global counts")
    if category_totals != index.global_counts:
        raise ValueError("candidate index category counts must match global counts")
    for poi_id, category in index.poi_categories.items():
        votes = {
            candidate_category: counts.get(poi_id, 0)
            for candidate_category, counts in index.category_counts.items()
        }
        expected = min(votes, key=lambda value: (-votes[value], value))
        if not votes[expected] or category != expected:
            raise ValueError("candidate index POI category mapping is inconsistent")

    incoming = {poi_id: 0 for poi_id in catalog}
    for source, counts in index.transition_counts.items():
        if sum(counts.values()) > index.global_counts[source]:
            raise ValueError("candidate index outgoing transition counts are inconsistent")
        for target, count in counts.items():
            incoming[target] += count
    if any(incoming[poi_id] > index.global_counts[poi_id] for poi_id in catalog):
        raise ValueError("candidate index incoming transition counts are inconsistent")


def _validate_flat_counts(values: Mapping[object, object], label: str) -> None:
    if not isinstance(values, dict):
        raise ValueError(f"candidate index {label} must be a JSON object")
    if any(not isinstance(key, str) or not key for key in values):
        raise ValueError(f"candidate index {label} keys must be non-empty strings")
    if any(not _is_positive_int(value) for value in values.values()):
        raise ValueError(f"candidate index {label} values must be positive integers")


def _validate_nested_counts(
    values: Mapping[object, object],
    label: str,
    catalog: set[str],
) -> None:
    if not isinstance(values, dict):
        raise ValueError(f"candidate index {label} must be a JSON object")
    if any(not isinstance(key, str) or not key for key in values):
        raise ValueError(f"candidate index {label} keys must be non-empty strings")
    for counts in values.values():
        if not isinstance(counts, dict) or not counts:
            raise ValueError(f"candidate index {label} values must be non-empty JSON objects")
        _validate_flat_counts(counts, label)
        if not set(counts).issubset(catalog):
            raise ValueError(f"candidate index {label} POIs must belong to the catalog")


def _sum_nested_by_poi(
    values: Mapping[str, Mapping[str, int]],
    catalog: set[str],
) -> dict[str, int]:
    totals = {poi_id: 0 for poi_id in catalog}
    for counts in values.values():
        for poi_id, count in counts.items():
            totals[poi_id] += count
    return totals


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
