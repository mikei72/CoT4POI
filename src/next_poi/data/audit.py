"""Read-only split overlap, range, order, and count audits."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

from next_poi.contracts import NormalizedEvent, SplitName
from next_poi.data._serialization import canonical_timestamp
from next_poi.data.readers import SPLIT_ORDER


def _identity_hash(parts: tuple[str, ...]) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def _event_identity(event: NormalizedEvent) -> str:
    return _identity_hash(
        (
            event.dataset,
            event.raw_user_id,
            canonical_timestamp(event.timestamp_utc),
            event.raw_poi_id,
        )
    )


def _session_identity(event: NormalizedEvent) -> str:
    return _identity_hash((event.dataset, event.raw_user_id, event.session_id))


@dataclass(frozen=True)
class SplitAuditSummary:
    split: SplitName
    event_count: int
    session_count: int
    min_timestamp: datetime | None
    max_timestamp: datetime | None
    input_time_ordered: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "split": self.split,
            "event_count": self.event_count,
            "session_count": self.session_count,
            "min_timestamp": (
                canonical_timestamp(self.min_timestamp) if self.min_timestamp is not None else None
            ),
            "max_timestamp": (
                canonical_timestamp(self.max_timestamp) if self.max_timestamp is not None else None
            ),
            "input_time_ordered": self.input_time_ordered,
        }


@dataclass(frozen=True)
class OverlapAudit:
    left_split: SplitName
    right_split: SplitName
    count: int
    identity_sha256: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "left_split": self.left_split,
            "right_split": self.right_split,
            "count": self.count,
            "identity_sha256": list(self.identity_sha256),
        }


@dataclass(frozen=True)
class TemporalBoundaryAudit:
    earlier_split: SplitName
    later_split: SplitName
    ordered: bool | None

    def to_dict(self) -> dict[str, object]:
        return {
            "earlier_split": self.earlier_split,
            "later_split": self.later_split,
            "ordered": self.ordered,
        }


@dataclass(frozen=True)
class SplitAuditReport:
    summaries: tuple[SplitAuditSummary, ...]
    event_overlaps: tuple[OverlapAudit, ...]
    session_overlaps: tuple[OverlapAudit, ...]
    temporal_boundaries: tuple[TemporalBoundaryAudit, ...]

    @property
    def has_event_overlap(self) -> bool:
        return any(item.count for item in self.event_overlaps)

    @property
    def has_session_overlap(self) -> bool:
        return any(item.count for item in self.session_overlaps)

    @property
    def temporal_splits_ordered(self) -> bool | None:
        values = [item.ordered for item in self.temporal_boundaries if item.ordered is not None]
        return all(values) if values else None

    def to_dict(self) -> dict[str, object]:
        return {
            "summaries": [item.to_dict() for item in self.summaries],
            "event_overlaps": [item.to_dict() for item in self.event_overlaps],
            "session_overlaps": [item.to_dict() for item in self.session_overlaps],
            "temporal_boundaries": [item.to_dict() for item in self.temporal_boundaries],
            "has_event_overlap": self.has_event_overlap,
            "has_session_overlap": self.has_session_overlap,
            "temporal_splits_ordered": self.temporal_splits_ordered,
        }


def audit_splits(
    splits: Mapping[SplitName, tuple[NormalizedEvent, ...] | list[NormalizedEvent]],
) -> SplitAuditReport:
    """Inspect explicit splits without sorting, filtering, or repartitioning them."""

    keys = set(splits)
    expected = set(SPLIT_ORDER)
    if keys != expected:
        raise ValueError(
            f"audit requires explicit train/validation/test splits; "
            f"missing={sorted(expected - keys)}, extra={sorted(keys - expected)}"
        )

    event_ids: dict[SplitName, set[str]] = {}
    session_ids: dict[SplitName, set[str]] = {}
    summaries: list[SplitAuditSummary] = []
    for split in SPLIT_ORDER:
        events = tuple(splits[split])
        mismatches = [event.split for event in events if event.split != split]
        if mismatches:
            raise ValueError(
                f"events labeled {mismatches[0]!r} were supplied under split {split!r}"
            )
        timestamps = [event.timestamp_utc for event in events]
        event_ids[split] = {_event_identity(event) for event in events}
        session_ids[split] = {_session_identity(event) for event in events}
        summaries.append(
            SplitAuditSummary(
                split=split,
                event_count=len(events),
                session_count=len(session_ids[split]),
                min_timestamp=min(timestamps) if timestamps else None,
                max_timestamp=max(timestamps) if timestamps else None,
                input_time_ordered=all(
                    left <= right
                    for left, right in zip(timestamps, timestamps[1:], strict=False)
                ),
            )
        )

    def overlaps(identities: Mapping[SplitName, set[str]]) -> tuple[OverlapAudit, ...]:
        records: list[OverlapAudit] = []
        for left_index, left in enumerate(SPLIT_ORDER):
            for right in SPLIT_ORDER[left_index + 1 :]:
                shared = tuple(sorted(identities[left] & identities[right]))
                records.append(OverlapAudit(left, right, len(shared), shared))
        return tuple(records)

    summary_by_split = {item.split: item for item in summaries}
    boundaries: list[TemporalBoundaryAudit] = []
    for earlier, later in zip(SPLIT_ORDER, SPLIT_ORDER[1:], strict=False):
        earlier_max = summary_by_split[earlier].max_timestamp
        later_min = summary_by_split[later].min_timestamp
        ordered = None if earlier_max is None or later_min is None else earlier_max <= later_min
        boundaries.append(TemporalBoundaryAudit(earlier, later, ordered))

    return SplitAuditReport(
        summaries=tuple(summaries),
        event_overlaps=overlaps(event_ids),
        session_overlaps=overlaps(session_ids),
        temporal_boundaries=tuple(boundaries),
    )
