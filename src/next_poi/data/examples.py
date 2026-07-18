"""Stable offline example identity and deterministic session-to-label conversion."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime

from next_poi.contracts import HistoryEvent, LabeledExample, NormalizedEvent, RecommendationRequest
from next_poi.data._serialization import canonical_timestamp


def compute_sample_id(
    *,
    dataset: str,
    split_protocol: str,
    raw_user_id: str,
    session_id: str,
    target_timestamp_utc: datetime,
    target_raw_poi_id: str,
) -> str:
    """Hash the documented six-field sample-ID preimage exactly once."""

    fields = (
        dataset,
        split_protocol,
        raw_user_id,
        session_id,
        canonical_timestamp(target_timestamp_utc),
        target_raw_poi_id,
    )
    if any(not field for field in fields[:-2]) or not target_raw_poi_id:
        raise ValueError("sample-ID identity fields must be non-empty")
    preimage = "|".join(fields)
    return hashlib.sha256(preimage.encode("utf-8")).hexdigest()


def _event_sort_key(event: NormalizedEvent) -> tuple[datetime, str, str, int]:
    return (
        event.timestamp_utc,
        event.raw_poi_id,
        event.category,
        -1 if event.model_poi_id is None else event.model_poi_id,
    )


def build_labeled_examples(
    events: Iterable[NormalizedEvent],
    *,
    split_protocol: str,
    min_history: int = 1,
    top_k: int = 10,
) -> tuple[LabeledExample, ...]:
    """Build examples by stable session identity, never by input list position."""

    if not split_protocol:
        raise ValueError("split_protocol must be non-empty")
    if min_history < 1:
        raise ValueError("min_history must be at least one")
    grouped: dict[tuple[str, str, str, str], list[NormalizedEvent]] = defaultdict(list)
    for event in events:
        key = (event.dataset, event.split, event.raw_user_id, event.session_id)
        grouped[key].append(event)

    examples: list[LabeledExample] = []
    seen_sample_ids: set[str] = set()
    for key in sorted(grouped):
        dataset, split, raw_user_id, session_id = key
        session_events = sorted(grouped[key], key=_event_sort_key)
        for target_index in range(min_history, len(session_events)):
            target = session_events[target_index]
            history_start = max(0, target_index - 128)
            history = tuple(
                HistoryEvent(
                    poi_id=item.raw_poi_id,
                    model_poi_id=item.model_poi_id,
                    category_name=item.category,
                    timestamp=item.timestamp_utc,
                    latitude=item.latitude,
                    longitude=item.longitude,
                )
                for item in session_events[history_start:target_index]
            )
            sample_id = compute_sample_id(
                dataset=dataset,
                split_protocol=split_protocol,
                raw_user_id=raw_user_id,
                session_id=session_id,
                target_timestamp_utc=target.timestamp_utc,
                target_raw_poi_id=target.raw_poi_id,
            )
            if sample_id in seen_sample_ids:
                raise ValueError(
                    "duplicate sample identity: stable identity fields do not uniquely identify "
                    f"target {target.raw_poi_id!r} at {canonical_timestamp(target.timestamp_utc)}"
                )
            seen_sample_ids.add(sample_id)
            examples.append(
                LabeledExample(
                    sample_id=sample_id,
                    split=split,
                    request=RecommendationRequest(
                        dataset=dataset,
                        history=history,
                        target_time=target.timestamp_utc,
                        top_k=top_k,
                    ),
                    target_poi_id=target.raw_poi_id,
                    target_model_poi_id=target.model_poi_id,
                    target_category=target.category,
                )
            )
    return tuple(sorted(examples, key=lambda item: item.sample_id))
