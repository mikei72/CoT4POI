"""Explicit split readers for repository-safe synthetic and legacy NYC CSVs."""

from __future__ import annotations

import csv
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

from next_poi.contracts import NormalizedEvent, SplitName

SPLIT_ORDER: tuple[SplitName, ...] = ("train", "validation", "test")


def _parse_timestamp(value: str) -> datetime:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError("timestamp must be non-empty")
    try:
        timestamp = datetime.fromisoformat(cleaned.replace("Z", "+00:00"))
    except ValueError:
        try:
            timestamp = datetime.strptime(cleaned, "%a %b %d %H:%M:%S %z %Y")
        except ValueError as exc:
            raise ValueError(f"unsupported timestamp format: {cleaned!r}") from exc
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("timestamp must include a timezone offset")
    return timestamp.astimezone(timezone.utc)


def _first(row: Mapping[str, str | None], names: tuple[str, ...], logical_name: str) -> str:
    for name in names:
        value = row.get(name)
        if value is not None and value.strip():
            return value.strip()
    raise ValueError(f"missing required column/value for {logical_name}: expected one of {names}")


def _optional(row: Mapping[str, str | None], names: tuple[str, ...]) -> str | None:
    for name in names:
        value = row.get(name)
        if value is not None and value.strip():
            return value.strip()
    return None


def _optional_int(row: Mapping[str, str | None], names: tuple[str, ...]) -> int | None:
    value = _optional(row, names)
    if value is None:
        return None
    parsed = int(value)
    if parsed < 0:
        raise ValueError("model POI ID must be non-negative")
    return parsed


def _coordinates(row: Mapping[str, str | None]) -> tuple[float | None, float | None]:
    latitude = _optional(row, ("latitude", "Latitude"))
    longitude = _optional(row, ("longitude", "Longitude"))
    if (latitude is None) != (longitude is None):
        raise ValueError("latitude and longitude must be provided together")
    if latitude is None:
        return None, None
    return float(latitude), float(cast(str, longitude))


def _validate_declared_split(row: Mapping[str, str | None], split: SplitName) -> None:
    declared = _optional(row, ("split", "SplitTag"))
    if declared is None:
        return
    normalized = "validation" if declared == "val" else declared
    if normalized != split:
        raise ValueError(
            f"row declares split {declared!r}, but caller supplied explicit split {split!r}"
        )


def _read_rows(
    path: str | Path,
    split: SplitName,
    normalizer: Callable[[Mapping[str, str | None], SplitName], NormalizedEvent],
) -> tuple[NormalizedEvent, ...]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"required split file not found: {source}")
    events: list[NormalizedEvent] = []
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {source}")
        for line_number, row in enumerate(reader, start=2):
            try:
                _validate_declared_split(row, split)
                events.append(normalizer(row, split))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid row at {source}:{line_number}: {exc}") from exc
    return tuple(events)


def _synthetic_event(
    row: Mapping[str, str | None], split: SplitName
) -> NormalizedEvent:
    latitude, longitude = _coordinates(row)
    if latitude is not None or longitude is not None:
        raise ValueError("synthetic data must not include coordinates")
    return NormalizedEvent(
        dataset="synthetic",
        split=split,
        raw_user_id=_first(row, ("raw_user_id",), "raw_user_id"),
        session_id=_first(row, ("session_id",), "session_id"),
        timestamp_utc=_parse_timestamp(
            _first(row, ("timestamp", "timestamp_utc"), "timestamp")
        ),
        raw_poi_id=_first(row, ("raw_poi_id",), "raw_poi_id"),
        model_poi_id=_optional_int(row, ("model_poi_id",)),
        category=_first(row, ("category",), "category"),
    )


def _nyc_event(row: Mapping[str, str | None], split: SplitName) -> NormalizedEvent:
    latitude, longitude = _coordinates(row)
    return NormalizedEvent(
        dataset="nyc",
        split=split,
        raw_user_id=_first(row, ("raw_user_id", "UserId", "user_id"), "raw_user_id"),
        session_id=_first(
            row,
            ("session_id", "pseudo_session_trajectory_id", "trajectory_id"),
            "session_id",
        ),
        timestamp_utc=_parse_timestamp(
            _first(row, ("timestamp", "timestamp_utc", "UTCTime"), "timestamp")
        ),
        raw_poi_id=_first(row, ("raw_poi_id", "PoiId", "poi_id"), "raw_poi_id"),
        model_poi_id=_optional_int(row, ("model_poi_id", "ModelPoiId")),
        category=_first(
            row,
            ("category", "PoiCategoryName", "poi_category_name", "PoiCategoryId"),
            "category",
        ),
        latitude=latitude,
        longitude=longitude,
    )


def read_synthetic_split(path: str | Path, split: SplitName) -> tuple[NormalizedEvent, ...]:
    """Read one explicitly named synthetic split without repartitioning it."""

    return _read_rows(path, split, _synthetic_event)


def read_nyc_split(path: str | Path, split: SplitName) -> tuple[NormalizedEvent, ...]:
    """Normalize one legacy NYC raw file while preserving its caller-supplied split."""

    return _read_rows(path, split, _nyc_event)


def _read_all_splits(
    split_files: Mapping[SplitName, str | Path],
    reader: Callable[[str | Path, SplitName], tuple[NormalizedEvent, ...]],
) -> dict[SplitName, tuple[NormalizedEvent, ...]]:
    keys = set(split_files)
    expected = set(SPLIT_ORDER)
    if keys != expected:
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        raise ValueError(f"split files must be explicit; missing={missing}, extra={extra}")
    return {split: reader(split_files[split], split) for split in SPLIT_ORDER}


def read_synthetic_splits(
    split_files: Mapping[SplitName, str | Path],
) -> dict[SplitName, tuple[NormalizedEvent, ...]]:
    return _read_all_splits(split_files, read_synthetic_split)


def read_nyc_splits(
    split_files: Mapping[SplitName, str | Path],
) -> dict[SplitName, tuple[NormalizedEvent, ...]]:
    return _read_all_splits(split_files, read_nyc_split)
