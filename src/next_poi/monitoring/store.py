"""Append-only JSONL storage for validated monitoring events."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from pydantic import ValidationError

from next_poi.monitoring.events import MonitoringEvent


class MonitoringStoreError(ValueError):
    """Raised when a stored event cannot be decoded safely."""


class JsonlEventStore:
    """Stable UTF-8 JSONL append/read storage owned by the event schema."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, event: MonitoringEvent) -> None:
        try:
            validated = MonitoringEvent.model_validate(event.model_dump(mode="json"))
        except ValidationError:
            raise MonitoringStoreError("invalid monitoring event for append") from None
        payload = json.dumps(
            validated.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(payload + "\n")
        except OSError:
            raise MonitoringStoreError("unable to append monitoring JSONL") from None

    def iter_events(self) -> Iterator[MonitoringEvent]:
        if not self.path.exists():
            return
        try:
            with self.path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        payload = json.loads(line)
                        yield MonitoringEvent.model_validate(payload)
                    except (json.JSONDecodeError, TypeError, ValidationError):
                        raise MonitoringStoreError(
                            f"invalid monitoring event at JSONL line {line_number}"
                        ) from None
        except OSError:
            raise MonitoringStoreError("unable to read monitoring JSONL") from None

    def read(self) -> tuple[MonitoringEvent, ...]:
        return tuple(self.iter_events())

    def read_all(self) -> tuple[MonitoringEvent, ...]:
        """Compatibility-friendly explicit name for callers reading a full window."""

        return self.read()
