"""Portable, deterministic encoders fitted exclusively from train events."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, cast

from next_poi.contracts import NormalizedEvent
from next_poi.data._serialization import write_stable_json

ENCODER_SCHEMA_VERSION = "1"
MAPPING_NAMES = ("category", "poi", "user")


def build_train_encoder(
    events: Iterable[NormalizedEvent],
    *,
    dataset: str,
    known_taxonomy: Iterable[str],
    id_offset: int = 0,
) -> dict[str, Any]:
    """Fit train identities while mapping categories from the fixed taxonomy."""

    if id_offset < 0:
        raise ValueError("id_offset must be non-negative")
    train_events = tuple(events)
    if not train_events:
        raise ValueError("at least one train event is required")
    if any(event.split != "train" for event in train_events):
        raise ValueError("encoder fitting accepts train events only")
    if any(event.dataset != dataset for event in train_events):
        raise ValueError("all encoder events must match the declared dataset")

    taxonomy = tuple(known_taxonomy)
    if not taxonomy:
        raise ValueError("known_taxonomy must contain at least one category")
    if any(not isinstance(category, str) or not category for category in taxonomy):
        raise ValueError("known_taxonomy categories must be non-empty strings")
    known_categories = set(taxonomy)
    unknown_train_categories = sorted(
        {event.category for event in train_events} - known_categories
    )
    if unknown_train_categories:
        raise ValueError(
            "train events contain categories outside known_taxonomy: "
            f"{unknown_train_categories}"
        )

    def mapping(values: Iterable[str]) -> dict[str, int]:
        return {
            value: index
            for index, value in enumerate(sorted(set(values)), start=id_offset)
        }

    return {
        "schema_version": ENCODER_SCHEMA_VERSION,
        "dataset": dataset,
        "fitted_split": "train",
        "id_offset": id_offset,
        "unknown_id": None,
        "mappings": {
            "category": mapping(known_categories),
            "poi": mapping(event.raw_poi_id for event in train_events),
            "user": mapping(event.raw_user_id for event in train_events),
        },
    }


def export_encoder_sidecar(path: str | Path, encoder: Mapping[str, Any]) -> str:
    """Validate and export a JSON sidecar, returning the exact file hash."""

    validated = _validate_encoder(dict(encoder))
    return write_stable_json(path, validated)


def fit_and_export_train_encoder(
    events: Iterable[NormalizedEvent],
    path: str | Path,
    *,
    dataset: str,
    known_taxonomy: Iterable[str],
    id_offset: int = 0,
) -> str:
    encoder = build_train_encoder(
        events,
        dataset=dataset,
        known_taxonomy=known_taxonomy,
        id_offset=id_offset,
    )
    return export_encoder_sidecar(path, encoder)


def load_encoder_sidecar(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"encoder sidecar not found: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid encoder JSON: {source}") from exc
    if not isinstance(payload, dict):
        raise ValueError("encoder sidecar must contain a JSON object")
    return _validate_encoder(payload)


def _validate_encoder(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema_version") != ENCODER_SCHEMA_VERSION:
        raise ValueError("unsupported encoder schema_version")
    if not isinstance(payload.get("dataset"), str) or not payload["dataset"]:
        raise ValueError("encoder dataset must be non-empty")
    if payload.get("fitted_split") != "train":
        raise ValueError("encoder fitted_split must be train")
    offset = payload.get("id_offset")
    if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
        raise ValueError("encoder id_offset must be a non-negative integer")
    if payload.get("unknown_id") is not None:
        raise ValueError("encoder unknown_id must be null")
    mappings = payload.get("mappings")
    if not isinstance(mappings, dict) or set(mappings) != set(MAPPING_NAMES):
        raise ValueError(f"encoder mappings must be exactly {MAPPING_NAMES}")

    normalized_mappings: dict[str, dict[str, int]] = {}
    for name in MAPPING_NAMES:
        raw_mapping = mappings[name]
        if not isinstance(raw_mapping, dict):
            raise ValueError(f"encoder mapping {name!r} must be an object")
        if any(not isinstance(key, str) or not key for key in raw_mapping):
            raise ValueError(f"encoder mapping {name!r} contains an invalid key")
        values = list(raw_mapping.values())
        if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
            raise ValueError(f"encoder mapping {name!r} values must be integers")
        expected = list(range(offset, offset + len(values)))
        if sorted(values) != expected:
            raise ValueError(f"encoder mapping {name!r} IDs must be contiguous from id_offset")
        normalized_mappings[name] = {
            key: cast(int, raw_mapping[key]) for key in sorted(raw_mapping)
        }

    return {
        "schema_version": ENCODER_SCHEMA_VERSION,
        "dataset": payload["dataset"],
        "fitted_split": "train",
        "id_offset": offset,
        "unknown_id": None,
        "mappings": normalized_mappings,
    }


def encoded_poi_id(raw_poi_id: str, encoder: Mapping[str, Any]) -> int | None:
    """Return a train-known POI ID; unseen validation/test POIs remain unknown."""

    validated = _validate_encoder(dict(encoder))
    return cast(dict[str, int], validated["mappings"]["poi"]).get(raw_poi_id)
