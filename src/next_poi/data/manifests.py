"""Deterministic data manifests and non-loading GPU artifact scans."""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from next_poi.contracts import (
    DataManifest,
    DataSplitSummary,
    FileDigest,
    ModelManifest,
    NormalizedEvent,
    SplitName,
)
from next_poi.data._serialization import (
    canonical_json_bytes,
    canonical_timestamp,
    sha256_bytes,
    sha256_file,
    write_stable_json,
)
from next_poi.data.encoders import load_encoder_sidecar
from next_poi.data.readers import SPLIT_ORDER

MISSING_CONTENT_SHA256 = sha256_bytes(b"")

BASE_MODEL_DIR = Path("model/Llama-2-7b-longlora-32k-ft")
FINE_MODEL_DIR = Path("datasets/ca/experiment/checkpoints/best_model")
ADAPTER_DIR = Path("experiment/checkpoint-n")

BASE_SHARDS: tuple[Path, ...] = (
    BASE_MODEL_DIR / "pytorch_model-00001-of-00002.bin",
    BASE_MODEL_DIR / "pytorch_model-00002-of-00002.bin",
)
FINE_WEIGHT_ALTERNATIVES: tuple[Path, ...] = (
    FINE_MODEL_DIR / "model.safetensors",
    FINE_MODEL_DIR / "pytorch_model.bin",
)
TRAINABLE_PARAMS = ADAPTER_DIR / "trainable_params.bin"
DEEPSPEED_STATE_MARKER = "experiment/checkpoint-n/global_step*/"

STATIC_SUPPORT_FILES: tuple[Path, ...] = (
    BASE_MODEL_DIR / "config.json",
    BASE_MODEL_DIR / "generation_config.json",
    BASE_MODEL_DIR / "pytorch_model.bin.index.json",
    BASE_MODEL_DIR / "tokenizer.model",
    BASE_MODEL_DIR / "tokenizer_config.json",
    FINE_MODEL_DIR / "config.json",
    FINE_MODEL_DIR / "sentencepiece.bpe.model",
    FINE_MODEL_DIR / "tokenizer_config.json",
    ADAPTER_DIR / "adapter_config.json",
    ADAPTER_DIR / "adapter_model.bin",
    ADAPTER_DIR / "trainer_state.json",
)

ARTIFACT_SCAN_DIRS: tuple[Path, ...] = (
    BASE_MODEL_DIR,
    FINE_MODEL_DIR,
    ADAPTER_DIR,
)


def hash_categories(categories: Iterable[str]) -> str:
    normalized = sorted(set(categories))
    if not normalized:
        raise ValueError("taxonomy must contain at least one category")
    if any(not category for category in normalized):
        raise ValueError("taxonomy categories must be non-empty")
    return sha256_bytes(canonical_json_bytes(normalized))


def _event_payload(event: NormalizedEvent) -> dict[str, object]:
    return {
        "dataset": event.dataset,
        "split": event.split,
        "raw_user_id": event.raw_user_id,
        "session_id": event.session_id,
        "timestamp_utc": canonical_timestamp(event.timestamp_utc),
        "raw_poi_id": event.raw_poi_id,
        "model_poi_id": event.model_poi_id,
        "category": event.category,
        "latitude": event.latitude,
        "longitude": event.longitude,
    }


def hash_events(events: Iterable[NormalizedEvent]) -> str:
    payloads = sorted(
        (_event_payload(event) for event in events),
        key=lambda item: canonical_json_bytes(item),
    )
    return sha256_bytes(canonical_json_bytes(payloads))


def build_data_manifest(
    splits: Mapping[SplitName, Sequence[NormalizedEvent]],
    *,
    dataset: str,
    split_protocol: str,
    encoder_path: str | Path,
    taxonomy: Iterable[str],
    schema_version: str = "1",
) -> DataManifest:
    keys = set(splits)
    expected = set(SPLIT_ORDER)
    if keys != expected:
        raise ValueError(
            f"data manifest requires explicit splits; "
            f"missing={sorted(expected - keys)}, extra={sorted(keys - expected)}"
        )
    if not split_protocol:
        raise ValueError("split_protocol must be non-empty")
    encoder = load_encoder_sidecar(encoder_path)
    if encoder["dataset"] != dataset:
        raise ValueError("encoder dataset does not match data manifest dataset")
    taxonomy_values = tuple(taxonomy)
    taxonomy_sha256 = hash_categories(taxonomy_values)
    taxonomy_categories = set(taxonomy_values)
    encoder_categories = set(encoder["mappings"]["category"])
    if encoder_categories != taxonomy_categories:
        raise ValueError("encoder category mapping must match the complete taxonomy")

    all_events: list[NormalizedEvent] = []
    summaries: list[DataSplitSummary] = []
    for split in SPLIT_ORDER:
        events = tuple(splits[split])
        if any(event.split != split for event in events):
            raise ValueError(f"event split does not match manifest split {split!r}")
        if any(event.dataset != dataset for event in events):
            raise ValueError(f"event dataset does not match declared dataset {dataset!r}")
        all_events.extend(events)
        timestamps = [event.timestamp_utc for event in events]
        summaries.append(
            DataSplitSummary(
                split=split,
                count=len(events),
                content_sha256=hash_events(events),
                min_timestamp=min(timestamps) if timestamps else None,
                max_timestamp=max(timestamps) if timestamps else None,
            )
        )

    unknown_event_categories = sorted(
        {event.category for event in all_events} - taxonomy_categories
    )
    if unknown_event_categories:
        raise ValueError(
            "data contains categories outside the complete taxonomy: "
            f"{unknown_event_categories}"
        )
    return DataManifest(
        schema_version=schema_version,
        dataset=dataset,
        split_protocol=split_protocol,
        taxonomy_sha256=taxonomy_sha256,
        encoder_sha256=sha256_file(encoder_path),
        splits=tuple(summaries),
    )


def write_data_manifest(path: str | Path, manifest: DataManifest) -> str:
    return write_stable_json(path, manifest.model_dump(mode="json"))


def _file_digest(root: Path, relative_path: Path) -> FileDigest:
    absolute_path = root / relative_path
    present = absolute_path.is_file() and not absolute_path.is_symlink()
    return FileDigest(
        path=relative_path.as_posix(),
        size_bytes=absolute_path.stat().st_size if present else 0,
        sha256=sha256_file(absolute_path) if present else MISSING_CONTENT_SHA256,
        present=present,
    )


def _discover_regular_files(root: Path) -> set[Path]:
    """Inventory known artifact trees without following symlinks."""

    discovered: set[Path] = set()
    for relative_directory in ARTIFACT_SCAN_DIRS:
        absolute_directory = root / relative_directory
        if not absolute_directory.is_dir() or absolute_directory.is_symlink():
            continue
        for current_root, directory_names, file_names in os.walk(
            absolute_directory,
            followlinks=False,
        ):
            current = Path(current_root)
            directory_names[:] = sorted(
                name for name in directory_names if not (current / name).is_symlink()
            )
            for name in sorted(file_names):
                path = current / name
                if path.is_symlink() or not path.is_file():
                    continue
                discovered.add(path.relative_to(root))
    return discovered


def scan_gpu_artifacts(
    repository_root: str | Path,
    *,
    model_name: str = "legacy-cot4poi-full-gpu",
    schema_version: str = "1",
) -> ModelManifest:
    """Hash static assets only; this function intentionally never imports or loads a model."""

    root = Path(repository_root)
    if not root.is_dir():
        raise FileNotFoundError(f"repository root not found: {root}")

    regular_expected = STATIC_SUPPORT_FILES + BASE_SHARDS + (TRAINABLE_PARAMS,)
    discovered_files = _discover_regular_files(root)
    digest_paths = set(regular_expected) | discovered_files
    expected_digests = [_file_digest(root, path) for path in regular_expected]
    missing = [digest.path for digest in expected_digests if not digest.present]

    fine_present = [
        path
        for path in FINE_WEIGHT_ALTERNATIVES
        if (root / path).is_file() and not (root / path).is_symlink()
    ]
    if fine_present:
        digest_paths.update(fine_present)
    else:
        fine_digests = [_file_digest(root, path) for path in FINE_WEIGHT_ALTERNATIVES]
        digest_paths.update(FINE_WEIGHT_ALTERNATIVES)
        missing.extend(digest.path for digest in fine_digests)

    adapter_depth = len(ADAPTER_DIR.parts)
    has_deepspeed_files = any(
        path.parts[:adapter_depth] == ADAPTER_DIR.parts
        and len(path.parts) > adapter_depth + 1
        and path.parts[adapter_depth].startswith("global_step")
        for path in discovered_files
    )
    if not has_deepspeed_files:
        missing.append(DEEPSPEED_STATE_MARKER)

    digests = [_file_digest(root, path) for path in sorted(digest_paths)]

    scan_config = {
        "schema_version": schema_version,
        "model_name": model_name,
        "expected_static_files": [path.as_posix() for path in regular_expected],
        "fine_weight_alternatives": [path.as_posix() for path in FINE_WEIGHT_ALTERNATIVES],
        "deepspeed_state_marker": DEEPSPEED_STATE_MARKER,
    }
    base_config = root / BASE_MODEL_DIR / "config.json"
    config_sha256 = (
        sha256_file(base_config)
        if base_config.is_file() and not base_config.is_symlink()
        else sha256_bytes(canonical_json_bytes(scan_config))
    )
    return ModelManifest(
        schema_version=schema_version,
        model_name=model_name,
        backend="full-gpu",
        runtime_status="static_only",
        dynamic_load_verified=False,
        files=tuple(sorted(digests, key=lambda item: item.path)),
        missing_files=tuple(sorted(set(missing))),
        config_sha256=config_sha256,
    )


build_static_gpu_manifest = scan_gpu_artifacts


def write_model_manifest(path: str | Path, manifest: ModelManifest) -> str:
    return write_stable_json(path, manifest.model_dump(mode="json"))
