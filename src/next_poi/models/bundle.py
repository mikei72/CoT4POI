"""Portable and strictly verified CPU smoke model bundles."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from next_poi.contracts import FileDigest, ModelManifest, VersionInfo
from next_poi.data._serialization import sha256_file, write_stable_json
from next_poi.models.index import (
    INDEX_SCHEMA_VERSION,
    TIME_BUCKET_VERSION,
    CandidateIndex,
)
from next_poi.models.predictor import SmokePredictor
from next_poi.models.ranker import SOURCE_WEIGHTS, VARIANT_SOURCES

BUNDLE_SCHEMA_VERSION = "1"
BUNDLE_FILENAMES = ("config.json", "index.json", "manifest.json")


@dataclass(frozen=True)
class BundleInfo:
    directory: Path
    manifest: ModelManifest
    manifest_sha256: str


def save_smoke_bundle(directory: str | Path, predictor: SmokePredictor) -> BundleInfo:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    if any((root / name).exists() for name in BUNDLE_FILENAMES):
        raise FileExistsError("smoke bundle target already contains bundle files")

    config = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "backend": "cpu-smoke",
        "variant": predictor.variant,
        "versions": predictor.versions.model_dump(mode="json"),
        "source_weights": SOURCE_WEIGHTS,
        "index_schema_version": INDEX_SCHEMA_VERSION,
        "time_bucket_version": TIME_BUCKET_VERSION,
        "train_data_sha256": predictor.index.train_data_sha256,
        "catalog_size": len(predictor.index.global_counts),
    }
    if predictor.variant not in VARIANT_SOURCES:
        raise ValueError("cannot save an unsupported smoke variant")
    config_path = root / "config.json"
    index_path = root / "index.json"
    write_stable_json(config_path, config)
    write_stable_json(index_path, predictor.index.to_dict())

    files = tuple(
        FileDigest(
            path=path.name,
            size_bytes=path.stat().st_size,
            sha256=sha256_file(path),
        )
        for path in (config_path, index_path)
    )
    manifest = ModelManifest(
        schema_version=BUNDLE_SCHEMA_VERSION,
        model_name=predictor.versions.model,
        backend="cpu-smoke",
        runtime_status="ready",
        dynamic_load_verified=True,
        files=files,
        config_sha256=sha256_file(config_path),
    )
    manifest_path = root / "manifest.json"
    write_stable_json(manifest_path, manifest.model_dump(mode="json"))
    return BundleInfo(root, manifest, sha256_file(manifest_path))


def load_smoke_bundle(directory: str | Path) -> tuple[SmokePredictor, BundleInfo]:
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError("smoke bundle directory not found")
    paths = {name: root / name for name in BUNDLE_FILENAMES}
    if any(not path.is_file() for path in paths.values()):
        raise ValueError("smoke bundle is incomplete")

    manifest_payload = _read_json_object(paths["manifest.json"], "manifest")
    if not _strict_manifest_payload(manifest_payload):
        raise ValueError("smoke bundle manifest fields are invalid")
    try:
        manifest = ModelManifest.model_validate(manifest_payload)
    except ValidationError as exc:
        raise ValueError("smoke bundle manifest is invalid") from exc
    if (
        manifest.schema_version != BUNDLE_SCHEMA_VERSION
        or manifest.backend != "cpu-smoke"
        or manifest.runtime_status != "ready"
        or not manifest.dynamic_load_verified
        or manifest.missing_files
    ):
        raise ValueError("smoke bundle manifest does not describe a ready CPU backend")
    expected_paths = {"config.json", "index.json"}
    if {item.path for item in manifest.files} != expected_paths:
        raise ValueError("smoke bundle manifest must cover config.json and index.json")
    for digest in manifest.files:
        path = root / digest.path
        if (
            not path.is_file()
            or path.stat().st_size != digest.size_bytes
            or sha256_file(path) != digest.sha256
        ):
            raise ValueError("smoke bundle file verification failed")

    config = _read_json_object(paths["config.json"], "config")
    required_config_fields = {
        "schema_version",
        "backend",
        "variant",
        "versions",
        "source_weights",
        "index_schema_version",
        "time_bucket_version",
        "train_data_sha256",
        "catalog_size",
    }
    if set(config) != required_config_fields:
        raise ValueError("smoke bundle config fields are invalid")
    if not _strict_config_types(config):
        raise ValueError("smoke bundle config field types are invalid")
    if (
        config["schema_version"] != BUNDLE_SCHEMA_VERSION
        or config["backend"] != "cpu-smoke"
        or config["variant"] not in VARIANT_SOURCES
        or config["source_weights"] != SOURCE_WEIGHTS
        or config["index_schema_version"] != INDEX_SCHEMA_VERSION
        or config["time_bucket_version"] != TIME_BUCKET_VERSION
        or config["train_data_sha256"] != index_payload_train_hash(paths["index.json"])
        or sha256_file(paths["config.json"]) != manifest.config_sha256
    ):
        raise ValueError("smoke bundle config is incompatible")
    try:
        versions = VersionInfo.model_validate(config["versions"])
    except ValidationError as exc:
        raise ValueError("smoke bundle versions are invalid") from exc
    if versions.model != manifest.model_name:
        raise ValueError("smoke bundle model version does not match manifest")

    index_payload = _read_json_object(paths["index.json"], "index")
    index = CandidateIndex.from_dict(index_payload)
    if (
        config["train_data_sha256"] != index.train_data_sha256
        or config["catalog_size"] != len(index.global_counts)
    ):
        raise ValueError("smoke bundle config and candidate index do not match")
    predictor = SmokePredictor(index, variant=str(config["variant"]), versions=versions)
    info = BundleInfo(root, manifest, sha256_file(paths["manifest.json"]))
    return predictor, info


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"smoke bundle {label} JSON is invalid") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"smoke bundle {label} must be a JSON object")
    return payload


def index_payload_train_hash(path: Path) -> object:
    """Read the config-linked index hash without accepting a partial index."""

    return _read_json_object(path, "index").get("train_data_sha256")


def _strict_manifest_payload(payload: dict[str, Any]) -> bool:
    if set(payload) != {
        "schema_version",
        "model_name",
        "backend",
        "runtime_status",
        "dynamic_load_verified",
        "files",
        "missing_files",
        "config_sha256",
    }:
        return False
    files = payload["files"]
    return (
        isinstance(payload["schema_version"], str)
        and isinstance(payload["model_name"], str)
        and isinstance(payload["backend"], str)
        and isinstance(payload["runtime_status"], str)
        and isinstance(payload["dynamic_load_verified"], bool)
        and isinstance(payload["config_sha256"], str)
        and isinstance(payload["missing_files"], list)
        and all(isinstance(path, str) for path in payload["missing_files"])
        and isinstance(files, list)
        and all(
            isinstance(item, dict)
            and set(item) == {"path", "size_bytes", "sha256", "present"}
            and isinstance(item["path"], str)
            and isinstance(item["size_bytes"], int)
            and not isinstance(item["size_bytes"], bool)
            and isinstance(item["sha256"], str)
            and isinstance(item["present"], bool)
            for item in files
        )
    )


def _strict_config_types(config: dict[str, Any]) -> bool:
    weights = config["source_weights"]
    return (
        isinstance(config["schema_version"], str)
        and isinstance(config["backend"], str)
        and isinstance(config["variant"], str)
        and isinstance(config["versions"], dict)
        and isinstance(weights, dict)
        and set(weights) == set(SOURCE_WEIGHTS)
        and all(type(value) is float for value in weights.values())
        and isinstance(config["index_schema_version"], str)
        and isinstance(config["time_bucket_version"], str)
        and isinstance(config["train_data_sha256"], str)
        and isinstance(config["catalog_size"], int)
        and not isinstance(config["catalog_size"], bool)
        and config["catalog_size"] > 0
    )
