"""Versioned smoke release manifests linking data, model, and config hashes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from next_poi.contracts import DataManifest, ModelManifest, ReleaseManifest
from next_poi.data._serialization import sha256_file, write_stable_json

RELEASE_SCHEMA_VERSION = "1"
DATA_MANIFEST_SCHEMA_VERSION = "1"
MODEL_MANIFEST_SCHEMA_VERSION = "1"


@dataclass(frozen=True)
class ReleaseInfo:
    path: Path
    manifest: ReleaseManifest
    sha256: str


def build_smoke_release(
    *,
    data_manifest_path: str | Path,
    model_manifest_path: str | Path,
    config_path: str | Path,
    output_path: str | Path,
    release_version: str,
) -> ReleaseInfo:
    """Validate linked manifests and write one deterministic smoke release."""

    data_path = Path(data_manifest_path)
    model_path = Path(model_manifest_path)
    resolved_config_path = Path(config_path)
    data_manifest = _load_manifest(data_path, DataManifest, "data")
    model_manifest = _load_manifest(model_path, ModelManifest, "model")
    if data_manifest.schema_version != DATA_MANIFEST_SCHEMA_VERSION:
        raise ValueError("smoke release requires data manifest schema version 1")
    if (
        model_manifest.schema_version != MODEL_MANIFEST_SCHEMA_VERSION
        or model_manifest.backend != "cpu-smoke"
        or model_manifest.runtime_status != "ready"
        or not model_manifest.dynamic_load_verified
    ):
        raise ValueError("smoke release requires a verified ready model manifest")
    model_root = model_path.parent
    expected_config_path = model_root / "config.json"
    if (
        not resolved_config_path.is_file()
        or resolved_config_path.is_symlink()
        or not expected_config_path.is_file()
        or expected_config_path.is_symlink()
    ):
        raise FileNotFoundError("smoke release config not found")
    if resolved_config_path.resolve() != expected_config_path.resolve():
        raise ValueError("smoke release config hash source must be the bundled config.json")

    config_entries = [item for item in model_manifest.files if item.path == "config.json"]
    if len(config_entries) != 1:
        raise ValueError("smoke release model manifest must declare config.json exactly once")
    for digest in model_manifest.files:
        artifact_path = model_root / digest.path
        if (
            not digest.present
            or not artifact_path.is_file()
            or artifact_path.is_symlink()
            or artifact_path.stat().st_size != digest.size_bytes
            or sha256_file(artifact_path) != digest.sha256
        ):
            raise ValueError("smoke release model artifact verification failed")

    config_sha256 = sha256_file(expected_config_path)
    if (
        config_entries[0].sha256 != config_sha256
        or config_sha256 != model_manifest.config_sha256
    ):
        raise ValueError("smoke release config hash does not match model manifest")
    if not data_manifest.splits:
        raise ValueError("smoke release data manifest contains no splits")

    manifest = ReleaseManifest(
        schema_version=RELEASE_SCHEMA_VERSION,
        release_version=release_version,
        profile="smoke",
        data_manifest_sha256=sha256_file(data_path),
        model_manifest_sha256=sha256_file(model_path),
        config_sha256=config_sha256,
    )
    destination = Path(output_path)
    digest = write_stable_json(destination, manifest.model_dump(mode="json"))
    return ReleaseInfo(path=destination, manifest=manifest, sha256=digest)


def _load_manifest(path: Path, model_type, label: str):
    if not path.is_file():
        raise FileNotFoundError(f"smoke release {label} manifest not found")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return model_type.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValidationError, TypeError):
        raise ValueError(f"smoke release {label} manifest is invalid") from None
