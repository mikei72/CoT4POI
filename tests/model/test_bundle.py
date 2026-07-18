from __future__ import annotations

import json
from pathlib import Path

import pytest

from next_poi.contracts import RecommendationRequest
from next_poi.data import read_synthetic_splits
from next_poi.data._serialization import sha256_file, write_stable_json
from next_poi.models import (
    CandidateIndex,
    SmokePredictor,
    load_smoke_bundle,
    save_smoke_bundle,
)

FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "synthetic"


def fixture_predictor() -> tuple[SmokePredictor, RecommendationRequest]:
    splits = read_synthetic_splits(
        {split: FIXTURE_ROOT / f"{split}.csv" for split in ("train", "validation", "test")}
    )
    taxonomy = {event.category for events in splits.values() for event in events}
    index = CandidateIndex.fit(splits["train"], taxonomy=taxonomy)
    target = splits["validation"][1]
    previous = splits["validation"][0]
    request = RecommendationRequest(
        dataset="synthetic",
        history=(
            {
                "poi_id": previous.raw_poi_id,
                "category_name": previous.category,
                "timestamp": previous.timestamp_utc,
            },
        ),
        target_time=target.timestamp_utc,
        top_k=5,
    )
    return SmokePredictor(index), request


def test_bundle_round_trip_and_hash_are_deterministic(tmp_path: Path) -> None:
    predictor, request = fixture_predictor()
    first_info = save_smoke_bundle(tmp_path / "first", predictor)
    second_info = save_smoke_bundle(tmp_path / "second", predictor)
    loaded, loaded_info = load_smoke_bundle(tmp_path / "first")

    assert first_info.manifest_sha256 == second_info.manifest_sha256
    assert first_info.manifest == second_info.manifest
    assert loaded_info.manifest_sha256 == first_info.manifest_sha256
    assert loaded.index == predictor.index
    assert [item.poi_id for item in loaded.predict(request).recommendations] == [
        item.poi_id for item in predictor.predict(request).recommendations
    ]


def test_bundle_rejects_tampered_index(tmp_path: Path) -> None:
    predictor, _ = fixture_predictor()
    root = tmp_path / "bundle"
    save_smoke_bundle(root, predictor)
    with (root / "index.json").open("a", encoding="utf-8") as handle:
        handle.write(" ")

    with pytest.raises(ValueError, match="verification failed"):
        load_smoke_bundle(root)


def test_bundle_rejects_rehashed_index_with_invalid_invariants(tmp_path: Path) -> None:
    predictor, _ = fixture_predictor()
    root = tmp_path / "bundle"
    save_smoke_bundle(root, predictor)
    index_path = root / "index.json"
    index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    first_poi = next(iter(index_payload["global_counts"]))
    index_payload["global_counts"][first_poi] = 0
    write_stable_json(index_path, index_payload)
    _refresh_manifest_digest(root, "index.json")

    with pytest.raises(ValueError, match="positive integers"):
        load_smoke_bundle(root)


def test_bundle_rejects_rehashed_manifest_with_wrong_schema(tmp_path: Path) -> None:
    predictor, _ = fixture_predictor()
    root = tmp_path / "bundle"
    save_smoke_bundle(root, predictor)
    manifest_path = root / "manifest.json"
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_payload["schema_version"] = "999"
    write_stable_json(manifest_path, manifest_payload)

    with pytest.raises(ValueError, match="ready CPU backend"):
        load_smoke_bundle(root)


def test_bundle_rejects_rehashed_index_with_extra_schema_field(tmp_path: Path) -> None:
    predictor, _ = fixture_predictor()
    root = tmp_path / "bundle"
    save_smoke_bundle(root, predictor)
    index_path = root / "index.json"
    index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    index_payload["unexpected"] = "field"
    write_stable_json(index_path, index_payload)
    _refresh_manifest_digest(root, "index.json")

    with pytest.raises(ValueError, match="fields are invalid"):
        load_smoke_bundle(root)


def test_bundle_rejects_rehashed_config_with_coerced_field_type(tmp_path: Path) -> None:
    predictor, _ = fixture_predictor()
    root = tmp_path / "bundle"
    save_smoke_bundle(root, predictor)
    config_path = root / "config.json"
    config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    config_payload["catalog_size"] = float(config_payload["catalog_size"])
    write_stable_json(config_path, config_payload)
    _refresh_manifest_digest(root, "config.json")

    with pytest.raises(ValueError, match="field types are invalid"):
        load_smoke_bundle(root)


def test_bundle_refuses_to_overwrite_existing_bundle(tmp_path: Path) -> None:
    predictor, _ = fixture_predictor()
    root = tmp_path / "bundle"
    save_smoke_bundle(root, predictor)
    with pytest.raises(FileExistsError):
        save_smoke_bundle(root, predictor)


def _refresh_manifest_digest(root: Path, filename: str) -> None:
    manifest_path = root / "manifest.json"
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_path = root / filename
    digest = next(item for item in manifest_payload["files"] if item["path"] == filename)
    digest["size_bytes"] = artifact_path.stat().st_size
    digest["sha256"] = sha256_file(artifact_path)
    if filename == "config.json":
        manifest_payload["config_sha256"] = digest["sha256"]
    write_stable_json(manifest_path, manifest_payload)
