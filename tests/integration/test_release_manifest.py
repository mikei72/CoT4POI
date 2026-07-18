from __future__ import annotations

import json
from pathlib import Path

import pytest

from next_poi.data import (
    build_data_manifest,
    build_train_encoder,
    export_encoder_sidecar,
    read_synthetic_splits,
    write_data_manifest,
)
from next_poi.data._serialization import write_stable_json
from next_poi.models import CandidateIndex, SmokePredictor, save_smoke_bundle
from next_poi.release import build_smoke_release

FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "synthetic"


def _release_inputs(tmp_path: Path):
    splits = read_synthetic_splits(
        {split: FIXTURE_ROOT / f"{split}.csv" for split in ("train", "validation", "test")}
    )
    taxonomy = {event.category for events in splits.values() for event in events}
    encoder_path = tmp_path / "encoder.json"
    export_encoder_sidecar(
        encoder_path,
        build_train_encoder(
            splits["train"], dataset="synthetic", known_taxonomy=taxonomy
        ),
    )
    data_manifest_path = tmp_path / "data_manifest.json"
    write_data_manifest(
        data_manifest_path,
        build_data_manifest(
            splits,
            dataset="synthetic",
            split_protocol="fixture-v1",
            encoder_path=encoder_path,
            taxonomy=taxonomy,
        ),
    )
    predictor = SmokePredictor(CandidateIndex.fit(splits["train"], taxonomy=taxonomy))
    bundle = save_smoke_bundle(tmp_path / "bundle", predictor)
    return data_manifest_path, bundle.directory


def test_release_links_verified_data_model_and_config_hashes(tmp_path: Path) -> None:
    data_manifest_path, bundle_path = _release_inputs(tmp_path)
    first = build_smoke_release(
        data_manifest_path=data_manifest_path,
        model_manifest_path=bundle_path / "manifest.json",
        config_path=bundle_path / "config.json",
        output_path=tmp_path / "release.json",
        release_version="fixture-r1",
    )
    second = build_smoke_release(
        data_manifest_path=data_manifest_path,
        model_manifest_path=bundle_path / "manifest.json",
        config_path=bundle_path / "config.json",
        output_path=tmp_path / "release-repeat.json",
        release_version="fixture-r1",
    )

    assert first.manifest == second.manifest
    assert first.sha256 == second.sha256
    assert first.manifest.profile == "smoke"


def test_release_rejects_config_that_is_not_the_bundled_config(tmp_path: Path) -> None:
    data_manifest_path, bundle_path = _release_inputs(tmp_path)
    other_config = tmp_path / "other.json"
    other_config.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="config hash"):
        build_smoke_release(
            data_manifest_path=data_manifest_path,
            model_manifest_path=bundle_path / "manifest.json",
            config_path=other_config,
            output_path=tmp_path / "release.json",
            release_version="fixture-r1",
        )


def test_release_rejects_rewritten_manifest_when_declared_file_digest_is_stale(
    tmp_path: Path,
) -> None:
    data_manifest_path, bundle_path = _release_inputs(tmp_path)
    index_path = bundle_path / "index.json"
    index_path.write_text(index_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    manifest_path = bundle_path / "manifest.json"
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    write_stable_json(manifest_path, manifest_payload)

    with pytest.raises(ValueError, match="artifact verification failed"):
        build_smoke_release(
            data_manifest_path=data_manifest_path,
            model_manifest_path=manifest_path,
            config_path=bundle_path / "config.json",
            output_path=tmp_path / "release.json",
            release_version="fixture-r1",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (("schema_version", "999"), ("backend", "full-gpu")),
)
def test_release_rejects_wrong_model_schema_or_backend(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    data_manifest_path, bundle_path = _release_inputs(tmp_path)
    manifest_path = bundle_path / "manifest.json"
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_payload[field] = value
    write_stable_json(manifest_path, manifest_payload)

    with pytest.raises(ValueError, match="verified ready model manifest"):
        build_smoke_release(
            data_manifest_path=data_manifest_path,
            model_manifest_path=manifest_path,
            config_path=bundle_path / "config.json",
            output_path=tmp_path / "release.json",
            release_version="fixture-r1",
        )
