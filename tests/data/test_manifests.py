from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from next_poi.data import (
    build_data_manifest,
    build_train_encoder,
    export_encoder_sidecar,
    hash_categories,
    read_synthetic_splits,
    scan_gpu_artifacts,
    write_data_manifest,
)

FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "synthetic"


class DataManifestTests(unittest.TestCase):
    def test_taxonomy_must_not_be_empty(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one category"):
            hash_categories(())

    def test_data_manifest_and_serialized_hash_repeat_exactly(self) -> None:
        splits = read_synthetic_splits({
            split: FIXTURE_ROOT / f"{split}.csv"
            for split in ("train", "validation", "test")
        })
        taxonomy = [event.category for events in splits.values() for event in events]
        encoder = build_train_encoder(
            splits["train"],
            dataset="synthetic",
            known_taxonomy=taxonomy,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            encoder_path = root / "encoder.json"
            export_encoder_sidecar(encoder_path, encoder)
            first = build_data_manifest(
                splits,
                dataset="synthetic",
                split_protocol="fixture-v1",
                encoder_path=encoder_path,
                taxonomy=reversed(taxonomy),
            )
            second = build_data_manifest(
                {key: tuple(reversed(value)) for key, value in splits.items()},
                dataset="synthetic",
                split_protocol="fixture-v1",
                encoder_path=encoder_path,
                taxonomy=taxonomy,
            )
            first_path = root / "first-manifest.json"
            second_path = root / "second-manifest.json"
            first_hash = write_data_manifest(first_path, first)
            second_hash = write_data_manifest(second_path, second)

        self.assertEqual(first, second)
        self.assertEqual(first_hash, second_hash)
        self.assertEqual([item.split for item in first.splits], ["train", "validation", "test"])
        self.assertEqual([item.count for item in first.splits], [12, 6, 6])
        self.assertEqual(first.taxonomy_sha256, hash_categories(taxonomy))

    def test_manifest_rejects_taxonomy_that_differs_from_encoder(self) -> None:
        splits = read_synthetic_splits({
            split: FIXTURE_ROOT / f"{split}.csv"
            for split in ("train", "validation", "test")
        })
        taxonomy = {event.category for events in splits.values() for event in events}
        encoder = build_train_encoder(
            splits["train"],
            dataset="synthetic",
            known_taxonomy=taxonomy,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            encoder_path = Path(temporary_directory) / "encoder.json"
            export_encoder_sidecar(encoder_path, encoder)
            with self.assertRaisesRegex(ValueError, "complete taxonomy"):
                build_data_manifest(
                    splits,
                    dataset="synthetic",
                    split_protocol="fixture-v1",
                    encoder_path=encoder_path,
                    taxonomy=taxonomy | {"Synthetic Unused Category"},
                )


class StaticGpuManifestTests(unittest.TestCase):
    def test_missing_assets_are_static_only_and_truthfully_listed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest = scan_gpu_artifacts(temporary_directory)
            repeated = scan_gpu_artifacts(temporary_directory)

        self.assertEqual(manifest, repeated)
        self.assertEqual(manifest.runtime_status, "static_only")
        self.assertFalse(manifest.dynamic_load_verified)
        self.assertIn(
            "model/Llama-2-7b-longlora-32k-ft/pytorch_model-00001-of-00002.bin",
            manifest.missing_files,
        )
        self.assertIn(
            "datasets/ca/experiment/checkpoints/best_model/model.safetensors",
            manifest.missing_files,
        )
        self.assertIn("experiment/checkpoint-n/trainable_params.bin", manifest.missing_files)
        self.assertIn("experiment/checkpoint-n/global_step*/", manifest.missing_files)
        self.assertTrue(any(not item.present for item in manifest.files))

    def test_present_static_files_are_hashed_but_never_claim_dynamic_load(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = root / "model/Llama-2-7b-longlora-32k-ft/config.json"
            config.parent.mkdir(parents=True)
            config.write_text('{"model_type":"llama"}\n', encoding="utf-8")
            attachment = root / "experiment/checkpoint-n/tokenizer.json"
            attachment.parent.mkdir(parents=True)
            attachment.write_text('{"version":"synthetic"}\n', encoding="utf-8")
            outside_directory = root / "outside-artifact-tree"
            outside_directory.mkdir()
            (outside_directory / "must-not-be-scanned.bin").write_bytes(b"not an artifact")
            (attachment.parent / "linked-outside").symlink_to(
                outside_directory,
                target_is_directory=True,
            )
            linked_expected = (
                root / "datasets/ca/experiment/checkpoints/best_model/config.json"
            )
            linked_expected.parent.mkdir(parents=True)
            linked_expected.symlink_to(outside_directory / "must-not-be-scanned.bin")

            manifest = scan_gpu_artifacts(root)
            repeated = scan_gpu_artifacts(root)

        config_entry = next(
            item
            for item in manifest.files
            if item.path == "model/Llama-2-7b-longlora-32k-ft/config.json"
        )
        attachment_entry = next(
            item
            for item in manifest.files
            if item.path == "experiment/checkpoint-n/tokenizer.json"
        )
        self.assertTrue(config_entry.present)
        self.assertGreater(config_entry.size_bytes, 0)
        self.assertTrue(attachment_entry.present)
        self.assertFalse(any("linked-outside" in item.path for item in manifest.files))
        linked_expected_entry = next(
            item
            for item in manifest.files
            if item.path == "datasets/ca/experiment/checkpoints/best_model/config.json"
        )
        self.assertFalse(linked_expected_entry.present)
        self.assertIn(linked_expected_entry.path, manifest.missing_files)
        self.assertEqual(manifest, repeated)
        self.assertEqual(manifest.runtime_status, "static_only")
        self.assertFalse(manifest.dynamic_load_verified)


if __name__ == "__main__":
    unittest.main()
