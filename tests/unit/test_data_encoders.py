from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from next_poi.data import (
    build_train_encoder,
    encoded_poi_id,
    export_encoder_sidecar,
    load_encoder_sidecar,
    read_synthetic_splits,
)

FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "synthetic"


class TrainOnlyEncoderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.splits = read_synthetic_splits({
            split: FIXTURE_ROOT / f"{split}.csv"
            for split in ("train", "validation", "test")
        })

    def test_export_is_deterministic_and_uses_only_train_values(self) -> None:
        train = self.splits["train"]
        validation_only = self.splits["validation"][0].model_copy(
            update={
                "category": "Synthetic Validation Only Category",
                "raw_poi_id": "synthetic-poi-validation-only",
                "raw_user_id": "synthetic-user-validation-only",
            }
        )
        taxonomy = {
            *(event.category for event in train),
            validation_only.category,
        }
        encoder = build_train_encoder(
            reversed(train),
            dataset="synthetic",
            known_taxonomy=taxonomy,
            id_offset=3,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            first = Path(temporary_directory) / "first.json"
            second = Path(temporary_directory) / "second.json"
            first_hash = export_encoder_sidecar(first, encoder)
            second_hash = export_encoder_sidecar(second, encoder)

            self.assertEqual(first_hash, second_hash)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            loaded = load_encoder_sidecar(first)

        self.assertEqual(loaded["fitted_split"], "train")
        self.assertEqual(loaded["id_offset"], 3)
        self.assertIn(validation_only.category, loaded["mappings"]["category"])
        self.assertIsNone(encoded_poi_id(validation_only.raw_poi_id, loaded))
        self.assertNotIn(validation_only.raw_user_id, loaded["mappings"]["user"])
        self.assertEqual(min(loaded["mappings"]["poi"].values()), 3)

    def test_rejects_validation_or_test_during_fit(self) -> None:
        with self.assertRaisesRegex(ValueError, "train events only"):
            build_train_encoder(
                self.splits["train"] + self.splits["validation"],
                dataset="synthetic",
                known_taxonomy={event.category for event in self.splits["train"]},
            )

    def test_rejects_train_category_outside_known_taxonomy(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside known_taxonomy"):
            build_train_encoder(
                self.splits["train"],
                dataset="synthetic",
                known_taxonomy={"Synthetic Missing Taxonomy"},
            )


if __name__ == "__main__":
    unittest.main()
