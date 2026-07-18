from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from next_poi.contracts import NormalizedEvent
from next_poi.data import build_labeled_examples, compute_sample_id, read_synthetic_splits

FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "synthetic"
EXPECTED_SAMPLE_ID = "d887ca3d1fdb80f37f548ef185c70405cadbbf6a80939076401f669c5b9ff3c3"


class SampleIdentityTests(unittest.TestCase):
    def test_long_session_uses_the_latest_128_history_events(self) -> None:
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        events = tuple(
            NormalizedEvent(
                dataset="synthetic",
                split="validation",
                raw_user_id="synthetic-user-long",
                session_id="synthetic-session-long",
                timestamp_utc=start + timedelta(minutes=index),
                raw_poi_id=f"synthetic-poi-{index:03d}",
                category="Synthetic Category",
            )
            for index in range(130)
        )

        examples = build_labeled_examples(events, split_protocol="fixture-v1")
        final = next(
            example
            for example in examples
            if example.target_poi_id == "synthetic-poi-129"
        )

        self.assertEqual(len(final.request.history), 128)
        self.assertEqual(final.request.history[0].poi_id, "synthetic-poi-001")
        self.assertEqual(final.request.history[-1].poi_id, "synthetic-poi-128")

    def test_documented_sample_id_preimage_has_fixed_sha256(self) -> None:
        sample_id = compute_sample_id(
            dataset="synthetic",
            split_protocol="fixture-v1",
            raw_user_id="synthetic-user-aurora",
            session_id="synthetic-session-train-aurora",
            target_timestamp_utc=datetime(2026, 1, 1, 9, tzinfo=timezone.utc),
            target_raw_poi_id="synthetic-poi-cafe",
        )

        self.assertEqual(sample_id, EXPECTED_SAMPLE_ID)

    def test_example_identity_and_history_do_not_depend_on_input_position(self) -> None:
        splits = read_synthetic_splits({
            split: FIXTURE_ROOT / f"{split}.csv"
            for split in ("train", "validation", "test")
        })
        events = tuple(event for split in splits.values() for event in split)

        examples = build_labeled_examples(events, split_protocol="fixture-v1")
        reversed_examples = build_labeled_examples(
            reversed(events), split_protocol="fixture-v1"
        )

        self.assertEqual(examples, reversed_examples)
        self.assertEqual(len(examples), 17)
        self.assertEqual(len({example.sample_id for example in examples}), len(examples))
        target = next(example for example in examples if example.sample_id == EXPECTED_SAMPLE_ID)
        self.assertEqual(target.target_poi_id, "synthetic-poi-cafe")
        self.assertEqual([item.poi_id for item in target.request.history], ["synthetic-poi-hub"])


if __name__ == "__main__":
    unittest.main()
