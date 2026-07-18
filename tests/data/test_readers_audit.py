from __future__ import annotations

import csv
import tempfile
import unittest
from datetime import timezone
from pathlib import Path

from next_poi.data import audit_splits, read_nyc_split, read_synthetic_splits

FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "synthetic"


def fixture_paths() -> dict[str, Path]:
    return {
        split: FIXTURE_ROOT / f"{split}.csv"
        for split in ("train", "validation", "test")
    }


class SyntheticReaderTests(unittest.TestCase):
    def test_fixture_is_repository_safe_and_preserves_explicit_splits(self) -> None:
        splits = read_synthetic_splits(fixture_paths())

        self.assertEqual({key: len(value) for key, value in splits.items()}, {
            "train": 12,
            "validation": 6,
            "test": 6,
        })
        for split, events in splits.items():
            self.assertTrue(events)
            for event in events:
                self.assertEqual(event.split, split)
                self.assertTrue(event.raw_user_id.startswith("synthetic-user-"))
                self.assertTrue(event.session_id.startswith("synthetic-session-"))
                self.assertTrue(event.raw_poi_id.startswith("synthetic-poi-"))
                self.assertTrue(event.category.startswith("Synthetic "))
                self.assertIsNone(event.latitude)
                self.assertIsNone(event.longitude)
                self.assertEqual(event.timestamp_utc.utcoffset(), timezone.utc.utcoffset(None))

        for path in fixture_paths().values():
            header = path.read_text(encoding="utf-8").splitlines()[0].lower()
            self.assertNotIn("latitude", header)
            self.assertNotIn("longitude", header)

    def test_reader_rejects_a_row_that_claims_another_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "train.csv"
            path.write_text(
                "raw_user_id,session_id,timestamp,raw_poi_id,category,split\n"
                "synthetic-user-a,synthetic-session-a,2026-01-01T00:00:00Z,"
                "synthetic-poi-a,Synthetic Cafe,test\n",
                encoding="utf-8",
            )
            paths = fixture_paths()
            paths["train"] = path

            with self.assertRaisesRegex(ValueError, "caller supplied explicit split"):
                read_synthetic_splits(paths)


class NycReaderTests(unittest.TestCase):
    def test_normalizes_legacy_headers_without_resplitting(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "NYC_val.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow([
                    "UserId",
                    "PoiId",
                    "PoiCategoryName",
                    "Latitude",
                    "Longitude",
                    "UTCTime",
                    "pseudo_session_trajectory_id",
                ])
                writer.writerow([
                    "invented-user",
                    "invented-poi",
                    "Invented Category",
                    "40.5",
                    "-73.5",
                    "2026-02-01 04:00:00+00:00",
                    "invented-session",
                ])

            events = read_nyc_split(path, "validation")

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].split, "validation")
        self.assertEqual(events[0].dataset, "nyc")
        self.assertEqual(events[0].timestamp_utc.tzinfo, timezone.utc)
        self.assertEqual((events[0].latitude, events[0].longitude), (40.5, -73.5))


class SplitAuditTests(unittest.TestCase):
    def test_reports_counts_ranges_order_and_overlap_without_mutation(self) -> None:
        splits = read_synthetic_splits(fixture_paths())
        original_train = tuple(splits["train"])
        duplicated = splits["train"][0].model_copy(update={"split": "validation"})
        audit_input = {
            "train": splits["train"],
            "validation": (duplicated,) + splits["validation"],
            "test": splits["test"],
        }

        report = audit_splits(audit_input)

        self.assertEqual([item.event_count for item in report.summaries], [12, 7, 6])
        self.assertTrue(report.has_event_overlap)
        self.assertTrue(report.has_session_overlap)
        self.assertFalse(report.temporal_splits_ordered)
        self.assertEqual(splits["train"], original_train)
        self.assertEqual(audit_input["validation"][0], duplicated)

    def test_clean_fixture_has_no_overlap_and_ordered_boundaries(self) -> None:
        report = audit_splits(read_synthetic_splits(fixture_paths()))

        self.assertFalse(report.has_event_overlap)
        self.assertFalse(report.has_session_overlap)
        self.assertTrue(report.temporal_splits_ordered)
        self.assertTrue(all(item.input_time_ordered for item in report.summaries))

    def test_event_overlap_does_not_depend_on_sessionization_identity(self) -> None:
        splits = read_synthetic_splits(fixture_paths())
        duplicated_event = splits["train"][0].model_copy(
            update={
                "split": "validation",
                "session_id": "synthetic-session-validation-reassigned",
            }
        )

        report = audit_splits({
            "train": splits["train"],
            "validation": (duplicated_event,) + splits["validation"],
            "test": splits["test"],
        })
        train_validation_event = next(
            item
            for item in report.event_overlaps
            if (item.left_split, item.right_split) == ("train", "validation")
        )
        train_validation_session = next(
            item
            for item in report.session_overlaps
            if (item.left_split, item.right_split) == ("train", "validation")
        )

        self.assertEqual(train_validation_event.count, 1)
        self.assertEqual(train_validation_session.count, 0)


if __name__ == "__main__":
    unittest.main()
