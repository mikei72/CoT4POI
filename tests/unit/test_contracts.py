from __future__ import annotations

import unittest
from datetime import datetime, timezone

from pydantic import ValidationError

from next_poi.contracts import (
    DataManifest,
    DataSplitSummary,
    FileDigest,
    HistoryEvent,
    LatencyBreakdown,
    ModelManifest,
    NormalizedEvent,
    Recommendation,
    RecommendationRequest,
    RecommendationResponse,
    VersionInfo,
)

SHA = "a" * 64


def history_event() -> HistoryEvent:
    return HistoryEvent(
        poi_id="poi-alpha",
        category_name="Cafe",
        timestamp=datetime(2026, 1, 1, 8, tzinfo=timezone.utc),
    )


class RecommendationRequestTests(unittest.TestCase):
    def test_forbids_leakage_fields(self) -> None:
        leakage_fields = (
            "target",
            "target_poi",
            "target_poi_id",
            "target_category",
            "label",
            "result",
        )
        for field in leakage_fields:
            with self.subTest(field=field), self.assertRaises(ValidationError):
                RecommendationRequest.model_validate(
                    {
                        "dataset": "synthetic",
                        "history": [history_event().model_dump()],
                        "target_time": "2026-01-01T09:00:00Z",
                        "top_k": 5,
                        field: "poi-secret",
                    }
                )

    def test_rejects_naive_time(self) -> None:
        with self.assertRaises(ValidationError):
            RecommendationRequest(
                dataset="synthetic",
                history=(history_event(),),
                target_time=datetime(2026, 1, 1, 9),
            )

    def test_rejects_target_before_history(self) -> None:
        with self.assertRaises(ValidationError):
            RecommendationRequest(
                dataset="synthetic",
                history=(history_event(),),
                target_time=datetime(2026, 1, 1, 7, tzinfo=timezone.utc),
            )


class ResponseContractTests(unittest.TestCase):
    def test_rejects_duplicate_recommendations(self) -> None:
        item = Recommendation(
            rank=1,
            poi_id="poi-alpha",
            category="Cafe",
            score=1.0,
            candidate_sources=("global_popularity",),
        )
        duplicate = item.model_copy(update={"rank": 2})

        with self.assertRaises(ValidationError):
            RecommendationResponse(
                recommendations=(item, duplicate),
                versions=VersionInfo(release="r1", data="d1", model="m1"),
                latency=LatencyBreakdown(total_ms=1.0),
                request_id="request-1",
            )

    def test_rejects_infinite_latency(self) -> None:
        with self.assertRaises(ValidationError):
            LatencyBreakdown(total_ms=float("inf"))


class NormalizedEventTests(unittest.TestCase):
    def test_timestamp_is_normalized_to_utc(self) -> None:
        event = NormalizedEvent(
            dataset="synthetic",
            split="train",
            raw_user_id="syn-user",
            session_id="syn-session",
            timestamp_utc="2026-01-01T08:00:00-05:00",
            raw_poi_id="syn-poi",
            category="Cafe",
        )

        self.assertEqual(event.timestamp_utc.isoformat(), "2026-01-01T13:00:00+00:00")


class ManifestContractTests(unittest.TestCase):
    def test_data_manifest_requires_all_frozen_splits(self) -> None:
        train = DataSplitSummary(split="train", count=1, content_sha256=SHA)
        with self.assertRaises(ValidationError):
            DataManifest(
                schema_version="1",
                dataset="synthetic",
                split_protocol="original",
                taxonomy_sha256=SHA,
                encoder_sha256=SHA,
                splits=(train,),
            )

    def test_static_only_cannot_claim_dynamic_verification(self) -> None:
        with self.assertRaises(ValidationError):
            ModelManifest(
                schema_version="1",
                model_name="legacy-gpu",
                backend="full-gpu",
                runtime_status="static_only",
                dynamic_load_verified=True,
                files=(),
                config_sha256=SHA,
            )

    def test_manifest_paths_are_relative(self) -> None:
        with self.assertRaises(ValidationError):
            FileDigest(path="/private/model.bin", size_bytes=1, sha256=SHA)

    def test_manifest_paths_are_platform_portable(self) -> None:
        with self.assertRaises(ValidationError):
            FileDigest(path=r"C:\private\model.bin", size_bytes=1, sha256=SHA)

    def test_missing_manifest_paths_are_relative(self) -> None:
        with self.assertRaises(ValidationError):
            ModelManifest(
                schema_version="1",
                model_name="legacy-gpu",
                backend="full-gpu",
                runtime_status="static_only",
                dynamic_load_verified=False,
                files=(),
                missing_files=("/private/model.bin",),
                config_sha256=SHA,
            )

    def test_ready_manifest_rejects_file_marked_absent(self) -> None:
        absent = FileDigest(path="model.bin", size_bytes=1, sha256=SHA, present=False)
        with self.assertRaises(ValidationError):
            ModelManifest(
                schema_version="1",
                model_name="smoke",
                backend="cpu",
                runtime_status="ready",
                dynamic_load_verified=True,
                files=(absent,),
                config_sha256=SHA,
            )

    def test_absent_file_must_be_listed_as_missing(self) -> None:
        absent = FileDigest(path="model.bin", size_bytes=1, sha256=SHA, present=False)
        with self.assertRaises(ValidationError):
            ModelManifest(
                schema_version="1",
                model_name="legacy-gpu",
                backend="full-gpu",
                runtime_status="static_only",
                dynamic_load_verified=False,
                files=(absent,),
                config_sha256=SHA,
            )

    def test_static_only_manifest_can_list_absent_file(self) -> None:
        absent = FileDigest(path="model.bin", size_bytes=1, sha256=SHA, present=False)
        manifest = ModelManifest(
            schema_version="1",
            model_name="legacy-gpu",
            backend="full-gpu",
            runtime_status="static_only",
            dynamic_load_verified=False,
            files=(absent,),
            missing_files=("model.bin",),
            config_sha256=SHA,
        )

        self.assertEqual(manifest.missing_files, ("model.bin",))

    def test_present_file_cannot_be_listed_as_missing(self) -> None:
        present = FileDigest(path="model.bin", size_bytes=1, sha256=SHA)
        with self.assertRaises(ValidationError):
            ModelManifest(
                schema_version="1",
                model_name="legacy-gpu",
                backend="full-gpu",
                runtime_status="static_only",
                dynamic_load_verified=False,
                files=(present,),
                missing_files=("model.bin",),
                config_sha256=SHA,
            )

    def test_manifest_file_paths_are_unique(self) -> None:
        file_digest = FileDigest(path="model.bin", size_bytes=1, sha256=SHA)
        with self.assertRaises(ValidationError):
            ModelManifest(
                schema_version="1",
                model_name="legacy-gpu",
                backend="full-gpu",
                runtime_status="static_only",
                dynamic_load_verified=False,
                files=(file_digest, file_digest),
                config_sha256=SHA,
            )


if __name__ == "__main__":
    unittest.main()
