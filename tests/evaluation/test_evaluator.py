from __future__ import annotations

import unittest
from pathlib import Path

from next_poi.contracts import RecommendationRequest, VersionInfo
from next_poi.data import build_labeled_examples, hash_events, read_synthetic_splits
from next_poi.evaluation import evaluate_examples
from next_poi.models import CandidateIndex, SmokePredictor

FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "synthetic"


def fixture_components():
    splits = read_synthetic_splits(
        {
            split: FIXTURE_ROOT / f"{split}.csv"
            for split in ("train", "validation", "test")
        }
    )
    taxonomy = {
        event.category for split_events in splits.values() for event in split_events
    }
    index = CandidateIndex.fit(splits["train"], taxonomy=taxonomy)
    examples = build_labeled_examples(
        (*splits["validation"], *splits["test"]),
        split_protocol="fixture-v1",
    )
    return splits, index, examples


class RecordingPredictor:
    def __init__(self, predictor: SmokePredictor) -> None:
        self._predictor = predictor
        self.index = predictor.index
        self.variant = predictor.variant
        self.versions = predictor.versions
        self.inputs: list[RecommendationRequest] = []

    def predict_with_trace(self, request: RecommendationRequest):
        self.inputs.append(request)
        return self._predictor.predict_with_trace(request)


class EvaluatorTests(unittest.TestCase):
    def test_fixture_report_has_full_metrics_slices_and_stable_core(self) -> None:
        splits, index, examples = fixture_components()
        versions = VersionInfo(
            release="fixture-v1",
            data="fixture-data-v1",
            model="smoke-b3-v1",
        )
        wrapped = RecordingPredictor(
            SmokePredictor(index, variant="b3", versions=versions)
        )
        fingerprint = hash_events(splits["train"])

        first = evaluate_examples(wrapped, examples, data_fingerprint=fingerprint)
        second = evaluate_examples(wrapped, examples, data_fingerprint=fingerprint)

        self.assertEqual(first.prediction_count, len(examples))
        self.assertTrue(all(isinstance(item, RecommendationRequest) for item in wrapped.inputs))
        self.assertEqual(first.report.core, second.report.core)
        self.assertEqual(first.report.core_sha256, second.report.core_sha256)
        self.assertNotEqual(first.report.runtime, second.report.runtime)
        metrics = first.report.core["metrics"]
        self.assertTrue(
            {
                "hit_at_1",
                "hit_at_5",
                "hit_at_10",
                "mrr",
                "ndcg_at_5",
                "ndcg_at_10",
                "candidate_recall_at_50",
                "candidate_recall_at_100",
                "macro_hit_at_1",
                "macro_hit_at_3",
                "macro_mrr",
                "coverage",
                "validity_rate",
                "invalid_rate",
                "duplicate_rate",
                "empty_rate",
            }.issubset(metrics)
        )
        self.assertEqual(
            set(first.report.core["slices"]),
            {"history_length", "target_rarity", "day_type"},
        )
        failure_ids = [
            item["sample_id"] for item in first.report.core["failure_cases"]
        ]
        self.assertEqual(failure_ids, sorted(failure_ids))

    def test_label_mutation_does_not_change_target_blind_predictions(self) -> None:
        splits, index, examples = fixture_components()
        predictor = SmokePredictor(index, variant="b3")
        original = evaluate_examples(
            predictor,
            examples,
            data_fingerprint=hash_events(splits["train"]),
        )
        mutated_examples = tuple(
            example.model_copy(
                update={"target_poi_id": "synthetic-poi-never-trained"}
            )
            for example in examples
        )
        mutated = evaluate_examples(
            predictor,
            mutated_examples,
            data_fingerprint=hash_events(splits["train"]),
        )

        self.assertEqual(
            [row.predictions for row in original.observations],
            [row.predictions for row in mutated.observations],
        )
        self.assertEqual(
            [row.candidates for row in original.observations],
            [row.candidates for row in mutated.observations],
        )


if __name__ == "__main__":
    unittest.main()
