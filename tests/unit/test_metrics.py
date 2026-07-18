from __future__ import annotations

import unittest

from next_poi.evaluation import (
    RankingObservation,
    aggregate_ranking_metrics,
    candidate_recall_at_k,
    hit_at_k,
    ndcg_at_k,
    reciprocal_rank,
)


class RankingMetricTests(unittest.TestCase):
    def test_single_relevant_item_metrics(self) -> None:
        predictions = ("poi-a", "poi-b", "poi-c")

        self.assertEqual(hit_at_k(predictions, "poi-b", 1), 0.0)
        self.assertEqual(hit_at_k(predictions, "poi-b", 5), 1.0)
        self.assertEqual(reciprocal_rank(predictions, "poi-b"), 0.5)
        self.assertAlmostEqual(ndcg_at_k(predictions, "poi-b", 5), 1.0 / 1.5849625007)
        self.assertEqual(candidate_recall_at_k(predictions, "poi-c", 2), 0.0)

    def test_empty_target_prediction_and_short_list_are_defined(self) -> None:
        self.assertEqual(hit_at_k((), "poi-a", 10), 0.0)
        self.assertEqual(reciprocal_rank(("poi-a",), None), 0.0)
        self.assertEqual(ndcg_at_k(("poi-a",), "poi-a", 10), 1.0)
        with self.assertRaisesRegex(ValueError, "at least one"):
            hit_at_k(("poi-a",), "poi-a", 0)

    def test_macro_metrics_use_ranked_categories(self) -> None:
        row = RankingObservation(
            "a" * 64,
            "poi-a",
            ("poi-a",),
            ("poi-a",),
            target_category="Cafe",
            macro_predictions=("Park", "Cafe"),
        )

        metrics = aggregate_ranking_metrics((row,), catalog=("poi-a",))

        self.assertEqual(metrics["macro_hit_at_1"], 0.0)
        self.assertEqual(metrics["macro_hit_at_3"], 1.0)
        self.assertEqual(metrics["macro_mrr"], 0.5)


class AggregateMetricTests(unittest.TestCase):
    def test_coverage_validity_duplicates_invalids_and_empties(self) -> None:
        rows = (
            RankingObservation("a" * 64, "poi-a", ("poi-a", "poi-a"), ("poi-a",)),
            RankingObservation("b" * 64, "poi-b", ("unknown",), ("poi-b",)),
            RankingObservation("c" * 64, "poi-c", (), ()),
        )

        metrics = aggregate_ranking_metrics(rows, catalog=("poi-a", "poi-b", "poi-c"))

        self.assertEqual(metrics["hit_at_1"], 1.0 / 3.0)
        self.assertEqual(metrics["coverage"], 1.0 / 3.0)
        self.assertEqual(metrics["validity_rate"], 0.0)
        self.assertEqual(metrics["duplicate_rate"], 1.0 / 3.0)
        self.assertEqual(metrics["invalid_rate"], 1.0 / 3.0)
        self.assertEqual(metrics["empty_rate"], 1.0 / 3.0)

    def test_empty_batch_returns_zeroes(self) -> None:
        metrics = aggregate_ranking_metrics((), catalog=("poi-a",))

        self.assertTrue(metrics)
        self.assertTrue(all(value == 0.0 for value in metrics.values()))


if __name__ == "__main__":
    unittest.main()
