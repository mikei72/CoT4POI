from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from next_poi.evaluation import build_report, write_report


class EvaluationReportTests(unittest.TestCase):
    def test_runtime_does_not_change_reproducible_core_hash(self) -> None:
        common = {
            "dataset": "synthetic",
            "variant": "b3",
            "model_version": "smoke-b3-v1",
            "data_fingerprint": "a" * 64,
            "sample_ids": ("b" * 64,),
            "metrics": {"mrr": 0.5, "hit_at_1": 0.0},
            "slices": {"history_length": {"1": {"mrr": 0.5}}},
            "failure_cases": (),
        }
        first = build_report(**common, runtime={"duration_ms": 1.0})
        second = build_report(**common, runtime={"duration_ms": 99.0})

        self.assertEqual(first.core, second.core)
        self.assertEqual(first.core_sha256, second.core_sha256)
        self.assertNotEqual(first.runtime, second.runtime)

        with tempfile.TemporaryDirectory() as temporary_directory:
            first_hash = write_report(Path(temporary_directory) / "report.json", first)
            repeated_hash = write_report(
                Path(temporary_directory) / "report-repeated.json", first
            )
        self.assertEqual(first_hash, repeated_hash)


if __name__ == "__main__":
    unittest.main()
