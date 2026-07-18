from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import mlflow

from next_poi.evaluation import build_report, write_report_artifacts
from next_poi.tracking import log_evaluation_run


class MlflowAdapterTests(unittest.TestCase):
    def test_records_local_rerun_lineage_metrics_and_artifacts(self) -> None:
        report = build_report(
            dataset="synthetic",
            variant="b3",
            model_version="smoke-b3-v1",
            data_fingerprint="a" * 64,
            sample_ids=("b" * 64,),
            metrics={"hit_at_1": 1.0, "mrr": 1.0},
            slices={},
            failure_cases=(),
            runtime={
                "evaluation_duration_ms": 1.25,
                "prediction_latency_ms": {
                    "mean": 0.5,
                    "p50": 0.4,
                    "p95": 0.7,
                    "max": 0.8,
                },
            },
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            artifacts = write_report_artifacts(root / "report", report)
            lineage_path = root / "config.json"
            lineage_path.write_text('{"variant":"b3"}\n', encoding="utf-8")
            info = log_evaluation_run(
                report,
                artifacts,
                tracking_directory=root / "mlruns",
                experiment_name="fixture-evaluation",
                lineage_artifacts=(lineage_path,),
            )
            client = mlflow.MlflowClient(tracking_uri=info.tracking_uri)
            run = client.get_run(info.run_id)
            artifact_names = {
                item.path for item in client.list_artifacts(info.run_id, "evaluation")
            }
            lineage_names = {
                item.path for item in client.list_artifacts(info.run_id, "lineage")
            }

        self.assertTrue(info.tracking_uri.startswith("file:"))
        self.assertEqual(run.data.tags["result_lineage"], "production_current")
        self.assertEqual(run.data.tags["result_source"], "rerun")
        self.assertEqual(run.data.params["result_lineage"], "production_current")
        self.assertEqual(run.data.params["result_source"], "rerun")
        self.assertEqual(run.data.params["config_version"], "smoke-scoring-v1")
        self.assertEqual(
            run.data.params["evaluation_core_sha256"], report.core_sha256
        )
        self.assertNotEqual(
            run.data.params["config_fingerprint"], report.core_sha256
        )
        self.assertEqual(
            run.data.params["sample_set_sha256"],
            "bd8aff2bc7e7d9450ce0f4b4acc9982d5fd2abaecaf1700ddce7c8b8e3222661",
        )
        self.assertEqual(run.data.metrics["hit_at_1"], 1.0)
        self.assertEqual(run.data.metrics["runtime_prediction_latency_p95_ms"], 0.7)
        self.assertEqual(
            artifact_names,
            {
                "evaluation/evaluation_core.json",
                "evaluation/evaluation_report.json",
                "evaluation/evaluation_report.md",
            },
        )
        self.assertEqual(lineage_names, {"lineage/config.json"})

    def test_missing_artifact_fails_explicitly(self) -> None:
        report = build_report(
            dataset="synthetic",
            variant="b0",
            model_version="smoke-b0-v1",
            data_fingerprint="a" * 64,
            sample_ids=(),
            metrics={},
            slices={},
            failure_cases=(),
            runtime={},
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            artifacts = write_report_artifacts(root / "report", report)
            artifacts.core_path.unlink()

            with self.assertRaisesRegex(FileNotFoundError, "artifact not found") as raised:
                log_evaluation_run(
                    report,
                    artifacts,
                    tracking_directory=root / "mlruns",
                )
            self.assertNotIn(str(root), str(raised.exception))


if __name__ == "__main__":
    unittest.main()
