from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import mlflow

from next_poi.evaluation import run_evaluation_pipeline

FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "synthetic"


class BatchPipelineTests(unittest.TestCase):
    def test_single_variant_fixture_pipeline_writes_all_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            runs = run_evaluation_pipeline(
                dataset="synthetic",
                train_path=FIXTURE_ROOT / "train.csv",
                validation_path=FIXTURE_ROOT / "validation.csv",
                test_path=FIXTURE_ROOT / "test.csv",
                output_directory=root / "output",
                tracking_directory=root / "mlruns",
                split_protocol="fixture-v1",
                variants=("b0",),
                experiment_name="gate2-cli",
            )

            self.assertEqual(len(runs), 1)
            run = runs[0]
            self.assertEqual(run.variant, "b0")
            self.assertTrue(run.bundle.directory.is_dir())
            self.assertTrue(run.data_artifacts.encoder_path.is_file())
            self.assertTrue(run.data_artifacts.manifest_path.is_file())
            self.assertTrue(run.data_artifacts.audit_path.is_file())
            self.assertTrue(run.release.path.is_file())
            self.assertEqual(
                run.release.manifest.data_manifest_sha256,
                run.data_artifacts.manifest_sha256,
            )
            self.assertEqual(
                run.release.manifest.model_manifest_sha256,
                run.bundle.manifest_sha256,
            )
            self.assertTrue(run.report_artifacts.core_path.is_file())
            self.assertTrue(run.report_artifacts.report_path.is_file())
            self.assertTrue(run.report_artifacts.markdown_path.is_file())
            self.assertTrue(run.tracking.tracking_uri.startswith("file:"))
            client = mlflow.MlflowClient(tracking_uri=run.tracking.tracking_uri)
            tracked_run = client.get_run(run.tracking.run_id)
            lineage_paths = {
                item.path
                for item in client.list_artifacts(run.tracking.run_id, "lineage")
            }
            self.assertEqual(
                tracked_run.data.params["data_manifest_sha256"],
                run.data_artifacts.manifest_sha256,
            )
            self.assertEqual(
                tracked_run.data.params["bundle_manifest_sha256"],
                run.bundle.manifest_sha256,
            )
            self.assertEqual(
                tracked_run.data.params["release_manifest_sha256"],
                run.release.sha256,
            )
            self.assertEqual(len(tracked_run.data.params["sample_set_sha256"]), 64)
            self.assertEqual(
                lineage_paths,
                {
                    "lineage/config.json",
                    "lineage/data_manifest.json",
                    "lineage/manifest.json",
                    "lineage/release_manifest.json",
                },
            )


if __name__ == "__main__":
    unittest.main()
