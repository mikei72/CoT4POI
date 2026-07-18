from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import mlflow

from next_poi.contracts import VersionInfo
from next_poi.data import build_labeled_examples, hash_events, read_synthetic_splits
from next_poi.evaluation import evaluate_examples, write_report_artifacts
from next_poi.models import (
    VARIANT_SOURCES,
    CandidateIndex,
    SmokePredictor,
    load_smoke_bundle,
    save_smoke_bundle,
)
from next_poi.tracking import log_evaluation_run

FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "synthetic"


def _recommendation_core(predictor: SmokePredictor, request):
    response = predictor.predict(request)
    return tuple(
        (item.poi_id, item.model_poi_id, item.category, item.score, item.candidate_sources)
        for item in response.recommendations
    )


class Gate2SmokePipelineTests(unittest.TestCase):
    def test_fixture_runs_all_variants_bundle_evaluation_and_mlflow(self) -> None:
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
        train_fingerprint = hash_events(splits["train"])
        core_hashes: dict[str, str] = {}
        bundle_hashes: dict[str, str] = {}

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for variant in VARIANT_SOURCES:
                predictor = SmokePredictor(
                    index,
                    variant=variant,
                    versions=VersionInfo(
                        release="fixture-v1",
                        data=train_fingerprint,
                        model=f"smoke-{variant}-v1",
                    ),
                )
                bundle_info = save_smoke_bundle(root / "bundles" / variant, predictor)
                loaded, loaded_info = load_smoke_bundle(root / "bundles" / variant)
                self.assertEqual(bundle_info.manifest_sha256, loaded_info.manifest_sha256)
                bundle_hashes[variant] = bundle_info.manifest_sha256
                for example in examples:
                    self.assertEqual(
                        _recommendation_core(predictor, example.request),
                        _recommendation_core(loaded, example.request),
                    )

                first = evaluate_examples(
                    loaded,
                    examples,
                    data_fingerprint=train_fingerprint,
                )
                repeated = evaluate_examples(
                    loaded,
                    examples,
                    data_fingerprint=train_fingerprint,
                )
                self.assertEqual(first.report.core, repeated.report.core)
                core_hashes[variant] = first.report.core_sha256
                artifacts = write_report_artifacts(root / "reports" / variant, first.report)
                log_evaluation_run(
                    first.report,
                    artifacts,
                    tracking_directory=root / "mlruns",
                    experiment_name="gate2-fixture-matrix",
                    lineage_artifacts=(
                        bundle_info.directory / "config.json",
                        bundle_info.directory / "manifest.json",
                    ),
                )

            duplicate_predictor = SmokePredictor(
                index,
                variant="b3",
                versions=VersionInfo(
                    release="fixture-v1",
                    data=train_fingerprint,
                    model="smoke-b3-v1",
                ),
            )
            duplicate_info = save_smoke_bundle(
                root / "bundles" / "b3-repeat", duplicate_predictor
            )
            self.assertEqual(bundle_hashes["b3"], duplicate_info.manifest_sha256)

            tracking_uri = (root / "mlruns").resolve().as_uri()
            client = mlflow.MlflowClient(tracking_uri=tracking_uri)
            experiment = client.get_experiment_by_name("gate2-fixture-matrix")
            self.assertIsNotNone(experiment)
            runs = client.search_runs([experiment.experiment_id])
            lineage_artifacts = {
                run.info.run_id: {
                    item.path
                    for item in client.list_artifacts(run.info.run_id, "lineage")
                }
                for run in runs
            }

        self.assertEqual(set(core_hashes), set(VARIANT_SOURCES))
        self.assertEqual(len(set(core_hashes.values())), len(VARIANT_SOURCES))
        self.assertEqual(len(runs), len(VARIANT_SOURCES))
        self.assertEqual(
            {run.data.params["variant"] for run in runs}, set(VARIANT_SOURCES)
        )
        self.assertTrue(
            all(run.data.tags["result_lineage"] == "production_current" for run in runs)
        )
        self.assertTrue(all(run.data.tags["result_source"] == "rerun" for run in runs))
        self.assertEqual(
            len({run.data.params["config_fingerprint"] for run in runs}),
            len(VARIANT_SOURCES),
        )
        self.assertEqual(len({run.data.params["sample_set_sha256"] for run in runs}), 1)
        self.assertTrue(
            all(
                paths == {"lineage/config.json", "lineage/manifest.json"}
                for paths in lineage_artifacts.values()
            )
        )


if __name__ == "__main__":
    unittest.main()
