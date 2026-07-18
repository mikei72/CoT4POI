"""Command-line entry point for the deterministic CPU evaluation pipeline."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from next_poi.evaluation.pipeline import run_evaluation_pipeline
from next_poi.models import VARIANT_SOURCES


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("synthetic", "nyc"), required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tracking-directory", type=Path)
    parser.add_argument("--split-protocol", required=True)
    parser.add_argument("--release-version", default="smoke-v1")
    parser.add_argument("--experiment-name", default="next-poi-smoke")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--variant",
        action="append",
        choices=tuple(VARIANT_SOURCES),
        dest="variants",
    )
    args = parser.parse_args(argv)
    runs = run_evaluation_pipeline(
        dataset=args.dataset,
        train_path=args.train,
        validation_path=args.validation,
        test_path=args.test,
        output_directory=args.output,
        split_protocol=args.split_protocol,
        variants=tuple(args.variants) if args.variants else tuple(VARIANT_SOURCES),
        release_version=args.release_version,
        top_k=args.top_k,
        tracking_directory=args.tracking_directory,
        experiment_name=args.experiment_name,
    )
    summary = [
        {
            "variant": run.variant,
            "core_sha256": run.core_sha256,
            "bundle_manifest_sha256": run.bundle.manifest_sha256,
            "data_manifest_sha256": run.data_artifacts.manifest_sha256,
            "release_manifest_sha256": run.release.sha256,
            "mlflow_run_id": run.tracking.run_id,
        }
        for run in runs
    ]
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
