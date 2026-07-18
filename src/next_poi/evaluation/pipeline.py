"""End-to-end CPU batch evaluation using only frozen public layer APIs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from next_poi.contracts import DatasetName, VersionInfo
from next_poi.data import (
    audit_splits,
    build_data_manifest,
    build_labeled_examples,
    build_train_encoder,
    export_encoder_sidecar,
    hash_events,
    read_nyc_splits,
    read_synthetic_splits,
    write_data_manifest,
)
from next_poi.data._serialization import write_stable_json
from next_poi.evaluation.evaluator import evaluate_examples
from next_poi.evaluation.report import ReportArtifacts, write_report_artifacts
from next_poi.models import (
    VARIANT_SOURCES,
    BundleInfo,
    CandidateIndex,
    SmokePredictor,
    load_smoke_bundle,
    save_smoke_bundle,
)
from next_poi.release import ReleaseInfo, build_smoke_release
from next_poi.tracking import MlflowRunInfo, log_evaluation_run


@dataclass(frozen=True)
class DataArtifacts:
    encoder_path: Path
    manifest_path: Path
    audit_path: Path
    manifest_sha256: str


@dataclass(frozen=True)
class PipelineRun:
    variant: str
    core_sha256: str
    bundle: BundleInfo
    data_artifacts: DataArtifacts
    release: ReleaseInfo
    report_artifacts: ReportArtifacts
    tracking: MlflowRunInfo


def run_evaluation_pipeline(
    *,
    dataset: DatasetName,
    train_path: str | Path,
    validation_path: str | Path,
    test_path: str | Path,
    output_directory: str | Path,
    split_protocol: str,
    variants: tuple[str, ...] = tuple(VARIANT_SOURCES),
    release_version: str = "smoke-v1",
    top_k: int = 10,
    tracking_directory: str | Path | None = None,
    experiment_name: str = "next-poi-smoke",
) -> tuple[PipelineRun, ...]:
    """Fit on train only, then bundle, evaluate, report, and track each variant."""

    if not split_protocol:
        raise ValueError("split_protocol must be non-empty")
    if not variants or len(variants) != len(set(variants)):
        raise ValueError("variants must be non-empty and unique")
    unsupported = sorted(set(variants) - set(VARIANT_SOURCES))
    if unsupported:
        raise ValueError(f"unsupported smoke variants: {unsupported}")

    split_paths = {
        "train": Path(train_path),
        "validation": Path(validation_path),
        "test": Path(test_path),
    }
    if dataset == "synthetic":
        splits = read_synthetic_splits(split_paths)
    elif dataset == "nyc":
        splits = read_nyc_splits(split_paths)
    else:
        raise ValueError("batch pipeline currently supports synthetic or nyc inputs")

    # Taxonomy is the allowed cross-split fixed label space. No validation/test
    # event is passed to CandidateIndex.fit or any learned count computation.
    taxonomy = {
        event.category for split_events in splits.values() for event in split_events
    }
    index = CandidateIndex.fit(splits["train"], taxonomy=taxonomy)
    examples = build_labeled_examples(
        (*splits["validation"], *splits["test"]),
        split_protocol=split_protocol,
        top_k=top_k,
    )
    train_fingerprint = hash_events(splits["train"])
    output_root = Path(output_directory)
    output_root.mkdir(parents=True, exist_ok=True)
    data_root = output_root / "data"
    if data_root.exists() and any(data_root.iterdir()):
        raise FileExistsError("evaluation data artifact output already exists")
    data_root.mkdir(parents=True, exist_ok=True)
    encoder_path = data_root / "encoder.json"
    export_encoder_sidecar(
        encoder_path,
        build_train_encoder(
            splits["train"],
            dataset=dataset,
            known_taxonomy=taxonomy,
        ),
    )
    data_manifest_path = data_root / "data_manifest.json"
    data_manifest = build_data_manifest(
        splits,
        dataset=dataset,
        split_protocol=split_protocol,
        encoder_path=encoder_path,
        taxonomy=taxonomy,
    )
    data_manifest_sha256 = write_data_manifest(data_manifest_path, data_manifest)
    audit_path = data_root / "split_audit.json"
    write_stable_json(audit_path, audit_splits(splits).to_dict())
    data_artifacts = DataArtifacts(
        encoder_path=encoder_path,
        manifest_path=data_manifest_path,
        audit_path=audit_path,
        manifest_sha256=data_manifest_sha256,
    )
    tracking_root = (
        Path(tracking_directory) if tracking_directory is not None else output_root / "mlruns"
    )

    runs: list[PipelineRun] = []
    for variant in variants:
        variant_root = output_root / variant
        if variant_root.exists() and any(variant_root.iterdir()):
            raise FileExistsError(f"evaluation output already exists for variant: {variant}")
        versions = VersionInfo(
            release=f"{release_version}-{variant}",
            data=data_manifest_sha256,
            model=f"smoke-{variant}-v1",
        )
        predictor = SmokePredictor(index, variant=variant, versions=versions)
        bundle = save_smoke_bundle(variant_root / "bundle", predictor)
        loaded_predictor, loaded_bundle = load_smoke_bundle(bundle.directory)
        if loaded_bundle.manifest_sha256 != bundle.manifest_sha256:
            raise ValueError("bundle manifest hash changed during round-trip")
        result = evaluate_examples(
            loaded_predictor,
            examples,
            data_fingerprint=data_manifest_sha256,
        )
        artifacts = write_report_artifacts(variant_root / "evaluation", result.report)
        release = build_smoke_release(
            data_manifest_path=data_manifest_path,
            model_manifest_path=bundle.directory / "manifest.json",
            config_path=bundle.directory / "config.json",
            output_path=variant_root / "release_manifest.json",
            release_version=versions.release,
        )
        tracking = log_evaluation_run(
            result.report,
            artifacts,
            tracking_directory=tracking_root,
            experiment_name=experiment_name,
            params={
                "train_data_sha256": train_fingerprint,
                "data_manifest_sha256": data_manifest_sha256,
                "bundle_manifest_sha256": bundle.manifest_sha256,
                "release_manifest_sha256": release.sha256,
                "release_version": versions.release,
            },
            lineage_artifacts=(
                data_manifest_path,
                bundle.directory / "config.json",
                bundle.directory / "manifest.json",
                release.path,
            ),
        )
        runs.append(
            PipelineRun(
                variant=variant,
                core_sha256=result.report.core_sha256,
                bundle=bundle,
                data_artifacts=data_artifacts,
                release=release,
                report_artifacts=artifacts,
                tracking=tracking,
            )
        )
    return tuple(runs)
