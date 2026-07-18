"""Explicit local-file MLflow tracking for deterministic evaluation reports."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from next_poi.data._serialization import canonical_json_bytes, sha256_bytes
from next_poi.evaluation.report import EvaluationReport, ReportArtifacts
from next_poi.models import SOURCE_WEIGHTS, TIME_BUCKET_VERSION, VARIANT_SOURCES

SCORING_CONFIG_VERSION = "smoke-scoring-v1"


class TrackingUnavailableError(RuntimeError):
    """Raised when the declared tracking extra is not installed."""


@dataclass(frozen=True)
class MlflowRunInfo:
    tracking_uri: str
    experiment_id: str
    run_id: str
    artifact_uri: str


def log_evaluation_run(
    report: EvaluationReport,
    artifacts: ReportArtifacts,
    *,
    tracking_directory: str | Path,
    experiment_name: str = "next-poi-smoke",
    run_name: str | None = None,
    params: dict[str, str | int | float | bool] | None = None,
    lineage_artifacts: Sequence[str | Path] = (),
) -> MlflowRunInfo:
    """Record one rerun in a caller-owned local MLflow file store."""

    mlflow = _require_mlflow()
    evaluation_artifacts = (
        artifacts.core_path,
        artifacts.report_path,
        artifacts.markdown_path,
    )
    lineage_paths = tuple(Path(path) for path in lineage_artifacts)
    artifact_paths = (*evaluation_artifacts, *lineage_paths)
    missing = [path for path in artifact_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"tracking artifact not found: {missing[0].name}")
    lineage_names = [path.name for path in lineage_paths]
    if len(lineage_names) != len(set(lineage_names)):
        raise ValueError("lineage artifact file names must be unique")

    tracking_root = Path(tracking_directory).expanduser().resolve()
    tracking_root.mkdir(parents=True, exist_ok=True)
    tracking_uri = tracking_root.as_uri()
    # MLflow 3 requires an explicit opt-in for its maintained local file backend.
    # This task deliberately uses that backend and never silently switches stores.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment(experiment_name)
    core = report.core
    variant = str(core["variant"])
    if variant not in VARIANT_SOURCES:
        raise ValueError(f"cannot track unsupported smoke variant: {variant}")
    config_fingerprint = sha256_bytes(
        canonical_json_bytes(
            {
                "config_version": SCORING_CONFIG_VERSION,
                "variant": variant,
                "active_sources": VARIANT_SOURCES[variant],
                "source_weights": SOURCE_WEIGHTS,
                "time_bucket_version": TIME_BUCKET_VERSION,
            }
        )
    )
    logged_params: dict[str, Any] = {
        "dataset": core["dataset"],
        "variant": variant,
        "model_version": core["model_version"],
        "data_fingerprint": core["data_fingerprint"],
        "sample_set_sha256": sha256_bytes(
            canonical_json_bytes(sorted(core["sample_ids"]))
        ),
        "config_version": SCORING_CONFIG_VERSION,
        "config_fingerprint": config_fingerprint,
        "evaluation_core_sha256": core["core_sha256"],
        "result_lineage": "production_current",
        "result_source": "rerun",
    }
    if params:
        reserved = set(logged_params) & set(params)
        if reserved:
            raise ValueError(f"tracking params cannot override reserved keys: {sorted(reserved)}")
        logged_params.update(params)

    resolved_run_name = run_name or f"{variant}-{core['core_sha256'][:12]}"
    with mlflow.start_run(
        experiment_id=experiment.experiment_id,
        run_name=resolved_run_name,
        tags={
            "result_lineage": "production_current",
            "result_source": "rerun",
        },
    ) as active_run:
        mlflow.log_params(logged_params)
        mlflow.log_metrics({name: float(value) for name, value in core["metrics"].items()})
        mlflow.log_metrics(_runtime_metrics(report.runtime))
        for path in evaluation_artifacts:
            mlflow.log_artifact(str(path), artifact_path="evaluation")
        for path in lineage_paths:
            mlflow.log_artifact(str(path), artifact_path="lineage")
        run_info = active_run.info

    return MlflowRunInfo(
        tracking_uri=tracking_uri,
        experiment_id=str(run_info.experiment_id),
        run_id=run_info.run_id,
        artifact_uri=run_info.artifact_uri,
    )


def _require_mlflow():
    try:
        import mlflow
    except ImportError as exc:
        raise TrackingUnavailableError(
            "MLflow tracking is required; install the project tracking extra"
        ) from exc
    return mlflow


def _runtime_metrics(runtime: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    duration = runtime.get("evaluation_duration_ms")
    if isinstance(duration, int | float):
        metrics["runtime_evaluation_duration_ms"] = float(duration)
    latency = runtime.get("prediction_latency_ms")
    if isinstance(latency, dict):
        for name in ("mean", "p50", "p95", "max"):
            value = latency.get(name)
            if isinstance(value, int | float):
                metrics[f"runtime_prediction_latency_{name}_ms"] = float(value)
    return metrics
