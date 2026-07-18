"""Stable evaluation report cores with runtime metadata kept out of core hashes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from next_poi.data._serialization import canonical_json_bytes, sha256_bytes, write_stable_json

REPORT_SCHEMA_VERSION = "1"


@dataclass(frozen=True)
class EvaluationReport:
    core: dict[str, Any]
    runtime: dict[str, Any]

    @property
    def core_sha256(self) -> str:
        return str(self.core["core_sha256"])

    def to_dict(self) -> dict[str, Any]:
        return {"core": self.core, "runtime": self.runtime}


@dataclass(frozen=True)
class ReportArtifacts:
    core_path: Path
    report_path: Path
    markdown_path: Path
    core_file_sha256: str
    report_file_sha256: str
    markdown_file_sha256: str


def build_report(
    *,
    dataset: str,
    variant: str,
    model_version: str,
    data_fingerprint: str,
    sample_ids: tuple[str, ...],
    metrics: dict[str, float],
    slices: dict[str, Any],
    failure_cases: tuple[dict[str, Any], ...],
    runtime: dict[str, Any],
) -> EvaluationReport:
    """Build a reproducible core; durations and wall-clock fields live in runtime."""

    core_payload: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "result_lineage": "production_current",
        "result_source": "rerun",
        "dataset": dataset,
        "variant": variant,
        "model_version": model_version,
        "data_fingerprint": data_fingerprint,
        "sample_ids": list(sorted(sample_ids)),
        "metrics": {name: metrics[name] for name in sorted(metrics)},
        "slices": slices,
        "failure_cases": list(failure_cases),
    }
    core_hash = sha256_bytes(canonical_json_bytes(core_payload))
    core = {**core_payload, "core_sha256": core_hash}
    return EvaluationReport(core=core, runtime=dict(runtime))


def write_report(path: str | Path, report: EvaluationReport) -> str:
    return write_stable_json(path, report.to_dict())


def write_report_artifacts(
    directory: str | Path,
    report: EvaluationReport,
) -> ReportArtifacts:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    core_path = root / "evaluation_core.json"
    report_path = root / "evaluation_report.json"
    markdown_path = root / "evaluation_report.md"
    core_hash = write_stable_json(core_path, report.core)
    report_hash = write_stable_json(report_path, report.to_dict())
    markdown = _render_markdown(report)
    markdown_path.write_text(markdown, encoding="utf-8")
    markdown_hash = sha256_bytes(markdown.encode("utf-8"))
    return ReportArtifacts(
        core_path=core_path,
        report_path=report_path,
        markdown_path=markdown_path,
        core_file_sha256=core_hash,
        report_file_sha256=report_hash,
        markdown_file_sha256=markdown_hash,
    )


def _render_markdown(report: EvaluationReport) -> str:
    core = report.core
    lines = [
        "# Deterministic smoke evaluation",
        "",
        f"- Dataset: `{core['dataset']}`",
        f"- Variant: `{core['variant']}`",
        f"- Model: `{core['model_version']}`",
        f"- Result lineage: `{core['result_lineage']}`",
        f"- Result source: `{core['result_source']}`",
        f"- Reproducible core SHA-256: `{core['core_sha256']}`",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    lines.extend(
        f"| {name} | {value:.12g} |" for name, value in core["metrics"].items()
    )
    lines.extend(
        [
            "",
            "## Runtime (excluded from core hash)",
            "",
            "```json",
            canonical_json_bytes(report.runtime).decode("utf-8"),
            "```",
            "",
        ]
    )
    return "\n".join(lines)
