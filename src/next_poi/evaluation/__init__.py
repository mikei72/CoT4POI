"""Target-aware offline evaluation for the shared smoke Predictor."""

from next_poi.evaluation.evaluator import EvaluationResult, evaluate_examples
from next_poi.evaluation.metrics import (
    RankingObservation,
    aggregate_ranking_metrics,
    candidate_recall_at_k,
    hit_at_k,
    ndcg_at_k,
    reciprocal_rank,
)
from next_poi.evaluation.pipeline import DataArtifacts, PipelineRun, run_evaluation_pipeline
from next_poi.evaluation.report import (
    EvaluationReport,
    ReportArtifacts,
    build_report,
    write_report,
    write_report_artifacts,
)

__all__ = [
    "EvaluationReport",
    "EvaluationResult",
    "DataArtifacts",
    "PipelineRun",
    "ReportArtifacts",
    "RankingObservation",
    "aggregate_ranking_metrics",
    "build_report",
    "candidate_recall_at_k",
    "evaluate_examples",
    "hit_at_k",
    "ndcg_at_k",
    "reciprocal_rank",
    "run_evaluation_pipeline",
    "write_report",
    "write_report_artifacts",
]
