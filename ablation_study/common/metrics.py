from __future__ import annotations

from typing import Iterable


def ranking_metrics(ranks: Iterable[int | None], ks: tuple[int, ...] = (1, 5, 10)) -> dict[str, float]:
    ranks = list(ranks)
    total = len(ranks)
    if total == 0:
        return {f"hit@{k}": 0.0 for k in ks} | {"mrr": 0.0, "total": 0}

    metrics: dict[str, float] = {}
    for k in ks:
        metrics[f"hit@{k}"] = sum(1 for rank in ranks if rank is not None and rank < k) / total
    metrics["mrr"] = sum((1.0 / (rank + 1)) for rank in ranks if rank is not None) / total
    metrics["total"] = total
    return metrics


def format_metrics(metrics: dict[str, float]) -> dict[str, float | int]:
    result: dict[str, float | int] = {}
    for key, value in metrics.items():
        if key == "total":
            result[key] = int(value)
        else:
            result[key] = round(float(value), 6)
    return result
