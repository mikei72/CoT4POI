"""In-process HTTP smoke test for a saved deterministic bundle."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from next_poi.serving.app import create_app


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path, help="smoke bundle directory")
    return parser


def _request_payload(dataset: str) -> dict[str, Any]:
    timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat()
    return {
        "dataset": dataset,
        "history": [
            {
                "poi_id": "smoke-client-unknown-poi",
                "category_name": "smoke-client-unknown-category",
                "timestamp": timestamp,
            }
        ],
        "target_time": timestamp,
        "top_k": 5,
        "profile": "smoke",
    }


def run(bundle: Path) -> dict[str, Any]:
    """Exercise lifecycle and all five HTTP endpoints through TestClient."""

    application = create_app(bundle)
    with TestClient(application, raise_server_exceptions=False) as client:
        health = client.get("/health")
        ready = client.get("/ready")
        version = client.get("/version")
        if health.status_code != 200 or ready.status_code != 200 or version.status_code != 200:
            return {
                "status": "failed",
                "health_status": health.status_code,
                "ready_status": ready.status_code,
                "version_status": version.status_code,
            }
        state = application.state.next_poi
        dataset = state.predictor.index.dataset
        recommendation = client.post("/recommend", json=_request_payload(dataset))
        metrics = client.get("/metrics")
        if recommendation.status_code != 200 or metrics.status_code != 200:
            return {
                "status": "failed",
                "recommend_status": recommendation.status_code,
                "metrics_status": metrics.status_code,
            }
        payload = recommendation.json()
        return {
            "status": "passed",
            "profile": "smoke",
            "recommendation_count": len(payload["recommendations"]),
            "request_id": payload["request_id"],
            "versions": version.json(),
        }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run(args.bundle)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
