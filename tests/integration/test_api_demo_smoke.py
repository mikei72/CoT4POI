from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from next_poi.contracts import RecommendationRequest
from next_poi.data import read_synthetic_splits
from next_poi.demo import ApiClient
from next_poi.models import CandidateIndex, SmokePredictor, save_smoke_bundle
from next_poi.serving import create_app

FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "synthetic"


def _bundle_and_request(tmp_path: Path) -> tuple[Path, RecommendationRequest]:
    splits = read_synthetic_splits(
        {split: FIXTURE_ROOT / f"{split}.csv" for split in ("train", "validation", "test")}
    )
    taxonomy = {event.category for events in splits.values() for event in events}
    predictor = SmokePredictor(CandidateIndex.fit(splits["train"], taxonomy=taxonomy))
    bundle_path = tmp_path / "synthetic-bundle"
    save_smoke_bundle(bundle_path, predictor)
    request = RecommendationRequest(
        dataset="synthetic",
        history=(
            {
                "poi_id": "unknown-http-client-poi",
                "category_name": "unknown-http-client-category",
                "timestamp": "2026-01-01T00:00:00+00:00",
            },
        ),
        target_time="2026-01-01T01:00:00+00:00",
        top_k=5,
    )
    return bundle_path, request


def test_demo_client_reaches_synthetic_bundle_api_only_through_http(tmp_path: Path) -> None:
    bundle_path, recommendation_request = _bundle_and_request(tmp_path)

    with TestClient(create_app(bundle_path)) as server:

        def bridge(request: httpx.Request) -> httpx.Response:
            request_json = json.loads(request.content) if request.content else None
            response = server.request(request.method, request.url.path, json=request_json)
            return httpx.Response(
                response.status_code,
                content=response.content,
                headers={"content-type": response.headers.get("content-type", "")},
                request=request,
            )

        with ApiClient(
            "http://api:8000", transport=httpx.MockTransport(bridge)
        ) as client:
            assert client.health()["status"] == "ok"
            assert client.ready()["profile"] == "smoke"
            assert client.version().release == "smoke-v1"
            response = client.recommend(recommendation_request)
            assert response.recommendations
            assert len({item.poi_id for item in response.recommendations}) == len(
                response.recommendations
            )
            assert client.metrics()["requests_total"] == 4


def test_smoke_test_entrypoint_runs_local_http_loop_without_exposing_path(
    tmp_path: Path,
) -> None:
    bundle_path, _request = _bundle_and_request(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "next_poi.serving.smoke_test",
            "--bundle",
            str(bundle_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    output = completed.stdout.strip()
    payload = json.loads(output)
    assert payload["status"] == "passed"
    assert payload["recommendation_count"] > 0
    assert str(bundle_path) not in output
