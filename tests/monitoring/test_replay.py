from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from next_poi.monitoring.events import MonitoringEvent, category_bucket
from next_poi.monitoring.store import JsonlEventStore


def make_event(index: int) -> MonitoringEvent:
    return MonitoringEvent(
        request_id=f"cli-{index}",
        versions={"release": "r1", "data": "d1", "model": "m1"},
        history_length_bucket="2-5",
        unknown_count=0,
        candidate_count=5,
        candidate_source_histogram={"global_popularity": 5},
        category_histogram={category_bucket("Cafe"): 5},
        score_entropy=0.4,
        stage_latency_ms={"candidate": 1.0, "model": 1.0},
        total_latency_ms=2.5,
        status="ok",
    )


def test_module_cli_runs_normal_and_injected_replays(tmp_path: Path) -> None:
    reference_path = tmp_path / "reference.jsonl"
    current_path = tmp_path / "current.jsonl"
    for index in range(12):
        event = make_event(index)
        JsonlEventStore(reference_path).append(event)
        JsonlEventStore(current_path).append(event)

    command = [
        sys.executable,
        "-m",
        "next_poi.monitoring.replay",
        "--reference",
        str(reference_path),
        "--current",
        str(current_path),
    ]
    normal = subprocess.run(command, check=True, capture_output=True, text=True)
    injected = subprocess.run(
        [*command, "--inject-drift"], check=True, capture_output=True, text=True
    )

    normal_report = json.loads(normal.stdout)
    injected_report = json.loads(injected.stdout)
    assert normal_report["overall_alert"] is False
    assert injected_report["overall_alert"] is True
    assert injected_report["distributions"]["history_length"]["alert"] is True
    assert injected_report["distributions"]["category"]["alert"] is True
    assert "accuracy" not in injected.stdout.lower()
