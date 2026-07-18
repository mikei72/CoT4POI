"""CLI and helpers for deterministic normal or injected monitoring replay."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from next_poi.monitoring.drift import DriftReport, DriftThresholds, build_drift_report
from next_poi.monitoring.events import (
    HistoryLengthBucket,
    MonitoringEvent,
    category_bucket,
)
from next_poi.monitoring.store import JsonlEventStore, MonitoringStoreError

_HISTORY_BUCKETS: tuple[HistoryLengthBucket, ...] = (
    "0",
    "1",
    "2-5",
    "6-10",
    "11-20",
    "21-50",
    "51-100",
    "101+",
)


def inject_category_history_drift(
    events: Sequence[MonitoringEvent],
) -> tuple[MonitoringEvent, ...]:
    """Return a reproducibly shifted window without adding label information."""

    if not events:
        return ()
    history_counts = Counter(event.history_length_bucket for event in events)
    dominant_bucket = history_counts.most_common(1)[0][0]
    drift_bucket = next(
        bucket for bucket in reversed(_HISTORY_BUCKETS) if bucket != dominant_bucket
    )

    existing_categories = {
        category for event in events for category in event.category_histogram
    }
    category = category_bucket("injected-drift-category")
    suffix = 1
    while category in existing_categories:
        category = category_bucket(f"injected-drift-category-{suffix}")
        suffix += 1

    injected: list[MonitoringEvent] = []
    for event in events:
        payload = event.model_dump(mode="json")
        payload["history_length_bucket"] = drift_bucket
        payload["category_histogram"] = {
            category: max(1, sum(event.category_histogram.values()))
        }
        injected.append(MonitoringEvent.model_validate(payload))
    return tuple(injected)


def replay(
    reference: Sequence[MonitoringEvent],
    current: Sequence[MonitoringEvent],
    *,
    inject_drift: bool = False,
    thresholds: DriftThresholds | None = None,
) -> DriftReport:
    replay_current = inject_category_history_drift(current) if inject_drift else current
    return build_drift_report(reference, replay_current, thresholds)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare privacy-safe reference and current monitoring JSONL windows."
    )
    parser.add_argument("--reference", required=True, help="Reference monitoring JSONL")
    parser.add_argument("--current", required=True, help="Current monitoring JSONL")
    parser.add_argument("--output", help="Optional caller-owned JSON report path")
    parser.add_argument(
        "--inject-drift",
        action="store_true",
        help="Deterministically inject category/history drift into the current window",
    )
    parser.add_argument("--js-threshold", type=float, default=0.10)
    parser.add_argument("--psi-threshold", type=float, default=0.20)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        thresholds = DriftThresholds(
            js_divergence=args.js_threshold,
            psi=args.psi_threshold,
        )
        report = replay(
            JsonlEventStore(args.reference).read(),
            JsonlEventStore(args.current).read(),
            inject_drift=args.inject_drift,
            thresholds=thresholds,
        )
        payload = json.dumps(
            report.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        ) + "\n"
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(payload, encoding="utf-8")
        else:
            print(payload, end="")
    except (MonitoringStoreError, ValueError, OSError):
        parser.exit(status=2, message="monitoring replay failed: invalid input or output\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
