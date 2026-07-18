"""FastAPI application factory for the validated smoke bundle."""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.responses import Response

from next_poi.contracts import RecommendationRequest, RecommendationResponse
from next_poi.models import BundleInfo, SmokePredictor, load_smoke_bundle
from next_poi.monitoring import JsonlEventStore, MonitoringStoreError
from next_poi.serving.monitoring import build_success_event

LOGGER = logging.getLogger(__name__)
BUNDLE_ENV_VAR = "NEXT_POI_BUNDLE"
MONITORING_ENV_VAR = "NEXT_POI_MONITORING_PATH"
_ENDPOINT_NAMES = {
    "/health": "health",
    "/ready": "ready",
    "/version": "version",
    "/recommend": "recommend",
    "/metrics": "metrics",
}
_PUBLIC_VALIDATION_FIELDS = {
    "body",
    "dataset",
    "history",
    "target_time",
    "top_k",
    "profile",
    "poi_id",
    "model_poi_id",
    "category_name",
    "timestamp",
    "latitude",
    "longitude",
    "target",
    "label",
    "result",
}


class ApiFault(Exception):
    """Expected public API failure with a stable, path-free payload."""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(code)
        self.status_code = status_code
        self.code = code
        self.message = message


@dataclass
class AggregateMetrics:
    """In-memory allowlisted counters; no request payload is retained."""

    requests_by_endpoint: dict[str, int] = field(
        default_factory=lambda: {
            **{name: 0 for name in _ENDPOINT_NAMES.values()},
            "other": 0,
        }
    )
    responses_by_status_class: dict[str, int] = field(
        default_factory=lambda: {"2xx": 0, "4xx": 0, "5xx": 0}
    )
    latency_count: int = 0
    latency_sum_ms: float = 0.0
    latency_max_ms: float = 0.0
    _lock: Lock = field(default_factory=Lock, repr=False)

    def record(self, path: str, status_code: int, latency_ms: float) -> None:
        endpoint = _ENDPOINT_NAMES.get(path, "other")
        status_class = f"{status_code // 100}xx"
        with self._lock:
            self.requests_by_endpoint[endpoint] += 1
            if status_class in self.responses_by_status_class:
                self.responses_by_status_class[status_class] += 1
            self.latency_count += 1
            self.latency_sum_ms += latency_ms
            self.latency_max_ms = max(self.latency_max_ms, latency_ms)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            count = self.latency_count
            total = self.latency_sum_ms
            return {
                "requests_total": sum(self.requests_by_endpoint.values()),
                "requests_by_endpoint": dict(self.requests_by_endpoint),
                "responses_by_status_class": dict(self.responses_by_status_class),
                "latency_ms": {
                    "count": count,
                    "sum": round(total, 3),
                    "average": round(total / count, 3) if count else 0.0,
                    "max": round(self.latency_max_ms, 3),
                },
            }


@dataclass
class ServiceState:
    bundle_path: Path | None
    event_store: JsonlEventStore | None = None
    predictor: SmokePredictor | None = None
    bundle_info: BundleInfo | None = None
    readiness_error_code: str = "bundle_not_loaded"
    metrics: AggregateMetrics = field(default_factory=AggregateMetrics)

    @property
    def ready(self) -> bool:
        return self.predictor is not None and self.bundle_info is not None


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message}},
    )


def _require_ready(state: ServiceState) -> tuple[SmokePredictor, BundleInfo]:
    if state.predictor is None or state.bundle_info is None:
        raise ApiFault(
            503,
            state.readiness_error_code,
            "validated smoke bundle is not ready",
        )
    return state.predictor, state.bundle_info


def _safe_validation_location(location: tuple[Any, ...]) -> str:
    parts = []
    for part in location:
        if isinstance(part, int):
            parts.append(str(part))
        elif part in _PUBLIC_VALIDATION_FIELDS:
            parts.append(str(part))
        else:
            parts.append("field")
    return ".".join(parts)


def create_app(
    bundle_path: str | Path | None = None,
    monitoring_path: str | Path | None = None,
) -> FastAPI:
    """Create an app that verifies and loads one smoke bundle during lifespan."""

    configured_path = bundle_path
    if configured_path is None:
        configured_path = os.environ.get(BUNDLE_ENV_VAR)
    configured_monitoring_path = monitoring_path
    if configured_monitoring_path is None:
        configured_monitoring_path = os.environ.get(MONITORING_ENV_VAR)
    service_state = ServiceState(
        bundle_path=Path(configured_path).expanduser() if configured_path else None,
        event_store=(
            JsonlEventStore(Path(configured_monitoring_path).expanduser())
            if configured_monitoring_path
            else None
        ),
    )

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        state: ServiceState = application.state.next_poi
        state.predictor = None
        state.bundle_info = None
        if state.bundle_path is None:
            state.readiness_error_code = "bundle_not_configured"
            LOGGER.warning("smoke service started without a configured bundle")
        else:
            try:
                predictor, bundle_info = load_smoke_bundle(state.bundle_path)
            except (FileNotFoundError, OSError, ValueError):
                state.readiness_error_code = "bundle_validation_failed"
                LOGGER.error("smoke bundle failed validation")
            else:
                state.predictor = predictor
                state.bundle_info = bundle_info
                state.readiness_error_code = ""
                LOGGER.info("validated smoke bundle loaded")
        yield
        state.predictor = None
        state.bundle_info = None

    application = FastAPI(
        title="Next-POI deterministic smoke API",
        version="1.0.0",
        debug=False,
        lifespan=lifespan,
    )
    application.state.next_poi = service_state

    @application.middleware("http")
    async def aggregate_request_metrics(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        started = perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            service_state.metrics.record(
                request.url.path,
                500,
                (perf_counter() - started) * 1000.0,
            )
            raise
        service_state.metrics.record(
            request.url.path,
            response.status_code,
            (perf_counter() - started) * 1000.0,
        )
        return response

    @application.exception_handler(ApiFault)
    async def api_fault_handler(_request: Request, exc: ApiFault) -> JSONResponse:
        return _error_response(exc.status_code, exc.code, exc.message)

    @application.exception_handler(RequestValidationError)
    async def validation_error_handler(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        details = sorted(
            (
                {
                    "location": _safe_validation_location(error["loc"]),
                    "message": str(error["msg"]),
                    "type": str(error["type"]),
                }
                for error in exc.errors()
            ),
            key=lambda item: (item["location"], item["type"], item["message"]),
        )
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "request_validation_error",
                    "message": "request validation failed",
                    "details": details,
                }
            },
        )

    @application.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.get("/ready")
    async def ready() -> dict[str, str]:
        predictor, bundle_info = _require_ready(service_state)
        return {
            "status": "ready",
            "profile": "smoke",
            "bundle_manifest_sha256": bundle_info.manifest_sha256,
            "model": predictor.versions.model,
        }

    @application.get("/version")
    async def version() -> dict[str, str]:
        predictor, _bundle_info = _require_ready(service_state)
        return predictor.versions.model_dump(mode="json")

    @application.post("/recommend", response_model=RecommendationResponse)
    async def recommend(request: RecommendationRequest) -> RecommendationResponse:
        predictor, _bundle_info = _require_ready(service_state)
        if request.dataset != predictor.index.dataset:
            raise ApiFault(
                422,
                "unsupported_dataset",
                "dataset is not supported by the loaded bundle",
            )
        trace = predictor.predict_with_trace(request)
        if service_state.event_store is not None:
            try:
                service_state.event_store.append(
                    build_success_event(request, trace, predictor)
                )
            except (MonitoringStoreError, ValueError):
                LOGGER.error("privacy-safe monitoring event write failed")
        return trace.response

    @application.get("/metrics")
    async def metrics() -> dict[str, Any]:
        return service_state.metrics.snapshot()

    return application


app = create_app()
