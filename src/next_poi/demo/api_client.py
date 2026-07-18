"""Typed, finite-timeout HTTP boundary used by the Streamlit demo."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any

import httpx
from pydantic import ValidationError

from next_poi.contracts import RecommendationRequest, RecommendationResponse, VersionInfo

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ApiClientError(RuntimeError):
    """Base user-facing API client error without raw response content."""

    def __init__(self, code: str, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


class ApiValidationError(ApiClientError):
    """The API rejected request fields with a 4xx validation response."""


class ApiUnavailableError(ApiClientError):
    """The API could not be reached or is not ready."""


class ApiServiceError(ApiClientError):
    """The API returned a non-validation failure."""


class ApiProtocolError(ApiClientError):
    """The API returned malformed or incompatible JSON."""


class ApiClient:
    """Small synchronous client; recommendation POSTs are never retried."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout: httpx.Timeout | float | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        normalized_url = base_url.rstrip("/")
        parsed_url = httpx.URL(normalized_url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.host:
            raise ValueError("API base URL must be an absolute HTTP(S) URL")
        request_timeout = timeout or httpx.Timeout(5.0, connect=2.0)
        self._client = httpx.Client(
            base_url=normalized_url,
            timeout=request_timeout,
            transport=transport,
        )

    def __enter__(self) -> ApiClient:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def health(self) -> dict[str, str]:
        payload = self._request("GET", "/health")
        if payload != {"status": "ok"}:
            raise _protocol_error("health response does not match the API contract")
        return {"status": "ok"}

    def ready(self) -> dict[str, str]:
        payload = self._request("GET", "/ready")
        required = {"status", "profile", "bundle_manifest_sha256", "model"}
        if set(payload) != required or not all(
            isinstance(payload[field], str) for field in required
        ):
            raise _protocol_error("readiness response does not match the API contract")
        if (
            payload["status"] != "ready"
            or payload["profile"] != "smoke"
            or not _SHA256_PATTERN.fullmatch(payload["bundle_manifest_sha256"])
            or not payload["model"]
        ):
            raise _protocol_error("readiness response contains invalid values")
        return {field: payload[field] for field in sorted(required)}

    def version(self) -> VersionInfo:
        payload = self._request("GET", "/version")
        try:
            return VersionInfo.model_validate(payload)
        except ValidationError as exc:
            raise _protocol_error("version response does not match the API contract") from exc

    def recommend(self, request: RecommendationRequest) -> RecommendationResponse:
        payload = self._request(
            "POST",
            "/recommend",
            json=request.model_dump(mode="json"),
        )
        try:
            return RecommendationResponse.model_validate(payload)
        except ValidationError as exc:
            raise _protocol_error(
                "recommendation response does not match the API contract"
            ) from exc

    def metrics(self) -> dict[str, Any]:
        payload = self._request("GET", "/metrics")
        required = {
            "requests_total",
            "requests_by_endpoint",
            "responses_by_status_class",
            "latency_ms",
        }
        if set(payload) != required or not _is_nonnegative_int(
            payload["requests_total"]
        ):
            raise _protocol_error("metrics response does not match the API contract")
        endpoints = payload["requests_by_endpoint"]
        statuses = payload["responses_by_status_class"]
        latency = payload["latency_ms"]
        if not isinstance(endpoints, dict) or set(endpoints) != {
            "health",
            "ready",
            "version",
            "recommend",
            "metrics",
            "other",
        }:
            raise _protocol_error("metrics response does not match the API contract")
        if not isinstance(statuses, dict) or set(statuses) != {"2xx", "4xx", "5xx"}:
            raise _protocol_error("metrics response does not match the API contract")
        if not isinstance(latency, dict) or set(latency) != {
            "count",
            "sum",
            "average",
            "max",
        }:
            raise _protocol_error("metrics response does not match the API contract")
        if not all(_is_nonnegative_int(value) for value in endpoints.values()):
            raise _protocol_error("metrics response does not match the API contract")
        if not all(_is_nonnegative_int(value) for value in statuses.values()):
            raise _protocol_error("metrics response does not match the API contract")
        if not _is_nonnegative_int(latency["count"]) or not all(
            _is_nonnegative_number(latency[field]) for field in ("sum", "average", "max")
        ):
            raise _protocol_error("metrics response does not match the API contract")
        if (
            payload["requests_total"] != sum(endpoints.values())
            or payload["requests_total"] != sum(statuses.values())
            or payload["requests_total"] != latency["count"]
        ):
            raise _protocol_error("metrics response does not match the API contract")
        return payload

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            response = self._client.request(method, path, json=json)
        except httpx.TimeoutException as exc:
            raise ApiUnavailableError("api_timeout", "API request timed out") from exc
        except httpx.RequestError as exc:
            raise ApiUnavailableError("api_unavailable", "API is unavailable") from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise _protocol_error("API returned invalid JSON") from exc
        if not isinstance(payload, Mapping):
            raise _protocol_error("API response must be a JSON object")
        decoded = dict(payload)

        if response.status_code >= 400:
            code, message = _decode_error(decoded)
            if response.status_code == 422:
                raise ApiValidationError(code, message, status_code=response.status_code)
            if response.status_code == 503:
                raise ApiUnavailableError(code, message, status_code=response.status_code)
            raise ApiServiceError(code, message, status_code=response.status_code)
        if not 200 <= response.status_code < 300:
            raise _protocol_error("API returned an unexpected HTTP status")
        return decoded


def _decode_error(payload: Mapping[str, Any]) -> tuple[str, str]:
    error = payload.get("error")
    if not isinstance(error, Mapping):
        return "api_error", "API request failed"
    code = error.get("code")
    message = error.get("message")
    if not isinstance(code, str) or not code or not isinstance(message, str) or not message:
        return "api_error", "API request failed"
    return code, message


def _protocol_error(message: str) -> ApiProtocolError:
    return ApiProtocolError("api_protocol_error", message)


def _is_nonnegative_int(value: object) -> bool:
    return type(value) is int and value >= 0


def _is_nonnegative_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value >= 0
    )
