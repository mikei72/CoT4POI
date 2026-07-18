"""Strict privacy allowlist for target-blind monitoring events."""

from __future__ import annotations

import hashlib
import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from next_poi.contracts import VersionInfo

HistoryLengthBucket = Literal["0", "1", "2-5", "6-10", "11-20", "21-50", "51-100", "101+"]
MonitoringStatus = Literal["ok", "error"]
CandidateSource = Literal[
    "global_popularity",
    "time_popularity",
    "transition",
    "recent_revisit",
    "history_category",
]
LatencyStage = Literal["candidate", "model"]

HistogramKey = Annotated[str, Field(min_length=1, max_length=128)]
HistogramCount = Annotated[int, Field(ge=0)]
LatencyMs = Annotated[float, Field(ge=0, allow_inf_nan=False)]

_OPAQUE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_ERROR_CODE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_VERSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:+-]{0,127}$")
_CATEGORY_BUCKET_PATTERN = re.compile(r"^category_[0-9a-f]{16}$")


def category_bucket(raw_category: str) -> str:
    """Return the only category representation allowed in monitoring events."""

    if not isinstance(raw_category, str) or not raw_category:
        raise ValueError("category must be a non-empty string")
    digest = hashlib.sha256(raw_category.encode("utf-8")).hexdigest()[:16]
    return f"category_{digest}"


def history_length_bucket(length: int) -> HistoryLengthBucket:
    """Bucket a history length without retaining trajectory details."""

    if length < 0:
        raise ValueError("history length must be non-negative")
    if length == 0:
        return "0"
    if length == 1:
        return "1"
    if length <= 5:
        return "2-5"
    if length <= 10:
        return "6-10"
    if length <= 20:
        return "11-20"
    if length <= 50:
        return "21-50"
    if length <= 100:
        return "51-100"
    return "101+"


class MonitoringEvent(BaseModel):
    """One append-only event containing only explicitly allowed aggregates.

    Raw identities, history items, coordinates, timestamps, targets, labels, and
    free-form error messages have no fields in this model and are rejected.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str
    versions: VersionInfo
    history_length_bucket: HistoryLengthBucket
    unknown_count: int = Field(ge=0)
    candidate_count: int = Field(ge=0)
    candidate_source_histogram: dict[CandidateSource, HistogramCount]
    category_histogram: dict[HistogramKey, HistogramCount]
    score_entropy: float = Field(ge=0, allow_inf_nan=False)
    stage_latency_ms: dict[LatencyStage, LatencyMs]
    total_latency_ms: float = Field(ge=0, allow_inf_nan=False)
    status: MonitoringStatus
    error_code: str | None = None

    @field_validator("request_id")
    @classmethod
    def opaque_request_id(cls, value: str) -> str:
        if not _OPAQUE_ID_PATTERN.fullmatch(value):
            raise ValueError("request_id must be an opaque token")
        return value

    @field_validator("versions")
    @classmethod
    def privacy_safe_versions(cls, value: VersionInfo) -> VersionInfo:
        if any(
            not _VERSION_PATTERN.fullmatch(part)
            for part in (value.release, value.data, value.model)
        ):
            raise ValueError("versions must be privacy-safe tokens")
        return value

    @field_validator("category_histogram")
    @classmethod
    def privacy_safe_category_keys(
        cls, value: dict[str, int]
    ) -> dict[str, int]:
        if any(not _CATEGORY_BUCKET_PATTERN.fullmatch(category) for category in value):
            raise ValueError("category histogram keys must be opaque category buckets")
        return value

    @field_validator("error_code")
    @classmethod
    def stable_error_code(cls, value: str | None) -> str | None:
        if value is not None and not _ERROR_CODE_PATTERN.fullmatch(value):
            raise ValueError("error_code must be a stable token")
        return value

    @model_validator(mode="after")
    def status_matches_error_code(self) -> MonitoringEvent:
        if self.status == "ok" and self.error_code is not None:
            raise ValueError("successful events must not include error_code")
        if self.status == "error" and self.error_code is None:
            raise ValueError("failed events require error_code")
        return self


PRIVACY_ALLOWLIST = frozenset(MonitoringEvent.model_fields)
