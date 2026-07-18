"""Canonical cross-layer contracts.

Offline labeled examples and online target-blind requests are intentionally
different types. Candidate generation and serving code accept only
``RecommendationRequest``.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from pathlib import PurePosixPath, PureWindowsPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DatasetName = Literal["nyc", "tky", "ca", "synthetic"]
SplitName = Literal["train", "validation", "test"]
ProfileName = Literal["smoke"]
RuntimeStatus = Literal["ready", "static_only"]

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class StrictModel(BaseModel):
    """Base model that rejects undeclared fields at every public boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)


def _require_aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must include a timezone offset")
    return value


def _require_portable_relative_path(value: str) -> str:
    posix_path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    if (
        not value.strip()
        or "\\" in value
        or posix_path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or ".." in posix_path.parts
        or not posix_path.parts
    ):
        raise ValueError("manifest paths must be portable relative paths")
    return value


class HistoryEvent(StrictModel):
    poi_id: str = Field(min_length=1)
    model_poi_id: int | None = Field(default=None, ge=0)
    category_name: str = Field(min_length=1)
    timestamp: datetime
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)

    _validate_timestamp = field_validator("timestamp")(_require_aware)

    @model_validator(mode="after")
    def coordinates_are_paired(self) -> HistoryEvent:
        if (self.latitude is None) != (self.longitude is None):
            raise ValueError("latitude and longitude must be provided together")
        return self


class RecommendationRequest(StrictModel):
    dataset: DatasetName
    history: tuple[HistoryEvent, ...] = Field(min_length=1, max_length=128)
    target_time: datetime
    top_k: int = Field(default=10, ge=1, le=100)
    profile: ProfileName = "smoke"

    _validate_target_time = field_validator("target_time")(_require_aware)

    @model_validator(mode="after")
    def target_follows_history(self) -> RecommendationRequest:
        latest = max(item.timestamp for item in self.history)
        if self.target_time < latest:
            raise ValueError("target_time must not precede the latest history event")
        return self


class LabeledExample(StrictModel):
    """Offline-only evaluation record; never accepted by a Predictor boundary."""

    sample_id: str
    split: SplitName
    request: RecommendationRequest
    target_poi_id: str = Field(min_length=1)
    target_model_poi_id: int | None = Field(default=None, ge=0)
    target_category: str = Field(min_length=1)

    @field_validator("sample_id")
    @classmethod
    def valid_sample_id(cls, value: str) -> str:
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("sample_id must be a lowercase SHA-256 hex digest")
        return value


class NormalizedEvent(StrictModel):
    dataset: DatasetName
    split: SplitName
    raw_user_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    timestamp_utc: datetime
    raw_poi_id: str = Field(min_length=1)
    model_poi_id: int | None = Field(default=None, ge=0)
    category: str = Field(min_length=1)
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)

    @field_validator("timestamp_utc")
    @classmethod
    def normalize_timestamp_utc(cls, value: datetime) -> datetime:
        return _require_aware(value).astimezone(timezone.utc)

    @model_validator(mode="after")
    def coordinates_are_paired(self) -> NormalizedEvent:
        if (self.latitude is None) != (self.longitude is None):
            raise ValueError("latitude and longitude must be provided together")
        return self


class Recommendation(StrictModel):
    rank: int = Field(ge=1)
    poi_id: str = Field(min_length=1)
    model_poi_id: int | None = Field(default=None, ge=0)
    category: str = Field(min_length=1)
    score: float
    candidate_sources: tuple[str, ...] = Field(min_length=1)

    @field_validator("score")
    @classmethod
    def finite_score(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("score must be finite")
        return value

    @field_validator("candidate_sources")
    @classmethod
    def unique_sources(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not source.strip() for source in value):
            raise ValueError("candidate sources must be non-empty")
        if len(value) != len(set(value)):
            raise ValueError("candidate sources must be unique")
        return value


class CategoryScore(StrictModel):
    category: str = Field(min_length=1)
    score: float

    _finite_score = field_validator("score")(Recommendation.finite_score)


class VersionInfo(StrictModel):
    release: str = Field(min_length=1)
    data: str = Field(min_length=1)
    model: str = Field(min_length=1)


class LatencyBreakdown(StrictModel):
    total_ms: float = Field(ge=0)
    candidate_ms: float = Field(default=0, ge=0)
    model_ms: float = Field(default=0, ge=0)

    @field_validator("total_ms", "candidate_ms", "model_ms")
    @classmethod
    def finite_latency(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("latency values must be finite")
        return value


class RecommendationResponse(StrictModel):
    recommendations: tuple[Recommendation, ...]
    macro: tuple[CategoryScore, ...] = ()
    versions: VersionInfo
    latency: LatencyBreakdown
    request_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def recommendations_are_unique_and_ranked(self) -> RecommendationResponse:
        poi_ids = [item.poi_id for item in self.recommendations]
        if len(poi_ids) != len(set(poi_ids)):
            raise ValueError("recommendation POI IDs must be unique")
        ranks = [item.rank for item in self.recommendations]
        if ranks != list(range(1, len(ranks) + 1)):
            raise ValueError("recommendation ranks must be contiguous and ordered")
        return self


class FileDigest(StrictModel):
    path: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    sha256: str
    present: bool = True

    @field_validator("path")
    @classmethod
    def relative_portable_path(cls, value: str) -> str:
        return _require_portable_relative_path(value)

    @field_validator("sha256")
    @classmethod
    def valid_sha256(cls, value: str) -> str:
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("sha256 must be a lowercase SHA-256 hex digest")
        return value


class DataSplitSummary(StrictModel):
    split: SplitName
    count: int = Field(ge=0)
    content_sha256: str
    min_timestamp: datetime | None = None
    max_timestamp: datetime | None = None

    _validate_sha256 = field_validator("content_sha256")(FileDigest.valid_sha256)

    @model_validator(mode="after")
    def ordered_time_range(self) -> DataSplitSummary:
        if self.min_timestamp is not None:
            _require_aware(self.min_timestamp)
        if self.max_timestamp is not None:
            _require_aware(self.max_timestamp)
        if (
            self.min_timestamp is not None
            and self.max_timestamp is not None
            and self.min_timestamp > self.max_timestamp
        ):
            raise ValueError("min_timestamp must not exceed max_timestamp")
        return self


class DataManifest(StrictModel):
    schema_version: str = Field(min_length=1)
    dataset: DatasetName
    split_protocol: str = Field(min_length=1)
    taxonomy_sha256: str
    encoder_sha256: str
    splits: tuple[DataSplitSummary, ...] = Field(min_length=1)

    _validate_taxonomy_sha256 = field_validator("taxonomy_sha256")(FileDigest.valid_sha256)
    _validate_encoder_sha256 = field_validator("encoder_sha256")(FileDigest.valid_sha256)

    @model_validator(mode="after")
    def complete_unique_splits(self) -> DataManifest:
        names = [item.split for item in self.splits]
        required = {"train", "validation", "test"}
        if len(names) != len(required) or set(names) != required:
            raise ValueError(
                "data manifest must contain train, validation, and test exactly once"
            )
        return self


class ModelManifest(StrictModel):
    schema_version: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    backend: str = Field(min_length=1)
    runtime_status: RuntimeStatus
    dynamic_load_verified: bool
    files: tuple[FileDigest, ...]
    missing_files: tuple[str, ...] = ()
    config_sha256: str

    _validate_config_sha256 = field_validator("config_sha256")(FileDigest.valid_sha256)

    @field_validator("missing_files")
    @classmethod
    def portable_missing_file_paths(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(_require_portable_relative_path(path) for path in value)

    @model_validator(mode="after")
    def runtime_status_is_truthful(self) -> ModelManifest:
        file_paths = [item.path for item in self.files]
        if len(file_paths) != len(set(file_paths)):
            raise ValueError("model manifest file paths must be unique")
        if self.runtime_status == "static_only" and self.dynamic_load_verified:
            raise ValueError("static_only models cannot be marked dynamically verified")
        if self.runtime_status == "ready" and (
            not self.dynamic_load_verified
            or self.missing_files
            or any(not item.present for item in self.files)
        ):
            raise ValueError("ready models require dynamic verification and no missing files")
        if len(self.missing_files) != len(set(self.missing_files)):
            raise ValueError("missing_files must be unique")
        missing_paths = set(self.missing_files)
        for item in self.files:
            if item.present and item.path in missing_paths:
                raise ValueError("present files cannot also be listed as missing")
            if not item.present and item.path not in missing_paths:
                raise ValueError("files marked absent must be listed in missing_files")
        return self


class ReleaseManifest(StrictModel):
    schema_version: str = Field(min_length=1)
    release_version: str = Field(min_length=1)
    profile: ProfileName = "smoke"
    data_manifest_sha256: str
    model_manifest_sha256: str
    config_sha256: str

    _validate_data_sha256 = field_validator("data_manifest_sha256")(FileDigest.valid_sha256)
    _validate_model_sha256 = field_validator("model_manifest_sha256")(FileDigest.valid_sha256)
    _validate_config_sha256 = field_validator("config_sha256")(FileDigest.valid_sha256)
