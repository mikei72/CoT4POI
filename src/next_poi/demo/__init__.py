"""Thin HTTP-only client for the smoke recommendation API."""

from next_poi.demo.api_client import (
    ApiClient,
    ApiClientError,
    ApiProtocolError,
    ApiServiceError,
    ApiUnavailableError,
    ApiValidationError,
)

__all__ = [
    "ApiClient",
    "ApiClientError",
    "ApiProtocolError",
    "ApiServiceError",
    "ApiUnavailableError",
    "ApiValidationError",
]
