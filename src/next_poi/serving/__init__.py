"""FastAPI service for the deterministic CPU smoke predictor."""

from next_poi.serving.app import app, create_app

__all__ = ["app", "create_app"]
