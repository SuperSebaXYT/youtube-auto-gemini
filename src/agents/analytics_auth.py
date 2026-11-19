"""Backward-compatible import for the shared Analytics OAuth module.

All code must use the same token validation and refresh implementation.
"""
from src.analytics_auth import (
    ANALYTICS_SCOPES,
    get_analytics_service,
    load_analytics_credentials,
)

__all__ = [
    "ANALYTICS_SCOPES",
    "get_analytics_service",
    "load_analytics_credentials",
]
