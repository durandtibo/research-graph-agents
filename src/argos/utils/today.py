r"""Contain utility functions for today."""

from __future__ import annotations

__all__ = ["get_today_date"]

from datetime import datetime
from zoneinfo import ZoneInfo


def get_today_date(timezone: str = "UTC", date_format: str = "%Y-%m-%d") -> str:
    r"""Return the current date for the specified timezone.

    Args:
        timezone: Timezone to use.
        date_format: Date format to use.

    Returns:
        A string representing the current date for the specified timezone.
    """
    return datetime.now(ZoneInfo(timezone)).strftime(date_format)
