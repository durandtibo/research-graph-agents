r"""Define some pytest fixtures for testing.

`pytest` is required to use these fixtures.
"""

from __future__ import annotations

__all__ = ["colorlog_available", "colorlog_not_available"]

import pytest

from argos.utils.imports import is_colorlog_available

colorlog_available: pytest.MarkDecorator = pytest.mark.skipif(
    not is_colorlog_available(), reason="Requires colorlog"
)
colorlog_not_available: pytest.MarkDecorator = pytest.mark.skipif(
    is_colorlog_available(), reason="Skip if colorlog is available"
)
