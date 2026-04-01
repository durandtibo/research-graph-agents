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
"""Pytest mark decorator that skips a test if ``colorlog`` is not installed.

Apply this decorator to tests that require the optional ``colorlog``
dependency to be present in the environment.

Example:
    ```python
    from argos.testing import colorlog_available

    @colorlog_available
    def test_colored_output() -> None:
        ...
    ```
"""

colorlog_not_available: pytest.MarkDecorator = pytest.mark.skipif(
    is_colorlog_available(), reason="Skip if colorlog is available"
)
"""Pytest mark decorator that skips a test if ``colorlog`` is installed.

Apply this decorator to tests that cover the fallback behaviour that
is only exercised when the optional ``colorlog`` dependency is absent.

Example:
    ```python
    from argos.testing import colorlog_not_available

    @colorlog_not_available
    def test_plain_logging_fallback() -> None:
        ...
    ```
"""
