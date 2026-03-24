r"""Contain utility functions to configure the standard logging
library."""

from __future__ import annotations

__all__ = ["configure_logging"]

import logging

from argos.utils.imports import is_colorlog_available

if is_colorlog_available():  # pragma: no cover
    import colorlog


def configure_logging(level: int = logging.INFO) -> None:
    r"""Configure the logging module with a colored formatter.

    If the ``colorlog`` package is installed, a colored formatter is
    used. Otherwise, the standard ``logging.basicConfig`` is called.

    Args:
        level: The minimum log level to capture. Defaults to
            ``logging.INFO``.

    Example:
        ```pycon
        >>> import logging
        >>> from argos.utils.logging import configure_logging
        >>> configure_logging(level=logging.DEBUG)

        ```
    """
    if not is_colorlog_available():
        logging.basicConfig(level=level)
        return

    handler = colorlog.StreamHandler()
    formatter = colorlog.ColoredFormatter(
        fmt=(
            "%(log_color)s(%(process)d) %(asctime)s [%(levelname)s] %(name)s:%(lineno)s%(reset)s "
            "%(message_log_color)s%(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "bold_yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        },
        secondary_log_colors={
            "message": {
                "DEBUG": "cyan",
                "INFO": "reset",
                "WARNING": "bold_yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            }
        },
    )
    handler.setFormatter(formatter)

    logging.basicConfig(level=level, handlers=[handler])
