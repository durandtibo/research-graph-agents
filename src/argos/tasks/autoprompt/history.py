r"""Contain utility functions to manage the history."""

from __future__ import annotations

__all__ = ["create_history_file"]

import logging
from typing import TYPE_CHECKING

from feu.utils.io import save_json

if TYPE_CHECKING:
    from argos.tasks.autoprompt.config import ExperimentConfig

logger: logging.Logger = logging.getLogger(__name__)


def create_history_file(config: ExperimentConfig) -> None:
    r"""Create the history file if it does not exist.

    Args:
        config: The experiment config.
    """
    if config.path_history.is_file():
        return
    logger.info("Creating the history file because it did not exist...")
    save_json([], config.path_history)
