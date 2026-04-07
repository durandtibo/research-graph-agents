r"""Contain utility functions to manage the history."""

from __future__ import annotations

__all__ = ["append_to_history", "create_history"]

import logging
from typing import TYPE_CHECKING

from feu.utils.io import load_json, save_json

if TYPE_CHECKING:
    from argos.tasks.autoprompt.config import ExperimentConfig

logger: logging.Logger = logging.getLogger(__name__)


def append_to_history(config: ExperimentConfig, data: dict) -> None:
    r"""Append data to the history file.

    Args:
        config: The experiment config.
        data: The data to append to the history file.
    """
    if not config.path_history.is_file():
        create_history(config)

    logger.info("Appending data to the history...")
    history = load_json(config.path_history)
    history.append(data)
    save_json(history, config.path_history, exist_ok=True)
    logger.info(f"The new history length is {len(history):,}")


def create_history(config: ExperimentConfig) -> None:
    r"""Create the history file if it does not exist.

    Args:
        config: The experiment config.
    """
    if config.path_history.is_file():
        return
    logger.info("Creating the history file because it did not exist...")
    save_json([], config.path_history)
