r"""Contain utility functions to manage the history."""

from __future__ import annotations

__all__ = ["BaseHistory", "InMemoryHistory", "JsonHistory"]

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from feu.utils.io import load_json, save_json

if TYPE_CHECKING:
    from pathlib import Path


logger: logging.Logger = logging.getLogger(__name__)


class BaseHistory(ABC):
    r"""Define the base class to implement a history."""

    @abstractmethod
    def append(self, data: dict[Any, Any]) -> None:
        r"""Append data to the history.

        Args:
            data: The data to append to the history.
        """

    @abstractmethod
    def get_values(self) -> list[dict[Any, Any]]:
        r"""Return the history values.

        Returns:
            The history values.
        """

    @abstractmethod
    def clear(self) -> None:
        r"""Clear the history."""


class InMemoryHistory(BaseHistory):
    r"""Implement a history that stores data in memory.

    Args:
        data: The initial data to store in the history.
    """

    def __init__(self, data: list[Any] | None = None) -> None:
        self._data = data or []

    def append(self, data: dict[Any, Any]) -> None:
        self._data.append(data)

    def get_values(self) -> list[dict[Any, Any]]:
        return self._data

    def clear(self) -> None:
        self._data.clear()


class JsonHistory(BaseHistory):
    r"""Implement a history that stores data in a JSON file.

    Args:
        path: The path to the history file.
    """

    def __init__(self, path: Path) -> None:
        self._path = path

        if self._path.is_file():
            logger.info(f"A history file already exists at {self._path}")
        else:
            logger.info(f"Creating the history file ({self._path}) because it did not exist...")
            save_json([], self._path)

    @property
    def path(self) -> Path:
        r"""The path to the JSON history file."""
        return self._path

    def append(self, data: dict[Any, Any]) -> None:
        logger.info("Appending data to the history...")
        history = load_json(self._path)
        history.append(data)
        save_json(history, self._path, exist_ok=True)
        logger.info(f"The new history length is {len(history):,}")

    def get_values(self) -> list[dict[Any, Any]]:
        return load_json(self._path)

    def clear(self) -> None:
        save_json([], self._path, exist_ok=True)
