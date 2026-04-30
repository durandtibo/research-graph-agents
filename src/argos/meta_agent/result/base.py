r"""Base class for storing and formatting metrics results."""

from __future__ import annotations

__all__ = ["BaseResult"]

from abc import ABC, abstractmethod
from typing import Any


class BaseResult(ABC):
    r"""Abstract base class for storing and formatting metrics results.

    Subclasses must implement methods to expose results in multiple
    representations: raw internal types, serialization-ready dicts,
    flat dicts, dataframes, and markdown.

    All ``to_*`` methods are non-destructive and return a new object
    or data structure without modifying the result instance.
    """

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        r"""Return the result values as serialization-ready native Python
        types.

        All values are converted to JSON-compatible Python types. For
        example, numpy arrays are converted to lists of Python floats,
        and numpy scalars to Python ints or floats.

        Returns:
            A dictionary mapping metric names to their converted values.
        """

    @abstractmethod
    def to_flat_dict(self) -> dict[str, Any]:
        r"""Return the result as a flat dictionary of native Python
        types.

        Similar to ``to_dict``, but nested structures are flattened into
        a single-level dictionary. Useful for logging, CSV export, or
        any context that does not support nested data.

        Returns:
            A flat dictionary mapping metric names to scalar native
                Python values, with no nested dicts or lists.
        """

    @abstractmethod
    def to_raw_dict(self) -> dict[str, Any]:
        r"""Return the result values in their raw internal
        representation.

        Unlike ``to_dict``, values are returned as-is without any type
        conversion. For example, numeric arrays are returned as numpy
        arrays rather than lists of Python floats.

        Returns:
            A dictionary mapping metric names to their raw values.
        """

    @abstractmethod
    def to_markdown(self) -> str:
        r"""Return the result formatted as a Markdown string.

        Produces a human-readable Markdown representation, typically
        as a table. Useful for reports, notebooks, or CLI output.

        Returns:
            A string containing the Markdown representation of the result.
        """
