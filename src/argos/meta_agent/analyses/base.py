r"""Base class for storing and formatting metrics analyses."""

from __future__ import annotations

__all__ = ["BaseAnalysis"]

from abc import ABC, abstractmethod
from typing import Any, Self

from coola.equality.tester import EqualNanEqualityTester, get_default_registry


class BaseAnalysis(ABC):
    r"""Abstract base class for storing and formatting metrics analyses.

    Subclasses must implement methods to expose analyses in multiple
    representations: raw internal types, serialization-ready dicts,
    flat dicts, dataframes, and markdown.

    Example:
        ```pycon
        >>> from argos.meta_agent.analyses import Analysis
        >>> analysis = Analysis("my custom analysis: blabla...")
        >>> analysis.to_dict()
        {'content': 'my custom analysis: blabla...'}
        >>> analysis.to_text()
        'my custom analysis: blabla...'

        ```
    """

    @abstractmethod
    def equal(self, other: object, equal_nan: bool = False) -> bool:
        r"""Return ``True`` if the two objects are equal, otherwise
        ``False``.

        Args:
            other: The value to compare with.
            equal_nan: Whether to compare NaN's as equal. If ``True``,
                NaN's in both objects will be considered equal.

        Returns:
            ``True`` if the two objects are equal, otherwise ``False``
        """

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        r"""Construct an instance from a plain dictionary.

        Args:
            data: Must contain the expected keys.

        Returns:
            A new instance of the calling subclass.
        """

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        r"""Return the analysis values as serialization-ready native
        Python types.

        All values are converted to JSON-compatible Python types. For
        example, numpy arrays are converted to lists of Python floats,
        and numpy scalars to Python ints or floats.

        Returns:
            A dictionary mapping metric names to their converted values.
        """

    @abstractmethod
    def to_text(self) -> str:
        r"""Return the analysis formatted as a Markdown string.

        Produces a human-readable Markdown representation. Useful
        for reports, notebooks, or CLI output.

        Returns:
            A string containing the Markdown representation of
                the analysis.

        Example:
            ```pycon
            >>> from argos.meta_agent.analyses import Analysis
            >>> analysis = Analysis("my custom analysis: blabla...")
            >>> analysis.to_text()
            'my custom analysis: blabla...'

            ```
        """


get_default_registry().register_many({BaseAnalysis: EqualNanEqualityTester()}, exist_ok=True)
