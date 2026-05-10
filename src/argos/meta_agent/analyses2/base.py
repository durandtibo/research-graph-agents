r"""Base class for storing and formatting analyses."""

from __future__ import annotations

__all__ = ["BaseAnalysis"]

import json
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, TypeAlias

import yaml
from coola.equality.tester import EqualNanEqualityTester, get_default_registry

PrimitiveType: TypeAlias = (
    Mapping[str, "PrimitiveType"] | Sequence["PrimitiveType"] | str | float | int | bool | None
)


class BaseAnalysis(ABC):
    r"""Abstract base class for storing and formatting analyses.

    Subclasses must implement:
    - ``equal``: equality check between two instances.
    - ``state_dict`` / ``from_state_dict``: full round-trip serialization
      for reconstruction.
    - ``to_primitive``: JSON-compatible representation for external use
      (e.g. passing to an LLM).

    Example:
        ```pycon
        >>> from argos.meta_agent.analyses2 import Analysis
        >>> analysis = Analysis("my custom analysis: blabla...")
        >>> analysis
        abc
        >>> analysis.to_primitive()
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
            ``True`` if the two objects are equal, otherwise ``False``.
        """

    @abstractmethod
    def to_primitive(self) -> PrimitiveType:
        r"""Return a JSON-compatible representation for external use.

        Converts the analysis to Python primitive types (dict, list,
        str, float, int, bool, or None) suitable for passing to an LLM,
        API, or other external consumer. Unlike ``state_dict``, this
        method may omit internal fields and is not required to support
        reconstruction.

        Returns:
            A JSON-compatible Python object (dict, list, str, float,
                int, bool, or None).

        Example:
            ```pycon
            >>> from argos.meta_agent.analyses2 import Analysis
            >>> analysis = Analysis("my custom analysis: blabla...")
            >>> analysis.to_primitive()
            'my custom analysis: blabla...'

            ```
        """

    def to_json(self, **kwargs: Any) -> str:
        r"""Return the analysis serialized as a JSON string.

        Calls ``to_primitive`` and serializes the result to JSON.
        Any keyword arguments are forwarded to ``json.dumps`` (e.g.
        ``indent=2`` for pretty-printing).

        Args:
            **kwargs: Additional keyword arguments forwarded to
                ``json.dumps``.

        Returns:
            A JSON string representation of the analysis.

        Example:
            ```pycon
            >>> from argos.meta_agent.analyses2 import Analysis
            >>> analysis = Analysis("my custom analysis: blabla...")
            >>> analysis.to_json()
            '"my custom analysis: blabla..."'
            >>> analysis.to_json(indent=2)
            '"my custom analysis: blabla..."'

            ```
        """
        return json.dumps(self.to_primitive(), **kwargs)

    def to_yaml(self, **kwargs: Any) -> str:
        r"""Return the analysis serialized as a JSON string.

        Calls ``to_primitive`` and serializes the result to JSON.
        Any keyword arguments are forwarded to ``json.dumps`` (e.g.
        ``indent=2`` for pretty-printing).

        Args:
            **kwargs: Additional keyword arguments forwarded to
                ``json.dumps``.

        Returns:
            A JSON string representation of the analysis.

        Example:
            ```pycon
            >>> from argos.meta_agent.analyses2 import Analysis
            >>> analysis = Analysis("my custom analysis: blabla...")
            >>> analysis.to_json()
            '"my custom analysis: blabla..."'
            >>> analysis.to_json(indent=2)
            '"my custom analysis: blabla..."'

            ```
        """
        return yaml.safe_dump(self.to_primitive(), **kwargs)


get_default_registry().register_many({BaseAnalysis: EqualNanEqualityTester()}, exist_ok=True)
