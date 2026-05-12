r"""Define the abstract class for entities."""

from __future__ import annotations

__all__ = ["BaseEntity"]

from abc import ABC, abstractmethod
from typing import Any, Self

from coola.equality import objects_are_equal
from coola.equality.tester import EqualNanEqualityTester, get_default_registry
from coola.nested import to_flat_dict


class BaseEntity(ABC):
    r"""Abstract base class defining the interface for a single labeled
    example.

    Subclasses must define all attributes and implement all methods.

    Attributes:
        id: A unique identifier for the example.
        metadata: Optional dictionary of auxiliary information.
            Defaults to ``None``.

    Example:
        ```pycon
        >>> from argos.meta_agent.entities import Record
        >>> x = Record(id="q1", input="What is 2+2?", target="4", prediction="5")
        >>> x
        Record(id='q1', input='What is 2+2?', target='4', prediction='5', metadata=None)
        >>> x.to_dict()
        {'id': 'q1', 'input': 'What is 2+2?', 'target': '4', 'prediction': '5', 'metadata': None}

        ```
    """

    id: str
    metadata: dict[str, Any] | None = None

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
        if type(other) is not type(self):
            return False
        return objects_are_equal(self.to_dict(), other.to_dict(), equal_nan=equal_nan)

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        r"""Construct an instance from a plain dictionary.

        Args:
            data: Must contain ``id``, ``input``, and ``target`` keys.
                ``metadata`` is optional.

        Returns:
            A new instance of the calling subclass.

        Example:
            ```pycon
            >>> from argos.meta_agent.entities import Record
            >>> x = Record.from_dict({"id": "q1", "input": "What is 2+2?", "target": "4"})
            >>> x
            Record(id='q1', input='What is 2+2?', target='4', prediction=None, metadata=None)

            ```
        """

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        r"""Serialize the example to a plain dictionary.

        Returns:
            A dictionary with keys ``id``, ``input``, ``target``, and
            ``metadata``.

        Example:
            ```pycon
            >>> from argos.meta_agent.entities import Record
            >>> x = Record(id="q1", input="What is 2+2?", target="4")
            >>> x.to_dict()
            {'id': 'q1', 'input': 'What is 2+2?', 'target': '4', 'prediction': None, 'metadata': None}

            ```
        """

    def to_flat_dict(self, separator: str = ".") -> dict[str, Any]:
        r"""Return the result as a flat dictionary of native Python
        types.

        Args:
            separator: The separator used to join nested keys when
                flattening. Defaults to ``"."``.

        Returns:
            A flat dictionary mapping metric names to scalar native
                Python values, with no nested dicts or lists.

        Example:
            ```pycon
            >>> from argos.meta_agent.entities import Record
            >>> x = Record(id="q1", input="What is 2+2?", target={"answer": 4, "style": "math"})
            >>> x.to_flat_dict()
            {'id': 'q1', 'input': 'What is 2+2?', 'target.answer': 4, 'target.style': 'math', 'prediction': None, 'metadata': None}

            ```
        """
        return to_flat_dict(self.to_dict(), separator=separator)


get_default_registry().register_many({BaseEntity: EqualNanEqualityTester()}, exist_ok=True)
