r"""Contain the base class to define a dataset example."""

from __future__ import annotations

__all__ = ["BaseExample"]

from abc import ABC, abstractmethod
from typing import Any, Generic, Self

from coola.equality.tester import EqualNanEqualityTester, get_default_registry
from coola.nested import to_flat_dict

from argos.meta_agent.typing import InputT, TargetT


class BaseExample(ABC, Generic[InputT, TargetT]):
    r"""Abstract base class defining the interface for a single labeled
    example.

    Subclasses must define all attributes and implement all methods.

    Attributes:
        id: A unique identifier for the example.
        input: The input passed to the agent.
        target: The expected ground-truth output.
        metadata: Optional dictionary of auxiliary information.
            Defaults to ``None``.

    Example:
        ```pycon
        >>> from argos.meta_agent.examples import BaseExample, Example
        >>> example = Example(id="q1", input="What is 2+2?", target="4")
        >>> example
        Example(id='q1', input='What is 2+2?', target='4', metadata=None)

        ```
    """

    id: str
    input: InputT
    target: TargetT
    metadata: dict[str, Any] | None = None

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

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        r"""Serialize the example to a plain dictionary.

        Returns:
            A dictionary with keys ``id``, ``input``, ``target``, and
            ``metadata``.

        Example:
            ```pycon
            >>> from argos.meta_agent.examples import Example
            >>> example = Example(id="q1", input="What is 2+2?", target="4")
            >>> example.to_dict()
            {'id': 'q1', 'input': 'What is 2+2?', 'target': '4', 'metadata': None}

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
            >>> from argos.meta_agent.examples import Example
            >>> example = Example(id="q1", input="What is 2+2?", target={"answer": 4, "style": "math"})
            >>> example.to_flat_dict()
            {'id': 'q1', 'input': 'What is 2+2?', 'target.answer': 4, 'target.style': 'math', 'metadata': None}

            ```
        """
        return to_flat_dict(self.to_dict(), separator=separator)

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
            >>> from argos.meta_agent.examples import Example
            >>> example = Example.from_dict({"id": "q1", "input": "What is 2+2?", "target": "4"})
            >>> example.id
            'q1'
            >>> example.target
            '4'

            ```
        """


get_default_registry().register_many({BaseExample: EqualNanEqualityTester()}, exist_ok=True)
