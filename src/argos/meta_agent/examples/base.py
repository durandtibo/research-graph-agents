r"""Contain the base class to define a dataset example."""

from __future__ import annotations

__all__ = ["BaseExample"]

from abc import ABC, abstractmethod
from typing import Any, Generic, Self

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
        >>> isinstance(example, BaseExample)
        True

        ```
    """

    id: str
    input: InputT
    target: TargetT
    metadata: dict[str, Any] | None = None

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        r"""Serialise the example to a plain dictionary.

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
