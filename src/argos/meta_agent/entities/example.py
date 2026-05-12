r"""Define the abstract class and implementations for examples."""

from __future__ import annotations

__all__ = ["BaseExample", "Example"]

from dataclasses import asdict, dataclass
from typing import Any, Generic, Self

from argos.meta_agent.entities.base import BaseEntity
from argos.meta_agent.typing import InputT


class BaseExample(BaseEntity, Generic[InputT]):
    r"""Abstract base class defining the interface for an example.

    Subclasses must define all attributes and implement all methods.

    Attributes:
        id: A unique identifier for the example.
        input: The input passed to the agent.
        metadata: Optional dictionary of auxiliary information.
            Defaults to ``None``.

    Example:
        ```pycon
        >>> from argos.meta_agent.entities import Example
        >>> example = Example(id="q1", input="What is 2+2?")
        >>> example
        Example(id='q1', input='What is 2+2?', metadata=None)
        >>> example.id
        'q1'
        >>> example.input
        'What is 2+2?'

        ```
    """

    input: InputT


@dataclass(frozen=True)
class Example(BaseExample[InputT]):
    r"""Define a concrete example for use in datasets.

    Args:
        id: A unique identifier for the example.
        input: The input passed to the agent.
        metadata: Optional dictionary of auxiliary information.
            Defaults to ``None``.

    Example:
        ```pycon
        >>> from argos.meta_agent.entities import Example
        >>> example = Example(id="q1", input="What is 2+2?")
        >>> example
        Example(id='q1', input='What is 2+2?', metadata=None)
        >>> example.id
        'q1'
        >>> example.input
        'What is 2+2?'

        ```
    """

    id: str
    input: InputT
    metadata: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(
            id=data["id"],
            input=data["input"],
            metadata=data.get("metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
