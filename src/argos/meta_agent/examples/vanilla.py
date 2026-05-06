r"""Contain a simple implementation of a dataset example."""

from __future__ import annotations

__all__ = ["Example"]

from dataclasses import dataclass
from typing import Any, Self

from coola.equality import objects_are_equal

from argos.meta_agent.examples.base import BaseExample
from argos.meta_agent.typing import InputT, TargetT


@dataclass(frozen=True)
class Example(BaseExample[InputT, TargetT]):
    r"""Define a concrete labeled example for use in datasets.

    Attributes:
        id: A unique identifier for the example.
        input: The input passed to the agent.
        target: The expected ground-truth output.
        metadata: Optional dictionary of auxiliary information.
            Defaults to ``None``.

    Example:
        ```pycon
        >>> from argos.meta_agent.examples import Example
        >>> example = Example(id="q1", input="What is 2+2?", target="4")
        >>> example.id
        'q1'
        >>> example.input
        'What is 2+2?'
        >>> example.target
        '4'

        ```
    """

    id: str
    input: InputT
    target: TargetT
    metadata: dict[str, Any] | None = None

    def equal(self, other: object, equal_nan: bool = False) -> bool:
        if type(other) is not type(self):
            return False
        return objects_are_equal(self.to_dict(), other.to_dict(), equal_nan=equal_nan)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "input": self.input,
            "target": self.target,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(
            id=data["id"],
            input=data["input"],
            target=data["target"],
            metadata=data.get("metadata"),
        )
