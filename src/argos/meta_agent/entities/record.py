r"""Define the abstract class and implementations for records."""

from __future__ import annotations

__all__ = ["BaseRecord", "Record"]

from dataclasses import dataclass
from typing import Any, Generic, Self

from coola.equality import objects_are_equal

from argos.meta_agent.entities.base import BaseEntity
from argos.meta_agent.typing import InputT, OutputT, TargetT


class BaseRecord(BaseEntity, Generic[InputT, TargetT, OutputT]):
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
        >>> from argos.meta_agent.entities import Record
        >>> example = Record(id="q1", input="What is 2+2?", target="4", prediction="5")
        >>> example
        Record(id='q1', input='What is 2+2?', target='4', prediction='5', metadata=None)
        >>> example.id
        'q1'
        >>> example.input
        'What is 2+2?'
        >>> example.target
        '4'
        >>> example.prediction
        '5'

        ```
    """

    id: str
    input: InputT | None = None
    target: TargetT | None = None
    prediction: OutputT | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class Record(BaseRecord[InputT, TargetT, OutputT]):
    r"""Define a concrete labeled example for use in datasets.

    Args:
        id: A unique identifier for the example.
        input: The input passed to the agent.
        target: The expected ground-truth output.
        metadata: Optional dictionary of auxiliary information.
            Defaults to ``None``.

    Example:
        ```pycon
        >>> from argos.meta_agent.entities import Record
        >>> example = Record(id="q1", input="What is 2+2?", target="4", prediction="5")
        >>> example
        Record(id='q1', input='What is 2+2?', target='4', prediction='5', metadata=None)
        >>> example.id
        'q1'
        >>> example.input
        'What is 2+2?'
        >>> example.target
        '4'
        >>> example.prediction
        '5'

        ```
    """

    id: str
    input: InputT | None = None
    target: TargetT | None = None
    prediction: OutputT | None = None
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
            "prediction": self.prediction,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(
            id=data["id"],
            input=data.get("input"),
            target=data.get("target"),
            prediction=data.get("prediction"),
            metadata=data.get("metadata"),
        )
