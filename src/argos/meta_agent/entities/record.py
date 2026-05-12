r"""Define the abstract class and implementations for records."""

from __future__ import annotations

__all__ = ["BaseRecord", "Record"]

from dataclasses import asdict, dataclass
from typing import Any, Generic, Self

from argos.meta_agent.entities.base import BaseEntity
from argos.meta_agent.typing import InputT, OutputT, TargetT


class BaseRecord(BaseEntity, Generic[InputT, TargetT, OutputT]):
    r"""Abstract base class defining the interface for a record.

    Subclasses must define all attributes and implement all methods.

    Attributes:
        id: A unique identifier for the record/example.
        input: The input passed to the agent.
        target: The expected ground-truth output.
        prediction: The predicted output.
        metadata: Optional dictionary of auxiliary information.
            Defaults to ``None``.

    Example:
        ```pycon
        >>> from argos.meta_agent.entities import Record
        >>> x = Record(id="q1", input="What is 2+2?", target="4", prediction="5")
        >>> x
        Record(id='q1', input='What is 2+2?', target='4', prediction='5', metadata=None)
        >>> x.id
        'q1'
        >>> x.input
        'What is 2+2?'
        >>> x.target
        '4'
        >>> x.prediction
        '5'

        ```
    """

    input: InputT | None = None
    target: TargetT | None = None
    prediction: OutputT | None = None


@dataclass(frozen=True)
class Record(BaseRecord[InputT, TargetT, OutputT]):
    r"""Define a concrete record for use in datasets.

    Args:
        id: A unique identifier for the example.
        input: The input passed to the agent.
        target: The expected ground-truth output.
        prediction: The predicted output.
        metadata: Optional dictionary of auxiliary information.
            Defaults to ``None``.

    Example:
        ```pycon
        >>> from argos.meta_agent.entities import Record
        >>> x = Record(id="q1", input="What is 2+2?", target="4", prediction="5")
        >>> x
        Record(id='q1', input='What is 2+2?', target='4', prediction='5', metadata=None)
        >>> x.id
        'q1'
        >>> x.input
        'What is 2+2?'
        >>> x.target
        '4'
        >>> x.prediction
        '5'

        ```
    """

    id: str
    input: InputT | None = None
    target: TargetT | None = None
    prediction: OutputT | None = None
    metadata: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(
            id=data["id"],
            input=data.get("input"),
            target=data.get("target"),
            prediction=data.get("prediction"),
            metadata=data.get("metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
