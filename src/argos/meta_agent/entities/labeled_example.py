r"""Define abstract and concrete entities for labeled examples."""

from __future__ import annotations

__all__ = ["BaseLabeledExample", "LabeledExample"]

from dataclasses import asdict, dataclass
from typing import Any, Generic, Self

from argos.meta_agent.entities.example import BaseExample
from argos.meta_agent.typing import InputT, TargetT


class BaseLabeledExample(BaseExample[InputT], Generic[InputT, TargetT]):
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
        >>> from argos.meta_agent.entities import LabeledExample
        >>> x = LabeledExample(id="q1", input="What is 2+2?", target="4")
        >>> x
        LabeledExample(id='q1', input='What is 2+2?', target='4', metadata=None)
        >>> x.id
        'q1'
        >>> x.input
        'What is 2+2?'

        ```
    """

    target: TargetT


@dataclass(frozen=True)
class LabeledExample(BaseLabeledExample[InputT, TargetT]):
    r"""Define a concrete labeled example for use in datasets.

    Args:
        id: A unique identifier for the example.
        input: The input passed to the agent.
        target: The expected ground-truth output.
        metadata: Optional dictionary of auxiliary information.
            Defaults to ``None``.

    Example:
        ```pycon
        >>> from argos.meta_agent.entities import LabeledExample
        >>> x = LabeledExample(id="q1", input="What is 2+2?", target="4")
        >>> x
        LabeledExample(id='q1', input='What is 2+2?', target='4', metadata=None)
        >>> x.id
        'q1'
        >>> x.input
        'What is 2+2?'

        ```

    Note:
        ``from_dict`` always requires ``id``. It reads ``input`` and
        ``target`` with ``dict.get``, so missing keys are converted to
        ``None``.
    """

    id: str
    input: InputT
    target: TargetT
    metadata: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(
            id=data["id"],
            input=data.get("input"),
            target=data.get("target"),
            metadata=data.get("metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
