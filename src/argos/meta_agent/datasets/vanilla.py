from __future__ import annotations

__all__ = ["Dataset"]

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import polars as pl

from argos.meta_agent.datasets import BaseDataset
from argos.meta_agent.typing import InputT, TargetT

if TYPE_CHECKING:
    from collections.abc import Sequence

    from argos.meta_agent.examples import BaseExample


@dataclass
class Dataset(BaseDataset[InputT, TargetT]):
    r"""Define a collection of labeled examples used for dataseting.

    The examples are indexed by their IDs.

    Attributes:
        examples: A mapping from example ID to
            :class:`BaseExample` instance.
        metadata: Optional dictionary of auxiliary information about
            the dataset. Defaults to ``None``.

    Example:
        ```pycon
        >>> from argos.meta_agent.datasets import Dataset
        >>> from argos.meta_agent.examples import Example
        >>> dataset = Dataset(
        ...     examples={
        ...         "q1": Example(id="q1", input="What is 2+2?", target="4"),
        ...         "q2": Example(id="q2", input="What is 3+3?", target="6"),
        ...     }
        ... )
        >>> len(dataset.examples)
        2

        ```
    """

    examples: dict[str, BaseExample[InputT, TargetT]]
    metadata: dict[str, Any] | None = None

    def to_dataframe(self) -> pl.DataFrame:
        return pl.DataFrame([example.to_dict() for example in self.examples.values()])

    @classmethod
    def from_examples(
        cls,
        examples: Sequence[BaseExample[InputT, TargetT]],
        metadata: dict[str, Any] | None = None,
    ) -> Dataset[InputT, TargetT]:
        data = {example.id: example for example in examples}
        if len(data) != len(examples):
            msg = "Some example IDs are duplicated"
            raise ValueError(msg)
        return cls(examples=data, metadata=metadata)
