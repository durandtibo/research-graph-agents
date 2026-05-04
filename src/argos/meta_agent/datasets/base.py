r"""Contain the base class to define a dataset."""

from __future__ import annotations

__all__ = ["BaseDataset"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic

from argos.meta_agent.typing import InputT, TargetT

if TYPE_CHECKING:
    from collections.abc import Sequence

    import polars as pl

    from argos.meta_agent.examples import BaseExample


class BaseDataset(ABC, Generic[InputT, TargetT]):
    r"""Abstract base class defining the interface for a single labeled
    example.

    Subclasses must define all attributes and implement all methods.
    """

    examples: dict[str, BaseExample[InputT, TargetT]]
    metadata: dict[str, Any] | None = None

    @abstractmethod
    def to_dataframe(self) -> pl.DataFrame:
        r"""Return a Pandas DataFrame representing the dataset examples.

        Note: the metadata are not included in the returned DataFrame.
        """

    @classmethod
    def from_examples(
        cls,
        examples: Sequence[BaseExample[InputT, TargetT]],
        metadata: dict[str, Any] | None = None,
    ) -> BaseDataset[InputT, TargetT]:
        r"""Create a dataset from a list of examples.

        Args:
            examples: A list of examples. The example IDs must be unique.
            metadata: The dataset metadata.

        Returns:
            The dataset instance.

        Raises:
            ValueError: If any example IDs are duplicated.

        Example:
            ```pycon
            >>> from argos.meta_agent.datasets import Dataset
            >>> from argos.meta_agent.examples import Example
            >>> dataset = Dataset.from_examples(
            ...     [
            ...         Example(id="q1", input="What is 2+2?", target="4"),
            ...         Example(id="q2", input="What is 3+3?", target="6"),
            ...     ]
            ... )
            >>> len(dataset.examples)
            2

            ```
        """
