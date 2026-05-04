r"""Contain the base class to define a dataset."""

from __future__ import annotations

__all__ = ["BaseDataset"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, Self

from coola.equality.tester import EqualNanEqualityTester, get_default_registry

from argos.meta_agent.examples import Example
from argos.meta_agent.typing import InputT, TargetT

if TYPE_CHECKING:
    from collections.abc import Sequence

    import polars as pl

    from argos.meta_agent.examples import BaseExample


class BaseDataset(ABC, Generic[InputT, TargetT]):
    r"""Abstract base class defining the interface for a dataset.

    Subclasses must define all attributes and implement all methods.

    Attributes:
        examples: A mapping from example ID to
            :class:`~argos.meta_agent.examples.BaseExample` instance.
        metadata: Optional dictionary of auxiliary information about
            the dataset. Defaults to ``None``.

    Example:
        ```pycon
        >>> from argos.meta_agent.datasets import BaseDataset, Dataset
        >>> from argos.meta_agent.examples import Example
        >>> dataset = Dataset.from_examples(
        ...     [
        ...         Example(id="q1", input="What is 2+2?", target="4"),
        ...         Example(id="q2", input="What is 3+3?", target="6"),
        ...     ]
        ... )
        >>> isinstance(dataset, BaseDataset)
        True

        ```
    """

    examples: dict[str, BaseExample[InputT, TargetT]]
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
    def to_dataframe(self) -> pl.DataFrame:
        r"""Return a Polars DataFrame representing the dataset examples.

        The dataset-level ``metadata`` attribute is not included in the
        returned DataFrame. Each row corresponds to one example, with
        columns derived from the example's ``to_dict`` representation.

        Returns:
            A Polars DataFrame with one row per example.

        Example:
            ```pycon
            >>> import polars as pl
            >>> from argos.meta_agent.datasets import Dataset
            >>> from argos.meta_agent.examples import Example
            >>> dataset = Dataset.from_examples(
            ...     [
            ...         Example(id="q1", input="What is 2+2?", target="4"),
            ...         Example(id="q2", input="What is 3+3?", target="6"),
            ...     ]
            ... )
            >>> isinstance(dataset.to_dataframe(), pl.DataFrame)
            True

            ```
        """

    @classmethod
    @abstractmethod
    def from_dataframe(
        cls,
        frame: pl.DataFrame,
        metadata: dict[str, Any] | None = None,
        example_type: type[BaseExample[InputT, TargetT]] = Example,
    ) -> Self:
        r"""Create a dataset from a list of examples.

        Args:
            frame: A Polars DataFrame where each row represents a single
                example. Column names must match the fields expected by
                ``example_type.from_dict``.
            metadata: The dataset metadata.
            example_type: The example class to instantiate for each row.
                Defaults to :class:`Example`.

        Returns:
            The dataset instance.
        """

    @classmethod
    @abstractmethod
    def from_examples(
        cls,
        examples: Sequence[BaseExample[InputT, TargetT]],
        metadata: dict[str, Any] | None = None,
    ) -> Self:
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


get_default_registry().register_many({BaseDataset: EqualNanEqualityTester()}, exist_ok=True)
