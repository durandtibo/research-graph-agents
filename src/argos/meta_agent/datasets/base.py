r"""Contain the base class to define a dataset."""

from __future__ import annotations

__all__ = ["BaseDataset"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic

from argos.meta_agent.typing import InputT, TargetT

if TYPE_CHECKING:
    import polars as pl

    from argos.meta_agent.examples import BaseExample


class BaseDataset(ABC, Generic[InputT, TargetT]):
    r"""Abstract base class defining the interface for a single labeled
    example.

    Subclasses must define all attributes and implement all methods.
    """

    @property
    @abstractmethod
    def examples(self) -> dict[str, BaseExample[InputT, TargetT]]:
        """The expected ground-truth output."""

    @property
    @abstractmethod
    def metadata(self) -> dict[str, Any] | None:
        """Optional dictionary of auxiliary information."""

    @abstractmethod
    def to_dataframe(self) -> pl.DataFrame:
        r"""Return a Pandas DataFrame representing the dataset examples.

        Note: the metadata are not included in the returned DataFrame.
        """

    @abstractmethod
    @classmethod
    def from_dataframe(
        cls, frame: pl.DataFrame, metadata: dict[str, Any] | None = None
    ) -> BaseDataset[InputT, TargetT]:
        r"""Return a dataset example from the provided dataframe.

        Args:
            frame: Pandas DataFrame representing the dataset examples.
            metadata: Optional dictionary of auxiliary information.
        """
