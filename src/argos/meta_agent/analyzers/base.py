r"""Define the base class to implement an analyzer."""

from __future__ import annotations

__all__ = ["BaseAnalyzer"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl

    from argos.meta_agent.analyses import BaseAnalysis


class BaseAnalyzer(ABC):
    r"""Define the base class to implement an analyzer.

    Subclasses must implement :meth:`analyze` to compare agent
    predictions against benchmark targets and return a dictionary of
    metrics.

    Example:
        ```pycon
        >>> import polars as pl
        >>> from argos.meta_agent.analyzers import NoOpAnalyzer
        >>> analyzer = NoOpAnalyzer()
        >>> data = pl.DataFrame({"id": ["q1", "q2", "q3"]})
        >>> analysis = analyzer.analyze(data)
        >>> analysis
        Result()

        ```
    """

    @abstractmethod
    def analyze(self, data: pl.DataFrame) -> BaseAnalysis:
        r"""Evaluate the performance of the given data.

        Args:
            data: The data used to analyze the performance.

        Returns:
            A dictionary mapping metric names to their computed
                values.

        Example:
            ```pycon
            >>> import polars as pl
            >>> from argos.meta_agent.analyzers import NoOpAnalyzer
            >>> analyzer = NoOpAnalyzer()
            >>> data = pl.DataFrame({"id": ["q1", "q2", "q3"]})
            >>> analysis = analyzer.analyze(data)
            >>> analysis
            Result()

            ```
        """
