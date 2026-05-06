r"""Define the base class to implement an analyzer."""

from __future__ import annotations

__all__ = ["BaseAnalyzer"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from coola.equality.tester import EqualNanEqualityTester, get_default_registry

if TYPE_CHECKING:
    import polars as pl

    from argos.meta_agent.analyses import BaseAnalysis


class BaseAnalyzer(ABC):
    r"""Define the base class to implement an analyzer.

    Subclasses must implement :meth:`analyze` to analyze the data
    and return a :class:`~argos.meta_agent.analyses.BaseAnalysis`
    object.

    Example:
        ```pycon
        >>> import polars as pl
        >>> from argos.meta_agent.analyses import Analysis
        >>> from argos.meta_agent.analyzers import Analyzer
        >>> analyzer = Analyzer(Analysis("my analysis blabla..."))
        >>> data = pl.DataFrame({"id": ["q1", "q2", "q3"]})
        >>> analysis = analyzer.analyze(data)
        >>> analysis
        Analysis(content_len=21, metadata=None)

        ```
    """

    @abstractmethod
    def analyze(self, data: pl.DataFrame) -> BaseAnalysis:
        r"""Analyze the given data and return a diagnostic analysis.

        Args:
            data: The data used to analyze the performance.

        Returns:
            An analysis object containing the diagnostic information.

        Example:
            ```pycon
            >>> import polars as pl
            >>> from argos.meta_agent.analyses import Analysis
            >>> from argos.meta_agent.analyzers import Analyzer
            >>> analyzer = Analyzer(Analysis("my analysis blabla..."))
            >>> data = pl.DataFrame({"id": ["q1", "q2", "q3"]})
            >>> analysis = analyzer.analyze(data)
            >>> analysis
            Analysis(content_len=21, metadata=None)

            ```
        """

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


get_default_registry().register_many({BaseAnalyzer: EqualNanEqualityTester()}, exist_ok=True)
