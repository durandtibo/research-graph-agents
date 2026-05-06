r"""Implement an analyzer that converts a DataFrame to a string format
analysis."""

from __future__ import annotations

__all__ = ["Data2StrAnalyzer"]

from typing import TYPE_CHECKING

from argos.meta_agent.analyses import Analysis
from argos.meta_agent.analyzers.base import BaseAnalyzer

if TYPE_CHECKING:
    import polars as pl


class Data2StrAnalyzer(BaseAnalyzer):
    r"""Implement an analyzer that converts a DataFrame to a string
    format.

    Example:
        ```pycon
        >>> import polars as pl
        >>> from argos.meta_agent.analyzers import Data2StrAnalyzer
        >>> analyzer = Data2StrAnalyzer()
        >>> data = pl.DataFrame({"id": ["q1", "q2"], "error": [True, False]})
        >>> analysis = analyzer.analyze(data)
        >>> isinstance(analysis.to_text(), str)
        True

        ```
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def analyze(self, data: pl.DataFrame) -> Analysis:
        return Analysis("_No data available._" if data.is_empty() else str(data))

    def equal(self, other: object, equal_nan: bool = False) -> bool:  # noqa: ARG002
        return type(other) is type(self)
