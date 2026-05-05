r"""Implement an analyzer that does nothing."""

from __future__ import annotations

__all__ = ["NoOpAnalyzer"]

from typing import TYPE_CHECKING

from argos.meta_agent.analyses import Analysis
from argos.meta_agent.analyzers.base import BaseAnalyzer

if TYPE_CHECKING:
    import polars as pl


class NoOpAnalyzer(BaseAnalyzer):
    r"""Implement an analyzer that does nothing.

    This analyzer should be used if no analysis is desired. It always
    returns an empty analysis.

    Example:
        ```pycon
        >>> import polars as pl
        >>> from argos.meta_agent.analyzers import NoOpAnalyzer
        >>> analyzer = NoOpAnalyzer()
        >>> data = pl.DataFrame({"id": ["q1", "q2", "q3"]})
        >>> analysis = analyzer.analyze(data)
        >>> analysis
        Analysis(content_len=0)

        ```
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def analyze(self, data: pl.DataFrame) -> Analysis:  # noqa: ARG002
        return Analysis("")

    def equal(self, other: object, equal_nan: bool = False) -> bool:  # noqa: ARG002
        return type(other) is type(self)
