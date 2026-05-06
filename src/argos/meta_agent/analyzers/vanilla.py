r"""Implement an analyzer that does nothing."""

from __future__ import annotations

__all__ = ["Analyzer"]

from typing import TYPE_CHECKING

from coola.utils.format import repr_indent, repr_mapping, str_indent, str_mapping

from argos.meta_agent.analyzers.base import BaseAnalyzer

if TYPE_CHECKING:
    import polars as pl

    from argos.meta_agent.analyses import BaseAnalysis


class Analyzer(BaseAnalyzer):
    r"""Implement an analyzer that always returns the same analysis.

    This analyzer should be used if no analysis is desired. It always
    returns an empty analysis.

    Example:
        ```pycon
        >>> import polars as pl
        >>> from argos.meta_agent.analyses import Analysis
        >>> from argos.meta_agent.analyzers import Analyzer
        >>> analyzer = Analyzer(Analysis("my analysis blabla..."))
        >>> analyzer
        Analyzer(
          (analysis): Analysis(content_len=21, metadata=None)
        )
        >>> data = pl.DataFrame({"id": ["q1", "q2", "q3"]})
        >>> analysis = analyzer.analyze(data)
        >>> analysis
        Analysis(content_len=21, metadata=None)

        ```
    """

    def __init__(self, analysis: BaseAnalysis) -> None:
        self._analysis = analysis

    def __repr__(self) -> str:
        args = repr_indent(repr_mapping({"analysis": self._analysis}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def __str__(self) -> str:
        args = str_indent(str_mapping({"analysis": self._analysis}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def analyze(self, data: pl.DataFrame) -> BaseAnalysis:  # noqa: ARG002
        return self._analysis

    def equal(self, other: object, equal_nan: bool = False) -> bool:
        if type(other) is not type(self):
            return False
        return self._analysis.equal(other._analysis, equal_nan=equal_nan)
