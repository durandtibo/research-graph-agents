r"""Implement a simple analysis implementation."""

from __future__ import annotations

__all__ = ["AnalysisDict"]

from dataclasses import dataclass

from coola.equality import objects_are_equal
from coola.utils.format import repr_indent, repr_mapping, str_indent, str_mapping

from argos.meta_agent.analyses2.base import BaseAnalysis, PrimitiveType


@dataclass(frozen=True)
class AnalysisDict(BaseAnalysis):
    r"""Implement an analysis that combines a dict of analyses.

    Args:
        analyses: The dict of analysis objects to combine.

    Example:
        ```pycon
        >>> from argos.meta_agent.analyses2 import Analysis, AnalysisDict
        >>> analysis = AnalysisDict(
        ...     {"style": Analysis("style analysis"), "semantic": Analysis("semantic analysis")}
        ... )
        >>> analysis
        AnalysisDict(
          (style): Analysis(content='style analysis', metadata=None)
          (semantic): Analysis(content='semantic analysis', metadata=None)
        )
        >>> analysis.to_primitive()
        {'style': 'style analysis', 'semantic': 'semantic analysis'}
        >>> print(analysis.to_json(indent=2))
        {
          "style": "style analysis",
          "semantic": "semantic analysis"
        }

        ```
    """

    analyses: dict[str, BaseAnalysis]

    def __repr__(self) -> str:
        if not self.analyses:
            return f"{self.__class__.__qualname__}()"

        args = repr_indent(repr_mapping(self.analyses))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def __str__(self) -> str:
        if not self.analyses:
            return f"{self.__class__.__qualname__}()"

        args = str_indent(str_mapping(self.analyses))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def equal(self, other: object, equal_nan: bool = False) -> bool:
        if type(other) is not type(self):
            return False
        return objects_are_equal(self.analyses, other.analyses, equal_nan=equal_nan)

    def to_primitive(self) -> PrimitiveType:
        return {key: value.to_primitive() for key, value in self.analyses.items()}
