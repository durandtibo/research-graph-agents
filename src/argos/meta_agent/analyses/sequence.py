r"""Contain analyses that combine a sequence of analysis objects."""

from __future__ import annotations

__all__ = ["AnalysisList", "IndentedListAnalysisList"]

from typing import TYPE_CHECKING, Any, Self

from coola.equality import objects_are_equal
from coola.utils.format import str_indent, str_sequence

from argos.meta_agent.analyses import BaseAnalysis

if TYPE_CHECKING:
    from collections.abc import Sequence


class AnalysisList(BaseAnalysis):
    r"""Implement an output that combines a sequence of analysis objects
    into a single output object.

    Args:
        analyses: The sequence of analysis objects to combine.

    Example:
        ```pycon
        >>> from argos.meta_agent.analyses import Analysis, AnalysisList
        >>> analysis = AnalysisList([Analysis("style analysis"), Analysis("semantic analysis")])
        >>> analysis
        AnalysisList(count=2)
        >>> analysis.to_dict()
        {'analyses': [Analysis(content_len=14, metadata=None), Analysis(content_len=17, metadata=None)]}
        >>> print(analysis.to_text())
        ['style analysis', 'semantic analysis']

        ```
    """

    def __init__(self, analyses: Sequence[BaseAnalysis]) -> None:
        self._analyses = list(analyses)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(count={len(self._analyses):,})"

    def __str__(self) -> str:
        args = f"\n  {str_indent(str_sequence(self._analyses))}\n" if self._analyses else ""
        return f"{self.__class__.__qualname__}({args})"

    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return objects_are_equal(self._analyses, other._analyses, equal_nan=equal_nan)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(data["analyses"])

    def to_dict(self) -> dict[str, Any]:
        return {"analyses": self._analyses}

    def to_text(self) -> str:
        return str([analysis.to_text() for analysis in self._analyses])


class IndentedListAnalysisList(AnalysisList):
    r"""Implement an output that combines a sequence of analysis objects
    with an indented list approach.

    Args:
        analyses: The sequence of analysis objects to combine.

    Example:
        ```pycon
        >>> from argos.meta_agent.analyses import Analysis, IndentedListAnalysisList
        >>> analysis = IndentedListAnalysisList(
        ...     [Analysis("style analysis"), Analysis("semantic analysis")]
        ... )
        >>> analysis
        IndentedListAnalysisList(count=2)
        >>> analysis.to_dict()
        {'analyses': [Analysis(content_len=14, metadata=None), Analysis(content_len=17, metadata=None)]}
        >>> print(analysis.to_text())
        - style analysis
        - semantic analysis

        ```
    """

    def to_text(self) -> str:
        items = [f"- {str_indent(analysis.to_text())}" for analysis in self._analyses]
        return "\n".join(items)
