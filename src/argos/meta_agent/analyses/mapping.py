r"""Contain an analysis which is a dictionary of analyses."""

from __future__ import annotations

__all__ = ["AnalysisDict", "BaseAnalysisDict", "BulletPointAnalysisDict", "YamlAnalysisDict"]

from typing import TYPE_CHECKING, Any, Self

from coola.equality import objects_are_equal
from coola.utils.format import str_indent, str_mapping

from argos.meta_agent.analyses import BaseAnalysis

if TYPE_CHECKING:
    from collections.abc import Mapping


class BaseAnalysisDict(BaseAnalysis):
    r"""Implement an output that combines a mapping of output objects
    into a single output object.

    Args:
        analyses: The mapping of output objects to combine.

    Example:
        ```pycon
        >>> from argos.meta_agent.analyses import Analysis, AnalysisDict
        >>> analysis = AnalysisDict(
        ...     {"style": Analysis("style analysis"), "semantic": Analysis("semantic analysis")}
        ... )
        >>> analysis
        AnalysisDict(count=2)
        >>> analysis.to_dict()
        {'analyses': {'style': Analysis(content_len=14, metadata=None), 'semantic': Analysis(content_len=17, metadata=None)}}
        >>> print(analysis.to_text())
        {'style': 'style analysis', 'semantic': 'semantic analysis'}

        ```
    """

    def __init__(self, analyses: Mapping[str, BaseAnalysis]) -> None:
        self._analyses = dict(analyses)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(count={len(self._analyses):,})"

    def __str__(self) -> str:
        if not self._analyses:
            return f"{self.__class__.__qualname__}()"

        args = str_indent(str_mapping(self._analyses))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return objects_are_equal(self._analyses, other._analyses, equal_nan=equal_nan)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(data["analyses"])

    def to_dict(self) -> dict[str, Any]:
        return {"analyses": self._analyses}

    def _format_item(self, key: str, value: BaseAnalysis, prefix: str = "") -> str:
        """Format a single analysis item with proper indentation."""
        value_text = str_indent(value.to_text())
        if isinstance(value, BaseAnalysisDict):
            return f"{prefix}{key}:\n  {value_text}"
        return f"{prefix}{key}: {value_text}"


class AnalysisDict(BaseAnalysisDict):
    r"""Implement an output that combines a mapping of output objects
    into a single output object.

    Args:
        analyses: The mapping of output objects to combine.

    Example:
        ```pycon
        >>> from argos.meta_agent.analyses import Analysis, AnalysisDict
        >>> analysis = AnalysisDict(
        ...     {"style": Analysis("style analysis"), "semantic": Analysis("semantic analysis")}
        ... )
        >>> analysis
        AnalysisDict(count=2)
        >>> analysis.to_dict()
        {'analyses': {'style': Analysis(content_len=14, metadata=None), 'semantic': Analysis(content_len=17, metadata=None)}}
        >>> print(analysis.to_text())
        {'style': 'style analysis', 'semantic': 'semantic analysis'}

        ```
    """

    def to_text(self) -> str:
        return str({key: value.to_text() for key, value in self._analyses.items()})


class BulletPointAnalysisDict(BaseAnalysisDict):
    r"""Implement an output that combines a mapping of output objects
    with a indented list approach.

    Args:
        analyses: The mapping of output objects to combine.

    Example:
        ```pycon
        >>> from argos.meta_agent.analyses import Analysis, BulletPointAnalysisDict
        >>> analysis = BulletPointAnalysisDict(
        ...     {"style": Analysis("style analysis"), "semantic": Analysis("semantic analysis")}
        ... )
        >>> analysis
        IndentedListAnalysisDict(count=2)
        >>> analysis.to_dict()
        {'analyses': {'style': Analysis(content_len=14, metadata=None), 'semantic': Analysis(content_len=17, metadata=None)}}
        >>> print(analysis.to_text())
        - style: style analysis
        - semantic: semantic analysis

        ```
    """

    def to_text(self) -> str:
        items = [
            self._format_item(key, value, prefix="- ") for key, value in self._analyses.items()
        ]
        return "\n".join(items)


class YamlAnalysisDict(BaseAnalysisDict):
    r"""Implement an output that combines a mapping of output objects
    with YAML style.

    Args:
        analyses: The mapping of output objects to combine.

    Example:
        ```pycon
        >>> from argos.meta_agent.analyses import Analysis, YamlAnalysisDict
        >>> analysis = YamlAnalysisDict(
        ...     {"style": Analysis("style analysis"), "semantic": Analysis("semantic analysis")}
        ... )
        >>> analysis
        YamlAnalysisDict(count=2)
        >>> analysis.to_dict()
        {'analyses': {'style': Analysis(content_len=14, metadata=None), 'semantic': Analysis(content_len=17, metadata=None)}}
        >>> print(analysis.to_text())
        style: style analysis
        semantic: semantic analysis

        ```
    """

    def to_text(self) -> str:
        items = [self._format_item(key, value) for key, value in self._analyses.items()]
        return "\n".join(items)
