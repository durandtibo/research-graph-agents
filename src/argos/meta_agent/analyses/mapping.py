r"""Contain an analysis which is a dictionary of analyses."""

from __future__ import annotations

__all__ = [
    "BaseAnalysisDict",
    "JsonAnalysisDict",
    "YamlAnalysisDict",
]

import json
from typing import TYPE_CHECKING, Any, Self

import yaml
from coola.equality import objects_are_equal
from coola.utils.format import str_indent, str_mapping

from argos.meta_agent.analyses import BaseAnalysis

if TYPE_CHECKING:
    from collections.abc import Mapping


class BaseAnalysisDict(BaseAnalysis):
    r"""Implement an analysis that combines a mapping of analysis objects
    into a single analysis object.

    Args:
        analyses: The mapping of analysis objects to combine.
        indent: The indentation to use.
        sort_keys: Whether or not to sort the analysis objects
            alphabetically when generating the text representation.
    """

    def __init__(
        self,
        analyses: Mapping[str, BaseAnalysis],
        indent: int | None = None,
        sort_keys: bool = False,
    ) -> None:
        self._analyses = dict(analyses)
        self._indent = indent
        self._sort_keys = sort_keys

    def __repr__(self) -> str:
        if not self._analyses:
            return f"{self.__class__.__qualname__}()"

        return (
            f"{self.__class__.__qualname__}(num_analyses={len(self._analyses):,}, "
            f"indent={self._indent}, sort_keys={self._sort_keys})"
        )

    def __str__(self) -> str:
        if not self._analyses:
            return f"{self.__class__.__qualname__}()"

        analyses = str_indent(str_mapping(self._analyses))
        args = str_indent(
            str_mapping(
                {
                    "indent": self._indent,
                    "sort_keys": self._sort_keys,
                    "analyses": f"\n  {analyses}",
                }
            )
        )
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        if type(other) is not type(self):
            return False
        return objects_are_equal(self.to_dict(), other.to_dict(), equal_nan=equal_nan)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(
            analyses=data["analyses"],
            indent=data.get("indent"),
            sort_keys=data.get("sort_keys", False),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"analyses": self._analyses, "indent": self._indent, "sort_keys": self._sort_keys}


class JsonAnalysisDict(BaseAnalysisDict):
    r"""Implement an analysis that combines a mapping of analyses with a
    JSON style.

    Args:
        analyses: The mapping of analysis objects to combine.
        indent: The indentation to use.
        sort_keys: Whether or not to sort the analysis objects
            alphabetically when generating the text representation.

    Example:
        ```pycon
        >>> from argos.meta_agent.analyses import Analysis, JsonAnalysisDict
        >>> analysis = JsonAnalysisDict(
        ...     {"style": Analysis("style analysis"), "semantic": Analysis("semantic analysis")}
        ... )
        >>> analysis
        JsonAnalysisDict(num_analyses=2, indent=None, sort_keys=False)
        >>> print(analysis)
        JsonAnalysisDict(
          (indent): None
          (sort_keys): False
          (analyses):
              (style): Analysis(content_len=14, metadata=None)
              (semantic): Analysis(content_len=17, metadata=None)
        )
        >>> analysis.to_dict()
        {'analyses': {'style': Analysis(content_len=14, metadata=None), 'semantic': Analysis(content_len=17, metadata=None)}, 'indent': None, 'sort_keys': False}
        >>> print(analysis.to_text())
        {"style": "style analysis", "semantic": "semantic analysis"}
        >>> analysis_with_indent = JsonAnalysisDict(
        ...     {"style": Analysis("style analysis"), "semantic": Analysis("semantic analysis")},
        ...     indent=2,
        ... )
        >>> print(analysis_with_indent.to_text())
        {
          "style": "style analysis",
          "semantic": "semantic analysis"
        }

        ```
    """

    def to_text(self) -> str:
        return json.dumps(
            {key: value.to_text() for key, value in self._analyses.items()}, indent=self._indent
        )


class YamlAnalysisDict(BaseAnalysisDict):
    r"""Implement an analysis that combines a mapping of analysis objects
    with YAML style.

    Args:
        analyses: The mapping of analysis objects to combine.
        indent: The indentation to use.
        sort_keys: Whether or not to sort the analysis objects
            alphabetically when generating the text representation.

    Example:
        ```pycon
        >>> from argos.meta_agent.analyses import Analysis, YamlAnalysisDict
        >>> analysis = YamlAnalysisDict(
        ...     {"style": Analysis("style analysis"), "semantic": Analysis("semantic analysis")}
        ... )
        >>> analysis
        YamlAnalysisDict(num_analyses=2, indent=None, sort_keys=False)
        >>> print(analysis)
        YamlAnalysisDict(
          (indent): None
          (sort_keys): False
          (analyses):
              (style): Analysis(content_len=14, metadata=None)
              (semantic): Analysis(content_len=17, metadata=None)
        )
        >>> analysis.to_dict()
        {'analyses': {'style': Analysis(content_len=14, metadata=None), 'semantic': Analysis(content_len=17, metadata=None)}, 'indent': None, 'sort_keys': False}
        >>> print(analysis.to_text())
        style: style analysis
        semantic: semantic analysis

        ```
    """

    def to_text(self) -> str:
        return yaml.safe_dump(
            {key: value.to_text() for key, value in self._analyses.items()},
            indent=self._indent,
            sort_keys=self._sort_keys,
        )
