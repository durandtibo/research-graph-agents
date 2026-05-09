r"""Define an analyzer that transform an analysis with an agent."""

from __future__ import annotations

__all__ = ["RefinedAnalyzer"]

import logging
from typing import TYPE_CHECKING, Any

from coola.equality import objects_are_equal
from coola.utils.format import repr_indent, repr_mapping, str_indent, str_mapping

from argos.meta_agent.analyses import Analysis
from argos.meta_agent.analyzers.base import BaseAnalyzer

if TYPE_CHECKING:
    import polars as pl

    from argos.meta_agent.agents import BaseAgent
    from argos.meta_agent.analyses import BaseAnalysis

logger: logging.Logger = logging.getLogger(__name__)


class RefinedAnalyzer(BaseAnalyzer):
    r"""Define an analyzer that refines an analysis using an agent.

    This analyzer first delegates to an inner analyzer to produce an
    initial analysis, then passes that analysis as text to an agent,
    whose response becomes the final analysis. This is useful for
    post-processing or summarizing a raw analysis with an LLM.

    Args:
        analyzer: The inner analyzer used to produce the initial
            analysis from the data.
        agent: The agent used to refine the initial analysis. It
            receives the text representation of the initial
            analysis and its response is used as the final content.

    Example:
        ```pycon
        >>> import polars as pl
        >>> from langchain_core.runnables import RunnableLambda
        >>> from argos.meta_agent.agents import Agent
        >>> from argos.meta_agent.analyses import Analysis
        >>> from argos.meta_agent.analyzers import RefinedAnalyzer, Analyzer
        >>> analyzer = RefinedAnalyzer(
        ...     analyzer=Analyzer(Analysis("raw analysis")),
        ...     agent=Agent(RunnableLambda(str.upper)),
        ... )
        >>> analyzer
        RefinedAnalyzer(
          (analyzer): Analyzer(
              (analysis): Analysis(content_len=12, metadata=None)
            )
          (agent): Agent(
              (runnable): RunnableLambda(upper)
            )
        )
        >>> analysis = analyzer.analyze(pl.DataFrame())
        >>> analysis.to_text()
        'RAW ANALYSIS'

        ```
    """

    def __init__(
        self,
        analyzer: BaseAnalyzer,
        agent: BaseAgent[str, str],
    ) -> None:
        self._analyzer = analyzer
        self._agent = agent

    def __repr__(self) -> str:
        args = repr_indent(repr_mapping(self._get_kwargs()))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def __str__(self) -> str:
        args = str_indent(str_mapping(self._get_kwargs()))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def analyze(self, data: pl.DataFrame) -> BaseAnalysis:
        logger.info("Generating an analysis of the data...")
        analysis = self._analyzer.analyze(data)
        logger.info("Calling an agent on the analysis...")
        content = self._agent.predict([analysis.to_text()])[0]
        return Analysis(content)

    def equal(self, other: object, equal_nan: bool = False) -> bool:
        if type(other) is not type(self):
            return False
        return objects_are_equal(self._get_kwargs(), other._get_kwargs(), equal_nan=equal_nan)

    def _get_kwargs(self) -> dict[str, Any]:
        return {"analyzer": self._analyzer, "agent": self._agent}
