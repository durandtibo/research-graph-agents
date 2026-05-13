r"""Define analyzer wrappers that cache the analysis result to a file."""

from __future__ import annotations

__all__ = ["BaseCacheAnalyzer", "PickleCacheAnalyzer"]

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from coola.equality import objects_are_equal
from coola.utils.format import repr_indent, repr_mapping, str_indent, str_mapping
from iden.io import load_pickle, save_pickle

from argos.meta_agent.analyzers.base import BaseAnalyzer

if TYPE_CHECKING:
    from pathlib import Path

    import polars as pl

    from argos.meta_agent.analyses import BaseAnalysis

logger: logging.Logger = logging.getLogger(__name__)


class BaseCacheAnalyzer(BaseAnalyzer):
    r"""Implement an abstract analyzer wrapper that caches the analysis to
    a file.

    On the first call to :meth:`analyze`, the inner ``analyzer`` is
    invoked and the resulting :class:`~argos.meta_agent.analyses.BaseAnalysis`
    object is persisted to ``path``.  On subsequent calls, the cached
    object is loaded directly from ``path``, skipping the inner
    analyzer. The returned analysis is the same whether it was freshly
    computed or loaded from cache, so the wrapper is transparent to
    callers.

    Args:
        analyzer: The inner analyzer whose output is cached.
        path: Destination path for the cache file.

    Example:
        ```pycon
        >>> import pathlib, tempfile, polars as pl
        >>> from argos.meta_agent.analyses import Analysis
        >>> from argos.meta_agent.analyzers import Analyzer, PickleCacheAnalyzer
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     path = pathlib.Path(tmp).joinpath("analysis.pickle")
        ...     analyzer = PickleCacheAnalyzer(
        ...         analyzer=Analyzer(Analysis("my analysis")), path=path
        ...     )
        ...     analyzer
        ...     analysis = analyzer.analyze(pl.DataFrame())
        ...     analysis
        ...     path.is_file()
        ...
        PickleCacheAnalyzer(
          (analyzer): Analyzer(
              (analysis): Analysis(content='my analysis', metadata=None)
            )
          (path): PosixPath('.../analysis.pickle')
        )
        Analysis(content='my analysis', metadata=None)
        True

        ```
    """

    def __init__(self, analyzer: BaseAnalyzer, path: Path, **kwargs: Any) -> None:
        self._analyzer = analyzer
        self._path = path
        self._kwargs = kwargs

    def __repr__(self) -> str:
        args = repr_indent(repr_mapping(self._get_kwargs()))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def __str__(self) -> str:
        args = str_indent(str_mapping(self._get_kwargs()))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def analyze(self, data: pl.DataFrame) -> BaseAnalysis:
        if self._path.is_file():
            return self._load()

        logger.info("Analyzing the data...")
        analysis = self._analyzer.analyze(data)
        logger.info(f"Caching the analysis to {self._path}")
        self._save(analysis)
        return analysis

    def equal(self, other: object, equal_nan: bool = False) -> bool:
        if type(other) is not type(self):
            return False
        return objects_are_equal(self._get_kwargs(), other._get_kwargs(), equal_nan=equal_nan)

    @abstractmethod
    def _load(self) -> BaseAnalysis:
        r"""Load the analysis from the cache.

        Returns:
            The cached analysis.
        """

    @abstractmethod
    def _save(self, analysis: BaseAnalysis) -> None:
        r"""Cache the analysis to a file.

        Args:
            analysis: The analysis to export.
        """

    def _get_kwargs(self) -> dict[str, Any]:
        return {"analyzer": self._analyzer, "path": self._path} | self._kwargs


class PickleCacheAnalyzer(BaseCacheAnalyzer):
    r"""Implement an analyzer wrapper that caches the analysis to a
    Pickle file.

    On the first call to :meth:`analyze`, the inner ``analyzer`` is
    invoked and the resulting
    :class:`~argos.meta_agent.analyses.BaseAnalysis` object is
    serialized in full to a Pickle file at ``path``.  On subsequent
    calls, the cached object is deserialized from ``path`` directly,
    bypassing the inner analyzer.

    Args:
        analyzer: The inner analyzer whose output is cached.
        path: Destination path for the Pickle file.

    Example:
        ```pycon
        >>> import pathlib, tempfile, polars as pl
        >>> from argos.meta_agent.analyses import Analysis
        >>> from argos.meta_agent.analyzers import Analyzer, PickleCacheAnalyzer
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     path = pathlib.Path(tmp).joinpath("analysis.pickle")
        ...     analyzer = PickleCacheAnalyzer(
        ...         analyzer=Analyzer(Analysis("my analysis")), path=path
        ...     )
        ...     analyzer
        ...     analysis = analyzer.analyze(pl.DataFrame())
        ...     analysis
        ...     path.is_file()
        ...
        PickleCacheAnalyzer(
          (analyzer): Analyzer(
              (analysis): Analysis(content='my analysis', metadata=None)
            )
          (path): PosixPath('.../analysis.pickle')
        )
        Analysis(content='my analysis', metadata=None)
        True

        ```
    """

    def _load(self) -> BaseAnalysis:
        return load_pickle(self._path)

    def _save(self, analysis: BaseAnalysis) -> None:
        save_pickle(analysis, self._path, **self._kwargs)
