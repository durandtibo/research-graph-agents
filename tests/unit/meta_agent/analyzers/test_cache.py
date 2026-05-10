from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest
from iden.io import load_pickle

from argos.meta_agent.analyses import Analysis, BaseAnalysis
from argos.meta_agent.analyzers import (
    Analyzer,
    NoOpAnalyzer,
    PickleCacheAnalyzer,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def analysis() -> Analysis:
    return Analysis("my analysis blabla...")


#########################################
#     Tests for PickleCacheAnalyzer     #
#########################################


def test_pickle_cache_analyzer_repr(tmp_path: Path, analysis: BaseAnalysis) -> None:
    path = tmp_path.joinpath("analysis.pickle")
    assert repr(PickleCacheAnalyzer(Analyzer(analysis), path)).startswith("PickleCacheAnalyzer(")


def test_pickle_cache_analyzer_repr_with_kwargs(tmp_path: Path, analysis: BaseAnalysis) -> None:
    path = tmp_path.joinpath("analysis.pickle")
    assert repr(PickleCacheAnalyzer(Analyzer(analysis), path, exist_ok=True)).startswith(
        "PickleCacheAnalyzer("
    )


def test_pickle_cache_analyzer_str(tmp_path: Path, analysis: BaseAnalysis) -> None:
    path = tmp_path.joinpath("analysis.pickle")
    assert str(PickleCacheAnalyzer(Analyzer(analysis), path)).startswith("PickleCacheAnalyzer(")


def test_pickle_cache_analyzer_str_with_kwargs(tmp_path: Path, analysis: BaseAnalysis) -> None:
    path = tmp_path.joinpath("analysis.pickle")
    assert str(PickleCacheAnalyzer(Analyzer(analysis), path, exist_ok=True)).startswith(
        "PickleCacheAnalyzer("
    )


def test_pickle_cache_analyzer_equal_true(tmp_path: Path, analysis: BaseAnalysis) -> None:
    path = tmp_path.joinpath("analysis.pickle")
    assert PickleCacheAnalyzer(Analyzer(analysis), path).equal(
        PickleCacheAnalyzer(Analyzer(analysis), path)
    )


def test_pickle_cache_analyzer_equal_false_different_analyzer(
    tmp_path: Path, analysis: BaseAnalysis
) -> None:
    path = tmp_path.joinpath("analysis.pickle")
    assert not PickleCacheAnalyzer(Analyzer(analysis), path).equal(
        PickleCacheAnalyzer(NoOpAnalyzer(), path)
    )


def test_pickle_cache_analyzer_equal_false_different_path(
    tmp_path: Path, analysis: BaseAnalysis
) -> None:
    assert not PickleCacheAnalyzer(Analyzer(analysis), tmp_path.joinpath("a.pickle")).equal(
        PickleCacheAnalyzer(Analyzer(analysis), tmp_path.joinpath("b.pickle"))
    )


def test_pickle_cache_analyzer_equal_false_different_kwargs(
    tmp_path: Path, analysis: BaseAnalysis
) -> None:
    path = tmp_path.joinpath("analysis.pickle")
    assert not PickleCacheAnalyzer(Analyzer(analysis), path).equal(
        PickleCacheAnalyzer(Analyzer(analysis), path, exist_ok=True)
    )


def test_pickle_cache_analyzer_equal_false_different_type(
    tmp_path: Path, analysis: BaseAnalysis
) -> None:
    assert not PickleCacheAnalyzer(Analyzer(analysis), tmp_path.joinpath("analysis.pickle")).equal(
        42
    )


def test_pickle_cache_analyzer_equal_false_different_type_child(
    tmp_path: Path, analysis: BaseAnalysis
) -> None:
    class Child(PickleCacheAnalyzer): ...

    path = tmp_path.joinpath("analysis.pickle")
    assert not PickleCacheAnalyzer(Analyzer(analysis), path).equal(Child(Analyzer(analysis), path))


def test_pickle_cache_analyzer_analyze_returns_analysis_and_creates_file(
    tmp_path: Path, analysis: BaseAnalysis
) -> None:
    path = tmp_path.joinpath("analysis.pickle")
    out = PickleCacheAnalyzer(Analyzer(analysis), path).analyze(pl.DataFrame())
    assert out.equal(analysis)
    assert path.is_file()
    assert load_pickle(path).equal(out)


def test_pickle_cache_analyzer_analyze_creates_parent_directory(
    tmp_path: Path, analysis: BaseAnalysis
) -> None:
    path = tmp_path.joinpath("subdir/analysis.pickle")
    out = PickleCacheAnalyzer(Analyzer(analysis), path).analyze(pl.DataFrame())
    assert out.equal(analysis)
    assert path.is_file()
    assert load_pickle(path).equal(out)
