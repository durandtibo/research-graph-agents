from __future__ import annotations

import polars as pl

from argos.meta_agent.analyses import Analysis
from argos.meta_agent.analyzers import Analyzer

##############################
#     Tests for Analyzer     #
##############################


def test_analyzer_repr() -> None:
    assert repr(Analyzer(Analysis("my analysis"))) == (
        "Analyzer(\n  (analysis): Analysis(content_len=11, metadata=None)\n)"
    )


def test_analyzer_str() -> None:
    assert str(Analyzer(Analysis("my analysis"))) == (
        "Analyzer(\n  (analysis): Analysis(content_len=11, metadata=None)\n)"
    )


def test_analyzer_equal_true() -> None:
    assert Analyzer(Analysis("my analysis")).equal(Analyzer(Analysis("my analysis")))


def test_analyzer_equal_false_different_analysis() -> None:
    assert not Analyzer(Analysis("my analysis")).equal(Analyzer(Analysis("other analysis")))


def test_analyzer_equal_false_different_type() -> None:
    assert not Analyzer(Analysis("my analysis")).equal("my analysis")


def test_analyzer_equal_nan_false_by_default() -> None:
    assert not Analyzer(Analysis(float("nan"))).equal(Analyzer(Analysis(float("nan"))))


def test_analyzer_equal_nan_true() -> None:
    assert Analyzer(Analysis(float("nan"))).equal(Analyzer(Analysis(float("nan"))), equal_nan=True)


def test_analyzer_analyze_returns_analysis() -> None:
    analysis = Analysis("my analysis")
    data = pl.DataFrame({"id": ["q1", "q2"], "score": [0.5, 0.9]})
    out = Analyzer(analysis).analyze(data)
    assert out is analysis
    assert out.equal(Analysis("my analysis"))


def test_analyzer_analyze_ignores_data() -> None:
    analysis = Analysis("my analysis")
    analyzer = Analyzer(analysis)
    out1 = analyzer.analyze(pl.DataFrame({"a": [1]}))
    out2 = analyzer.analyze(pl.DataFrame({"b": [2, 3]}))
    assert out1 is out2


def test_analyzer_analyze_empty_dataframe() -> None:
    analysis = Analysis("my analysis")
    out = Analyzer(analysis).analyze(pl.DataFrame())
    assert out is analysis
    assert out.equal(Analysis("my analysis"))
