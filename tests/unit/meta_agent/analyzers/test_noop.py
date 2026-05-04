from __future__ import annotations

import polars as pl

from argos.meta_agent.analyses import Analysis
from argos.meta_agent.analyzers import NoOpAnalyzer

##################################
#     Tests for NoOpAnalyzer     #
##################################


def test_no_op_analyzer_repr() -> None:
    assert repr(NoOpAnalyzer()) == "NoOpAnalyzer()"


def test_no_op_analyzer_str() -> None:
    assert str(NoOpAnalyzer()) == "NoOpAnalyzer()"


def test_no_op_analyzer_equal_true() -> None:
    assert NoOpAnalyzer().equal(NoOpAnalyzer())


def test_no_op_analyzer_equal_false_different_type() -> None:
    assert not NoOpAnalyzer().equal("abc")


def test_no_op_analyzer_equal_false_none() -> None:
    assert not NoOpAnalyzer().equal(None)


def test_no_op_analyzer_analyze_returns_analysis() -> None:
    assert NoOpAnalyzer().analyze(pl.DataFrame()).equal(Analysis(""))


def test_no_op_analyzer_analyze_non_empty_dataframe() -> None:
    data = pl.DataFrame({"id": ["q1", "q2"], "score": [0.5, 0.9]})
    assert NoOpAnalyzer().analyze(data).equal(Analysis(""))


def test_no_op_analyzer_analyze_ignores_data() -> None:
    analyzer = NoOpAnalyzer()
    assert analyzer.analyze(pl.DataFrame({"a": [1]})).equal(
        analyzer.analyze(pl.DataFrame({"b": [2, 3]}))
    )
