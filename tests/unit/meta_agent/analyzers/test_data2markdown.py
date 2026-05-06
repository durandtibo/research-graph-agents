from __future__ import annotations

import polars as pl

from argos.meta_agent.analyses import Analysis
from argos.meta_agent.analyzers import Data2MarkdownAnalyzer

###########################################
#     Tests for Data2MarkdownAnalyzer     #
###########################################


def test_data2markdown_analyzer_repr() -> None:
    assert repr(Data2MarkdownAnalyzer()) == "Data2MarkdownAnalyzer()"


def test_data2markdown_analyzer_str() -> None:
    assert str(Data2MarkdownAnalyzer()) == "Data2MarkdownAnalyzer()"


def test_data2markdown_analyzer_equal_true() -> None:
    assert Data2MarkdownAnalyzer().equal(Data2MarkdownAnalyzer())


def test_data2markdown_analyzer_equal_false_different_type() -> None:
    assert not Data2MarkdownAnalyzer().equal("abc")


def test_data2markdown_analyzer_equal_false_none() -> None:
    assert not Data2MarkdownAnalyzer().equal(None)


def test_data2markdown_analyzer_analyze_empty_dataframe() -> None:
    assert Data2MarkdownAnalyzer().analyze(pl.DataFrame()).equal(Analysis("_No data available._"))


def test_data2markdown_analyzer_analyze_non_empty_dataframe() -> None:
    data = pl.DataFrame({"id": ["q1", "q2"], "score": [0.5, 0.9]})
    assert (
        Data2MarkdownAnalyzer()
        .analyze(data)
        .equal(Analysis("| id | score |\n|----|-------|\n| q1 | 0.5   |\n| q2 | 0.9   |"))
    )
