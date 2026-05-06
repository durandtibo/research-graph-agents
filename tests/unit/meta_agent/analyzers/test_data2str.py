from __future__ import annotations

import polars as pl

from argos.meta_agent.analyses import Analysis
from argos.meta_agent.analyzers import Data2StrAnalyzer

######################################
#     Tests for Data2StrAnalyzer     #
######################################


def test_data2str_analyzer_repr() -> None:
    assert repr(Data2StrAnalyzer()) == "Data2StrAnalyzer()"


def test_data2str_analyzer_str() -> None:
    assert str(Data2StrAnalyzer()) == "Data2StrAnalyzer()"


def test_data2str_analyzer_equal_true() -> None:
    assert Data2StrAnalyzer().equal(Data2StrAnalyzer())


def test_data2str_analyzer_equal_false_different_type() -> None:
    assert not Data2StrAnalyzer().equal("abc")


def test_data2str_analyzer_equal_false_none() -> None:
    assert not Data2StrAnalyzer().equal(None)


def test_data2str_analyzer_analyze_empty_dataframe() -> None:
    assert Data2StrAnalyzer().analyze(pl.DataFrame()).equal(Analysis("_No data available._"))


def test_data2str_analyzer_analyze_non_empty_dataframe() -> None:
    data = pl.DataFrame({"id": ["q1", "q2"], "score": [0.5, 0.9]})
    assert (
        Data2StrAnalyzer()
        .analyze(data)
        .equal(
            Analysis(
                "shape: (2, 2)\n"
                "┌─────┬───────┐\n"
                "│ id  ┆ score │\n"
                "│ --- ┆ ---   │\n"
                "│ str ┆ f64   │\n"
                "╞═════╪═══════╡\n"
                "│ q1  ┆ 0.5   │\n"
                "│ q2  ┆ 0.9   │\n"
                "└─────┴───────┘"
            )
        )
    )
