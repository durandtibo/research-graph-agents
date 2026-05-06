from __future__ import annotations

import polars as pl

from argos.meta_agent.analyses import Analysis
from argos.meta_agent.analyzers import Data2CsvAnalyzer

######################################
#     Tests for Data2CsvAnalyzer     #
######################################


def test_data2csv_analyzer_repr() -> None:
    assert repr(Data2CsvAnalyzer()) == "Data2CsvAnalyzer()"


def test_data2csv_analyzer_str() -> None:
    assert str(Data2CsvAnalyzer()) == "Data2CsvAnalyzer()"


def test_data2csv_analyzer_equal_true() -> None:
    assert Data2CsvAnalyzer().equal(Data2CsvAnalyzer())


def test_data2csv_analyzer_equal_false_different_type() -> None:
    assert not Data2CsvAnalyzer().equal("abc")


def test_data2csv_analyzer_equal_false_none() -> None:
    assert not Data2CsvAnalyzer().equal(None)


def test_data2csv_analyzer_analyze_empty_dataframe() -> None:
    assert Data2CsvAnalyzer().analyze(pl.DataFrame()).equal(Analysis("_No data available._"))


def test_data2csv_analyzer_analyze_non_empty_dataframe() -> None:
    data = pl.DataFrame({"id": ["q1", "q2"], "score": [0.5, 0.9]})
    assert (
        Data2CsvAnalyzer()
        .analyze(data)
        .equal(Analysis("Schema: id: String, score: Float64\n\nid,score\nq1,0.5\nq2,0.9\n"))
    )
