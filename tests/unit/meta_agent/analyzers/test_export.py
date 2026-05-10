from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest
from feu.utils.io import load_json

from argos.meta_agent.analyses import Analysis, BaseAnalysis
from argos.meta_agent.analyzers import Analyzer, JsonExportAnalyzer, NoOpAnalyzer

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def analysis() -> Analysis:
    return Analysis("my analysis blabla...")


########################################
#     Tests for JsonExportAnalyzer     #
########################################


def test_json_export_analyzer_repr(tmp_path: Path, analysis: BaseAnalysis) -> None:
    path = tmp_path.joinpath("analysis.json")
    assert repr(JsonExportAnalyzer(Analyzer(analysis), path)).startswith("JsonExportAnalyzer(")


def test_json_export_analyzer_repr_with_kwargs(tmp_path: Path, analysis: BaseAnalysis) -> None:
    path = tmp_path.joinpath("analysis.json")
    assert repr(JsonExportAnalyzer(Analyzer(analysis), path, exist_ok=True)).startswith(
        "JsonExportAnalyzer("
    )


def test_json_export_analyzer_str(tmp_path: Path, analysis: BaseAnalysis) -> None:
    path = tmp_path.joinpath("analysis.json")
    assert str(JsonExportAnalyzer(Analyzer(analysis), path)).startswith("JsonExportAnalyzer(")


def test_json_export_analyzer_str_with_kwargs(tmp_path: Path, analysis: BaseAnalysis) -> None:
    path = tmp_path.joinpath("analysis.json")
    assert str(JsonExportAnalyzer(Analyzer(analysis), path, exist_ok=True)).startswith(
        "JsonExportAnalyzer("
    )


def test_json_export_analyzer_equal_true(tmp_path: Path, analysis: BaseAnalysis) -> None:
    path = tmp_path.joinpath("analysis.json")
    assert JsonExportAnalyzer(Analyzer(analysis), path).equal(
        JsonExportAnalyzer(Analyzer(analysis), path)
    )


def test_json_export_analyzer_equal_false_different_analyzer(
    tmp_path: Path, analysis: BaseAnalysis
) -> None:
    path = tmp_path.joinpath("analysis.json")
    assert not JsonExportAnalyzer(Analyzer(analysis), path).equal(
        JsonExportAnalyzer(NoOpAnalyzer(), path)
    )


def test_json_export_analyzer_equal_false_different_path(
    tmp_path: Path, analysis: BaseAnalysis
) -> None:
    assert not JsonExportAnalyzer(Analyzer(analysis), tmp_path.joinpath("a.json")).equal(
        JsonExportAnalyzer(Analyzer(analysis), tmp_path.joinpath("b.json"))
    )


def test_json_export_analyzer_equal_false_different_kwargs(
    tmp_path: Path, analysis: BaseAnalysis
) -> None:
    path = tmp_path.joinpath("analysis.json")
    assert not JsonExportAnalyzer(Analyzer(analysis), path).equal(
        JsonExportAnalyzer(Analyzer(analysis), path, exist_ok=True)
    )


def test_json_export_analyzer_equal_false_different_type(
    tmp_path: Path, analysis: BaseAnalysis
) -> None:
    assert not JsonExportAnalyzer(Analyzer(analysis), tmp_path.joinpath("analysis.json")).equal(42)


def test_json_export_analyzer_equal_false_different_type_child(
    tmp_path: Path, analysis: BaseAnalysis
) -> None:
    class Child(JsonExportAnalyzer): ...

    path = tmp_path.joinpath("analysis.json")
    assert not JsonExportAnalyzer(Analyzer(analysis), path).equal(Child(Analyzer(analysis), path))


def test_json_export_analyzer_analyze_returns_analysis_and_creates_file(
    tmp_path: Path, analysis: BaseAnalysis
) -> None:
    path = tmp_path.joinpath("analysis.json")
    out = JsonExportAnalyzer(Analyzer(analysis), path).analyze(pl.DataFrame())
    assert out.equal(analysis)
    assert path.is_file()
    assert load_json(path) == "my analysis blabla..."


def test_json_export_analyzer_analyze_creates_parent_directory(
    tmp_path: Path, analysis: BaseAnalysis
) -> None:
    path = tmp_path.joinpath("subdir/analysis.json")
    out = JsonExportAnalyzer(Analyzer(analysis), path).analyze(pl.DataFrame())
    assert out.equal(analysis)
    assert path.is_file()
    assert load_json(path) == "my analysis blabla..."
