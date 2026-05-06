from __future__ import annotations

import polars as pl
import pytest
from langchain_core.runnables import RunnableLambda

from argos.meta_agent.agents import Agent, BaseAgent
from argos.meta_agent.analyses import Analysis, BaseAnalysis
from argos.meta_agent.analyzers import Analyzer, NoOpAnalyzer, RefinedAnalyzer


@pytest.fixture
def agent() -> BaseAgent:
    return Agent(RunnableLambda(str.upper))


@pytest.fixture
def analysis() -> BaseAnalysis:
    return Analysis("my analysis")


#####################################
#     Tests for RefinedAnalyzer     #
#####################################


def test_refined_analyzer_repr(agent: BaseAgent) -> None:
    assert repr(RefinedAnalyzer(NoOpAnalyzer(), agent)).startswith("RefinedAnalyzer(")


def test_refined_analyzer_str(agent: BaseAgent) -> None:
    assert str(RefinedAnalyzer(NoOpAnalyzer(), agent)).startswith("RefinedAnalyzer(")


def test_refined_analyzer_equal_true(agent: BaseAgent) -> None:
    assert RefinedAnalyzer(NoOpAnalyzer(), agent).equal(RefinedAnalyzer(NoOpAnalyzer(), agent))


def test_refined_analyzer_equal_false_different_analyzer(agent: BaseAgent) -> None:
    assert not RefinedAnalyzer(NoOpAnalyzer(), agent).equal(
        RefinedAnalyzer(Analyzer(Analysis("my analysis")), agent)
    )


def test_refined_analyzer_equal_false_different_agent(agent: BaseAgent) -> None:
    assert not RefinedAnalyzer(NoOpAnalyzer(), agent).equal(
        RefinedAnalyzer(NoOpAnalyzer(), Agent(RunnableLambda(str.lower)))
    )


def test_refined_analyzer_equal_false_different_type(agent: BaseAgent) -> None:
    assert not RefinedAnalyzer(NoOpAnalyzer(), agent).equal(NoOpAnalyzer())


def test_refined_analyzer_equal_false_different_type_child(agent: BaseAgent) -> None:
    class Child(RefinedAnalyzer): ...

    assert not RefinedAnalyzer(NoOpAnalyzer(), agent).equal(Child(NoOpAnalyzer(), agent))


def test_refined_analyzer_analyze(agent: BaseAgent, analysis: BaseAnalysis) -> None:
    out = RefinedAnalyzer(Analyzer(analysis), agent).analyze(pl.DataFrame())
    assert out.equal(Analysis("MY ANALYSIS"))
