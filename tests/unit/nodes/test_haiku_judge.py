from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
from langchain_core.language_models import BaseChatModel

from argos.models.haiku_judge import HaikuJudgeOutput
from argos.nodes.haiku_judge import (
    HaikuJudgeState,
    make_haiku_judge_node,
)


@pytest.fixture
def mock_llm() -> BaseChatModel:
    return Mock(spec=BaseChatModel, with_structured_output=Mock())


@pytest.fixture
def mock_judge_result() -> HaikuJudgeOutput:
    return HaikuJudgeOutput(
        overall_prediction=True,
        score_prediction=8,
        score_reasoning="score explanation",
        structure_prediction=True,
        structure_reasoning="structure explanation",
        topic_prediction=True,
        topic_reasoning="topic explanation",
    )


#####################################
#     Tests for HaikuJudgeState     #
#####################################


def test_haiku_judge_state_keys() -> None:
    assert set(HaikuJudgeState.__annotations__) == {"topic", "haiku", "evaluation"}


###########################################
#     Tests for make_haiku_judge_node     #
###########################################


def test_make_haiku_judge_node_returns_callable(mock_llm: BaseChatModel) -> None:
    node = make_haiku_judge_node(llm=mock_llm)
    assert callable(node)


def test_make_haiku_judge_node_with_structured_output_called(mock_llm: BaseChatModel) -> None:
    make_haiku_judge_node(llm=mock_llm)
    mock_llm.with_structured_output.assert_called_once_with(HaikuJudgeOutput)


def test_make_haiku_judge_node_call(
    mock_llm: BaseChatModel, mock_judge_result: HaikuJudgeOutput
) -> None:
    chain_mock = Mock(invoke=Mock(return_value=mock_judge_result))
    with patch("argos.nodes.haiku_judge.create_haiku_judge_model", return_value=chain_mock):
        node = make_haiku_judge_node(mock_llm)
        out = node({"topic": "fog", "haiku": "grey mist hides the hills"})
        assert out == {"evaluation": mock_judge_result}
    chain_mock.invoke.assert_called_once_with(
        {"haiku": "grey mist hides the hills", "topic": "fog"}
    )
