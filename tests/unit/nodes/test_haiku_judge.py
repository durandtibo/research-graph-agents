from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
from langchain_core.language_models import BaseChatModel

from argos.nodes.haiku_judge import (
    HaikuJudgeResult,
    HaikuJudgeState,
    make_haiku_judge_node,
)


@pytest.fixture
def mock_llm() -> BaseChatModel:
    return Mock(spec=BaseChatModel, with_structured_output=Mock())


@pytest.fixture
def mock_judge_result() -> HaikuJudgeResult:
    return HaikuJudgeResult(
        structure_passed=True,
        topic_passed=True,
        score=8,
        reasoning="Great imagery and strong structure.",
        passed=True,
    )


######################################
#     Tests for HaikuJudgeResult     #
######################################


def test_haiku_judge_result_valid_passed() -> None:
    result = HaikuJudgeResult(
        structure_passed=True,
        topic_passed=True,
        score=8,
        reasoning="Great imagery and strong structure.",
        passed=True,
    )
    assert result.structure_passed
    assert result.topic_passed
    assert result.score == 8
    assert result.passed


def test_haiku_judge_result_valid_failed() -> None:
    result = HaikuJudgeResult(
        structure_passed=False,
        topic_passed=True,
        score=8,
        reasoning="Great imagery but incorrect structure.",
        passed=False,
    )
    assert not result.structure_passed
    assert result.topic_passed
    assert result.score == 8
    assert not result.passed


def test_haiku_judge_result_invalid_score_too_low() -> None:
    with pytest.raises(ValueError, match=r"score\n  Input should be greater than or equal to 1"):
        HaikuJudgeResult(
            structure_passed=True,
            topic_passed=True,
            score=0,
            reasoning="meow",
            passed=True,
        )


def test_haiku_judge_result_invalid_score_too_high() -> None:
    with pytest.raises(ValueError, match=r"score\n  Input should be less than or equal to 10"):
        HaikuJudgeResult(
            structure_passed=True,
            topic_passed=True,
            score=11,
            reasoning="meow",
            passed=True,
        )


def test_haiku_judge_result_valid_inconsistent_passed_and_structure_passed() -> None:
    with pytest.raises(ValueError, match=r"passed .* does not match structure_passed"):
        HaikuJudgeResult(
            structure_passed=False,
            topic_passed=True,
            score=8,
            reasoning="meow",
            passed=True,
        )


def test_haiku_judge_result_valid_inconsistent_passed_and_topic_passed() -> None:
    with pytest.raises(ValueError, match=r"passed .* does not match topic_passed"):
        HaikuJudgeResult(
            structure_passed=True,
            topic_passed=False,
            score=8,
            reasoning="meow",
            passed=True,
        )


def test_haiku_judge_result_valid_inconsistent_passed_and_score() -> None:
    with pytest.raises(ValueError, match=r"passed .* does not match score"):
        HaikuJudgeResult(
            structure_passed=True,
            topic_passed=True,
            score=6,
            reasoning="meow",
            passed=True,
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
    mock_llm.with_structured_output.assert_called_once_with(HaikuJudgeResult)


def test_make_haiku_judge_node_call(
    mock_llm: BaseChatModel, mock_judge_result: HaikuJudgeResult
) -> None:
    chain_mock = Mock(invoke=Mock(return_value=mock_judge_result))
    with patch("langchain_core.prompts.ChatPromptTemplate.__or__", return_value=chain_mock):
        node = make_haiku_judge_node(mock_llm)
        out = node({"topic": "fog", "haiku": "grey mist hides the hills"})
        assert out == {"evaluation": mock_judge_result}
    chain_mock.invoke.assert_called_once_with(
        {"topic": "fog", "haiku": "grey mist hides the hills"}
    )
