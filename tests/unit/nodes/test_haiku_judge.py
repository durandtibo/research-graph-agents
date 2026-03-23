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
        line_count=3,
        line_count_passed=True,
        syllable_breakdown=[5, 7, 5],
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
        line_count=3,
        line_count_passed=True,
        syllable_breakdown=[5, 7, 5],
        structure_passed=True,
        topic_passed=True,
        score=8,
        reasoning="Great imagery and strong structure.",
        passed=True,
    )
    assert result.line_count == 3
    assert result.line_count_passed
    assert result.syllable_breakdown == [5, 7, 5]
    assert result.structure_passed
    assert result.topic_passed
    assert result.score == 8
    assert result.passed


def test_haiku_judge_result_valid_failed() -> None:
    result = HaikuJudgeResult(
        line_count=4,
        line_count_passed=False,
        syllable_breakdown=[5, 7, 5, 3],
        structure_passed=False,
        topic_passed=True,
        score=8,
        reasoning="Great imagery but incorrect structure.",
        passed=False,
    )
    assert result.line_count == 4
    assert not result.line_count_passed
    assert result.syllable_breakdown == [5, 7, 5, 3]
    assert not result.structure_passed
    assert result.topic_passed
    assert result.score == 8
    assert not result.passed


@pytest.mark.parametrize("line_count", [-1, -2])
def test_haiku_judge_result_invalid_line_count(line_count: int) -> None:
    with pytest.raises(
        ValueError, match=r"line_count\n  Input should be greater than or equal to 0"
    ):
        HaikuJudgeResult(
            line_count=line_count,
            line_count_passed=False,
            syllable_breakdown=[5, 7, 5],
            structure_passed=True,
            topic_passed=True,
            score=8,
            reasoning="meow",
            passed=True,
        )


def test_haiku_judge_result_invalid_score_too_low() -> None:
    with pytest.raises(ValueError, match=r"score\n  Input should be greater than or equal to 1"):
        HaikuJudgeResult(
            line_count=3,
            line_count_passed=False,
            syllable_breakdown=[5, 7, 5],
            structure_passed=True,
            topic_passed=True,
            score=0,
            reasoning="meow",
            passed=True,
        )


def test_haiku_judge_result_invalid_score_too_high() -> None:
    with pytest.raises(ValueError, match=r"score\n  Input should be less than or equal to 10"):
        HaikuJudgeResult(
            line_count=3,
            line_count_passed=False,
            syllable_breakdown=[5, 7, 5],
            structure_passed=True,
            topic_passed=True,
            score=11,
            reasoning="meow",
            passed=True,
        )


@pytest.mark.parametrize("line_count", [0, 1, 2, 4])
def test_haiku_judge_result_valid_inconsistent_line_count(line_count: int) -> None:
    with pytest.raises(ValueError, match=r"line_count_passed .* does not match line_count"):
        HaikuJudgeResult(
            line_count=line_count,
            line_count_passed=True,
            syllable_breakdown=[5, 7, 5],
            structure_passed=True,
            topic_passed=True,
            score=8,
            reasoning="meow",
            passed=True,
        )


@pytest.mark.parametrize("syllable_breakdown", [[5], [5, 7], [5, 7, 5, 3]])
def test_haiku_judge_result_valid_inconsistent_syllable_breakdown(
    syllable_breakdown: list[int],
) -> None:
    with pytest.raises(ValueError, match=r"line_count .* does not match syllable_breakdown"):
        HaikuJudgeResult(
            line_count=3,
            line_count_passed=True,
            syllable_breakdown=syllable_breakdown,
            structure_passed=False,
            topic_passed=True,
            score=8,
            reasoning="meow",
            passed=True,
        )


@pytest.mark.parametrize("syllable_breakdown", [[5, 5, 7], [1, 1, 1], [7, 5, 5]])
def test_haiku_judge_result_valid_inconsistent_structure_passed(
    syllable_breakdown: list[int],
) -> None:
    with pytest.raises(ValueError, match=r"structure_passed .* does not match syllable_breakdown"):
        HaikuJudgeResult(
            line_count=3,
            line_count_passed=True,
            syllable_breakdown=syllable_breakdown,
            structure_passed=True,
            topic_passed=True,
            score=8,
            reasoning="meow",
            passed=True,
        )


def test_haiku_judge_result_valid_inconsistent_passed_and_line_count_passed() -> None:
    with pytest.raises(ValueError, match=r"passed .* does not match line_count_passed"):
        HaikuJudgeResult(
            line_count=4,
            line_count_passed=False,
            syllable_breakdown=[5, 7, 5, 3],
            structure_passed=False,
            topic_passed=True,
            score=8,
            reasoning="meow",
            passed=True,
        )


def test_haiku_judge_result_valid_inconsistent_passed_and_structure_passed() -> None:
    with pytest.raises(ValueError, match=r"passed .* does not match structure_passed"):
        HaikuJudgeResult(
            line_count=3,
            line_count_passed=True,
            syllable_breakdown=[5, 7, 3],
            structure_passed=False,
            topic_passed=True,
            score=8,
            reasoning="meow",
            passed=True,
        )


def test_haiku_judge_result_valid_inconsistent_passed_and_topic_passed() -> None:
    with pytest.raises(ValueError, match=r"passed .* does not match topic_passed"):
        HaikuJudgeResult(
            line_count=3,
            line_count_passed=True,
            syllable_breakdown=[5, 7, 5],
            structure_passed=True,
            topic_passed=False,
            score=8,
            reasoning="meow",
            passed=True,
        )


def test_haiku_judge_result_valid_inconsistent_passed_and_score() -> None:
    with pytest.raises(ValueError, match=r"passed .* does not match score"):
        HaikuJudgeResult(
            line_count=3,
            line_count_passed=True,
            syllable_breakdown=[5, 7, 5],
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


def test_make_haiku_judge_node_returns_callable() -> None:
    node = make_haiku_judge_node(llm=Mock(spec=BaseChatModel))
    assert callable(node)


def test_make_haiku_judge_node_with_structured_output_called_with_haiku_judge_result(
    mock_llm: BaseChatModel,
) -> None:
    make_haiku_judge_node(llm=mock_llm)
    mock_llm.with_structured_output.assert_called_once_with(HaikuJudgeResult)


def test_make_haiku_judge_node_call(
    mock_llm: BaseChatModel, mock_judge_result: HaikuJudgeResult
) -> None:
    chain_mock = Mock(invoke=Mock(return_value=mock_judge_result))
    with patch("langchain_core.prompts.ChatPromptTemplate.__or__", return_value=chain_mock):
        node = make_haiku_judge_node(mock_llm)
        node({"topic": "fog", "haiku": "grey mist hides the hills"})
    chain_mock.invoke.assert_called_once_with(
        {"topic": "fog", "haiku": "grey mist hides the hills"}
    )
