from __future__ import annotations

from unittest.mock import Mock

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableLambda

from argos.models.haiku_judge import HaikuJudgeResult, create_haiku_judge_model
from argos.prompts.haiku_judge import HAIKU_JUDGE_SYSTEM_PROMPT


@pytest.fixture
def judge_result() -> HaikuJudgeResult:
    return HaikuJudgeResult(
        structure_passed=True,
        topic_passed=True,
        score=8,
        reasoning="Great imagery and strong structure.",
        passed=True,
    )


@pytest.fixture
def mock_llm(judge_result: HaikuJudgeResult) -> BaseChatModel:
    return Mock(
        spec=BaseChatModel,
        with_structured_output=Mock(
            return_value=RunnableLambda(lambda x: judge_result)  # noqa: ARG005
        ),
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


def test_haiku_judge_result_passed_auto_corrected_structure_failed() -> None:
    result = HaikuJudgeResult(
        structure_passed=False,
        topic_passed=True,
        score=8,
        reasoning="meow",
        passed=True,  # LLM inconsistency: overridden to False
    )
    assert not result.passed


def test_haiku_judge_result_passed_auto_corrected_topic_failed() -> None:
    result = HaikuJudgeResult(
        structure_passed=True,
        topic_passed=False,
        score=8,
        reasoning="meow",
        passed=True,  # LLM inconsistency: overridden to False
    )
    assert not result.passed


def test_haiku_judge_result_passed_auto_corrected_score_too_low() -> None:
    result = HaikuJudgeResult(
        structure_passed=True,
        topic_passed=True,
        score=6,
        reasoning="meow",
        passed=True,  # LLM inconsistency: overridden to False
    )
    assert not result.passed


def test_haiku_judge_result_passed_auto_corrected_all_pass() -> None:
    result = HaikuJudgeResult(
        structure_passed=True,
        topic_passed=True,
        score=7,
        reasoning="meow",
        passed=False,  # LLM inconsistency: overridden to True
    )
    assert result.passed


##############################################
#     Tests for create_haiku_judge_model     #
##############################################


def test_create_haiku_judge_model_returns_runnable(mock_llm: BaseChatModel) -> None:
    model = create_haiku_judge_model(mock_llm)
    assert isinstance(model, Runnable)


def test_create_haiku_judge_model_uses_structured_output(mock_llm: BaseChatModel) -> None:
    create_haiku_judge_model(mock_llm)
    mock_llm.with_structured_output.assert_called_once_with(HaikuJudgeResult)


def test_create_haiku_judge_model_uses_default_system_prompt(mock_llm: BaseChatModel) -> None:
    model = create_haiku_judge_model(mock_llm)
    system_message = model.first.messages[0]
    assert system_message.prompt.template == HAIKU_JUDGE_SYSTEM_PROMPT


def test_create_haiku_judge_model_uses_custom_system_prompt(mock_llm: BaseChatModel) -> None:
    custom_prompt = "You are a custom haiku judge."
    model = create_haiku_judge_model(mock_llm, system_prompt=custom_prompt)
    system_message = model.first.messages[0]
    assert system_message.prompt.template == custom_prompt


def test_create_haiku_judge_model_prompt_contains_expected_input_variables(
    mock_llm: BaseChatModel,
) -> None:
    model = create_haiku_judge_model(mock_llm)
    assert model.first.input_variables == ["haiku", "topic"]


def test_create_haiku_judge_model_invokes_with_topic_and_haiku(
    mock_llm: BaseChatModel, judge_result: HaikuJudgeResult
) -> None:
    model = create_haiku_judge_model(mock_llm)
    result = model.invoke(
        {
            "topic": "autumn",
            "haiku": "Leaves fall silently\nCrisp air and golden colors\nWinter is coming",
        }
    )
    assert result == judge_result
