from __future__ import annotations

from unittest.mock import Mock

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableLambda

from argos.models.haiku_judge import create_haiku_judge_model
from argos.nodes.haiku_judge import HaikuJudgeResult
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
