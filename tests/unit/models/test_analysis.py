from __future__ import annotations

from unittest.mock import Mock

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableSequence

from argos.models.analysis import ANALYZER_SYSTEM_PROMPT, create_analyzer_model


@pytest.fixture
def mock_llm() -> BaseChatModel:
    return Mock(spec=BaseChatModel, model="gpt-4o", temperature=0)


###########################################
#     Tests for create_analyzer_model     #
###########################################


def test_create_analyzer_model_returns_runnable_sequence(mock_llm: BaseChatModel) -> None:
    model = create_analyzer_model(mock_llm)
    assert isinstance(model, RunnableSequence)


def test_create_analyzer_model_uses_default_system_prompt(mock_llm: BaseChatModel) -> None:
    model = create_analyzer_model(mock_llm)
    prompt_step = model.first
    system_message = prompt_step.messages[0]
    assert system_message.prompt.template == ANALYZER_SYSTEM_PROMPT


def test_create_analyzer_model_uses_custom_system_prompt(mock_llm: BaseChatModel) -> None:
    custom_prompt = "You are a custom analyzer."
    model = create_analyzer_model(mock_llm, system_prompt=custom_prompt)
    prompt_step = model.first
    system_message = prompt_step.messages[0]
    assert system_message.prompt.template == custom_prompt


@pytest.mark.parametrize("system_prompt", ["", " ", "\n\n"])
def test_create_analyzer_model_uses_empty_system_prompt(
    mock_llm: BaseChatModel, system_prompt: str
) -> None:
    with pytest.raises(ValueError, match="system_prompt must be a non-empty string"):
        create_analyzer_model(mock_llm, system_prompt=system_prompt)


def test_create_analyzer_model_prompt_contains_text_placeholder(mock_llm: BaseChatModel) -> None:
    model = create_analyzer_model(mock_llm)
    prompt_step = model.first
    assert prompt_step.input_variables == ["text"]


def test_create_analyzer_model_invokes_llm_with_text(mock_llm: BaseChatModel) -> None:
    mock_llm.invoke = Mock(return_value=AIMessage(content="Analysis result"))
    model = create_analyzer_model(mock_llm)
    result = model.invoke({"text": "Some text to analyze"})
    assert result.content == "Analysis result"
