from __future__ import annotations

from unittest.mock import Mock

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableLambda, RunnableSequence

from argos.models.prompt_generation import (
    PromptGeneratorOutput,
    create_prompt_generator_model,
)
from argos.prompts.prompt_generation import PROMPT_GENERATOR_SYSTEM_PROMPT
from tests.unit.utils.test_prompt import EMPTY_PROMPTS


@pytest.fixture
def mock_output() -> PromptGeneratorOutput:
    return PromptGeneratorOutput(reasoning="blabla", prompt="my new prompt")


@pytest.fixture
def mock_llm(mock_output: PromptGeneratorOutput) -> BaseChatModel:
    return Mock(
        spec=BaseChatModel,
        model="gpt-4o",
        temperature=0,
        with_structured_output=Mock(
            return_value=RunnableLambda(lambda x: mock_output)  # noqa: ARG005
        ),
    )


###################################################
#     Tests for create_prompt_generator_model     #
###################################################


def test_create_prompt_generator_model_returns_runnable_sequence(mock_llm: BaseChatModel) -> None:
    model = create_prompt_generator_model(mock_llm)
    assert isinstance(model, RunnableSequence)


def test_create_prompt_generator_model_uses_default_system_prompt(mock_llm: BaseChatModel) -> None:
    model = create_prompt_generator_model(mock_llm)
    prompt_step = model.first
    system_message = prompt_step.messages[0]
    assert system_message.prompt.template == PROMPT_GENERATOR_SYSTEM_PROMPT


def test_create_prompt_generator_model_uses_custom_system_prompt(mock_llm: BaseChatModel) -> None:
    custom_prompt = "a custom prompt"
    model = create_prompt_generator_model(mock_llm, system_prompt=custom_prompt)
    prompt_step = model.first
    system_message = prompt_step.messages[0]
    assert system_message.prompt.template == custom_prompt


@pytest.mark.parametrize("system_prompt", EMPTY_PROMPTS)
def test_create_prompt_generator_model_uses_empty_system_prompt(
    mock_llm: BaseChatModel, system_prompt: str
) -> None:
    with pytest.raises(ValueError, match="system_prompt must be a non-empty string"):
        create_prompt_generator_model(mock_llm, system_prompt=system_prompt)


def test_create_prompt_generator_model_prompt_contains_text_placeholder(
    mock_llm: BaseChatModel,
) -> None:
    model = create_prompt_generator_model(mock_llm)
    prompt_step = model.first
    assert prompt_step.input_variables == ["history"]


def test_create_prompt_generator_model_invokes_llm_with_history(
    mock_llm: BaseChatModel, mock_output: PromptGeneratorOutput
) -> None:
    model = create_prompt_generator_model(mock_llm)
    out = model.invoke({"history": "['prompt1', 'prompt2']"})
    assert out == mock_output
