from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
from langchain_core.language_models import BaseChatModel

from argos.autoprompt.haiku.chat_model import create_chat_model
from argos.autoprompt.haiku.config import LlmConfig

MODULE = "argos.autoprompt.haiku.chat_model"


@pytest.fixture
def config() -> LlmConfig:
    return LlmConfig(model="gpt-4o", system_prompt="You are a haiku judge.")


@pytest.fixture
def mock_model() -> BaseChatModel:
    return Mock(spec=BaseChatModel, model="gpt-4o-mini", temperature=0.7)


#######################################
#     Tests for create_chat_model     #
#######################################


def test_create_chat_model(config: LlmConfig, mock_model: BaseChatModel) -> None:
    mock_init = Mock(return_value=mock_model)
    with patch(f"{MODULE}.init_chat_model", mock_init):
        model = create_chat_model(config)
        assert model is mock_model
        mock_init.assert_called_once_with(model="gpt-4o", temperature=0.0, max_retries=9999)


def test_create_chat_model_custom_temperature(config: LlmConfig, mock_model: BaseChatModel) -> None:
    config.temperature = 0.7
    mock_init = Mock(return_value=mock_model)
    with patch(f"{MODULE}.init_chat_model", mock_init):
        model = create_chat_model(config)
        assert model is mock_model
        mock_init.assert_called_once_with(model="gpt-4o", temperature=0.7, max_retries=9999)


def test_create_chat_model_custom_max_retries(config: LlmConfig, mock_model: BaseChatModel) -> None:
    config.max_retries = 0
    mock_init = Mock(return_value=mock_model)
    with patch(f"{MODULE}.init_chat_model", mock_init):
        model = create_chat_model(config)
        assert model is mock_model
        mock_init.assert_called_once_with(model="gpt-4o", temperature=0.0, max_retries=0)


def test_create_chat_model_custom_init_kwargs(config: LlmConfig, mock_model: BaseChatModel) -> None:
    config.init_kwargs = {"max_tokens": 1000}
    mock_init = Mock(return_value=mock_model)
    with patch(f"{MODULE}.init_chat_model", mock_init):
        model = create_chat_model(config)
        assert model is mock_model
        mock_init.assert_called_once_with(
            model="gpt-4o", temperature=0.0, max_retries=9999, max_tokens=1000
        )
