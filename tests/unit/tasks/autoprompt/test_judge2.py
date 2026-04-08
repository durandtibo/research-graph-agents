from __future__ import annotations

import logging
from unittest.mock import Mock, patch

import pytest
from langchain_core.language_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph

from argos.tasks.autoprompt.config import LlmConfig
from argos.tasks.autoprompt.judge2 import create_judge_graph

MODULE = "argos.tasks.autoprompt.judge2"


@pytest.fixture
def config() -> LlmConfig:
    return LlmConfig(model="gpt-4o", system_prompt="You are a haiku judge.")


@pytest.fixture
def mock_llm() -> BaseChatModel:
    return Mock(spec=BaseChatModel, model="gpt-4o", temperature=0)


########################################
#     Tests for create_judge_graph     #
########################################


def test_create_judge_graph_returns_compiled_state_graph(
    config: LlmConfig, mock_llm: BaseChatModel
) -> None:
    """create_judge_graph should return a CompiledStateGraph
    instance."""
    with patch(f"{MODULE}.init_chat_model", return_value=mock_llm):
        graph = create_judge_graph(config)
        assert isinstance(graph, CompiledStateGraph)
        assert "judge" in graph.nodes


def test_create_judge_graph_init_chat_model_called_with_correct_model(
    config: LlmConfig, mock_llm: BaseChatModel
) -> None:
    """init_chat_model must be called with the model name passed to
    create_judge_graph."""
    with patch(f"{MODULE}.init_chat_model", return_value=mock_llm) as mock_init:
        create_judge_graph(config)
        mock_init.assert_called_once_with(model="gpt-4o", temperature=0, max_retries=9999)


def test_create_judge_graph_make_haiku_judge_node_receives_llm_and_system_prompt(
    config: LlmConfig, mock_llm: BaseChatModel
) -> None:
    """make_haiku_judge_node must be called with the LLM and the judge
    system prompt."""
    with (
        patch(f"{MODULE}.init_chat_model", return_value=mock_llm),
        patch(f"{MODULE}.make_haiku_judge_node") as mock_node_factory,
    ):
        create_judge_graph(config)
        mock_node_factory.assert_called_once_with(mock_llm, system_prompt=config.system_prompt)


def test_create_judge_graph_multiple_calls_return_distinct_graphs(
    config: LlmConfig, mock_llm: BaseChatModel
) -> None:
    """Each invocation must return a new, independent
    CompiledStateGraph."""
    with patch(f"{MODULE}.init_chat_model", return_value=mock_llm):
        graph_a = create_judge_graph(config)
        graph_b = create_judge_graph(config)
        assert graph_a is not graph_b


def test_create_judge_graph_temperature_warning(caplog: pytest.LogCaptureFixture) -> None:
    with (
        caplog.at_level(logging.WARNING),
        patch(
            f"{MODULE}.init_chat_model",
            return_value=Mock(spec=BaseChatModel, model="gpt-4o", temperature=0.2),
        ),
    ):
        create_judge_graph(
            LlmConfig(model="gpt-4o", system_prompt="You are a haiku judge.", temperature=0.2)
        )
        assert (
            caplog.messages[-1]
            == "It is recommended to set temperature to 0 to have a deterministic judge"
        )
