from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from argos.nodes.haiku_generator import HaikuState, make_haiku_generator_node


@pytest.fixture
def mock_llm() -> BaseChatModel:
    return Mock(spec=BaseChatModel)


################################
#     Tests for HaikuState     #
################################


def test_haiku_judge_state_keys() -> None:
    assert set(HaikuState.__annotations__) == {"topic", "haiku"}


###############################################
#     Tests for make_haiku_generator_node     #
###############################################


def test_make_haiku_generator_node_returns_callable(mock_llm: BaseChatModel) -> None:
    node = make_haiku_generator_node(llm=mock_llm)
    assert callable(node)


def test_make_haiku_generator_node_call(mock_llm: BaseChatModel) -> None:
    chain_mock = Mock(invoke=Mock(return_value=AIMessage("meow")))
    with patch("langchain_core.prompts.ChatPromptTemplate.__or__", return_value=chain_mock):
        node = make_haiku_generator_node(mock_llm)
        out = node({"topic": "cat"})
        assert out == {"haiku": "meow"}
    chain_mock.invoke.assert_called_once_with({"topic": "cat"})
