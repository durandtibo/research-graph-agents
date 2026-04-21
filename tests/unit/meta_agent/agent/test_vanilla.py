r"""Unit tests for Agent."""

from unittest.mock import Mock

import pytest
from langchain_core.runnables import Runnable

from argos.meta_agent.agent import Agent


@pytest.fixture
def mock_model() -> Runnable:
    return Mock(spec=Runnable, batch=Mock(return_value=[{"answer": "hi"}, {"answer": "cat"}]))


###########################
#     Tests for Agent     #
###########################


def test_agent_stores_model(mock_model: Runnable) -> None:
    agent = Agent(model=mock_model)
    assert agent._model is mock_model


def test_agent_predict_calls_batch(mock_model: Runnable) -> None:
    inputs = [{"query": "hello"}, {"query": "world"}]
    agent = Agent(model=mock_model)
    assert agent.predict(inputs) == [{"answer": "hi"}, {"answer": "cat"}]
    mock_model.batch.assert_called_once_with(inputs)


def test_agent_predict_with_empty_inputs(mock_model: Runnable) -> None:
    mock_model.batch.return_value = []
    agent = Agent(model=mock_model)
    assert agent.predict([]) == []
    mock_model.batch.assert_called_once_with([])


def test_predict_propagates_exception(mock_model: Runnable) -> None:
    mock_model.batch.side_effect = RuntimeError("model failure")
    agent = Agent(model=mock_model)
    with pytest.raises(RuntimeError, match="model failure"):
        agent.predict([{"query": "hello"}])
