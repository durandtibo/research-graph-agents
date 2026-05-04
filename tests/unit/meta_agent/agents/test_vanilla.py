from __future__ import annotations

from unittest.mock import Mock

import pytest
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda

from argos.meta_agent.agents import Agent, BaseAgent
from tests.unit.helpers.runnable import (
    ConfigCaptureRunnable,
    DoubleRunnable,
    IdentityRunnable,
    RaisingErrorRunnable,
)


@pytest.fixture
def mock_model() -> Runnable:
    return Mock(spec=Runnable, batch=Mock(return_value=[{"answer": "hi"}, {"answer": "cat"}]))


###########################
#     Tests for Agent     #
###########################


def test_agent_stores_model(mock_model: Runnable) -> None:
    agent = Agent(model=mock_model)
    assert agent._model is mock_model


def test_agent_is_instance_of_base_agent() -> None:
    assert isinstance(Agent(DoubleRunnable()), BaseAgent)


def test_agent_predict_returns_model_output() -> None:
    agent = Agent(DoubleRunnable())
    assert agent.predict([1, 2, 3]) == [2, 4, 6]


def test_agent_predict_empty_input() -> None:
    agent = Agent(DoubleRunnable())
    assert agent.predict([]) == []


def test_agent_predict_passes_config() -> None:
    runnable = ConfigCaptureRunnable()
    agent = Agent(runnable)
    config = RunnableConfig(tags=["test"])
    assert agent.predict(["x"], config=config) == ["x"]
    assert runnable.last_config == config


def test_agent_predict_calls_batch(mock_model: Runnable) -> None:
    inputs = [{"query": "hello"}, {"query": "world"}]
    agent = Agent(model=mock_model)
    assert agent.predict(inputs) == [{"answer": "hi"}, {"answer": "cat"}]
    mock_model.batch.assert_called_once_with(inputs=inputs, config=None)


def test_agent_predict_calls_batch_with_config(mock_model: Runnable) -> None:
    inputs = [{"query": "hello"}, {"query": "world"}]
    config = RunnableConfig(max_concurrency=2)
    agent = Agent(model=mock_model)
    assert agent.predict(inputs, config=config) == [{"answer": "hi"}, {"answer": "cat"}]
    mock_model.batch.assert_called_once_with(inputs=inputs, config=config)


def test_agent_predict_propagates_exception() -> None:
    agent = Agent(model=RaisingErrorRunnable())
    with pytest.raises(RuntimeError, match="model failure"):
        agent.predict([{"query": "hello"}])


def test_agent_predict_single_input() -> None:
    agent = Agent(IdentityRunnable())
    assert agent.predict(["only one"]) == ["only one"]


@pytest.mark.parametrize(
    ("inputs", "expected"),
    [
        pytest.param(["hello"], ["HELLO"], id="single"),
        pytest.param(["hello", "world"], ["HELLO", "WORLD"], id="two"),
        pytest.param(["a", "b", "c"], ["A", "B", "C"], id="three"),
    ],
)
def test_agent_predict_various_inputs(inputs: list[str], expected: list[str]) -> None:
    agent = Agent(RunnableLambda(str.upper))
    assert agent.predict(inputs) == expected
