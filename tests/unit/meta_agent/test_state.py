r"""Unit tests for MetaAgentState."""

import pytest

from argos.meta_agent import MetaAgentState
from argos.meta_agent.agent import AgentConfig

####################################
#     Tests for MetaAgentState     #
####################################


@pytest.fixture
def state() -> MetaAgentState:
    return MetaAgentState(
        config=AgentConfig(components={"llm": "gpt-4"}),
        metrics={"accuracy": 0.95},
        diagnostic={"status": "ok"},
        history=[{"step": 1, "action": "init"}],
        iteration=0,
    )


def test_state_contains_expected_keys(state: MetaAgentState) -> None:
    assert state["config"] == AgentConfig(components={"llm": "gpt-4"})
    assert state["metrics"] == {"accuracy": 0.95}
    assert state["diagnostic"] == {"status": "ok"}
    assert state["history"] == [{"step": 1, "action": "init"}]
    assert state["iteration"] == 0


def test_state_with_empty_metrics() -> None:
    state = MetaAgentState(
        config=AgentConfig(components={"llm": "gpt-4"}),
        metrics={},
        diagnostic={},
        history=[],
        iteration=0,
    )
    assert state["metrics"] == {}


def test_state_with_empty_diagnostic() -> None:
    state = MetaAgentState(
        config=AgentConfig(components={"llm": "gpt-4"}),
        metrics={},
        diagnostic={},
        history=[],
        iteration=0,
    )
    assert state["diagnostic"] == {}


def test_state_with_empty_history() -> None:
    state = MetaAgentState(
        config=AgentConfig(components={"llm": "gpt-4"}),
        metrics={},
        diagnostic={},
        history=[],
        iteration=0,
    )
    assert state["history"] == []


def test_state_iteration_can_be_incremented(state: MetaAgentState) -> None:
    state["iteration"] += 1
    assert state["iteration"] == 1


def test_state_history_can_be_appended(state: MetaAgentState) -> None:
    assert len(state["history"]) == 1
    state["history"].append({"step": 2, "action": "run"})
    assert len(state["history"]) == 2
    assert state["history"][-1] == {"step": 2, "action": "run"}


def test_state_metrics_can_be_updated(state: MetaAgentState) -> None:
    state["metrics"]["loss"] = 0.05
    assert state["metrics"]["loss"] == 0.05


def test_state_config_can_be_replaced(state: MetaAgentState) -> None:
    state["config"] = AgentConfig(components={"llm": "claude"})
    assert state["config"] == AgentConfig(components={"llm": "claude"})
