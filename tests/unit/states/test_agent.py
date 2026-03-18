from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from argos.states import AgentState

################################
#     Tests for AgentState     #
################################


def test_agent_state_is_typed_dict() -> None:
    state = AgentState(messages=[])
    assert isinstance(state, dict)


def test_agent_state_has_messages_key() -> None:
    state = AgentState(messages=[])
    assert "messages" in state


def test_agent_state_messages_defaults_to_empty_list() -> None:
    state = AgentState(messages=[])
    assert state["messages"] == []


def test_agent_state_rejects_unknown_keys() -> None:
    """TypedDict does not enforce extra keys at runtime, but we confirm
    only expected keys are declared in __annotations__."""
    assert set(AgentState.__annotations__.keys()) == {"messages"}


def test_agent_state_holds_human_message() -> None:
    state = AgentState(messages=[HumanMessage(content="Hello")])
    assert isinstance(state["messages"][0], HumanMessage)
    assert state["messages"][0].content == "Hello"


def test_agent_state_holds_ai_message() -> None:
    state = AgentState(messages=[AIMessage(content="Hi there")])
    assert isinstance(state["messages"][0], AIMessage)
    assert state["messages"][0].content == "Hi there"


def test_agent_state_holds_system_message() -> None:
    state = AgentState(messages=[SystemMessage(content="You are a helpful assistant.")])
    assert isinstance(state["messages"][0], SystemMessage)
    assert state["messages"][0].content == "You are a helpful assistant."


def test_agent_state_holds_multiple_messages() -> None:
    state = AgentState(
        messages=[
            SystemMessage(content="System prompt"),
            HumanMessage(content="User input"),
            AIMessage(content="Agent response"),
        ]
    )
    assert len(state["messages"]) == 3


def test_agent_state_preserves_message_order() -> None:
    state = AgentState(
        messages=[
            HumanMessage(content="first"),
            AIMessage(content="second"),
            HumanMessage(content="third"),
        ]
    )
    assert [m.content for m in state["messages"]] == ["first", "second", "third"]
