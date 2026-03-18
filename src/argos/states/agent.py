r"""Contain a state to represent the messages of an agent."""

from __future__ import annotations

__all__ = ["AgentState"]

from typing import TYPE_CHECKING, Annotated

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    r"""State to represent the messages of an agent."""

    # 'add_messages' ensures new prompts/responses are appended to the list
    messages: Annotated[list[BaseMessage], add_messages]
