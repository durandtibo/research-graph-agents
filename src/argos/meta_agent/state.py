r"""Define the meta-agent state."""

from __future__ import annotations

__all__ = ["MetaAgentState"]

from typing import TYPE_CHECKING, Any, TypedDict

if TYPE_CHECKING:
    from argos.meta_agent.agents.config import AgentConfig


class MetaAgentState(TypedDict):
    r"""Define the meta-agent state."""

    config: AgentConfig
    metrics: dict[Any, Any]
    diagnostic: dict[Any, Any]
    history: list[dict[Any, Any]]
    iteration: int
