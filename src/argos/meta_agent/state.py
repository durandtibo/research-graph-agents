r"""Define the meta-agent state."""

from __future__ import annotations

__all__ = ["MetaAgentState"]

from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from argos.meta_agent.agent.config import AgentConfig


class MetaAgentState(TypedDict):
    r"""Define the meta-agent state."""

    config: AgentConfig
    metrics: dict
    diagnostic: dict
    history: list[dict]
    iteration: int
