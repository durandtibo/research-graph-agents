r"""Define the meta-agent graph state shared across optimization iterations."""

from __future__ import annotations

__all__ = ["MetaAgentState"]

from typing import TYPE_CHECKING, Any, TypedDict

if TYPE_CHECKING:
    from argos.meta_agent.agents.config import AgentConfig


class MetaAgentState(TypedDict):
    r"""Define the meta-agent state.

    Attributes:
        config: The current agent configuration being optimized.
        metrics: A dictionary of evaluation metrics produced at the
            current iteration.
        diagnostic: A dictionary of diagnostic information (e.g.
            error analysis) produced at the current iteration.
        history: A list of dicts recording the configuration and
            metrics from all previous iterations.
        iteration: The current iteration index (zero-based).

    Example:
        ```pycon
        >>> from argos.meta_agent.agents import AgentConfig
        >>> from argos.meta_agent.state import MetaAgentState
        >>> state: MetaAgentState = {
        ...     "config": AgentConfig(components={}),
        ...     "metrics": {},
        ...     "diagnostic": {},
        ...     "history": [],
        ...     "iteration": 0,
        ... }
        >>> state["iteration"]
        0

        ```
    """

    config: AgentConfig
    metrics: dict[Any, Any]
    diagnostic: dict[Any, Any]
    history: list[dict[Any, Any]]
    iteration: int
