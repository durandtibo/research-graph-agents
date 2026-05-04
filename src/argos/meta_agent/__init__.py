r"""Contain the implementation of the meta-agent.

This sub-package provides the core abstractions and concrete
implementations for building meta-agents that iteratively optimize an
agent's configuration using feedback from evaluation.
"""

from __future__ import annotations

__all__ = ["MetaAgentState"]

from argos.meta_agent.state import MetaAgentState
