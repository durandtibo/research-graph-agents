r"""Provide the building blocks for meta-agent optimization loops.

The package models the objects exchanged across an optimization cycle:
agent configurations, benchmark examples, predictions, evaluation
results, and analyses. A typical workflow creates an agent from a
configuration, runs a predictor on a benchmark, evaluates the
predictions, and then analyzes the resulting records to decide how to
update the next configuration.
"""

from __future__ import annotations

__all__ = ["MetaAgentState"]

from argos.meta_agent.state import MetaAgentState
