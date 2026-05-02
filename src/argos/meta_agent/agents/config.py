r"""Define a generic agent configuration."""

from __future__ import annotations

__all__ = ["AgentConfig"]

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentConfig:
    r"""Define a generic agent configuration.

    Attributes:
        components: A dictionary of modular building blocks that define
            the agent (e.g. model, prompt, retriever).
        metadata: A dictionary for auxiliary information such as
            versioning and lineage. Defaults to an empty dict.
    """

    components: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
