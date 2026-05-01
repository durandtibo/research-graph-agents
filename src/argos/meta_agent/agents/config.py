r"""Define a generic agent configuration."""

from __future__ import annotations

__all__ = ["AgentConfig"]

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentConfig:
    r"""Define a generic agent configuration."""

    components: dict[str, Any]  # modular building blocks
    metadata: dict[str, Any] = field(default_factory=dict)  # versioning, lineage, etc.
