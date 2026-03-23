r"""Define a haiku generator."""

from __future__ import annotations

__all__ = ["HaikuState"]

from typing import TypedDict


class HaikuState(TypedDict):
    r"""Define the state to generate a haiku."""

    topic: str
    haiku: str
