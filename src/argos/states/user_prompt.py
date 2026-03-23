r"""Contain a state to represent a user prompt."""

from __future__ import annotations

__all__ = ["UserPromptState"]

from typing_extensions import TypedDict


class UserPromptState(TypedDict):
    r"""A simple state that contains only a user prompt."""

    user_prompt: str
