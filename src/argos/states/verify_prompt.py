r"""Contain a state to represent a user prompt."""

from __future__ import annotations

__all__ = ["PromptVerification", "UserPromptVerificationState"]

from typing import TypedDict

from argos.states.user_prompt import UserPromptState


class PromptVerification(TypedDict):
    is_valid: bool
    error: str


class UserPromptVerificationState(UserPromptState):
    r"""A simple state that contains only a user prompt."""

    verification: PromptVerification
