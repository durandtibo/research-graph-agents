r"""Contain utility functions for prompts."""

from __future__ import annotations

__all__ = ["check_non_empty_prompt"]


def check_non_empty_prompt(prompt: str, name: str = "prompt") -> None:
    r"""Check if the input prompt is not empty.

    Args:
        prompt: The input prompt to check.
        name: The name of the input prompt to show in the error message if empty.

    Raises:
        ValueError: If ``prompt`` is an empty or whitespace-only string.
    """
    if not prompt.strip():
        msg = f"{name} must be a non-empty string"
        raise ValueError(msg)
