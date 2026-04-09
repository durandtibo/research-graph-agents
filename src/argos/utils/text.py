r"""Contain utility functions to process text."""

from __future__ import annotations

__all__ = ["count_lines", "count_syllables", "remove_empty_lines"]

import re

import syllables


def count_lines(text: str) -> int:
    r"""Count the number of lines in a string.

    Args:
        text: The input string to count lines in.

    Returns:
        The number of lines in the input string.

    Example:
        ```pycon
        >>> from argos.utils.text import count_lines
        >>> count_lines("Hello\nWorld")
        2
        >>> count_lines("Hello\nWorld\nFoo")
        3

        ```
    """
    if not text:
        return 0

    return len(text.splitlines())


def count_syllables(text: str) -> int:
    r"""Count the number of syllables in a string using the syllables
    library.

    Args:
        text: The input string, which can be a word or a sentence.

    Returns:
        The total number of syllables found in the input string.

    Example:
        ```pycon
        >>> from argos.utils.text import count_syllables
        >>> count_syllables("Hello")
        2
        >>> count_syllables("The quick brown fox")
        4

        ```
    """
    if not text:
        return 0
    words = re.findall(r"[a-zA-Z]+", text)
    return sum(syllables.estimate(word) for word in words)


def remove_empty_lines(text: str) -> str:
    r"""Remove empty lines from a string.

    Args:
        text: The input string from which empty lines will be removed.

    Returns:
        A new string with all empty or whitespace-only lines removed.

    Example:
        ```pycon
        >>> from argos.utils.text import remove_empty_lines
        >>> remove_empty_lines("Hello\n\nWorld\n\n\nFoo")
        'Hello\nWorld\nFoo'
        >>> remove_empty_lines("\n\nOnly empty lines\n\n")
        'Only empty lines'

        ```
    """
    return "\n".join([line for line in text.splitlines() if line.strip()])
