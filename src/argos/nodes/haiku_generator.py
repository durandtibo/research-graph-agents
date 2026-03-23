r"""Define a haiku generator."""

from __future__ import annotations

__all__ = ["HAIKU_GENERATOR_SYSTEM_PROMPT", "HaikuState", "make_haiku_generator_node"]

from typing import TYPE_CHECKING, TypedDict

from langchain_core.prompts import ChatPromptTemplate

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_core.language_models import BaseChatModel


class HaikuState(TypedDict):
    r"""Define the state used throughout haiku generation.

    Attributes:
        topic: The subject or theme the haiku should be written about.
        haiku: The generated haiku text, consisting of three lines
            separated by newlines.
    """

    topic: str
    haiku: str


HAIKU_GENERATOR_SYSTEM_PROMPT = """# Role
You are a master poet specializing in traditional haiku.

# Task
Write exactly one haiku based on the user's topic.

# Constraints (Strict)

## 1. Structure (5-7-5 syllables)
- Line 1: exactly 5 syllables
- Line 2: exactly 7 syllables
- Line 3: exactly 5 syllables
- Count syllables phonetically before finalizing.

## 2. Topic Fidelity
- The haiku must clearly reflect the user's topic.
- Avoid drift, vagueness, or unrelated imagery.

## 3. Poetic Quality
- Use vivid, sensory imagery (show, don't tell).
- Evoke a specific moment or scene.
- Prefer subtle juxtaposition or contrast.
- Avoid literal, generic, or explanatory phrasing.

# Output Rules
- Output only the three haiku lines.
- No titles, explanations, quotes, or extra text."""


def make_haiku_generator_node(
    llm: BaseChatModel, system_prompt: str = HAIKU_GENERATOR_SYSTEM_PROMPT
) -> Callable[[HaikuState], dict]:
    r"""Create a haiku generator node.

    The returned node reads ``topic`` from the graph state, calls the
    LLM to compose a haiku, and returns the result under the ``haiku``
    key.

    Args:
        llm: The LLM used to generate the haiku.
        system_prompt: The system prompt that instructs the LLM on how
            to write a haiku. Defaults to
            ``HAIKU_GENERATOR_SYSTEM_PROMPT``.

    Returns:
        A callable node that accepts a :class:`HaikuState` and returns
            a dict with the generated haiku under the ``haiku`` key.

    Example:
        ```pycon
        >>> from unittest.mock import Mock
        >>> from langchain_core.language_models import BaseChatModel
        >>> from argos.nodes.haiku_generator import make_haiku_generator_node
        >>> llm = Mock(spec=BaseChatModel)
        >>> node = make_haiku_generator_node(llm=llm)
        >>> callable(node)
        True

        ```
    """
    generator_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("user", "Topic: {topic}"),
        ]
    )
    llm_generator = generator_prompt | llm

    def llm_node(state: HaikuState) -> dict:
        response = llm_generator.invoke({"topic": state["topic"]})
        return {"haiku": response.content}

    return llm_node
