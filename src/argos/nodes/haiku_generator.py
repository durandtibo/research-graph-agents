r"""Define a haiku generator."""

from __future__ import annotations

__all__ = ["HAIKU_GENERATOR_SYSTEM_PROMPT", "HaikuState", "make_haiku_generator_node"]

from typing import TYPE_CHECKING, TypedDict

from langchain_core.prompts import ChatPromptTemplate

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_core.language_models import BaseChatModel


class HaikuState(TypedDict):
    r"""Define the state to generate a haiku."""

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

    Args:
        llm: The LLM used to generate the haiku.
        system_prompt: The haiku generator system prompt.

    Returns:
        The haiku generator node.
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
