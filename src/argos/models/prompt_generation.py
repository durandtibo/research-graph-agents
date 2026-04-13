r"""Contain models to analyze text."""

from __future__ import annotations

__all__ = ["PromptGeneratorInput", "create_prompt_generator_model"]

import logging
from typing import TYPE_CHECKING, TypedDict

from langchain_core.prompts import ChatPromptTemplate

from argos.prompts.prompt_generation import PROMPT_GENERATOR_SYSTEM_PROMPT_0
from argos.utils.prompt import check_non_empty_prompt

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.runnables import RunnableSequence

logger: logging.Logger = logging.getLogger(__name__)


# Generate a new prompt based on the history of previous prompt.
# Follow best practices to write a prompt.
# Maximize the performances (accuracy and F1 score).


class PromptGeneratorInput(TypedDict):
    r"""Define the input to analyze text.

    Attributes:
        history: The history of previous prompt.
    """

    history: str


def create_prompt_generator_model(
    llm: BaseChatModel, system_prompt: str = PROMPT_GENERATOR_SYSTEM_PROMPT_0
) -> RunnableSequence[PromptGeneratorInput, AIMessage]:
    r"""Create a simple prompt generator model.

    Args:
        llm: The LLM used to process the text.
        system_prompt: The system prompt that instructs the LLM on how
            to generate the next prompt.

    Returns:
        An :class:`~langchain_core.runnables.RunnableSequence` chain
            that accepts a dict with a ``history`` key and returns an
            :class:`~langchain_core.messages.AIMessage` containing the
            analysis.
    """
    check_non_empty_prompt(prompt=system_prompt, name="system_prompt")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{history}"),
        ]
    )
    return prompt | llm
