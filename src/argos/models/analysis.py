r"""Contain models to analyze text."""

from __future__ import annotations

__all__ = ["AnalyzerInput", "create_analyzer_model"]

import logging
from typing import TYPE_CHECKING, Any, TypedDict

from langchain_core.prompts import ChatPromptTemplate

from argos.utils.prompt import check_non_empty_prompt

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.runnables import RunnableSequence

logger: logging.Logger = logging.getLogger(__name__)

ANALYZER_SYSTEM_PROMPT = "Analyze the content of the text and provide a short analysis of it."


class AnalyzerInput(TypedDict):
    r"""Define the input to analyze text."""

    text: str


def create_analyzer_model(
    llm: BaseChatModel, system_prompt: str = ANALYZER_SYSTEM_PROMPT
) -> RunnableSequence[dict[Any, Any], AIMessage]:
    r"""Create a simple analyzer model.

    Args:
        llm: The LLM used to process the text.
        system_prompt: The system prompt used to define the analysis of the text.

    Returns:
        An :class:`~langchain_core.runnables.RunnableSequence` chain
            that accepts a dict with a ``text`` key and returns an
            :class:`~langchain_core.messages.AIMessage` containing the
            analysis.
    """
    check_non_empty_prompt(prompt=system_prompt, name="system_prompt")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{text}"),
        ]
    )
    return prompt | llm
