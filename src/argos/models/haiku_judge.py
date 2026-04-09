r"""Contain model factory functions for the haiku judge."""

from __future__ import annotations

__all__ = ["create_haiku_judge_model"]

import logging
from typing import TYPE_CHECKING

from langchain_core.prompts import ChatPromptTemplate

from argos.nodes.haiku_judge import HaikuJudgeResult
from argos.prompts.haiku_judge import HAIKU_JUDGE_SYSTEM_PROMPT

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.runnables import Runnable

logger: logging.Logger = logging.getLogger(__name__)


def create_haiku_judge_model(
    llm: BaseChatModel, system_prompt: str = HAIKU_JUDGE_SYSTEM_PROMPT
) -> Runnable[dict[str, str], HaikuJudgeResult]:
    r"""Create a simple haiku judge model.

    Args:
        llm: The LLM used to build the judge. The LLM must support
            structured output via
            :meth:`~langchain_core.language_models.BaseChatModel.with_structured_output`.
        system_prompt: The system prompt that instructs the LLM on how
            to evaluate a haiku. Defaults to
            ``HAIKU_JUDGE_SYSTEM_PROMPT``.

    Returns:
        A simple haiku judge model.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("user", "Topic: {topic}\n\nHaiku: {haiku}"),
        ]
    )
    return prompt | llm.with_structured_output(HaikuJudgeResult)
