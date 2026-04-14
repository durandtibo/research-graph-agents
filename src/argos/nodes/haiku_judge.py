r"""Define a haiku judge."""

from __future__ import annotations

__all__ = ["HaikuJudgeState", "make_haiku_judge_node"]

from typing import TYPE_CHECKING

from argos.models.haiku_judge import HaikuJudgeOutput, create_haiku_judge_model
from argos.nodes.haiku_generator import HaikuState
from argos.prompts.haiku_judge import HAIKU_JUDGE_SYSTEM_PROMPT

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_core.language_models import BaseChatModel


class HaikuJudgeState(HaikuState):
    r"""Define the graph state used during haiku evaluation.

    Extends :class:`HaikuState` with an ``evaluation`` field that
    holds the structured output produced by the judge node.

    Attributes:
        evaluation: The structured evaluation result returned by the
            haiku judge LLM.
    """

    evaluation: HaikuJudgeOutput


def make_haiku_judge_node(
    llm: BaseChatModel, system_prompt: str = HAIKU_JUDGE_SYSTEM_PROMPT
) -> Callable[[HaikuState], dict]:
    r"""Create a judge node for haiku evaluation.

    The returned node reads ``topic`` and ``haiku`` from the graph
    state, invokes the LLM with structured output to produce a
    :class:`HaikuJudgeOutput`, and returns it under the ``evaluation``
    key.

    Args:
        llm: The LLM used to build the judge. The LLM must support
            structured output via
            :meth:`~langchain_core.language_models.BaseChatModel.with_structured_output`.
        system_prompt: The system prompt that instructs the LLM on how
            to evaluate a haiku. Defaults to
            ``HAIKU_JUDGE_SYSTEM_PROMPT``.

    Returns:
        A callable node that accepts a :class:`HaikuState` and returns
            a dict with the evaluation result under the ``evaluation``
            key.

    Example:
        ```pycon
        >>> from langchain_ollama import ChatOllama
        >>> from argos.nodes.haiku_judge import make_haiku_judge_node
        >>> llm = ChatOllama(model="gemma3:1b")
        >>> node = make_haiku_judge_node(llm=llm)

        ```
    """
    model = create_haiku_judge_model(llm=llm, system_prompt=system_prompt)

    def llm_node(state: HaikuState) -> dict:
        response = model.invoke({"topic": state["topic"], "haiku": state["haiku"]})
        return {"evaluation": response}

    return llm_node
