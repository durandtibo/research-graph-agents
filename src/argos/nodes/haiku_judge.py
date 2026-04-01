r"""Define a haiku judge."""

from __future__ import annotations

__all__ = [
    "HAIKU_JUDGE_SYSTEM_PROMPT",
    "HaikuJudgeResult",
    "HaikuJudgeState",
    "make_haiku_judge_node",
]

from typing import TYPE_CHECKING, Self

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, model_validator

from argos.nodes.haiku_generator import HaikuState

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_core.language_models import BaseChatModel


class HaikuJudgeResult(BaseModel):
    r"""Define the structured result produced by the haiku judge LLM.

    ``structure_passed``, ``topic_passed``, ``score``, and
    ``reasoning`` are populated by the LLM via structured output.
    ``passed`` is always derived automatically from those fields by the
    model validator, so it is guaranteed to be consistent.

    Attributes:
        structure_passed: ``True`` only if the haiku has exactly three
            lines with syllable counts of 5, 7, and 5 respectively.
        topic_passed: ``True`` if the haiku meaningfully addresses the
            target topic, otherwise ``False``.
        score: Quality score from 1 to 10 based on imagery, emotional
            resonance, and word choice. Constrained to the range
            ``[1, 10]``.
        reasoning: A brief explanation justifying the score, topic
            adherence, and structure evaluation.
        passed: Derived automatically: ``True`` only if
            ``structure_passed`` and ``topic_passed`` are both ``True``
            and ``score >= 7``. Any LLM-provided value is overwritten
            by the :meth:`compute_passed` model validator.
    """

    structure_passed: bool = Field(
        description="True ONLY if the haiku has exactly 3 lines with syllable counts of 5, 7, and 5 respectively."
    )
    topic_passed: bool = Field(
        description="True if the haiku meaningfully addresses the target topic, otherwise False."
    )
    score: int = Field(
        ge=1,
        le=10,
        description=(
            "Quality score from 1-10 based on imagery, emotional resonance, and word choice."
        ),
    )
    reasoning: str = Field(
        description="A brief explanation justifying the score, topic adherence, and structure."
    )
    passed: bool = Field(
        default=False,
        description="Derived automatically: True ONLY if structure_passed AND topic_passed AND score >= 7.",
    )

    @model_validator(mode="after")
    def compute_passed(self) -> Self:
        r"""Compute ``passed`` from the contributing fields.

        Overrides any LLM-provided value to guarantee that ``passed``
        is always consistent with ``structure_passed``,
        ``topic_passed``, and ``score``.
        """
        self.passed = self.structure_passed and self.topic_passed and self.score >= 7
        return self


class HaikuJudgeState(HaikuState):
    r"""Define the graph state used during haiku evaluation.

    Extends :class:`HaikuState` with an ``evaluation`` field that
    holds the structured output produced by the judge node.

    Attributes:
        evaluation: The structured evaluation result returned by the
            haiku judge LLM.
    """

    evaluation: HaikuJudgeResult


HAIKU_JUDGE_SYSTEM_PROMPT = """You are a strict haiku evaluator. Evaluate the given haiku against the target topic and return a structured result.

## Structure (`structure_passed`)
True ONLY if the haiku has exactly 3 lines with syllable counts of 5, 7, and 5 respectively.
Count syllables phonetically. Any deviation makes this False.

## Topic (`topic_passed`)
True if the haiku clearly and meaningfully reflects the given topic. Otherwise False.

## Quality Score (`score`)
Rate the haiku from 1 to 10:
- 1-3: Literal, dull, or incoherent
- 4-6: Adequate but weak imagery
- 7-8: Vivid imagery and effective juxtaposition
- 9-10: Exceptional, precise, and evocative

## Reasoning (`reasoning`)
1-3 concise sentences covering syllable accuracy, topic adherence, and quality."""


def make_haiku_judge_node(
    llm: BaseChatModel, system_prompt: str = HAIKU_JUDGE_SYSTEM_PROMPT
) -> Callable[[HaikuState], dict]:
    r"""Create a judge node for haiku evaluation.

    The returned node reads ``topic`` and ``haiku`` from the graph
    state, invokes the LLM with structured output to produce a
    :class:`HaikuJudgeResult`, and returns it under the ``evaluation``
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
    judge_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("user", "Topic: {topic}\n\nHaiku: {haiku}"),
        ]
    )
    llm_judge = judge_prompt | llm.with_structured_output(HaikuJudgeResult)

    def llm_node(state: HaikuState) -> dict:
        response = llm_judge.invoke({"topic": state["topic"], "haiku": state["haiku"]})
        return {"evaluation": response}

    return llm_node
