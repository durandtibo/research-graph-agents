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

    All fields are populated by the LLM via structured output and are
    then validated by the model validators to ensure internal
    consistency.
    """

    line_count: int = Field(description="Number of lines in the input haiku.", ge=0)
    line_count_passed: bool = Field(description="True ONLY if line_count == 3.")
    syllable_breakdown: list[int] = Field(
        description="The exact syllable count for each line (e.g., [5, 7, 5])."
    )
    structure_passed: bool = Field(
        description="True ONLY if line_count_passed AND syllable_breakdown == [5, 7, 5]."
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
        description=(
            "True ONLY if line_count_passed AND structure_passed AND topic_passed AND score >= 7."
        )
    )

    @model_validator(mode="after")
    def check_line_count_passed(self) -> Self:
        r"""Validate that ``line_count_passed`` is consistent with
        ``line_count``.

        Raises:
            ValueError: If ``line_count_passed`` is ``True`` but
                ``line_count`` is not ``3``.
        """
        if self.line_count_passed and self.line_count != 3:
            msg = f"line_count_passed ({self.line_count_passed}) does not match line_count ({self.line_count})"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def check_syllable_breakdown(self) -> Self:
        r"""Validate that ``syllable_breakdown`` length matches
        ``line_count``.

        Raises:
            ValueError: If ``len(syllable_breakdown)`` does not equal
                ``line_count``.
        """
        if self.line_count != len(self.syllable_breakdown):
            msg = f"line_count ({self.line_count}) does not match syllable_breakdown ({self.syllable_breakdown})"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def check_structure_passed(self) -> Self:
        r"""Validate that ``structure_passed`` is consistent with
        ``syllable_breakdown``.

        Raises:
            ValueError: If ``structure_passed`` is ``True`` but
                ``syllable_breakdown`` is not ``[5, 7, 5]``.
        """
        if self.structure_passed and self.syllable_breakdown != [5, 7, 5]:
            msg = f"structure_passed ({self.structure_passed}) does not match syllable_breakdown ({self.syllable_breakdown})"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def check_passed(self) -> Self:
        r"""Validate that ``passed`` is consistent with all contributing
        fields.

        Raises:
            ValueError: If ``passed`` is ``True`` but any of the
                following conditions are not met: ``line_count_passed``
                is ``True``, ``structure_passed`` is ``True``,
                ``topic_passed`` is ``True``, and ``score >= 7``.
        """
        if self.passed and not self.line_count_passed:
            msg = (
                f"passed ({self.passed}) does not match line_count_passed "
                f"({self.line_count_passed})"
            )
            raise ValueError(msg)
        if self.passed and not self.structure_passed:
            msg = (
                f"passed ({self.passed}) does not match structure_passed ({self.structure_passed})"
            )
            raise ValueError(msg)
        if self.passed and not self.topic_passed:
            msg = f"passed ({self.passed}) does not match topic_passed ({self.topic_passed})"
            raise ValueError(msg)
        if self.passed and self.score < 7:
            msg = f"passed ({self.passed}) does not match score ({self.score})"
            raise ValueError(msg)
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


HAIKU_JUDGE_SYSTEM_PROMPT = """# Role
You are a strict haiku evaluator.

# Task
Evaluate a generated haiku against a target topic and return a structured result that strictly matches the required schema.

# Inputs
- Target Topic
- Generated Haiku (expected: exactly 3 lines)

# Evaluation Rules (Deterministic)

## 0. Line Count Check
- Split the input by newline.
- line_count = number of lines.
- line_count_passed = True ONLY if line_count == 3.
- If False, still attempt evaluation but mark structure_passed = False.

## 1. Syllable Counting
- Count syllables per line phonetically.
- If line_count != 3, return best-effort counts for available lines.
- Return as: [L1, L2, L3] (use 0 for missing lines if needed).

## 2. Structure
- structure_passed = True ONLY if:
  - line_count_passed == True AND
  - syllable_breakdown == [5, 7, 5]

## 3. Topic Fidelity
- topic_passed = True if the haiku clearly and meaningfully reflects the topic.
- Otherwise False.

## 4. Quality Score (1-10)
- 1-3: Literal, dull, or incoherent
- 4-6: Adequate but weak imagery
- 7-8: Vivid imagery and effective juxtaposition
- 9-10: Exceptional, precise, and evocative

## 5. Final Pass
- passed = True ONLY if:
  - line_count_passed == True
  - structure_passed == True
  - topic_passed == True
  - score >= 7

## 6. Reasoning
- 1-3 concise sentences covering:
  - line count (if incorrect)
  - syllable issues (if any)
  - topic adherence
  - quality justification"""


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
