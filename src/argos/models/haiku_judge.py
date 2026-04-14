r"""Contain model factory functions for the haiku judge."""

from __future__ import annotations

__all__ = [
    "HaikuJudgeInput",
    "HaikuJudgeInputValidator",
    "HaikuJudgeOutput",
    "create_haiku_judge_model",
    "validate_haiku_judge_input",
]

from typing import TYPE_CHECKING, Any, Self, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableSequence
from pydantic import BaseModel, Field, field_validator, model_validator

from argos.prompts.haiku_judge import HAIKU_JUDGE_SYSTEM_PROMPT
from argos.utils.prompt import check_non_empty_prompt

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


class HaikuJudgeInput(TypedDict):
    r"""Define the input to haiku judge."""

    haiku: str
    topic: str


class HaikuJudgeInputValidator(BaseModel):
    r"""Define the input validator to haiku judge."""

    haiku: str = Field(
        min_length=1,
        description="The haiku text to evaluate, expected to consist of three lines.",
    )
    topic: str = Field(
        min_length=1,
        description="The target topic that the haiku should meaningfully address.",
    )


class RawHaikuJudgeOutput(BaseModel):
    r"""Define the raw structured output produced by the haiku judge LLM.

    This class intentionally excludes ``overall_prediction``, which is a
    derived field computed from the other attributes. Excluding it prevents
    the LLM from attempting to populate it via structured output, which could
    produce inconsistent outputs. Use :class:`HaikuJudgeResult` instead when
    consuming the judge's output, as it extends this class with the derived
    ``overall_prediction`` field.

    Attributes:
        structure_reasoning: A brief explanation justifying the
            ``structure_prediction`` decision.
        structure_prediction: ``True`` only if the haiku has exactly
            three lines with syllable counts of 5, 7, and 5
            respectively.
        topic_reasoning: A brief explanation justifying the
            ``topic_prediction`` decision.
        topic_prediction: ``True`` if the haiku meaningfully addresses
            the target topic, otherwise ``False``.
        score_reasoning: A brief explanation justifying the
            ``score_prediction`` decision.
        score_prediction: Quality score from 1 to 10 based on imagery, emotional
            resonance, and word choice. Constrained to the range
            ``[1, 10]``.
    """

    structure_reasoning: str = Field(
        description=(
            "A non-empty, clear and concise explanation justifying the "
            "structure_prediction decision."
        ),
    )
    structure_prediction: bool = Field(
        description=(
            "True if the haiku follows the 5-7-5 syllable structure across exactly 3 lines, "
            "False otherwise."
        )
    )

    topic_reasoning: str = Field(
        description=(
            "A non-empty, clear and concise explanation justifying the topic_prediction decision."
        ),
    )
    topic_prediction: bool = Field(
        description="True if the haiku meaningfully addresses the target topic, otherwise False."
    )

    score_reasoning: str = Field(
        description="A non-empty, clear and concise explanation justifying the score decision.",
    )
    score_prediction: int = Field(
        ge=1,
        le=10,
        description=(
            "Quality score from 1-10 based on imagery, emotional resonance, and word choice."
        ),
    )

    @field_validator("score_prediction", mode="before")
    @classmethod
    def coerce_score_to_int(cls, v: object) -> int:
        r"""Coerce ``score_prediction`` to ``int``."""
        return int(v)


class HaikuJudgeOutput(RawHaikuJudgeOutput):
    r"""Define the structured output produced by the haiku judge LLM.

    ``structure_prediction``, ``topic_prediction``, ``score_prediction``,
    and ``score_reasoning`` are populated by the LLM via structured
    output. ``overall_prediction`` is always derived automatically from
    those fields by the model validator, so it is guaranteed to be
    consistent.

    Attributes:
        structure_reasoning: A brief explanation justifying the
            ``structure_prediction`` decision.
        structure_prediction: ``True`` only if the haiku has exactly
            three lines with syllable counts of 5, 7, and 5
            respectively.
        topic_reasoning: A brief explanation justifying the
            ``topic_prediction`` decision.
        topic_prediction: ``True`` if the haiku meaningfully addresses
            the target topic, otherwise ``False``.
        score_reasoning: A brief explanation justifying the
            ``score_prediction`` decision.
        score_prediction: Quality score from 1 to 10 based on imagery, emotional
            resonance, and word choice. Constrained to the range
            ``[1, 10]``.
        overall_prediction: Derived automatically: ``True`` only if
            ``structure_prediction`` and ``topic_prediction`` are both
            ``True`` and ``score_prediction >= 7``. Any LLM-provided
            value is overwritten by the :meth:`compute_passed` model
            validator.
    """

    overall_prediction: bool = Field(
        default=False,
        description=(
            "Derived automatically: True ONLY if structure_prediction AND topic_prediction "
            "AND score_prediction >= 7."
        ),
    )

    @model_validator(mode="after")
    def compute_overall_prediction(self) -> Self:
        r"""Compute ``overall_prediction`` from the contributing fields.

        Overrides any LLM-provided value to guarantee that
        ``overall_prediction`` is always consistent with
        ``structure_prediction``, ``topic_prediction``, and ``score_prediction``.

        Returns:
            The updated model instance with ``overall_prediction`` set.
        """
        self.overall_prediction = (
            self.structure_prediction and self.topic_prediction and self.score_prediction >= 7
        )
        return self


def validate_haiku_judge_input(data: dict[str, Any]) -> HaikuJudgeInput:
    r"""Validate the input data for the haiku judge.

    Runs ``data`` through :class:`HaikuJudgeInputValidator` to enforce
    field constraints such as minimum length. Raises a
    :class:`~pydantic.ValidationError` if validation fails.

    Args:
        data: A dict with ``topic`` and ``haiku`` keys to validate.

    Returns:
        The original ``data`` unchanged if validation passes.

    Raises:
        ValidationError: If ``data`` does not satisfy the
            :class:`HaikuJudgeInputValidator` constraints.

    Example:
        ```pycon
        >>> from argos.models.haiku_judge import validate_haiku_judge_input
        >>> validate_haiku_judge_input({"topic": "nature", "haiku": "old pond"})
        {'topic': 'nature', 'haiku': 'old pond'}

        ```
    """
    HaikuJudgeInputValidator.model_validate(data)
    return data


def create_haiku_judge_model(
    llm: BaseChatModel, system_prompt: str = HAIKU_JUDGE_SYSTEM_PROMPT
) -> RunnableSequence[HaikuJudgeInput, HaikuJudgeOutput]:
    r"""Create a simple haiku judge model.

    Args:
        llm: The LLM used to build the judge. The LLM must support
            structured output via
            :meth:`~langchain_core.language_models.BaseChatModel.with_structured_output`.
        system_prompt: The system prompt that instructs the LLM on how
            to evaluate a haiku. Defaults to
            ``HAIKU_JUDGE_SYSTEM_PROMPT``.

    Returns:
        A :class:`~langchain_core.runnables.RunnableSequence` that accepts a
            dict with ``topic`` and ``haiku`` keys and returns a
            :class:`HaikuJudgeOutput` with the structured evaluation.
    """
    check_non_empty_prompt(prompt=system_prompt, name="system_prompt")
    return RunnableSequence(
        RunnableLambda(validate_haiku_judge_input),
        ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("user", "Haiku: {haiku}\n\nTopic: {topic}"),
            ]
        ),
        llm.with_structured_output(RawHaikuJudgeOutput),
        RunnableLambda(lambda output: HaikuJudgeOutput.model_validate(output.model_dump())),
    )
