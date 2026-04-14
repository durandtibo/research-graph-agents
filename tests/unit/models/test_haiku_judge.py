from __future__ import annotations

from unittest.mock import Mock

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import ValidationError

from argos.models.haiku_judge import (
    HaikuJudgeInput,
    HaikuJudgeInputValidator,
    HaikuJudgeOutput,
    RawHaikuJudgeOutput,
    create_haiku_judge_model,
    validate_haiku_judge_input,
)
from argos.prompts.haiku_judge import HAIKU_JUDGE_SYSTEM_PROMPT
from tests.unit.utils.test_prompt import EMPTY_PROMPTS


@pytest.fixture
def judge_output() -> HaikuJudgeOutput:
    return HaikuJudgeOutput(
        score_prediction=8,
        score_reasoning="score explanation",
        structure_prediction=True,
        structure_reasoning="structure explanation",
        topic_prediction=True,
        topic_reasoning="topic explanation",
    )


@pytest.fixture
def mock_llm() -> BaseChatModel:
    output = RawHaikuJudgeOutput(
        score_prediction=8,
        score_reasoning="score explanation",
        structure_prediction=True,
        structure_reasoning="structure explanation",
        topic_prediction=True,
        topic_reasoning="topic explanation",
    )
    return Mock(
        spec=BaseChatModel,
        with_structured_output=Mock(return_value=RunnableLambda(lambda x: output)),  # noqa: ARG005
    )


#####################################
#     Tests for HaikuJudgeInput     #
#####################################


def test_haiku_judge_input_keys() -> None:
    assert set(HaikuJudgeInput.__annotations__) == {"topic", "haiku"}


##############################################
#     Tests for HaikuJudgeInputValidator     #
##############################################


def test_haiku_judge_input_validator_valid_input() -> None:
    validator = HaikuJudgeInputValidator(
        topic="nature",
        haiku="An old silent pond\nA frog jumps into the pond\nSplash! Silence again",
    )
    assert validator.topic == "nature"
    assert (
        validator.haiku == "An old silent pond\nA frog jumps into the pond\nSplash! Silence again"
    )


def test_haiku_judge_input_validator_topic_missing() -> None:
    with pytest.raises(
        ValidationError,
        match="topic\n  Field required",
    ):
        HaikuJudgeInputValidator(haiku="An old silent pond\nA frog jumps\nSplash!")


def test_haiku_judge_input_validator_haiku_missing() -> None:
    with pytest.raises(
        ValidationError,
        match="haiku\n  Field required",
    ):
        HaikuJudgeInputValidator(topic="nature")


def test_haiku_judge_input_validator_topic_empty_string() -> None:
    with pytest.raises(
        ValidationError,
        match="topic\n  String should have at least 1 character",
    ):
        HaikuJudgeInputValidator(topic="", haiku="An old silent pond\nA frog jumps\nSplash!")


def test_haiku_judge_input_validator_haiku_empty_string() -> None:
    with pytest.raises(
        ValidationError,
        match="haiku\n  String should have at least 1 character",
    ):
        HaikuJudgeInputValidator(topic="nature", haiku="")


def test_haiku_judge_input_validator_both_fields_empty_string() -> None:
    with pytest.raises(
        ValidationError,
        match="haiku\n  String should have at least 1 character",
    ):
        HaikuJudgeInputValidator(topic="", haiku="")


def test_haiku_judge_input_validator_topic_whitespace_only() -> None:
    # Whitespace-only strings have length > 0, so they pass min_length=1
    validator = HaikuJudgeInputValidator(
        topic="   ", haiku="An old silent pond\nA frog jumps\nSplash!"
    )
    assert validator.topic == "   "


def test_haiku_judge_input_validator_haiku_single_line() -> None:
    # Single-line haiku is technically valid (min_length only, no line count enforcement)
    validator = HaikuJudgeInputValidator(topic="nature", haiku="A single line haiku")
    assert validator.haiku == "A single line haiku"


def test_haiku_judge_input_validator_topic_single_character() -> None:
    validator = HaikuJudgeInputValidator(
        topic="a", haiku="An old silent pond\nA frog jumps\nSplash!"
    )
    assert validator.topic == "a"


def test_haiku_judge_input_validator_haiku_single_character() -> None:
    validator = HaikuJudgeInputValidator(topic="nature", haiku="x")
    assert validator.haiku == "x"


def test_haiku_judge_input_validator_topic_none() -> None:
    with pytest.raises(ValidationError, match="topic\n  Input should be a valid string"):
        HaikuJudgeInputValidator(topic=None, haiku="An old silent pond\nA frog jumps\nSplash!")


def test_haiku_judge_input_validator_haiku_none() -> None:
    with pytest.raises(ValidationError, match="haiku\n  Input should be a valid string"):
        HaikuJudgeInputValidator(topic="nature", haiku=None)


def test_haiku_judge_input_validator_extra_fields_ignored() -> None:
    # Pydantic v2 ignores extra fields by default
    validator = HaikuJudgeInputValidator(
        topic="nature", haiku="Line one\nLine two\nLine three", extra_field="ignored"
    )
    assert not hasattr(validator, "extra_field")


######################################
#     Tests for HaikuJudgeOutput     #
######################################


def test_haiku_judge_output_valid_passed() -> None:
    output = HaikuJudgeOutput(
        overall_prediction=True,
        score_prediction=8,
        score_reasoning="score explanation",
        structure_prediction=True,
        structure_reasoning="structure explanation",
        topic_prediction=True,
        topic_reasoning="topic explanation",
    )
    assert output.structure_prediction
    assert output.topic_prediction
    assert output.score_prediction == 8
    assert output.overall_prediction


def test_haiku_judge_output_valid_failed() -> None:
    output = HaikuJudgeOutput(
        overall_prediction=False,
        score_prediction=8,
        score_reasoning="score explanation",
        structure_prediction=False,
        structure_reasoning="structure explanation",
        topic_prediction=True,
        topic_reasoning="topic explanation",
    )
    assert not output.structure_prediction
    assert output.topic_prediction
    assert output.score_prediction == 8
    assert not output.overall_prediction


def test_haiku_judge_output_score_prediction_float_to_int() -> None:
    output = HaikuJudgeOutput(
        overall_prediction=True,
        score_prediction=8.1,
        score_reasoning="score explanation",
        structure_prediction=True,
        structure_reasoning="structure explanation",
        topic_prediction=True,
        topic_reasoning="topic explanation",
    )
    assert output.score_prediction == 8


def test_haiku_judge_output_invalid_score_too_low() -> None:
    with pytest.raises(
        ValueError, match=r"score_prediction\n  Input should be greater than or equal to 1"
    ):
        HaikuJudgeOutput(
            overall_prediction=True,
            score_prediction=0,
            score_reasoning="score explanation",
            structure_prediction=True,
            structure_reasoning="structure explanation",
            topic_prediction=True,
            topic_reasoning="topic explanation",
        )


def test_haiku_judge_output_invalid_score_too_high() -> None:
    with pytest.raises(
        ValueError, match=r"score_prediction\n  Input should be less than or equal to 10"
    ):
        HaikuJudgeOutput(
            overall_prediction=True,
            score_prediction=11,
            score_reasoning="score explanation",
            structure_prediction=True,
            structure_reasoning="structure explanation",
            topic_prediction=True,
            topic_reasoning="topic explanation",
        )


def test_haiku_judge_output_passed_auto_corrected_structure_failed() -> None:
    output = HaikuJudgeOutput(
        overall_prediction=True,  # LLM inconsistency: overridden to False
        score_prediction=8,
        score_reasoning="score explanation",
        structure_prediction=False,
        structure_reasoning="structure explanation",
        topic_prediction=True,
        topic_reasoning="topic explanation",
    )
    assert not output.overall_prediction


def test_haiku_judge_output_passed_auto_corrected_topic_failed() -> None:
    output = HaikuJudgeOutput(
        overall_prediction=True,  # LLM inconsistency: overridden to False
        score_prediction=8,
        score_reasoning="score explanation",
        structure_prediction=True,
        structure_reasoning="structure explanation",
        topic_prediction=False,
        topic_reasoning="topic explanation",
    )
    assert not output.overall_prediction


def test_haiku_judge_output_passed_auto_corrected_score_too_low() -> None:
    output = HaikuJudgeOutput(
        overall_prediction=True,  # LLM inconsistency: overridden to False
        score_prediction=6,
        score_reasoning="score explanation",
        structure_prediction=True,
        structure_reasoning="structure explanation",
        topic_prediction=True,
        topic_reasoning="topic explanation",
    )
    assert not output.overall_prediction


def test_haiku_judge_output_passed_auto_corrected_all_pass() -> None:
    output = HaikuJudgeOutput(
        overall_prediction=False,  # LLM inconsistency: overridden to True
        score_prediction=7,
        score_reasoning="score explanation",
        structure_prediction=True,
        structure_reasoning="structure explanation",
        topic_prediction=True,
        topic_reasoning="topic explanation",
    )
    assert output.overall_prediction


def test_haiku_judge_output_with_all_reasoning_fields() -> None:
    output = HaikuJudgeOutput(
        score_prediction=9,
        score_reasoning="Exceptional imagery and structure.",
        structure_prediction=True,
        structure_reasoning="All lines follow 5-7-5.",
        topic_prediction=True,
        topic_reasoning="The haiku clearly evokes autumn.",
    )
    assert output.structure_reasoning == "All lines follow 5-7-5."
    assert output.topic_reasoning == "The haiku clearly evokes autumn."
    assert output.score_reasoning == "Exceptional imagery and structure."


def test_haiku_judge_output_with_empty_structure_reasoning() -> None:
    with pytest.raises(
        ValidationError, match=r"structure_reasoning\n  String should have at least 1 character"
    ):
        HaikuJudgeOutput(
            score_prediction=9,
            score_reasoning="Exceptional imagery and structure.",
            structure_prediction=True,
            structure_reasoning="",
            topic_prediction=True,
            topic_reasoning="The haiku clearly evokes autumn.",
        )


def test_haiku_judge_output_with_empty_topic_reasoning() -> None:
    with pytest.raises(
        ValidationError, match=r"topic_reasoning\n  String should have at least 1 character"
    ):
        HaikuJudgeOutput(
            score_prediction=9,
            score_reasoning="Exceptional imagery and structure.",
            structure_prediction=True,
            structure_reasoning="All lines follow 5-7-5.",
            topic_prediction=True,
            topic_reasoning="",
        )


def test_haiku_judge_output_with_empty_score_reasoning() -> None:
    with pytest.raises(
        ValidationError, match=r"score_reasoning\n  String should have at least 1 character"
    ):
        HaikuJudgeOutput(
            score_prediction=9,
            score_reasoning="",
            structure_prediction=True,
            structure_reasoning="All lines follow 5-7-5.",
            topic_prediction=True,
            topic_reasoning="The haiku clearly evokes autumn.",
        )


################################################
#     Tests for validate_haiku_judge_input     #
################################################


def test_validate_haiku_judge_input_returns_original_dict() -> None:
    data = {"topic": "nature", "haiku": "old pond\nfrog leaps in\nwater's sound"}
    output = validate_haiku_judge_input(data)
    assert output is data


def test_validate_haiku_judge_input_valid_data_unchanged() -> None:
    output = validate_haiku_judge_input({"topic": "winter", "haiku": "cold wind blows"})
    assert output == {"topic": "winter", "haiku": "cold wind blows"}


def test_validate_haiku_judge_input_empty_topic_raises() -> None:
    with pytest.raises(ValidationError, match="topic\n  String should have at least 1 character"):
        validate_haiku_judge_input({"topic": "", "haiku": "cold wind blows"})


def test_validate_haiku_judge_input_empty_haiku_raises() -> None:
    with pytest.raises(ValidationError, match="haiku\n  String should have at least 1 character"):
        validate_haiku_judge_input({"topic": "winter", "haiku": ""})


def test_validate_haiku_judge_input_missing_topic_raises() -> None:
    with pytest.raises(ValidationError, match="topic\n  Field required"):
        validate_haiku_judge_input({"haiku": "cold wind blows"})


def test_validate_haiku_judge_input_missing_haiku_raises() -> None:
    with pytest.raises(ValidationError, match="haiku\n  Field required"):
        validate_haiku_judge_input({"topic": "winter"})


def test_validate_haiku_judge_input_empty_dict_raises() -> None:
    with pytest.raises(ValidationError, match="2 validation errors for HaikuJudgeInputValidator"):
        validate_haiku_judge_input({})


##############################################
#     Tests for create_haiku_judge_model     #
##############################################


def test_create_haiku_judge_model_returns_runnable(mock_llm: BaseChatModel) -> None:
    model = create_haiku_judge_model(mock_llm)
    assert isinstance(model, Runnable)


def test_create_haiku_judge_model_uses_structured_output(mock_llm: BaseChatModel) -> None:
    create_haiku_judge_model(mock_llm)
    mock_llm.with_structured_output.assert_called_once_with(RawHaikuJudgeOutput)


def test_create_haiku_judge_model_uses_default_system_prompt(mock_llm: BaseChatModel) -> None:
    model = create_haiku_judge_model(mock_llm)
    system_message = model.steps[1].messages[0]
    assert system_message.prompt.template == HAIKU_JUDGE_SYSTEM_PROMPT


def test_create_haiku_judge_model_uses_custom_system_prompt(mock_llm: BaseChatModel) -> None:
    custom_prompt = "You are a custom haiku judge."
    model = create_haiku_judge_model(mock_llm, system_prompt=custom_prompt)
    system_message = model.steps[1].messages[0]
    assert system_message.prompt.template == custom_prompt


@pytest.mark.parametrize("system_prompt", EMPTY_PROMPTS)
def test_create_haiku_judge_model_uses_empty_system_prompt(
    mock_llm: BaseChatModel, system_prompt: str
) -> None:
    with pytest.raises(ValueError, match="system_prompt must be a non-empty string"):
        create_haiku_judge_model(mock_llm, system_prompt=system_prompt)


def test_create_haiku_judge_model_prompt_contains_expected_input_variables(
    mock_llm: BaseChatModel,
) -> None:
    model = create_haiku_judge_model(mock_llm)
    assert model.steps[1].input_variables == ["haiku", "topic"]


def test_create_haiku_judge_model_invokes_with_topic_and_haiku(
    mock_llm: BaseChatModel, judge_output: HaikuJudgeOutput
) -> None:
    model = create_haiku_judge_model(mock_llm)
    output = model.invoke(
        {
            "topic": "autumn",
            "haiku": "Leaves fall silently\nCrisp air and golden colors\nWinter is coming",
        }
    )
    assert output == judge_output


def test_create_haiku_judge_model_invokes_without_topic(mock_llm: BaseChatModel) -> None:
    model = create_haiku_judge_model(mock_llm)
    with pytest.raises(ValidationError, match="topic\n  Field required"):
        model.invoke(
            {"haiku": "Leaves fall silently\nCrisp air and golden colors\nWinter is coming"}
        )


def test_create_haiku_judge_model_invokes_without_haiku(mock_llm: BaseChatModel) -> None:
    model = create_haiku_judge_model(mock_llm)
    with pytest.raises(ValidationError, match="haiku\n  Field required"):
        model.invoke({"topic": "autumn"})
