from __future__ import annotations

from pathlib import Path

import pytest

from argos.tasks.autoprompt.config import ExperimentConfig, LlmConfig

#############################
#     Tests for LlmConfig   #
#############################


def test_llm_config_required_fields() -> None:
    config = LlmConfig(model="gpt-4o", system_prompt="You are a judge.")
    assert config.model == "gpt-4o"
    assert config.system_prompt == "You are a judge."


def test_llm_config_default_batch_size() -> None:
    config = LlmConfig(model="gpt-4o", system_prompt="prompt")
    assert config.batch_size == 1


def test_llm_config_default_max_retries() -> None:
    config = LlmConfig(model="gpt-4o", system_prompt="prompt")
    assert config.max_retries == 9999


def test_llm_config_default_temperature() -> None:
    config = LlmConfig(model="gpt-4o", system_prompt="prompt")
    assert config.temperature == 0.0


def test_llm_config_default_init_kwargs() -> None:
    config = LlmConfig(model="gpt-4o", system_prompt="prompt")
    assert config.init_kwargs is None


def test_llm_config_custom_values() -> None:
    config = LlmConfig(
        model="claude-3",
        system_prompt="Be helpful.",
        batch_size=5,
        max_retries=3,
        temperature=0.7,
        init_kwargs={"timeout": 30},
    )
    assert config.model == "claude-3"
    assert config.system_prompt == "Be helpful."
    assert config.batch_size == 5
    assert config.max_retries == 3
    assert config.temperature == 0.7
    assert config.init_kwargs == {"timeout": 30}


def test_llm_config_equality() -> None:
    c1 = LlmConfig(model="gpt-4o", system_prompt="prompt")
    c2 = LlmConfig(model="gpt-4o", system_prompt="prompt")
    assert c1 == c2


def test_llm_config_inequality_different_model() -> None:
    c1 = LlmConfig(model="gpt-4o", system_prompt="prompt")
    c2 = LlmConfig(model="gpt-3.5", system_prompt="prompt")
    assert c1 != c2


def test_llm_config_inequality_different_prompt() -> None:
    c1 = LlmConfig(model="gpt-4o", system_prompt="prompt A")
    c2 = LlmConfig(model="gpt-4o", system_prompt="prompt B")
    assert c1 != c2


def test_llm_config_is_mutable() -> None:
    config = LlmConfig(model="gpt-4o", system_prompt="prompt")
    config.temperature = 0.5
    assert config.temperature == 0.5


####################################
#     Tests for ExperimentConfig   #
####################################


def test_experiment_config_required_fields(tmp_path: Path) -> None:
    config = ExperimentConfig(
        judge_model="gpt-4o",
        judge_system_prompt="You are a judge.",
        path_experiment=tmp_path,
    )
    assert config.judge_model == "gpt-4o"
    assert config.judge_system_prompt == "You are a judge."
    assert config.path_experiment == tmp_path


def test_experiment_config_default_batch_size(tmp_path: Path) -> None:
    config = ExperimentConfig(
        judge_model="gpt-4o",
        judge_system_prompt="prompt",
        path_experiment=tmp_path,
    )
    assert config.batch_size == 20


def test_experiment_config_default_iteration(tmp_path: Path) -> None:
    config = ExperimentConfig(
        judge_model="gpt-4o",
        judge_system_prompt="prompt",
        path_experiment=tmp_path,
    )
    assert config.iteration == 0


def test_experiment_config_default_optional_fields_are_none(tmp_path: Path) -> None:
    config = ExperimentConfig(
        judge_model="gpt-4o",
        judge_system_prompt="prompt",
        path_experiment=tmp_path,
    )
    assert config.judge is None
    assert config.prompt_generator is None
    assert config.error_analyzer is None


def test_experiment_config_custom_values(tmp_path: Path) -> None:
    judge_cfg = LlmConfig(model="gpt-4o", system_prompt="judge prompt")
    generator_cfg = LlmConfig(model="gpt-3.5", system_prompt="generator prompt")
    analyzer_cfg = LlmConfig(model="claude-3", system_prompt="analyzer prompt")

    config = ExperimentConfig(
        judge_model="gpt-4o",
        judge_system_prompt="You are a judge.",
        path_experiment=tmp_path,
        batch_size=5,
        iteration=3,
        judge=judge_cfg,
        prompt_generator=generator_cfg,
        error_analyzer=analyzer_cfg,
    )
    assert config.batch_size == 5
    assert config.iteration == 3
    assert config.judge == judge_cfg
    assert config.prompt_generator == generator_cfg
    assert config.error_analyzer == analyzer_cfg


def test_experiment_config_path_history(tmp_path: Path) -> None:
    config = ExperimentConfig(
        judge_model="gpt-4o",
        judge_system_prompt="prompt",
        path_experiment=tmp_path,
    )
    assert config.path_history == tmp_path / "history.json"


def test_experiment_config_path_artifact_default_iteration(tmp_path: Path) -> None:
    config = ExperimentConfig(
        judge_model="gpt-4o",
        judge_system_prompt="prompt",
        path_experiment=tmp_path,
    )
    assert config.path_artifact == tmp_path / "artifacts/0000"


def test_experiment_config_path_artifact_custom_iteration(tmp_path: Path) -> None:
    config = ExperimentConfig(
        judge_model="gpt-4o",
        judge_system_prompt="prompt",
        path_experiment=tmp_path,
        iteration=42,
    )
    assert config.path_artifact == tmp_path / "artifacts/0042"


def test_experiment_config_path_artifact_large_iteration(tmp_path: Path) -> None:
    config = ExperimentConfig(
        judge_model="gpt-4o",
        judge_system_prompt="prompt",
        path_experiment=tmp_path,
        iteration=9999,
    )
    assert config.path_artifact == tmp_path / "artifacts/9999"


@pytest.mark.parametrize("iteration", [0, 1, 10, 100, 1000])
def test_experiment_config_path_artifact_zero_padded(tmp_path: Path, iteration: int) -> None:
    config = ExperimentConfig(
        judge_model="gpt-4o",
        judge_system_prompt="prompt",
        path_experiment=tmp_path,
        iteration=iteration,
    )
    expected = tmp_path / f"artifacts/{iteration:04d}"
    assert config.path_artifact == expected


def test_experiment_config_equality(tmp_path: Path) -> None:
    c1 = ExperimentConfig(
        judge_model="gpt-4o",
        judge_system_prompt="prompt",
        path_experiment=tmp_path,
    )
    c2 = ExperimentConfig(
        judge_model="gpt-4o",
        judge_system_prompt="prompt",
        path_experiment=tmp_path,
    )
    assert c1 == c2


def test_experiment_config_inequality_different_model(tmp_path: Path) -> None:
    c1 = ExperimentConfig(
        judge_model="gpt-4o",
        judge_system_prompt="prompt",
        path_experiment=tmp_path,
    )
    c2 = ExperimentConfig(
        judge_model="gpt-3.5",
        judge_system_prompt="prompt",
        path_experiment=tmp_path,
    )
    assert c1 != c2
