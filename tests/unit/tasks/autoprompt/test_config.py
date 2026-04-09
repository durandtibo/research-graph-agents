from __future__ import annotations

from pathlib import Path

import pytest

from argos.tasks.autoprompt.config import ExperimentConfig, LlmConfig


##############################
#     Tests for LlmConfig    #
##############################


def test_llm_config_required_fields() -> None:
    config = LlmConfig(model="openai:gpt-4o", system_prompt="You are a judge.")
    assert config.model == "openai:gpt-4o"
    assert config.system_prompt == "You are a judge."


def test_llm_config_default_batch_size() -> None:
    config = LlmConfig(model="openai:gpt-4o", system_prompt="You are a judge.")
    assert config.batch_size == 1


def test_llm_config_default_max_retries() -> None:
    config = LlmConfig(model="openai:gpt-4o", system_prompt="You are a judge.")
    assert config.max_retries == 9999


def test_llm_config_default_temperature() -> None:
    config = LlmConfig(model="openai:gpt-4o", system_prompt="You are a judge.")
    assert config.temperature == 0.0


def test_llm_config_default_init_kwargs() -> None:
    config = LlmConfig(model="openai:gpt-4o", system_prompt="You are a judge.")
    assert config.init_kwargs is None


def test_llm_config_custom_values() -> None:
    config = LlmConfig(
        model="ollama:gemma3:1b",
        system_prompt="Custom prompt.",
        batch_size=5,
        max_retries=3,
        temperature=0.7,
        init_kwargs={"timeout": 30},
    )
    assert config.model == "ollama:gemma3:1b"
    assert config.system_prompt == "Custom prompt."
    assert config.batch_size == 5
    assert config.max_retries == 3
    assert config.temperature == 0.7
    assert config.init_kwargs == {"timeout": 30}


def test_llm_config_equality() -> None:
    config1 = LlmConfig(model="openai:gpt-4o", system_prompt="You are a judge.")
    config2 = LlmConfig(model="openai:gpt-4o", system_prompt="You are a judge.")
    assert config1 == config2


def test_llm_config_inequality() -> None:
    config1 = LlmConfig(model="openai:gpt-4o", system_prompt="You are a judge.")
    config2 = LlmConfig(model="openai:gpt-4o", system_prompt="Different prompt.")
    assert config1 != config2


###################################
#     Tests for ExperimentConfig  #
###################################


@pytest.fixture
def experiment_config(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig(
        judge_model="openai:gpt-4o",
        judge_system_prompt="You are a haiku judge.",
        path_experiment=tmp_path,
    )


def test_experiment_config_required_fields(tmp_path: Path) -> None:
    config = ExperimentConfig(
        judge_model="openai:gpt-4o",
        judge_system_prompt="You are a haiku judge.",
        path_experiment=tmp_path,
    )
    assert config.judge_model == "openai:gpt-4o"
    assert config.judge_system_prompt == "You are a haiku judge."
    assert config.path_experiment == tmp_path


def test_experiment_config_default_batch_size(experiment_config: ExperimentConfig) -> None:
    assert experiment_config.batch_size == 20


def test_experiment_config_default_iteration(experiment_config: ExperimentConfig) -> None:
    assert experiment_config.iteration == 0


def test_experiment_config_default_judge_is_none(experiment_config: ExperimentConfig) -> None:
    assert experiment_config.judge is None


def test_experiment_config_default_prompt_generator_is_none(
    experiment_config: ExperimentConfig,
) -> None:
    assert experiment_config.prompt_generator is None


def test_experiment_config_default_error_analyzer_is_none(
    experiment_config: ExperimentConfig,
) -> None:
    assert experiment_config.error_analyzer is None


def test_experiment_config_path_history(tmp_path: Path) -> None:
    config = ExperimentConfig(
        judge_model="openai:gpt-4o",
        judge_system_prompt="You are a haiku judge.",
        path_experiment=tmp_path,
    )
    assert config.path_history == tmp_path / "history.json"


def test_experiment_config_path_history_is_path_instance(tmp_path: Path) -> None:
    config = ExperimentConfig(
        judge_model="openai:gpt-4o",
        judge_system_prompt="You are a haiku judge.",
        path_experiment=tmp_path,
    )
    assert isinstance(config.path_history, Path)


def test_experiment_config_path_artifact_default_iteration(tmp_path: Path) -> None:
    config = ExperimentConfig(
        judge_model="openai:gpt-4o",
        judge_system_prompt="You are a haiku judge.",
        path_experiment=tmp_path,
        iteration=0,
    )
    assert config.path_artifact == tmp_path / "artifacts/0000"


def test_experiment_config_path_artifact_nonzero_iteration(tmp_path: Path) -> None:
    config = ExperimentConfig(
        judge_model="openai:gpt-4o",
        judge_system_prompt="You are a haiku judge.",
        path_experiment=tmp_path,
        iteration=3,
    )
    assert config.path_artifact == tmp_path / "artifacts/0003"


def test_experiment_config_path_artifact_large_iteration(tmp_path: Path) -> None:
    config = ExperimentConfig(
        judge_model="openai:gpt-4o",
        judge_system_prompt="You are a haiku judge.",
        path_experiment=tmp_path,
        iteration=1234,
    )
    assert config.path_artifact == tmp_path / "artifacts/1234"


def test_experiment_config_path_artifact_zero_padded(tmp_path: Path) -> None:
    config = ExperimentConfig(
        judge_model="openai:gpt-4o",
        judge_system_prompt="You are a haiku judge.",
        path_experiment=tmp_path,
        iteration=7,
    )
    assert config.path_artifact.name == "0007"


def test_experiment_config_path_artifact_is_path_instance(tmp_path: Path) -> None:
    config = ExperimentConfig(
        judge_model="openai:gpt-4o",
        judge_system_prompt="You are a haiku judge.",
        path_experiment=tmp_path,
    )
    assert isinstance(config.path_artifact, Path)


def test_experiment_config_with_llm_configs(tmp_path: Path) -> None:
    judge_config = LlmConfig(model="openai:gpt-4o", system_prompt="Judge prompt.")
    generator_config = LlmConfig(model="openai:gpt-4o-mini", system_prompt="Generator prompt.")
    analyzer_config = LlmConfig(model="openai:gpt-4o", system_prompt="Analyzer prompt.")
    config = ExperimentConfig(
        judge_model="openai:gpt-4o",
        judge_system_prompt="You are a haiku judge.",
        path_experiment=tmp_path,
        judge=judge_config,
        prompt_generator=generator_config,
        error_analyzer=analyzer_config,
    )
    assert config.judge == judge_config
    assert config.prompt_generator == generator_config
    assert config.error_analyzer == analyzer_config
