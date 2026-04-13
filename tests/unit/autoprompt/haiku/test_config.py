from __future__ import annotations

from typing import TYPE_CHECKING

from argos.autoprompt.haiku.config import ChatModelConfig, ExperimentConfig

if TYPE_CHECKING:
    from pathlib import Path

#############################
#     Tests for LlmConfig   #
#############################


def test_llm_config_required_fields() -> None:
    config = ChatModelConfig(model="gpt-4o", system_prompt="You are a judge.")
    assert config.model == "gpt-4o"
    assert config.system_prompt == "You are a judge."


def test_llm_config_default_batch_size() -> None:
    config = ChatModelConfig(model="gpt-4o", system_prompt="prompt")
    assert config.batch_size == 1


def test_llm_config_default_max_retries() -> None:
    config = ChatModelConfig(model="gpt-4o", system_prompt="prompt")
    assert config.max_retries == 9999


def test_llm_config_default_temperature() -> None:
    config = ChatModelConfig(model="gpt-4o", system_prompt="prompt")
    assert config.temperature == 0.0


def test_llm_config_default_init_kwargs() -> None:
    config = ChatModelConfig(model="gpt-4o", system_prompt="prompt")
    assert config.init_kwargs is None


def test_llm_config_custom_values() -> None:
    config = ChatModelConfig(
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
    c1 = ChatModelConfig(model="gpt-4o", system_prompt="prompt")
    c2 = ChatModelConfig(model="gpt-4o", system_prompt="prompt")
    assert c1 == c2


def test_llm_config_inequality_different_model() -> None:
    c1 = ChatModelConfig(model="gpt-4o", system_prompt="prompt")
    c2 = ChatModelConfig(model="gpt-3.5", system_prompt="prompt")
    assert c1 != c2


def test_llm_config_inequality_different_prompt() -> None:
    c1 = ChatModelConfig(model="gpt-4o", system_prompt="prompt A")
    c2 = ChatModelConfig(model="gpt-4o", system_prompt="prompt B")
    assert c1 != c2


def test_llm_config_is_mutable() -> None:
    config = ChatModelConfig(model="gpt-4o", system_prompt="prompt")
    config.temperature = 0.5
    assert config.temperature == 0.5


####################################
#     Tests for ExperimentConfig   #
####################################


def test_experiment_config_required_fields(tmp_path: Path) -> None:
    config = ExperimentConfig(path_experiment=tmp_path)
    assert config.path_experiment == tmp_path


def test_experiment_config_default_iteration(tmp_path: Path) -> None:
    config = ExperimentConfig(path_experiment=tmp_path)
    assert config.iteration == 0


def test_experiment_config_default_optional_fields_are_none(tmp_path: Path) -> None:
    config = ExperimentConfig(path_experiment=tmp_path)
    assert config.judge is None
    assert config.prompt_generator is None
    assert config.error_analyzer is None


def test_experiment_config_custom_values(tmp_path: Path) -> None:
    judge_cfg = ChatModelConfig(model="gpt-4o", system_prompt="judge prompt")
    generator_cfg = ChatModelConfig(model="gpt-3.5", system_prompt="generator prompt")
    analyzer_cfg = ChatModelConfig(model="claude-3", system_prompt="analyzer prompt")

    config = ExperimentConfig(
        path_experiment=tmp_path,
        iteration=3,
        judge=judge_cfg,
        prompt_generator=generator_cfg,
        error_analyzer=analyzer_cfg,
    )
    assert config.iteration == 3
    assert config.judge == judge_cfg
    assert config.prompt_generator == generator_cfg
    assert config.error_analyzer == analyzer_cfg


def test_experiment_config_path_history(tmp_path: Path) -> None:
    config = ExperimentConfig(path_experiment=tmp_path)
    assert config.path_history == tmp_path.joinpath("history.json")


def test_experiment_config_path_artifact_default_iteration(tmp_path: Path) -> None:
    config = ExperimentConfig(path_experiment=tmp_path)
    assert config.path_artifact == tmp_path.joinpath("artifacts/0000")


def test_experiment_config_path_artifact_custom_iteration(tmp_path: Path) -> None:
    config = ExperimentConfig(path_experiment=tmp_path, iteration=42)
    assert config.path_artifact == tmp_path.joinpath("artifacts/0042")


def test_experiment_config_equality(tmp_path: Path) -> None:
    c1 = ExperimentConfig(path_experiment=tmp_path)
    c2 = ExperimentConfig(path_experiment=tmp_path)
    assert c1 == c2


def test_experiment_config_inequality_different_iteration(tmp_path: Path) -> None:
    c1 = ExperimentConfig(path_experiment=tmp_path, iteration=42)
    c2 = ExperimentConfig(path_experiment=tmp_path)
    assert c1 != c2
