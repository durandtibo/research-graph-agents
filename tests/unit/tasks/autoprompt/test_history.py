from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from feu.utils.io import save_json
from iden.io import load_json

from argos.tasks.autoprompt.config import ExperimentConfig
from argos.tasks.autoprompt.history import append_to_history, create_history

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def config(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig(
        judge_model="llama",
        judge_system_prompt="my prompt",
        path_experiment=tmp_path.joinpath("data"),
    )


#######################################
#     Tests for append_to_history     #
#######################################


def test_append_to_history_empty(config: ExperimentConfig) -> None:
    append_to_history(config, {"hello": "world"})
    assert config.path_history.is_file()
    assert load_json(config.path_history) == [{"hello": "world"}]


def test_append_to_history_empty_not_empty(config: ExperimentConfig) -> None:
    save_json([{"key": "value"}], config.path_history)
    append_to_history(config, {"hello": "world"})
    assert config.path_history.is_file()
    assert load_json(config.path_history) == [{"key": "value"}, {"hello": "world"}]


####################################
#     Tests for create_history     #
####################################


def test_create_history_does_not_exist(config: ExperimentConfig) -> None:
    create_history(config)
    assert config.path_history.is_file()
    assert load_json(config.path_history) == []


def test_create_history_exists(config: ExperimentConfig) -> None:
    save_json([{"key": "value"}], config.path_history)
    create_history(config)
    assert config.path_history.is_file()
    assert load_json(config.path_history) == [{"key": "value"}]
