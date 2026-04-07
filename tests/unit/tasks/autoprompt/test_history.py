from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from feu.utils.io import save_json
from iden.io import load_json

from argos.tasks.autoprompt.config import ExperimentConfig
from argos.tasks.autoprompt.history import create_history_file

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def config(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig(
        judge_model="llama",
        judge_system_prompt="my prompt",
        path_experiment=tmp_path.joinpath("data"),
    )


#########################################
#     Tests for create_history_file     #
#########################################


def test_create_history_file_does_not_exist(config: ExperimentConfig) -> None:
    create_history_file(config)
    assert config.path_history.is_file()
    assert load_json(config.path_history) == []


def test_create_history_file_exists(config: ExperimentConfig) -> None:
    save_json([{"key": "value"}], config.path_history)
    create_history_file(config)
    assert config.path_history.is_file()
    assert load_json(config.path_history) == [{"key": "value"}]
