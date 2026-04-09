from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from feu.utils.io import save_json
from iden.io import load_json

from argos.tasks.autoprompt.history import (
    JsonHistory,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def json_path(tmp_path: Path) -> Path:
    path = tmp_path.joinpath("data").joinpath("history.json")
    save_json([{"key": "value"}], path)
    return path


#################################
#     Tests for JsonHistory     #
#################################


def test_json_history_does_not_exist(tmp_path: Path) -> None:
    history = JsonHistory(tmp_path.joinpath("data"))
    assert history.path.is_file()
    assert load_json(history.path) == []


def test_json_history_already_exists(json_path: Path) -> None:
    history = JsonHistory(json_path)
    assert history.path.is_file()
    assert load_json(history.path) == [{"key": "value"}]


def test_json_history_append_empty(tmp_path: Path) -> None:
    history = JsonHistory(tmp_path.joinpath("data").joinpath("history.json"))
    history.append({"hello": "world"})
    assert history.path.is_file()
    assert load_json(history.path) == [{"hello": "world"}]


def test_json_history_append_not_empty(json_path: Path) -> None:
    history = JsonHistory(json_path)
    history.append({"hello": "world"})
    assert history.path.is_file()
    assert load_json(history.path) == [{"key": "value"}, {"hello": "world"}]


def test_json_history_get_values_empty(tmp_path: Path) -> None:
    history = JsonHistory(tmp_path.joinpath("data").joinpath("history.json"))
    assert history.get_values() == []


def test_json_history_get_values_not_empty(json_path: Path) -> None:
    history = JsonHistory(json_path)
    assert history.get_values() == [{"key": "value"}]


def test_json_history_clear_empty(tmp_path: Path) -> None:
    history = JsonHistory(tmp_path.joinpath("data").joinpath("history.json"))
    history.clear()
    assert history.get_values() == []


def test_json_history_clear_not_empty(json_path: Path) -> None:
    history = JsonHistory(json_path)
    history.clear()
    assert history.get_values() == []
