from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

import pytest
from iden.io import load_json
from langchain_core.runnables import Runnable

from argos.autoprompt.haiku.prompt_generator import HistoryPromptGenerator
from argos.models.prompt_generation import PromptGeneratorInput, PromptGeneratorOutput

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def history() -> list[dict[str, Any]]:
    return [
        {"system_prompt": "prompt 1", "error analysis": "blabla 1", "accuracy": 0.1},
        {"system_prompt": "prompt 2", "error analysis": "blabla 2", "accuracy": 0.2},
        {"system_prompt": "prompt 3", "error analysis": "blabla 3", "accuracy": 0.3},
    ]


@pytest.fixture
def mock_model() -> Runnable[PromptGeneratorInput, PromptGeneratorOutput]:
    return Mock(
        spec=Runnable,
        invoke=Mock(
            side_effect=[PromptGeneratorOutput(reasoning="blabla", prompt="my new prompt")]
        ),
    )


_HISTORY_PROMPT = """The prompt history is provided below as a JSON array. Items are listed in order of execution, starting with the first iteration and ending with the most recent.
[{'system_prompt': 'prompt 1', 'error analysis': 'blabla 1', 'accuracy': 0.1}, {'system_prompt': 'prompt 2', 'error analysis': 'blabla 2', 'accuracy': 0.2}, {'system_prompt': 'prompt 3', 'error analysis': 'blabla 3', 'accuracy': 0.3}]"""

############################################
#     Tests for HistoryPromptGenerator     #
############################################


def test_history_prompt_generator_repr(history: list, mock_model: Runnable) -> None:
    assert repr(HistoryPromptGenerator(history=history, model=mock_model)).startswith(
        "HistoryPromptGenerator("
    )


def test_history_prompt_generator_str(history: list, mock_model: Runnable) -> None:
    assert str(HistoryPromptGenerator(history=history, model=mock_model)).startswith(
        "HistoryPromptGenerator("
    )


def test_history_prompt_generator_generate(history: list, mock_model: Runnable) -> None:
    assert HistoryPromptGenerator(history=history, model=mock_model).generate() == "my new prompt"
    mock_model.invoke.assert_called_once_with({"history": _HISTORY_PROMPT})


def test_history_prompt_generator_generate_with_path(
    history: list,
    mock_model: Runnable,
    tmp_path: Path,
) -> None:
    path = tmp_path.joinpath("data").joinpath("prompt.json")
    assert (
        HistoryPromptGenerator(history=history, model=mock_model, path=path).generate()
        == "my new prompt"
    )
    mock_model.invoke.assert_called_once_with({"history": _HISTORY_PROMPT})
    assert path.is_file()
    assert load_json(path) == {"reasoning": "blabla", "prompt": "my new prompt"}
