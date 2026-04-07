r"""Contain the experiment configuration."""

from __future__ import annotations

__all__ = ["ExperimentConfig"]

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class ExperimentConfig:
    r"""The experiment configuration."""

    judge_model: str
    judge_system_prompt: str
    path_experiment: Path
    batch_size: int = 20
    iteration: int = 0
    # path_history: Path
    # prompt_model: str

    @property
    def path_history(self) -> Path:
        r"""Path to the history."""
        return self.path_experiment.joinpath("history.json")
