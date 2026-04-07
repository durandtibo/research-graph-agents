r"""Contain the experiment configuration."""

from __future__ import annotations

__all__ = ["ExperimentConfig", "LlmConfig"]

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class LlmConfig:
    r"""A generic LLM configuration."""

    model: str
    system_prompt: str
    batch_size: int = 1
    # Optional keyword arguments that will be passed to `init_chat_model`
    max_retries: int = 9999
    temperature: float = 0.0
    init_kwargs: dict[str, Any] | None = None


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
    # judge: LlmConfig
    # prompt: LlmConfig
    # analyzer: LlmConfig

    @property
    def path_history(self) -> Path:
        r"""Path to the history."""
        return self.path_experiment.joinpath("history.json")
