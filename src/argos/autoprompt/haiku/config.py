r"""Contain the experiment configuration."""

from __future__ import annotations

__all__ = ["ChatModelConfig", "ExperimentConfig"]

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class ChatModelConfig:
    r"""A generic LLM configuration.

    Attributes:
        model: The model identifier string passed to ``init_chat_model``
            (e.g. ``"openai:gpt-4o"`` or ``"ollama:gemma3:1b"``).
        system_prompt: The system prompt that instructs the LLM on its
            role and task.
        batch_size: Number of examples to process concurrently per
            inference batch. Defaults to ``1``.
        max_retries: Maximum number of retries on failed LLM calls.
            Defaults to ``9999``.
        temperature: Sampling temperature passed to the LLM. Set to
            ``0.0`` for deterministic outputs. Defaults to ``0.0``.
        init_kwargs: Optional extra keyword arguments forwarded to
            ``init_chat_model``. Defaults to ``None``.
    """

    model: str
    system_prompt: str
    batch_size: int = 1
    # Optional keyword arguments that will be passed to `init_chat_model`
    max_retries: int = 9999
    temperature: float = 0.0
    init_kwargs: dict[str, Any] | None = None


@dataclass
class ExperimentConfig:
    r"""The experiment configuration.

    Attributes:
        path_experiment: Root directory where all experiment outputs
            (results, artifacts, history) are stored.
        iteration: Current iteration index used to namespace artifact
            subdirectories. Defaults to ``0``.
        prompt_generator: Optional :class:`ChatModelConfig` for the
            prompt generator model. Defaults to ``None``.
        judge: Optional :class:`ChatModelConfig` for the judge model.
            Defaults to ``None``.
        error_analyzer: Optional :class:`ChatModelConfig` for the
            error analyzer model. Defaults to ``None``.
    """

    path_experiment: Path
    iteration: int = 0
    prompt_generator: ChatModelConfig = None
    judge: ChatModelConfig = None
    error_analyzer: ChatModelConfig = None

    @property
    def path_history(self) -> Path:
        r"""Return the path to the history JSON file.

        Returns:
            ``<path_experiment>/history.json``
        """
        return self.path_experiment.joinpath("history.json")

    @property
    def path_artifact(self) -> Path:
        r"""Return the artifact directory path for the current iteration.

        Returns:
            ``<path_experiment>/artifacts/<iteration:04d>``
        """
        return self.path_experiment.joinpath(f"artifacts/{self.iteration:04d}")
