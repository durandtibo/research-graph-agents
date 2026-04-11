r"""Define a script to test the performance of the haiku judge."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from dotenv import load_dotenv

from argos.autoprompt.haiku.config import ExperimentConfig
from argos.autoprompt.haiku.judge import (
    run_experiment,
)
from argos.prompts.haiku_judge import HAIKU_JUDGE_SYSTEM_PROMPT
from argos.utils.logging import configure_logging

logger = logging.getLogger(__name__)


def run_evaluation(judge_model: str, judge_system_prompt: str) -> None:
    r"""Run haiku judge evaluation.

    Args:
        judge_model: The name of the judge model.
        judge_system_prompt: The prompt of the judge-system-prompt.
    """
    path_experiment = (
        Path(__file__)
        .resolve()
        .parent.parent.joinpath("results")
        .joinpath("haiku_judge")
        .joinpath(hashlib.sha256(bytes(str(judge_system_prompt), "utf-8")).hexdigest())
        .joinpath(judge_model.replace(":", "_"))
    )

    config = ExperimentConfig(
        judge_model=judge_model,
        judge_system_prompt=judge_system_prompt,
        path_experiment=path_experiment,
    )
    run_experiment(config)


def main() -> None:
    r"""Define the main function to test the haiku judge system."""
    # model = "ollama:smollm:135m"
    # model = "ollama:gemma3:1b"
    # model = "anthropic:claude-haiku-4-5-20251001"
    # model = "anthropic:claude-sonnet-4-6"
    # model = "anthropic:claude-opus-4-6"

    models = [
        # "ollama:smollm:135m",
        # "ollama:gemma3:1b",
        # "ollama:gemma3:4b",
        # "ollama:gemma3:12b",
        # "anthropic:claude-haiku-4-5-20251001",
        # "anthropic:claude-sonnet-4-6",
        # "anthropic:claude-opus-4-6",
        "google_genai:gemini-3.1-flash-lite-preview",
        # "google_genai:gemini-3-flash-preview",
        # "google_genai:gemini-3.1-pro-preview",
        # "openai:gpt-5.4-nano",
        # "openai:gpt-5.4-mini",
        # "openai:gpt-5.4",
    ]
    judge_system_prompts = [
        HAIKU_JUDGE_SYSTEM_PROMPT,
        # HAIKU_JUDGE_SYSTEM_PROMPT1,
        # HAIKU_JUDGE_SYSTEM_PROMPT2,
        # HAIKU_JUDGE_SYSTEM_PROMPT3,
        # HAIKU_JUDGE_SYSTEM_PROMPT4,
        # HAIKU_JUDGE_SYSTEM_PROMPT_CLAUDE_HAIKU_4_6,
        # HAIKU_JUDGE_SYSTEM_PROMPT_CLAUDE_SONNET_4_6,
        # HAIKU_JUDGE_SYSTEM_PROMPT_GPT_5_3,
        # HAIKU_JUDGE_SYSTEM_PROMPT_GEMINI_3_1_FAST,
        # HAIKU_JUDGE_SYSTEM_PROMPT_GEMINI_3_1_PRO,
    ]

    for model in models:
        for judge_system_prompt in judge_system_prompts:
            run_evaluation(judge_model=model, judge_system_prompt=judge_system_prompt)


if __name__ == "__main__":
    configure_logging(level=logging.INFO)
    load_dotenv()

    main()
