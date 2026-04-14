r"""Demo examples of chat models."""

from __future__ import annotations

import logging

from coola.utils.timing import timeblock
from dotenv import load_dotenv

from argos.autoprompt.haiku import columns
from argos.autoprompt.haiku.chat_model import create_chat_model
from argos.autoprompt.haiku.config import ChatModelConfig
from argos.autoprompt.haiku.judge import create_judge_graph
from argos.models.analysis import create_analyzer_model
from argos.models.haiku_judge import create_haiku_judge_model
from argos.prompts.haiku_error_analysis import HAIKU_ERROR_ANALYSIS_SYSTEM_PROMPT_1
from argos.utils.logging import configure_logging

logger: logging.Logger = logging.getLogger(__name__)


def create_text_input() -> str:
    r"""Create text input prompt."""
    return "blah blah blah"


def main_chat_model(config: ChatModelConfig) -> None:
    r"""Define a demo a simple chat model."""
    logger.info("\n\n" + "=" * 50 + " chat model " + "=" * 50 + "\n")
    llm = create_chat_model(config)
    logger.info(llm)

    inp = create_text_input()
    logger.info(f"input: {inp}")
    with timeblock():
        out = llm.invoke(inp)
    logger.info(f"output ({type(out)})")
    logger.info(f"output.content:\n{out.content}")


def main_chat_model_with_template(config: ChatModelConfig) -> None:
    r"""Define a demo a simple chat model."""
    logger.info("\n\n" + "=" * 50 + " chat model with template " + "=" * 50 + "\n")
    llm = create_chat_model(config)
    model = create_analyzer_model(llm)
    logger.info(model)

    inp = {"text": create_text_input()}
    logger.info(f"input: {inp}")
    with timeblock():
        out = model.invoke(inp)
    logger.info(f"output ({type(out)})\n{out.content}")


def main_chat_model_with_structured_output(config: ChatModelConfig) -> None:
    r"""Define a demo a simple chat model."""
    logger.info("\n\n" + "=" * 50 + " chat model with structured output " + "=" * 50 + "\n")
    llm = create_chat_model(config)
    model = create_haiku_judge_model(llm)
    logger.info(model)

    inp = {
        columns.TOPIC: "cat",
        columns.HAIKU: (
            "Soft fur, warm light gleam,\nSilent paws upon the floor,\nSunbeam, peace descends."
        ),
    }
    logger.info(f"input: {inp}")
    with timeblock():
        out = model.invoke(inp)
    logger.info(f"output ({type(out)})\n{out}")


def main_graph(config: ChatModelConfig) -> None:
    r"""Define a demo a simple chat model."""
    logger.info("\n\n" + "=" * 50 + " graph " + "=" * 50 + "\n")
    model = create_judge_graph(config)
    logger.info(model)

    inp = {
        columns.TOPIC: "cat",
        columns.HAIKU: (
            "Soft fur, warm light gleam,\nSilent paws upon the floor,\nSunbeam, peace descends."
        ),
    }
    logger.info(f"input: {inp}")
    with timeblock():
        out = model.invoke(inp)
    logger.info(f"output ({type(out)})\n{out}")


def main() -> None:
    r"""Define the main function."""
    config = ChatModelConfig(
        model="ollama:smollm:135m",
        # model="ollama:gemma3:1b",
        system_prompt=HAIKU_ERROR_ANALYSIS_SYSTEM_PROMPT_1,
    )
    logger.info(config)
    main_chat_model(config)
    main_chat_model_with_template(config)
    main_chat_model_with_structured_output(config)
    main_graph(config)


if __name__ == "__main__":
    configure_logging(level=logging.INFO)
    load_dotenv()

    main()
