# noqa: INP001
r"""Script to create or update the package versions."""

from __future__ import annotations

import logging
from pathlib import Path

from feu.utils.io import save_json
from feu.version import (
    fetch_latest_major_versions,
    fetch_latest_minor_versions,
)

logger: logging.Logger = logging.getLogger(__name__)


def fetch_package_versions() -> dict[str, list[str]]:
    r"""Get the versions for each package.

    Returns:
        A dictionary with the versions for each package.
    """
    return {
        "coola": list(fetch_latest_minor_versions("coola", lower="1.1")),
        "grandalf": list(fetch_latest_minor_versions("grandalf", lower="0.8")),
        "iden": list(fetch_latest_minor_versions("iden", lower="0.4")),
        "langchain": list(fetch_latest_minor_versions("langchain", lower="1.3")),
        "langchain-core": list(fetch_latest_minor_versions("langchain-core", lower="1.4")),
        "langgraph": list(fetch_latest_major_versions("langgraph", lower="1.1")),
        "polars": list(fetch_latest_major_versions("polars", lower="1.40")),
        "python-dotenv": list(fetch_latest_major_versions("python-dotenv", lower="1.2")),
        # LangChain optional packages
        "langchain-anthropic": list(
            fetch_latest_major_versions("langchain-anthropic", lower="1.4")
        ),
        "langchain-google-genai": list(
            fetch_latest_major_versions("langchain-google-genai", lower="4.2")
        ),
        "langchain-ollama": list(fetch_latest_major_versions("langchain-ollama", lower="1.1")),
        "langchain-openai": list(fetch_latest_major_versions("langchain-openai", lower="1.2")),
    }


def main() -> None:
    r"""Generate the package versions and save them in a JSON file."""
    versions = fetch_package_versions()
    logger.info(f"{versions=}")
    path = Path(__file__).parent.parent.joinpath("dev/config").joinpath("package_versions.json")
    logger.info(f"Saving package versions to {path}")
    save_json(versions, path, exist_ok=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
