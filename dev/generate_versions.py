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
        "langchain-core": list(fetch_latest_minor_versions("langchain-core", lower="1.2")),
        "langchain-google-genai": list(
            fetch_latest_major_versions("langchain-google-genai", lower="4.2")
        ),
        "langgraph": list(fetch_latest_major_versions("langgraph", lower="1.1")),
        "python-dotenv": list(fetch_latest_major_versions("python-dotenv", lower="1.2")),
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
