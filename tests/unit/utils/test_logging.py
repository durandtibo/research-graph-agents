from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from argos.testing.fixtures import rich_available
from argos.utils.logging import configure_logging, log_dict_pretty, log_markdown

MODULE = "argos.utils.logging"


@pytest.fixture(autouse=True)
def _reset_logging() -> None:
    logging.basicConfig()


#######################################
#     Tests for configure_logging     #
#######################################


def test_configure_logging_without_colorlog() -> None:
    with (
        patch(f"{MODULE}.is_colorlog_available", return_value=False),
        patch(f"{MODULE}.logging.basicConfig") as basic_config,
    ):
        configure_logging()

    basic_config.assert_called_once_with(level=logging.INFO)


def test_configure_logging_with_colorlog() -> None:
    handler = MagicMock()
    formatter = MagicMock()
    fake_colorlog = MagicMock()
    fake_colorlog.StreamHandler.return_value = handler
    fake_colorlog.ColoredFormatter.return_value = formatter

    with (
        patch(f"{MODULE}.is_colorlog_available", return_value=True),
        patch(f"{MODULE}.colorlog", fake_colorlog, create=True),
        patch(f"{MODULE}.logging.basicConfig") as basic_config,
    ):
        configure_logging(level=logging.WARNING)

    fake_colorlog.StreamHandler.assert_called_once_with()
    fake_colorlog.ColoredFormatter.assert_called_once()
    handler.setFormatter.assert_called_once_with(formatter)
    basic_config.assert_called_once_with(level=logging.WARNING, handlers=[handler])


@pytest.mark.parametrize("level", [logging.INFO, logging.WARNING, logging.ERROR])
def test_configure_logging_level(level: int) -> None:
    with patch(f"{MODULE}.logging.basicConfig") as bc:
        configure_logging(level)
        assert bc.call_args.kwargs["level"] == level


##################################
#     Tests for log_markdown     #
##################################


@rich_available
def test_log_markdown_with_rich() -> None:
    with patch(f"{MODULE}.logger") as mock_logger:
        log_markdown("# Hello")

    mock_logger.log.assert_not_called()


@rich_available
def test_log_markdown_with_rich_with_title() -> None:
    with patch(f"{MODULE}.logger") as mock_logger:
        log_markdown("# Hello", title="cats")

    mock_logger.log.assert_not_called()


def test_log_markdown_without_rich() -> None:
    with (
        patch(f"{MODULE}.is_rich_available", return_value=False),
        patch(f"{MODULE}.logger") as mock_logger,
    ):
        log_markdown("# Hello")

    mock_logger.log.assert_called_once_with(logging.INFO, "# Hello")


def test_log_markdown_with_title_without_rich() -> None:
    with (
        patch(f"{MODULE}.is_rich_available", return_value=False),
        patch(f"{MODULE}.logger") as mock_logger,
    ):
        log_markdown("# Hello", title="cats")

    mock_logger.log.assert_called_once_with(logging.INFO, "cats:\n# Hello")


def test_log_markdown_passes_custom_level_to_logger() -> None:
    with (
        patch(f"{MODULE}.is_rich_available", return_value=False),
        patch(f"{MODULE}.logger") as mock_logger,
    ):
        log_markdown("# Hello", level=logging.WARNING)

    mock_logger.log.assert_called_once_with(logging.WARNING, "# Hello")


#####################################
#     Tests for log_dict_pretty     #
#####################################


@rich_available
def test_log_dict_pretty_with_rich() -> None:
    with patch(f"{MODULE}.logger") as mock_logger:
        log_dict_pretty({"hello": "world"})

    mock_logger.log.assert_not_called()


@rich_available
def test_log_dict_pretty_with_rich_with_title() -> None:
    with patch(f"{MODULE}.logger") as mock_logger:
        log_dict_pretty({"hello": "world"}, title="cats")

    mock_logger.log.assert_not_called()


def test_log_dict_pretty_without_rich() -> None:
    with (
        patch(f"{MODULE}.is_rich_available", return_value=False),
        patch(f"{MODULE}.logger") as mock_logger,
    ):
        log_dict_pretty({"hello": "world"})

    mock_logger.log.assert_called_once_with(logging.INFO, {"hello": "world"})


def test_log_dict_pretty_with_title_without_rich() -> None:
    with (
        patch(f"{MODULE}.is_rich_available", return_value=False),
        patch(f"{MODULE}.logger") as mock_logger,
    ):
        log_dict_pretty({"hello": "world"}, title="cats")

    mock_logger.log.assert_called_once_with(logging.INFO, "cats:\n{'hello': 'world'}")


def test_log_dict_pretty_passes_custom_level_to_logger() -> None:
    with (
        patch(f"{MODULE}.is_rich_available", return_value=False),
        patch(f"{MODULE}.logger") as mock_logger,
    ):
        log_dict_pretty({"hello": "world"}, level=logging.WARNING)

    mock_logger.log.assert_called_once_with(logging.WARNING, {"hello": "world"})
