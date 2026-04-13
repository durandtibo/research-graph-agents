r"""Contain utility functions for chat models."""

from __future__ import annotations

__all__ = ["create_chat_model"]

import logging
from typing import TYPE_CHECKING

from langchain.chat_models import init_chat_model

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from argos.autoprompt.haiku.config import ChatModelConfig

logger: logging.Logger = logging.getLogger(__name__)


def create_chat_model(config: ChatModelConfig) -> BaseChatModel:
    r"""Create the chat model based on the config.

    Args:
        config: The chat model config.

    Returns:
        The instantiated chat model.
    """
    init_kwargs = config.init_kwargs or {}
    model = init_chat_model(
        model=config.model,
        temperature=config.temperature,
        max_retries=config.max_retries,
        **init_kwargs,
    )
    model_version = getattr(model, "model", getattr(model, "model_name", "Unknown"))
    logger.info(
        f"class: {type(model).__name__} | model: {model_version} | temperature: {model.temperature}"
    )
    return model
