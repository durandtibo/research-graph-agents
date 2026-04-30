from __future__ import annotations

__all__ = ["ConfigCaptureRunnable", "DoubleRunnable", "IdentityRunnable", "RaisingErrorRunnable"]

from typing import Any, Generic, TypeVar

from langchain_core.runnables import Runnable, RunnableConfig

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class DoubleRunnable(Runnable[int, int]):
    """Doubles every integer input."""

    def invoke(
        self,
        input: int,  # noqa: A002
        config: RunnableConfig | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> int:
        return input * 2

    def batch(
        self,
        inputs: list[int],
        config: RunnableConfig | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> list[int]:
        return [x * 2 for x in inputs]


class IdentityRunnable(Runnable[InputT, InputT], Generic[InputT]):
    """Return inputs unchanged — the simplest possible runnable stub."""

    def invoke(
        self,
        input: InputT,  # noqa: A002
        config: RunnableConfig | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> InputT:
        return input

    def batch(
        self,
        inputs: list[InputT],
        config: RunnableConfig | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> list[InputT]:
        return list(inputs)


class ConfigCaptureRunnable(IdentityRunnable[InputT], Generic[InputT]):
    """Records the config it receives so tests can assert on it."""

    def __init__(self) -> None:
        self.last_config = None

    def invoke(
        self,
        input: InputT,  # noqa: A002
        config: RunnableConfig | None = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> InputT:
        self.last_config = config
        return input

    def batch(
        self,
        inputs: list[InputT],
        config: RunnableConfig | None = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> list[InputT]:
        self.last_config = config
        return list(inputs)


class RaisingErrorRunnable(Runnable[InputT, OutputT]):
    """Always raises, to verify Agent does not swallow exceptions."""

    def invoke(
        self,
        input: InputT,  # noqa: A002, ARG002
        config: RunnableConfig | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> OutputT:
        msg = "model failure"
        raise RuntimeError(msg)

    def batch(
        self,
        inputs: list[InputT],  # noqa: ARG002
        config: RunnableConfig | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> list[OutputT]:
        msg = "model failure"
        raise RuntimeError(msg)
