r"""Contain the implementation of a benchmark example."""

from __future__ import annotations

__all__ = ["Benchmark", "BenchmarkExample"]

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic

from argos.meta_agent.typing import InputT, TargetT

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass
class BenchmarkExample(Generic[InputT, TargetT]):
    r"""Define the benchmark example class."""

    id: str
    input: InputT
    target: TargetT
    metadata: dict[str, Any] | None = None


@dataclass
class Benchmark(Generic[InputT, TargetT]):
    r"""Define the benchmark class.

    The examples are indexed by their IDs.
    """

    examples: dict[str, BenchmarkExample[InputT, TargetT]]
    metadata: dict[str, Any] | None = None

    @classmethod
    def from_examples(
        cls,
        examples: Sequence[BenchmarkExample[InputT, TargetT]],
        metadata: dict[str, Any] | None = None,
    ) -> Benchmark[InputT, TargetT]:
        r"""Create a benchmark from a list of examples.

        Args:
            examples: A list of examples. The example IDs must be unique.
            metadata: The benchmark metadata.

        Returns:
            The benchmark instance.
        """
        data = {example.id: example for example in examples}
        if len(data) != len(examples):
            msg = "Some example IDs are duplicated"
            raise ValueError(msg)
        return cls(examples=data, metadata=metadata)
