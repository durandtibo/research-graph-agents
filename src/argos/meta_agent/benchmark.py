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
    r"""Define a single labeled example used for benchmarking.

    Attributes:
        id: A unique identifier for the example.
        input: The input passed to the agent.
        target: The expected ground-truth output.
        metadata: Optional dictionary of auxiliary information.
            Defaults to ``None``.

    Example:
        ```pycon
        >>> from argos.meta_agent.benchmark import BenchmarkExample
        >>> example = BenchmarkExample(id="q1", input="What is 2+2?", target="4")
        >>> example.id
        'q1'
        >>> example.input
        'What is 2+2?'
        >>> example.target
        '4'

        ```
    """

    id: str
    input: InputT
    target: TargetT
    metadata: dict[str, Any] | None = None


@dataclass
class Benchmark(Generic[InputT, TargetT]):
    r"""Define a collection of labeled examples used for benchmarking.

    The examples are indexed by their IDs.

    Attributes:
        examples: A mapping from example ID to
            :class:`BenchmarkExample` instance.
        metadata: Optional dictionary of auxiliary information about
            the benchmark. Defaults to ``None``.

    Example:
        ```pycon
        >>> from argos.meta_agent.benchmark import Benchmark, BenchmarkExample
        >>> benchmark = Benchmark(
        ...     examples={
        ...         "q1": BenchmarkExample(id="q1", input="What is 2+2?", target="4"),
        ...         "q2": BenchmarkExample(id="q2", input="What is 3+3?", target="6"),
        ...     }
        ... )
        >>> len(benchmark.examples)
        2

        ```
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

        Raises:
            ValueError: If any example IDs are duplicated.

        Example:
            ```pycon
            >>> from argos.meta_agent.benchmark import Benchmark, BenchmarkExample
            >>> benchmark = Benchmark.from_examples(
            ...     [
            ...         BenchmarkExample(id="q1", input="What is 2+2?", target="4"),
            ...         BenchmarkExample(id="q2", input="What is 3+3?", target="6"),
            ...     ]
            ... )
            >>> len(benchmark.examples)
            2

            ```
        """
        data = {example.id: example for example in examples}
        if len(data) != len(examples):
            msg = "Some example IDs are duplicated"
            raise ValueError(msg)
        return cls(examples=data, metadata=metadata)
