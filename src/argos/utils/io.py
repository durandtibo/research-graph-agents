r"""Contain I/O utility functions."""

from __future__ import annotations

__all__ = ["read_jsonl_in_batches", "write_jsonl"]

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path


def read_jsonl_in_batches(
    path: Path, batch_size: int = 1
) -> Generator[list[dict[Any, Any]], None, None]:
    r"""Read a JSONL file and yield its records in batches.

    Blank lines in the file are silently skipped. The last batch may
    contain fewer records than ``batch_size`` if the total number of
    lines is not evenly divisible.

    Args:
        path: Path to the JSONL file.
        batch_size: Number of records to include in each batch.
            Defaults to ``1``.

    Yields:
        A list of parsed JSON objects (dicts) with at most
            ``batch_size`` entries.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        json.JSONDecodeError: If a non-blank line cannot be parsed
            as valid JSON.

    Example:
        ```pycon
        >>> import pathlib, tempfile, json
        >>> from argos.utils.io import read_jsonl_in_batches
        >>> with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        ...     _ = f.write('{"a": 1}\n{"a": 2}\n{"a": 3}\n')
        ...     path = pathlib.Path(f.name)
        ...
        >>> list(read_jsonl_in_batches(path, batch_size=2))
        [[{'a': 1}, {'a': 2}], [{'a': 3}]]

        ```
    """
    batch = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped_line = line.strip()
            if not stripped_line:
                continue
            batch.append(json.loads(stripped_line))
            if len(batch) == batch_size:
                yield batch
                batch = []

    if batch:
        yield batch


def write_jsonl(path: Path, records: list[dict[Any, Any]]) -> None:
    r"""Write a list of dicts to a JSONL file.

    Each dict is serialised to a single JSON line. The file is created
    (or overwritten) at ``path``.

    Args:
        path: Destination path for the JSONL file.
        records: List of dicts to serialise. Each dict must be
            JSON-serialisable.

    Raises:
        TypeError: If any value in ``records`` is not JSON-serialisable.
        OSError: If the file cannot be opened for writing (e.g. due to
            permission errors or a missing parent directory).

    Example:
        ```pycon
        >>> import pathlib, tempfile
        >>> from argos.utils.io import write_jsonl, read_jsonl_in_batches
        >>> with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        ...     path = pathlib.Path(f.name)
        ...
        >>> write_jsonl(path, [{"x": 1}, {"x": 2}])
        >>> list(read_jsonl_in_batches(path, batch_size=10))
        [[{'x': 1}, {'x': 2}]]

        ```
    """
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
