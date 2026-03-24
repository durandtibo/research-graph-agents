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
    """Read a JSONL file in batches.

    Args:
        path: Path to the JSONL file as a pathlib.Path object.
        batch_size: Number of records per batch.

    Yields:
        List of parsed JSON objects (dicts) of length <= batch_size.
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
    """Write a list of dicts to a JSONL file.

    Args:
        path: Path to the JSONL file as a pathlib.Path object.
        records: List of dicts to write to a JSONL file.
    """
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
