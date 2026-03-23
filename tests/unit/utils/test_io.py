import json
from itertools import chain
from pathlib import Path

import pytest

from argos.utils.io import read_jsonl_in_batches, write_jsonl

###########################################
#     Tests for read_jsonl_in_batches     #
###########################################


def test_read_jsonl_in_batches_single_full_batch(tmp_path: Path) -> None:
    """Records fit exactly into one batch."""
    path = tmp_path.joinpath("data.jsonl")
    write_jsonl(path, records=[{"id": i} for i in range(3)])

    batches = list(read_jsonl_in_batches(path, batch_size=3))
    assert batches == [[{"id": 0}, {"id": 1}, {"id": 2}]]


def test_read_jsonl_in_batches_multiple_full_batches(tmp_path: Path) -> None:
    """Records split evenly across multiple batches."""
    path = tmp_path.joinpath("data.jsonl")
    write_jsonl(path, records=[{"id": i} for i in range(6)])

    batches = list(read_jsonl_in_batches(path, batch_size=2))
    assert batches == [
        [{"id": 0}, {"id": 1}],
        [{"id": 2}, {"id": 3}],
        [{"id": 4}, {"id": 5}],
    ]


def test_read_jsonl_in_batches_partial_last_batch(tmp_path: Path) -> None:
    """Last batch is smaller than batch_size and is not dropped."""
    path = tmp_path.joinpath("data.jsonl")
    write_jsonl(path, records=[{"id": i} for i in range(5)])

    batches = list(read_jsonl_in_batches(path, batch_size=3))
    assert batches == [
        [{"id": 0}, {"id": 1}, {"id": 2}],
        [{"id": 3}, {"id": 4}],
    ]


def test_read_jsonl_in_batches_batch_size_larger_than_file(tmp_path: Path) -> None:
    """batch_size exceeds total records — should yield one partial
    batch."""
    path = tmp_path.joinpath("data.jsonl")
    write_jsonl(path, records=[{"id": i} for i in range(3)])

    batches = list(read_jsonl_in_batches(path, batch_size=100))
    assert batches == [[{"id": 0}, {"id": 1}, {"id": 2}]]


def test_read_jsonl_in_batches_empty_file(tmp_path: Path) -> None:
    """Empty file yields no batches."""
    path = tmp_path.joinpath("data.jsonl")
    write_jsonl(path, records=[])

    batches = list(read_jsonl_in_batches(path, batch_size=10))
    assert batches == []


def test_read_jsonl_in_batches_blank_lines_are_skipped(tmp_path: Path) -> None:
    """Blank lines in the file are ignored."""
    path = tmp_path.joinpath("data.jsonl")
    with path.open("w", encoding="utf-8") as f:
        f.write('{"id": 0}\n')
        f.write("\n")
        f.write('{"id": 1}\n')
        f.write("\n\n")
        f.write('{"id": 2}\n')

    batches = list(read_jsonl_in_batches(path, batch_size=10))

    assert batches == [[{"id": 0}, {"id": 1}, {"id": 2}]]


def test_read_jsonl_in_batches_all_records_are_returned(tmp_path: Path) -> None:
    """Flattening all batches recovers the original records in order."""
    path = tmp_path.joinpath("data.jsonl")
    write_jsonl(path, records=[{"id": i} for i in range(20)])

    batches = list(read_jsonl_in_batches(path, batch_size=6))
    assert list(chain.from_iterable(batches)) == [{"id": i} for i in range(20)]


def test_read_jsonl_in_batches_batch_size_of_one(tmp_path: Path) -> None:
    """batch_size=1 yields each record in its own batch."""
    path = tmp_path.joinpath("data.jsonl")
    write_jsonl(path, records=[{"id": i} for i in range(4)])

    batches = list(read_jsonl_in_batches(path, batch_size=1))
    assert batches == [[{"id": 0}], [{"id": 1}], [{"id": 2}], [{"id": 3}]]


def test_read_jsonl_in_batches_invalid_json_raises(tmp_path: Path) -> None:
    """Malformed JSON line raises json.JSONDecodeError."""
    path = tmp_path.joinpath("data.jsonl")
    path.write_text('{"id": 1}\nnot-valid-json\n{"id": 3}\n', encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        list(read_jsonl_in_batches(path, batch_size=10))


def test_read_jsonl_in_batches_file_not_found_raises(tmp_path: Path) -> None:
    """Non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="No such file or directory"):
        list(read_jsonl_in_batches(tmp_path.joinpath("nonexistent/file.jsonl"), batch_size=10))
