from __future__ import annotations

import pytest

from argos.utils.batching import batchify

##############################
#     Tests for batchify     #
##############################


def test_batchify_exact_multiple() -> None:
    assert batchify([1, 2, 3, 4, 5, 6], size=3) == [(1, 2, 3), (4, 5, 6)]


def test_batchify_with_remainder() -> None:
    assert batchify([1, 2, 3, 4, 5, 6, 7, 8], size=3) == [(1, 2, 3), (4, 5, 6), (7, 8)]


def test_batchify_size_one() -> None:
    assert batchify([1, 2, 3], size=1) == [(1,), (2,), (3,)]


def test_batchify_size_larger_than_list() -> None:
    assert batchify([1, 2, 3], size=10) == [(1, 2, 3)]


def test_batchify_size_equals_list() -> None:
    assert batchify([1, 2, 3], size=3) == [(1, 2, 3)]


def test_batchify_empty_list() -> None:
    assert batchify([], size=3) == []


def test_batchify_string_items() -> None:
    assert batchify(["a", "b", "c", "d"], size=2) == [("a", "b"), ("c", "d")]


def test_batchify_single_item() -> None:
    assert batchify([42], size=3) == [(42,)]


@pytest.mark.parametrize("size", [0, -1, -10])
def test_batchify_invalid_size(size: int) -> None:
    with pytest.raises(ValueError, match="size must be >= 1"):
        batchify([1, 2, 3], size=size)
