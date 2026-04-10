from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from pydantic import BaseModel

from argos.utils.mapping import to_dict

#############################
#     Tests for to_dict     #
#############################


@dataclass
class Point:
    x: int
    y: int


class UserModel(BaseModel):
    name: str
    age: int


def test_to_dict_pydantic_model() -> None:
    assert to_dict(UserModel(name="Alice", age=30)) == {"name": "Alice", "age": 30}


def test_to_dict_dataclass() -> None:
    assert to_dict(Point(x=1, y=2)) == {"x": 1, "y": 2}


@pytest.mark.parametrize(
    "data",
    [
        pytest.param({"x": 1, "y": 2}, id="dict"),
        pytest.param([1, 2, 3], id="list"),
        pytest.param("hello", id="str"),
        pytest.param(42, id="int"),
        pytest.param(None, id="none"),
    ],
)
def test_to_dict_passthrough(data: Any) -> None:
    assert to_dict(data) == data
