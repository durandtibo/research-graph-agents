from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from pydantic import BaseModel

from argos.utils.mapping import recursive_to_dict, to_dict

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
    "obj",
    [
        pytest.param({"x": 1, "y": 2}, id="dict"),
        pytest.param([1, 2, 3], id="list"),
        pytest.param("hello", id="str"),
        pytest.param(42, id="int"),
        pytest.param(None, id="none"),
    ],
)
def test_to_dict_passthrough(obj: Any) -> None:
    assert to_dict(obj) == obj


#######################################
#     Tests for recursive_to_dict     #
#######################################


def test_recursive_to_dict_dataclass() -> None:
    assert recursive_to_dict(Point(4, 2)) == {"x": 4, "y": 2}


def test_recursive_to_dict_list() -> None:
    assert recursive_to_dict([Point(4, 2)]) == [{"x": 4, "y": 2}]


def test_recursive_to_dict_dict() -> None:
    assert recursive_to_dict({"p1": Point(4, 2), "p2": Point(3, 7)}) == {
        "p1": {"x": 4, "y": 2},
        "p2": {"x": 3, "y": 7},
    }


def test_recursive_to_dict_non_dataclass_string() -> None:
    assert recursive_to_dict("hello") == "hello"


def test_recursive_to_dict_non_dataclass_int() -> None:
    assert recursive_to_dict(42) == 42


def test_recursive_to_dict_non_dataclass_none() -> None:
    assert recursive_to_dict(None) is None


def test_recursive_to_dict_empty_list() -> None:
    assert recursive_to_dict([]) == []


def test_recursive_to_dict_list_of_non_dataclass() -> None:
    assert recursive_to_dict([1, 2, 3]) == [1, 2, 3]


def test_recursive_to_dict_mixed_list() -> None:
    assert recursive_to_dict([Point(1, 2), "text", 99]) == [{"x": 1, "y": 2}, "text", 99]


def test_recursive_to_dict_list_of_multiple_dataclasses() -> None:
    assert recursive_to_dict([Point(1, 2), Point(3, 4)]) == [
        {"x": 1, "y": 2},
        {"x": 3, "y": 4},
    ]


def test_recursive_to_dict_empty_dict() -> None:
    assert recursive_to_dict({}) == {}


def test_recursive_to_dict_pydantic_model() -> None:
    assert recursive_to_dict(UserModel(name="Alice", age=30)) == {"name": "Alice", "age": 30}


def test_recursive_to_dict_pydantic_model_in_list() -> None:
    assert recursive_to_dict([UserModel(name="Alice", age=30), UserModel(name="Bob", age=25)]) == [
        {"name": "Alice", "age": 30},
        {"name": "Bob", "age": 25},
    ]


def test_recursive_to_dict_pydantic_model_in_dict() -> None:
    assert recursive_to_dict({"u": UserModel(name="Alice", age=30)}) == {
        "u": {"name": "Alice", "age": 30}
    }


@dataclass
class Line:
    start: Point
    end: Point


def test_recursive_to_dict_nested_dataclass() -> None:
    assert recursive_to_dict(Line(start=Point(0, 0), end=Point(1, 1))) == {
        "start": {"x": 0, "y": 0},
        "end": {"x": 1, "y": 1},
    }


def test_recursive_to_dict_tuple_of_dataclasses() -> None:
    assert recursive_to_dict((Point(1, 2), Point(3, 4))) == ({"x": 1, "y": 2}, {"x": 3, "y": 4})


def test_recursive_to_dict_nested_list() -> None:
    assert recursive_to_dict([[Point(1, 2)], [Point(3, 4)]]) == [
        [{"x": 1, "y": 2}],
        [{"x": 3, "y": 4}],
    ]


def test_recursive_to_dict_float() -> None:
    assert recursive_to_dict(3.14) == 3.14


def test_recursive_to_dict_bool() -> None:
    assert recursive_to_dict(True) is True
