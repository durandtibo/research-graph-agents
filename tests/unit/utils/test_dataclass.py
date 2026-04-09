from __future__ import annotations

from dataclasses import dataclass

from argos.utils.dataclass import dataclass_to_dict


@dataclass
class Point:
    x: float
    y: float


@dataclass
class Line:
    start: Point
    end: Point


#######################################
#     Tests for dataclass_to_dict     #
#######################################


def test_dataclass_to_dict_dataclass() -> None:
    assert dataclass_to_dict(Point(4, 2)) == {"x": 4, "y": 2}


def test_dataclass_to_dict_list() -> None:
    assert dataclass_to_dict([Point(4, 2)]) == [{"x": 4, "y": 2}]


def test_dataclass_to_dict_dict() -> None:
    assert dataclass_to_dict({"p1": Point(4, 2), "p2": Point(3, 7)}) == {
        "p1": {"x": 4, "y": 2},
        "p2": {"x": 3, "y": 7},
    }


def test_dataclass_to_dict_non_dataclass_string() -> None:
    assert dataclass_to_dict("hello") == "hello"


def test_dataclass_to_dict_non_dataclass_int() -> None:
    assert dataclass_to_dict(42) == 42


def test_dataclass_to_dict_non_dataclass_none() -> None:
    assert dataclass_to_dict(None) is None


def test_dataclass_to_dict_nested_dataclass() -> None:
    line = Line(start=Point(0, 0), end=Point(1, 1))
    assert dataclass_to_dict(line) == {"start": {"x": 0, "y": 0}, "end": {"x": 1, "y": 1}}


def test_dataclass_to_dict_list_of_non_dataclass() -> None:
    assert dataclass_to_dict([1, 2, 3]) == [1, 2, 3]


def test_dataclass_to_dict_mixed_list() -> None:
    assert dataclass_to_dict([Point(1, 2), "text", 3]) == [{"x": 1, "y": 2}, "text", 3]


def test_dataclass_to_dict_empty_list() -> None:
    assert dataclass_to_dict([]) == []


def test_dataclass_to_dict_empty_dict() -> None:
    assert dataclass_to_dict({}) == {}
