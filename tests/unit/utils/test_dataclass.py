from __future__ import annotations

from dataclasses import dataclass

from argos.utils.dataclass import dataclass_to_dict


@dataclass
class Point:
    x: float
    y: float


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
