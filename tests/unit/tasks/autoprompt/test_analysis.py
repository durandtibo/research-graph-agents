from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from iden.io import load_text

from argos.tasks.autoprompt.analysis import (
    analyze_errors,
    find_errors,
    format_incorrect_structure_haiku,
    format_incorrect_topic_haiku,
)

if TYPE_CHECKING:
    from pathlib import Path

STRUCTURE_MSG_4 = """4 haikus have incorrect structure predictions. Here is the list of haikus with the true label (target) and the predicted label (target):
- (haiku): A (target): false (prediction): true
- (haiku): B (target): false (prediction): true
- (haiku): C (target): true (prediction): false
- (haiku): D (target): true (prediction): false
"""
STRUCTURE_MSG_2 = """2 haikus have incorrect structure predictions. Here is the list of haikus with the true label (target) and the predicted label (target):
- (haiku): A (target): false (prediction): true
- (haiku): D (target): true (prediction): false
"""
STRUCTURE_MSG_EMPTY = """0 haikus have incorrect structure predictions. Here is the list of haikus with the true label (target) and the predicted label (target):
"""

TOPIC_MSG_4 = """4 haikus have incorrect topic predictions. Here is the list of haikus with the true label (target) and the predicted label (target):
- (haiku): A (target): false (prediction): true
- (haiku): B (target): false (prediction): true
- (haiku): C (target): true (prediction): false
- (haiku): D (target): true (prediction): false
"""
TOPIC_MSG_2 = """2 haikus have incorrect topic predictions. Here is the list of haikus with the true label (target) and the predicted label (target):
- (haiku): A (target): false (prediction): true
- (haiku): D (target): true (prediction): false
"""
TOPIC_MSG_EMPTY = """0 haikus have incorrect topic predictions. Here is the list of haikus with the true label (target) and the predicted label (target):
"""


####################################
#     Tests for analyze_errors     #
####################################


def test_analyze_errors_all_correct(tmp_path: Path) -> None:
    analyze_errors(
        results=pl.DataFrame(
            {
                "haiku": ["A", "B", "C", "D"],
                "passed": [True, True, False, False],
                "target": [True, True, False, False],
                "structure_passed": [True, True, False, False],
                "structure_target": [True, True, False, False],
                "topic_passed": [True, True, False, False],
                "topic_target": [True, True, False, False],
            }
        ),
        path=tmp_path.joinpath("data"),
    )
    file_struct = tmp_path.joinpath("data").joinpath("error_analysis_structure.md")
    assert file_struct.is_file()
    assert load_text(file_struct) == STRUCTURE_MSG_EMPTY

    file_topic = tmp_path.joinpath("data").joinpath("error_analysis_topic.md")
    assert file_topic.is_file()
    assert load_text(file_topic) == TOPIC_MSG_EMPTY


def test_analyze_errors_all_incorrect(tmp_path: Path) -> None:
    analyze_errors(
        results=pl.DataFrame(
            {
                "haiku": ["A", "B", "C", "D"],
                "passed": [True, True, False, False],
                "target": [False, False, True, True],
                "structure_passed": [True, True, False, False],
                "structure_target": [False, False, True, True],
                "topic_passed": [True, True, False, False],
                "topic_target": [False, False, True, True],
            }
        ),
        path=tmp_path.joinpath("data"),
    )
    file_struct = tmp_path.joinpath("data").joinpath("error_analysis_structure.md")
    assert file_struct.is_file()
    assert load_text(file_struct) == STRUCTURE_MSG_4

    file_topic = tmp_path.joinpath("data").joinpath("error_analysis_topic.md")
    assert file_topic.is_file()
    assert load_text(file_topic) == TOPIC_MSG_4


def test_analyze_errors_empty(tmp_path: Path) -> None:
    analyze_errors(
        results=pl.DataFrame(
            {
                "haiku": [],
                "passed": [],
                "target": [],
                "structure_passed": [],
                "structure_target": [],
                "topic_passed": [],
                "topic_target": [],
            }
        ),
        path=tmp_path.joinpath("data"),
    )
    file_struct = tmp_path.joinpath("data").joinpath("error_analysis_structure.md")
    assert file_struct.is_file()
    assert load_text(file_struct) == STRUCTURE_MSG_EMPTY

    file_topic = tmp_path.joinpath("data").joinpath("error_analysis_topic.md")
    assert file_topic.is_file()
    assert load_text(file_topic) == TOPIC_MSG_EMPTY


#################################
#     Tests for find_errors     #
#################################


def test_find_errors_all_correct() -> None:
    assert (
        find_errors(
            results=pl.DataFrame(
                {
                    "haiku": ["A", "B", "C", "D"],
                    "passed": [True, True, False, False],
                    "target": [True, True, False, False],
                }
            ),
            col_target="target",
            col_prediction="passed",
        )
        == []
    )


def test_find_errors_all_incorrect() -> None:
    assert find_errors(
        results=pl.DataFrame(
            {
                "haiku": ["A", "B", "C", "D"],
                "passed": [True, True, False, False],
                "target": [False, False, True, True],
            }
        ),
        col_target="target",
        col_prediction="passed",
    ) == [
        {"haiku": "A", "target": False, "passed": True},
        {"haiku": "B", "target": False, "passed": True},
        {"haiku": "C", "target": True, "passed": False},
        {"haiku": "D", "target": True, "passed": False},
    ]


def test_find_errors_partially_incorrect() -> None:
    assert find_errors(
        results=pl.DataFrame(
            {
                "haiku": ["A", "B", "C", "D"],
                "passed": [True, True, False, False],
                "target": [False, True, False, True],
            }
        ),
        col_target="target",
        col_prediction="passed",
    ) == [
        {"haiku": "A", "target": False, "passed": True},
        {"haiku": "D", "target": True, "passed": False},
    ]


def test_find_incorrect_structure_haiku_empty() -> None:
    assert (
        find_errors(
            results=pl.DataFrame({"haiku": [], "passed": [], "target": []}),
            col_target="target",
            col_prediction="passed",
        )
        == []
    )


######################################################
#     Tests for format_incorrect_structure_haiku     #
######################################################


def test_format_incorrect_structure_haiku_all_correct() -> None:
    assert (
        format_incorrect_structure_haiku(
            results=pl.DataFrame(
                {
                    "haiku": ["A", "B", "C", "D"],
                    "structure_passed": [True, True, False, False],
                    "structure_target": [True, True, False, False],
                }
            )
        )
        == STRUCTURE_MSG_EMPTY
    )


def test_format_incorrect_structure_haiku_all_incorrect() -> None:
    assert (
        format_incorrect_structure_haiku(
            results=pl.DataFrame(
                {
                    "haiku": ["A", "B", "C", "D"],
                    "structure_passed": [True, True, False, False],
                    "structure_target": [False, False, True, True],
                }
            )
        )
        == STRUCTURE_MSG_4
    )


def test_format_incorrect_structure_haiku_partially_incorrect() -> None:
    assert (
        format_incorrect_structure_haiku(
            results=pl.DataFrame(
                {
                    "haiku": ["A", "B", "C", "D"],
                    "structure_passed": [True, True, False, False],
                    "structure_target": [False, True, False, True],
                }
            )
        )
        == STRUCTURE_MSG_2
    )


def test_format_incorrect_structure_haiku_empty() -> None:
    assert (
        format_incorrect_structure_haiku(
            results=pl.DataFrame({"haiku": [], "structure_target": [], "structure_passed": []})
        )
        == STRUCTURE_MSG_EMPTY
    )


##################################################
#     Tests for format_incorrect_topic_haiku     #
##################################################


def test_format_incorrect_topic_haiku_all_correct() -> None:
    assert (
        format_incorrect_topic_haiku(
            results=pl.DataFrame(
                {
                    "haiku": ["A", "B", "C", "D"],
                    "topic_passed": [True, True, False, False],
                    "topic_target": [True, True, False, False],
                }
            )
        )
        == TOPIC_MSG_EMPTY
    )


def test_format_incorrect_topic_haiku_all_incorrect() -> None:
    assert (
        format_incorrect_topic_haiku(
            results=pl.DataFrame(
                {
                    "haiku": ["A", "B", "C", "D"],
                    "topic_passed": [True, True, False, False],
                    "topic_target": [False, False, True, True],
                }
            )
        )
        == TOPIC_MSG_4
    )


def test_format_incorrect_topic_haiku_partially_incorrect() -> None:
    assert (
        format_incorrect_topic_haiku(
            results=pl.DataFrame(
                {
                    "haiku": ["A", "B", "C", "D"],
                    "topic_passed": [True, True, False, False],
                    "topic_target": [False, True, False, True],
                }
            )
        )
        == TOPIC_MSG_2
    )


def test_format_incorrect_topic_haiku_empty() -> None:
    assert (
        format_incorrect_topic_haiku(
            results=pl.DataFrame({"haiku": [], "topic_target": [], "topic_passed": []})
        )
        == TOPIC_MSG_EMPTY
    )
