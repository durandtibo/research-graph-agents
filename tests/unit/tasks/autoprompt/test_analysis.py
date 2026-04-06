from __future__ import annotations

import polars as pl

from argos.tasks.autoprompt.analysis import (
    find_incorrect_predictions,
    format_incorrect_structure_haiku,
    format_incorrect_topic_haiku,
)

################################################
#     Tests for find_incorrect_predictions     #
################################################


def test_find_incorrect_predictions_all_correct() -> None:
    assert (
        find_incorrect_predictions(
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


def test_find_incorrect_predictions_all_incorrect() -> None:
    assert find_incorrect_predictions(
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
        "(haiku): A (target): false (prediction): true",
        "(haiku): B (target): false (prediction): true",
        "(haiku): C (target): true (prediction): false",
        "(haiku): D (target): true (prediction): false",
    ]


def test_find_incorrect_predictions_partially_incorrect() -> None:
    assert find_incorrect_predictions(
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
        "(haiku): A (target): false (prediction): true",
        "(haiku): D (target): true (prediction): false",
    ]


def test_find_incorrect_structure_haiku_empty() -> None:
    assert (
        find_incorrect_predictions(
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
    output = """0 haikus have incorrect structure predictions. Here is the list of haikus with the true label (target) and the predicted label (target):
"""
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
        == output
    )


def test_format_incorrect_structure_haiku_all_incorrect() -> None:
    output = """4 haikus have incorrect structure predictions. Here is the list of haikus with the true label (target) and the predicted label (target):
- (haiku): A (target): false (prediction): true
- (haiku): B (target): false (prediction): true
- (haiku): C (target): true (prediction): false
- (haiku): D (target): true (prediction): false
"""
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
        == output
    )


def test_format_incorrect_structure_haiku_partially_incorrect() -> None:
    output = """2 haikus have incorrect structure predictions. Here is the list of haikus with the true label (target) and the predicted label (target):
- (haiku): A (target): false (prediction): true
- (haiku): D (target): true (prediction): false
"""
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
        == output
    )


def test_format_incorrect_structure_haiku_empty() -> None:
    output = """0 haikus have incorrect structure predictions. Here is the list of haikus with the true label (target) and the predicted label (target):
"""
    assert (
        format_incorrect_structure_haiku(
            results=pl.DataFrame({"haiku": [], "structure_target": [], "structure_passed": []})
        )
        == output
    )


##################################################
#     Tests for format_incorrect_topic_haiku     #
##################################################


def test_format_incorrect_topic_haiku_all_correct() -> None:
    output = """0 haikus have incorrect topic predictions. Here is the list of haikus with the true label (target) and the predicted label (target):
"""
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
        == output
    )


def test_format_incorrect_topic_haiku_all_incorrect() -> None:
    output = """4 haikus have incorrect topic predictions. Here is the list of haikus with the true label (target) and the predicted label (target):
- (haiku): A (target): false (prediction): true
- (haiku): B (target): false (prediction): true
- (haiku): C (target): true (prediction): false
- (haiku): D (target): true (prediction): false
"""
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
        == output
    )


def test_format_incorrect_topic_haiku_partially_incorrect() -> None:
    output = """2 haikus have incorrect topic predictions. Here is the list of haikus with the true label (target) and the predicted label (target):
- (haiku): A (target): false (prediction): true
- (haiku): D (target): true (prediction): false
"""
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
        == output
    )


def test_format_incorrect_topic_haiku_empty() -> None:
    output = """0 haikus have incorrect topic predictions. Here is the list of haikus with the true label (target) and the predicted label (target):
"""
    assert (
        format_incorrect_topic_haiku(
            results=pl.DataFrame({"haiku": [], "topic_target": [], "topic_passed": []})
        )
        == output
    )
