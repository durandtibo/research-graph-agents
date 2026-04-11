from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from iden.io import load_json

from argos.autoprompt.haiku.analysis import (
    analyze_errors,
)

if TYPE_CHECKING:
    from pathlib import Path


SINGLE_ERROR = [
    {
        "topic": "A",
        "haiku": "Soft paws on the rug\nPurring in her sleep\nDreaming of a mouse",
        "target": False,
        "prediction": True,
    }
]

SINGLE_ERROR_TABLE = (
    "| # | Topic | Haiku | Target | Prediction |\n"
    "|----|----|----|----|----|\n"
    "| 1 | A | Soft paws on the rug / Purring in her sleep / Dreaming of a mouse | False | True |"
)

MULTIPLE_ERRORS = [
    {
        "topic": "A",
        "haiku": "Soft paws on the rug\nPurring in her sleep\nDreaming of a mouse",
        "target": False,
        "prediction": True,
    },
    {
        "topic": "B",
        "haiku": "Eyes of glowing green\nWatching from the dark\nReady for a pounce",
        "target": True,
        "prediction": False,
    },
    {
        "topic": "C",
        "haiku": "Tail is standing high\nRubbing on my leg\nBegging for a treat",
        "target": False,
        "prediction": False,
    },
]

MULTIPLE_ERRORS_TABLE = (
    "| # | Topic | Haiku | Target | Prediction |\n"
    "|----|----|----|----|----|\n"
    "| 1 | A | Soft paws on the rug / Purring in her sleep / Dreaming of a mouse | False | True |\n"
    "| 2 | B | Eyes of glowing green / Watching from the dark / Ready for a pounce | True | False |\n"
    "| 3 | C | Tail is standing high / Rubbing on my leg / Begging for a treat | False | False |"
)


####################################
#     Tests for analyze_errors     #
####################################


def test_analyze_errors_all_correct(tmp_path: Path) -> None:
    analyze_errors(
        results=pl.DataFrame(
            {
                "topic": ["u", "v", "w", "x"],
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
    file_struct = tmp_path.joinpath("data").joinpath("error_analysis_structure.json")
    assert file_struct.is_file()
    assert load_json(file_struct) == []

    file_topic = tmp_path.joinpath("data").joinpath("error_analysis_topic.json")
    assert file_topic.is_file()
    assert load_json(file_topic) == []


def test_analyze_errors_all_incorrect(tmp_path: Path) -> None:
    analyze_errors(
        results=pl.DataFrame(
            {
                "topic": ["u", "v", "w", "x"],
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
    file_struct = tmp_path.joinpath("data").joinpath("error_analysis_structure.json")
    assert file_struct.is_file()
    assert load_json(file_struct) == [
        {"topic": "u", "haiku": "A", "target": False, "prediction": True},
        {"topic": "v", "haiku": "B", "target": False, "prediction": True},
        {"topic": "w", "haiku": "C", "target": True, "prediction": False},
        {"topic": "x", "haiku": "D", "target": True, "prediction": False},
    ]

    file_topic = tmp_path.joinpath("data").joinpath("error_analysis_topic.json")
    assert file_topic.is_file()
    assert load_json(file_topic) == [
        {"topic": "u", "haiku": "A", "target": False, "prediction": True},
        {"topic": "v", "haiku": "B", "target": False, "prediction": True},
        {"topic": "w", "haiku": "C", "target": True, "prediction": False},
        {"topic": "x", "haiku": "D", "target": True, "prediction": False},
    ]


def test_analyze_errors_empty(tmp_path: Path) -> None:
    analyze_errors(
        results=pl.DataFrame(
            {
                "topic": [],
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
    file_struct = tmp_path.joinpath("data").joinpath("error_analysis_structure.json")
    assert file_struct.is_file()
    assert load_json(file_struct) == []

    file_topic = tmp_path.joinpath("data").joinpath("error_analysis_topic.json")
    assert file_topic.is_file()
    assert load_json(file_topic) == []
