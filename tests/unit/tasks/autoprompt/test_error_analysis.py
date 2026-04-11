from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from iden.io import load_json

from argos.tasks.autoprompt.error_analysis import (
    find_errors,
    find_structure_errors,
    format_errors_as_markdown,
    format_errors_as_markdown_table,
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


###########################################
#     Tests for find_structure_errors     #
###########################################


def test_find_structure_errors_all_correct(tmp_path: Path) -> None:
    path = tmp_path.joinpath("data").joinpath("error_analysis_structure.json")
    out = find_structure_errors(
        predictions=pl.DataFrame(
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
        path=path,
    )
    assert out == []
    assert path.is_file()
    assert load_json(path) == []


def test_find_structure_errors_all_incorrect(tmp_path: Path) -> None:
    path = tmp_path.joinpath("data").joinpath("error_analysis_structure.json")
    out = find_structure_errors(
        predictions=pl.DataFrame(
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
        path=path,
    )
    assert out == [
        {"topic": "u", "haiku": "A", "target": False, "prediction": True},
        {"topic": "v", "haiku": "B", "target": False, "prediction": True},
        {"topic": "w", "haiku": "C", "target": True, "prediction": False},
        {"topic": "x", "haiku": "D", "target": True, "prediction": False},
    ]

    assert path.is_file()
    assert load_json(path) == [
        {"topic": "u", "haiku": "A", "target": False, "prediction": True},
        {"topic": "v", "haiku": "B", "target": False, "prediction": True},
        {"topic": "w", "haiku": "C", "target": True, "prediction": False},
        {"topic": "x", "haiku": "D", "target": True, "prediction": False},
    ]


def test_find_structure_errors_empty(tmp_path: Path) -> None:
    path = tmp_path.joinpath("data").joinpath("error_analysis_structure.json")
    out = find_structure_errors(
        predictions=pl.DataFrame(
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
        path=path,
    )
    assert out == []
    assert path.is_file()
    assert load_json(path) == []


def test_find_structure_errors_without_path() -> None:
    out = find_structure_errors(
        predictions=pl.DataFrame(
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
    )
    assert out == []


#################################
#     Tests for find_errors     #
#################################


def test_find_errors_all_correct() -> None:
    assert (
        find_errors(
            predictions=pl.DataFrame(
                {
                    "topic": ["u", "v", "w", "x"],
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
        predictions=pl.DataFrame(
            {
                "topic": ["u", "v", "w", "x"],
                "haiku": ["A", "B", "C", "D"],
                "passed": [True, True, False, False],
                "target": [False, False, True, True],
            }
        ),
        col_target="target",
        col_prediction="passed",
    ) == [
        {"topic": "u", "haiku": "A", "target": False, "prediction": True},
        {"topic": "v", "haiku": "B", "target": False, "prediction": True},
        {"topic": "w", "haiku": "C", "target": True, "prediction": False},
        {"topic": "x", "haiku": "D", "target": True, "prediction": False},
    ]


def test_find_errors_partially_incorrect() -> None:
    assert find_errors(
        predictions=pl.DataFrame(
            {
                "topic": ["u", "v", "w", "x"],
                "haiku": ["A", "B", "C", "D"],
                "passed": [True, True, False, False],
                "target": [False, True, False, True],
            }
        ),
        col_target="target",
        col_prediction="passed",
    ) == [
        {"topic": "u", "haiku": "A", "target": False, "prediction": True},
        {"topic": "x", "haiku": "D", "target": True, "prediction": False},
    ]


def test_find_incorrect_structure_haiku_empty() -> None:
    assert (
        find_errors(
            predictions=pl.DataFrame({"topic": [], "haiku": [], "passed": [], "target": []}),
            col_target="target",
            col_prediction="passed",
        )
        == []
    )


###############################################
#     Tests for format_errors_as_markdown     #
###############################################


def test_format_errors_as_markdown_single_error() -> None:
    assert format_errors_as_markdown(SINGLE_ERROR, error_type="structure") == (
        "1 haikus have incorrect structure predictions. "
        "The table below details these errors:\n"
        f"- **Topic**: The topic of the haiku (valid only if the topic target is true).\n"
        "- **Haiku**: The evaluated text, with line breaks (`\\n`) replaced by slashes (` / `)\n"
        "- **Target**: The true, correct structure label.\n"
        "- **Prediction**: The model's output structure label.\n"
        f"\n{SINGLE_ERROR_TABLE}\n"
    )


def test_format_errors_as_markdown_multiple_errors() -> None:
    assert format_errors_as_markdown(MULTIPLE_ERRORS, error_type="topic") == (
        "3 haikus have incorrect topic predictions. "
        "The table below details these errors:\n"
        f"- **Topic**: The topic of the haiku (valid only if the topic target is true).\n"
        "- **Haiku**: The evaluated text, with line breaks (`\\n`) replaced by slashes (` / `)\n"
        "- **Target**: The true, correct topic label.\n"
        "- **Prediction**: The model's output topic label.\n"
        f"\n{MULTIPLE_ERRORS_TABLE}\n"
    )


#####################################################
#     Tests for format_errors_as_markdown_table     #
#####################################################


def test_format_errors_as_markdown_table_empty_list() -> None:
    assert (
        format_errors_as_markdown_table([])
        == "| # | Topic | Haiku | Target | Prediction |\n|----|----|----|----|----|"
    )


def test_format_errors_as_markdown_table_single_error() -> None:
    assert format_errors_as_markdown_table(SINGLE_ERROR) == SINGLE_ERROR_TABLE


def test_format_errors_as_markdown_table_multiple_errors() -> None:
    assert format_errors_as_markdown_table(MULTIPLE_ERRORS) == MULTIPLE_ERRORS_TABLE
