from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from iden.io import load_json

from argos.autoprompt.haiku import columns
from argos.autoprompt.haiku.error_analysis import (
    find_errors,
    find_structure_errors,
    format_errors_as_markdown,
    format_errors_as_markdown_table,
)

if TYPE_CHECKING:
    from pathlib import Path

SINGLE_ERROR = [
    {
        columns.TOPIC: "A",
        columns.HAIKU: "Soft paws on the rug\nPurring in her sleep\nDreaming of a mouse",
        columns.TARGET: False,
        columns.PREDICTION: True,
    }
]

SINGLE_ERROR_TABLE = (
    "| # | Topic | Haiku | Target | Prediction |\n"
    "|----|----|----|----|----|\n"
    "| 1 | A | Soft paws on the rug / Purring in her sleep / Dreaming of a mouse | False | True |"
)

MULTIPLE_ERRORS = [
    {
        columns.TOPIC: "A",
        columns.HAIKU: "Soft paws on the rug\nPurring in her sleep\nDreaming of a mouse",
        columns.TARGET: False,
        columns.PREDICTION: True,
    },
    {
        columns.TOPIC: "B",
        columns.HAIKU: "Eyes of glowing green\nWatching from the dark\nReady for a pounce",
        columns.TARGET: True,
        columns.PREDICTION: False,
    },
    {
        columns.TOPIC: "C",
        columns.HAIKU: "Tail is standing high\nRubbing on my leg\nBegging for a treat",
        columns.TARGET: False,
        columns.PREDICTION: False,
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
                columns.TOPIC: ["u", "v", "w", "x"],
                columns.HAIKU: ["A", "B", "C", "D"],
                columns.STRUCTURE_PREDICTION: [True, True, False, False],
                columns.STRUCTURE_TARGET: [True, True, False, False],
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
                columns.TOPIC: ["u", "v", "w", "x"],
                columns.HAIKU: ["A", "B", "C", "D"],
                columns.STRUCTURE_PREDICTION: [True, True, False, False],
                columns.STRUCTURE_TARGET: [False, False, True, True],
            }
        ),
        path=path,
    )
    assert out == [
        {columns.TOPIC: "u", columns.HAIKU: "A", columns.TARGET: False, columns.PREDICTION: True},
        {columns.TOPIC: "v", columns.HAIKU: "B", columns.TARGET: False, columns.PREDICTION: True},
        {columns.TOPIC: "w", columns.HAIKU: "C", columns.TARGET: True, columns.PREDICTION: False},
        {columns.TOPIC: "x", columns.HAIKU: "D", columns.TARGET: True, columns.PREDICTION: False},
    ]

    assert path.is_file()
    assert load_json(path) == [
        {columns.TOPIC: "u", columns.HAIKU: "A", columns.TARGET: False, columns.PREDICTION: True},
        {columns.TOPIC: "v", columns.HAIKU: "B", columns.TARGET: False, columns.PREDICTION: True},
        {columns.TOPIC: "w", columns.HAIKU: "C", columns.TARGET: True, columns.PREDICTION: False},
        {columns.TOPIC: "x", columns.HAIKU: "D", columns.TARGET: True, columns.PREDICTION: False},
    ]


def test_find_structure_errors_empty(tmp_path: Path) -> None:
    path = tmp_path.joinpath("data").joinpath("error_analysis_structure.json")
    out = find_structure_errors(
        predictions=pl.DataFrame(
            {
                columns.TOPIC: [],
                columns.HAIKU: [],
                columns.STRUCTURE_PREDICTION: [],
                columns.STRUCTURE_TARGET: [],
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
                columns.TOPIC: ["u", "v", "w", "x"],
                columns.HAIKU: ["A", "B", "C", "D"],
                columns.STRUCTURE_PREDICTION: [True, True, False, False],
                columns.STRUCTURE_TARGET: [True, True, False, False],
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
                    columns.TOPIC: ["u", "v", "w", "x"],
                    columns.HAIKU: ["A", "B", "C", "D"],
                    columns.OVERALL_PREDICTION: [True, True, False, False],
                    columns.OVERALL_TARGET: [True, True, False, False],
                }
            ),
            target_col=columns.OVERALL_TARGET,
            prediction_col=columns.OVERALL_PREDICTION,
        )
        == []
    )


def test_find_errors_all_incorrect() -> None:
    assert find_errors(
        predictions=pl.DataFrame(
            {
                columns.TOPIC: ["u", "v", "w", "x"],
                columns.HAIKU: ["A", "B", "C", "D"],
                columns.OVERALL_PREDICTION: [True, True, False, False],
                columns.OVERALL_TARGET: [False, False, True, True],
            }
        ),
        target_col=columns.OVERALL_TARGET,
        prediction_col=columns.OVERALL_PREDICTION,
    ) == [
        {columns.TOPIC: "u", columns.HAIKU: "A", columns.TARGET: False, columns.PREDICTION: True},
        {columns.TOPIC: "v", columns.HAIKU: "B", columns.TARGET: False, columns.PREDICTION: True},
        {columns.TOPIC: "w", columns.HAIKU: "C", columns.TARGET: True, columns.PREDICTION: False},
        {columns.TOPIC: "x", columns.HAIKU: "D", columns.TARGET: True, columns.PREDICTION: False},
    ]


def test_find_errors_partially_incorrect() -> None:
    assert find_errors(
        predictions=pl.DataFrame(
            {
                columns.TOPIC: ["u", "v", "w", "x"],
                columns.HAIKU: ["A", "B", "C", "D"],
                columns.OVERALL_PREDICTION: [True, True, False, False],
                columns.OVERALL_TARGET: [False, True, False, True],
            }
        ),
        target_col=columns.OVERALL_TARGET,
        prediction_col=columns.OVERALL_PREDICTION,
    ) == [
        {columns.TOPIC: "u", columns.HAIKU: "A", columns.TARGET: False, columns.PREDICTION: True},
        {columns.TOPIC: "x", columns.HAIKU: "D", columns.TARGET: True, columns.PREDICTION: False},
    ]


def test_find_incorrect_structure_haiku_empty() -> None:
    assert (
        find_errors(
            predictions=pl.DataFrame(
                {
                    columns.TOPIC: [],
                    columns.HAIKU: [],
                    columns.OVERALL_PREDICTION: [],
                    columns.OVERALL_TARGET: [],
                }
            ),
            target_col=columns.OVERALL_TARGET,
            prediction_col=columns.OVERALL_PREDICTION,
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
    assert format_errors_as_markdown(MULTIPLE_ERRORS, error_type=columns.TOPIC) == (
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
