from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from iden.io import load_json

from argos.autoprompt.haiku import columns
from argos.autoprompt.haiku.error_analysis import (
    find_errors,
    find_structure_errors,
    find_topic_errors,
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
        columns.REASONING: "blabla...",
    }
]

SINGLE_ERROR_TABLE = (
    "| # | Topic | Haiku | Target | Prediction | Reasoning |\n"
    "|----|----|----|----|----|----|\n"
    "| 1 | A | Soft paws on the rug / Purring in her sleep / Dreaming of a mouse | False | True | blabla... |"
)

MULTIPLE_ERRORS = [
    {
        columns.TOPIC: "A",
        columns.HAIKU: "Soft paws on the rug\nPurring in her sleep\nDreaming of a mouse",
        columns.TARGET: False,
        columns.PREDICTION: True,
        columns.REASONING: "bla...",
    },
    {
        columns.TOPIC: "B",
        columns.HAIKU: "Eyes of glowing green\nWatching from the dark\nReady for a pounce",
        columns.TARGET: True,
        columns.PREDICTION: False,
        columns.REASONING: "blabla...",
    },
    {
        columns.TOPIC: "C",
        columns.HAIKU: "Tail is standing high\nRubbing on my leg\nBegging for a treat",
        columns.TARGET: False,
        columns.PREDICTION: False,
        columns.REASONING: "blablabla...",
    },
]

MULTIPLE_ERRORS_TABLE = (
    "| # | Topic | Haiku | Target | Prediction | Reasoning |\n"
    "|----|----|----|----|----|----|\n"
    "| 1 | A | Soft paws on the rug / Purring in her sleep / Dreaming of a mouse | False | True | bla... |\n"
    "| 2 | B | Eyes of glowing green / Watching from the dark / Ready for a pounce | True | False | blabla... |\n"
    "| 3 | C | Tail is standing high / Rubbing on my leg / Begging for a treat | False | False | blablabla... |"
)


###########################################
#     Tests for find_structure_errors     #
###########################################


def test_find_structure_errors_all_correct(tmp_path: Path) -> None:
    path = tmp_path.joinpath("data").joinpath("error.json")
    out = find_structure_errors(
        predictions=pl.DataFrame(
            {
                columns.TOPIC: ["t1", "t2", "t3", "t4"],
                columns.HAIKU: ["h1", "h2", "h3", "h4"],
                columns.STRUCTURE_PREDICTION: [True, True, False, False],
                columns.STRUCTURE_TARGET: [True, True, False, False],
                columns.STRUCTURE_REASONING: ["r1", "r2", "r3", "r4"],
            }
        ),
        path=path,
    )
    assert out == []
    assert path.is_file()
    assert load_json(path) == []


def test_find_structure_errors_all_incorrect(tmp_path: Path) -> None:
    path = tmp_path.joinpath("data").joinpath("error.json")
    out = find_structure_errors(
        predictions=pl.DataFrame(
            {
                columns.TOPIC: ["t1", "t2", "t3", "t4"],
                columns.HAIKU: ["h1", "h2", "h3", "h4"],
                columns.STRUCTURE_PREDICTION: [True, True, False, False],
                columns.STRUCTURE_TARGET: [False, False, True, True],
                columns.STRUCTURE_REASONING: ["r1", "r2", "r3", "r4"],
            }
        ),
        path=path,
    )

    expected = [
        {
            columns.TOPIC: "t1",
            columns.HAIKU: "h1",
            columns.TARGET: False,
            columns.PREDICTION: True,
            columns.REASONING: "r1",
        },
        {
            columns.TOPIC: "t2",
            columns.HAIKU: "h2",
            columns.TARGET: False,
            columns.PREDICTION: True,
            columns.REASONING: "r2",
        },
        {
            columns.TOPIC: "t3",
            columns.HAIKU: "h3",
            columns.TARGET: True,
            columns.PREDICTION: False,
            columns.REASONING: "r3",
        },
        {
            columns.TOPIC: "t4",
            columns.HAIKU: "h4",
            columns.TARGET: True,
            columns.PREDICTION: False,
            columns.REASONING: "r4",
        },
    ]
    assert out == expected
    assert path.is_file()
    assert load_json(path) == expected


def test_find_structure_errors_empty(tmp_path: Path) -> None:
    path = tmp_path.joinpath("data").joinpath("error.json")
    out = find_structure_errors(
        predictions=pl.DataFrame(
            {
                columns.TOPIC: [],
                columns.HAIKU: [],
                columns.STRUCTURE_PREDICTION: [],
                columns.STRUCTURE_TARGET: [],
                columns.STRUCTURE_REASONING: [],
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
                columns.TOPIC: ["t1", "t2", "t3", "t4"],
                columns.HAIKU: ["h1", "h2", "h3", "h4"],
                columns.STRUCTURE_PREDICTION: [True, True, False, False],
                columns.STRUCTURE_TARGET: [True, True, False, False],
                columns.STRUCTURE_REASONING: ["r1", "r2", "r3", "r4"],
            }
        ),
    )
    assert out == []


#######################################
#     Tests for find_topic_errors     #
#######################################


def test_find_topic_errors_all_correct(tmp_path: Path) -> None:
    path = tmp_path.joinpath("data").joinpath("error.json")
    out = find_topic_errors(
        predictions=pl.DataFrame(
            {
                columns.TOPIC: ["t1", "t2", "t3", "t4"],
                columns.HAIKU: ["h1", "h2", "h3", "h4"],
                columns.TOPIC_PREDICTION: [True, True, False, False],
                columns.TOPIC_TARGET: [True, True, False, False],
                columns.TOPIC_REASONING: ["r1", "r2", "r3", "r4"],
            }
        ),
        path=path,
    )
    assert out == []
    assert path.is_file()
    assert load_json(path) == []


def test_find_topic_errors_all_incorrect(tmp_path: Path) -> None:
    path = tmp_path.joinpath("data").joinpath("error.json")
    out = find_topic_errors(
        predictions=pl.DataFrame(
            {
                columns.TOPIC: ["t1", "t2", "t3", "t4"],
                columns.HAIKU: ["h1", "h2", "h3", "h4"],
                columns.TOPIC_PREDICTION: [True, True, False, False],
                columns.TOPIC_TARGET: [False, False, True, True],
                columns.TOPIC_REASONING: ["r1", "r2", "r3", "r4"],
            }
        ),
        path=path,
    )
    expected = [
        {
            columns.TOPIC: "t1",
            columns.HAIKU: "h1",
            columns.TARGET: False,
            columns.PREDICTION: True,
            columns.REASONING: "r1",
        },
        {
            columns.TOPIC: "t2",
            columns.HAIKU: "h2",
            columns.TARGET: False,
            columns.PREDICTION: True,
            columns.REASONING: "r2",
        },
        {
            columns.TOPIC: "t3",
            columns.HAIKU: "h3",
            columns.TARGET: True,
            columns.PREDICTION: False,
            columns.REASONING: "r3",
        },
        {
            columns.TOPIC: "t4",
            columns.HAIKU: "h4",
            columns.TARGET: True,
            columns.PREDICTION: False,
            columns.REASONING: "r4",
        },
    ]
    assert out == expected
    assert path.is_file()
    assert load_json(path) == expected


def test_find_topic_errors_empty(tmp_path: Path) -> None:
    path = tmp_path.joinpath("data").joinpath("error.json")
    out = find_topic_errors(
        predictions=pl.DataFrame(
            {
                columns.TOPIC: [],
                columns.HAIKU: [],
                columns.TOPIC_PREDICTION: [],
                columns.TOPIC_TARGET: [],
                columns.TOPIC_REASONING: [],
            }
        ),
        path=path,
    )
    assert out == []
    assert path.is_file()
    assert load_json(path) == []


def test_find_topic_errors_without_path() -> None:
    out = find_topic_errors(
        predictions=pl.DataFrame(
            {
                columns.TOPIC: ["t1", "t2", "t3", "t4"],
                columns.HAIKU: ["h1", "h2", "h3", "h4"],
                columns.TOPIC_PREDICTION: [True, True, False, False],
                columns.TOPIC_TARGET: [True, True, False, False],
                columns.TOPIC_REASONING: ["r1", "r2", "r3", "r4"],
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
                    columns.TOPIC: ["t1", "t2", "t3", "t4"],
                    columns.HAIKU: ["h1", "h2", "h3", "h4"],
                    "prediction": [True, True, False, False],
                    "target": [True, True, False, False],
                    "reasoning": ["r1", "r2", "r3", "r4"],
                }
            ),
            target_col="target",
            prediction_col="prediction",
            reasoning_col="reasoning",
        )
        == []
    )


def test_find_errors_all_incorrect() -> None:
    assert find_errors(
        predictions=pl.DataFrame(
            {
                columns.TOPIC: ["t1", "t2", "t3", "t4"],
                columns.HAIKU: ["h1", "h2", "h3", "h4"],
                "prediction": [True, True, False, False],
                "target": [False, False, True, True],
                "reasoning": ["r1", "r2", "r3", "r4"],
            }
        ),
        target_col="target",
        prediction_col="prediction",
        reasoning_col="reasoning",
    ) == [
        {
            columns.TOPIC: "t1",
            columns.HAIKU: "h1",
            columns.TARGET: False,
            columns.PREDICTION: True,
            columns.REASONING: "r1",
        },
        {
            columns.TOPIC: "t2",
            columns.HAIKU: "h2",
            columns.TARGET: False,
            columns.PREDICTION: True,
            columns.REASONING: "r2",
        },
        {
            columns.TOPIC: "t3",
            columns.HAIKU: "h3",
            columns.TARGET: True,
            columns.PREDICTION: False,
            columns.REASONING: "r3",
        },
        {
            columns.TOPIC: "t4",
            columns.HAIKU: "h4",
            columns.TARGET: True,
            columns.PREDICTION: False,
            columns.REASONING: "r4",
        },
    ]


def test_find_errors_partially_incorrect() -> None:
    assert find_errors(
        predictions=pl.DataFrame(
            {
                columns.TOPIC: ["t1", "t2", "t3", "t4"],
                columns.HAIKU: ["h1", "h2", "h3", "h4"],
                "prediction": [True, True, False, False],
                "target": [False, True, False, True],
                "reasoning": ["r1", "r2", "r3", "r4"],
            }
        ),
        target_col="target",
        prediction_col="prediction",
        reasoning_col="reasoning",
    ) == [
        {
            columns.TOPIC: "t1",
            columns.HAIKU: "h1",
            columns.TARGET: False,
            columns.PREDICTION: True,
            columns.REASONING: "r1",
        },
        {
            columns.TOPIC: "t4",
            columns.HAIKU: "h4",
            columns.TARGET: True,
            columns.PREDICTION: False,
            columns.REASONING: "r4",
        },
    ]


def test_find_incorrect_structure_haiku_empty() -> None:
    assert (
        find_errors(
            predictions=pl.DataFrame(
                {
                    columns.TOPIC: [],
                    columns.HAIKU: [],
                    "prediction": [],
                    "target": [],
                    "reasoning": [],
                }
            ),
            target_col="target",
            prediction_col="prediction",
            reasoning_col="reasoning",
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
        "- **Reasoning**: The explanation behind the model's prediction.\n"
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
        "- **Reasoning**: The explanation behind the model's prediction.\n"
        f"\n{MULTIPLE_ERRORS_TABLE}\n"
    )


#####################################################
#     Tests for format_errors_as_markdown_table     #
#####################################################


def test_format_errors_as_markdown_table_empty_list() -> None:
    assert (
        format_errors_as_markdown_table([])
        == "| # | Topic | Haiku | Target | Prediction | Reasoning |\n|----|----|----|----|----|----|"
    )


def test_format_errors_as_markdown_table_single_error() -> None:
    assert format_errors_as_markdown_table(SINGLE_ERROR) == SINGLE_ERROR_TABLE


def test_format_errors_as_markdown_table_multiple_errors() -> None:
    assert format_errors_as_markdown_table(MULTIPLE_ERRORS) == MULTIPLE_ERRORS_TABLE
