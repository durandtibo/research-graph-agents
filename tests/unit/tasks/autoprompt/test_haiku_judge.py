from __future__ import annotations

from unittest.mock import patch

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from argos.tasks.autoprompt.haiku_judge import prepare_dataset


@pytest.fixture
def dataset() -> pl.DataFrame:
    return pl.from_dicts(
        [
            {
                "topic": "rain",
                "haiku": (
                    "Gray sky descends slow,\n"
                    "Cool drops kiss the thirsty ground,\n"
                    "Silence finds the leaf."
                ),
                "structure_target": True,
                "topic_target": True,
                "target": True,
            },
            {
                "topic": "cat",
                "haiku": (
                    "Soft fur, warm light gleam,\n"
                    "Silent paws upon the floor,\n"
                    "Sunbeam, peace descends."
                ),
                "structure_target": True,
                "topic_target": True,
                "target": True,
            },
            {
                "topic": "mountain",
                "haiku": (
                    "Snow upon the peak\nClouds are resting on the stone\nQuiet, cold, and still"
                ),
                "structure_target": True,
                "topic_target": True,
                "target": True,
            },
        ]
    )


#####################################
#     Tests for prepare_dataset     #
#####################################


def test_prepare_dataset_returns_dataframe(dataset: pl.DataFrame) -> None:
    with patch("argos.tasks.autoprompt.haiku_judge.generate_haiku_dataset", return_value=dataset):
        assert_frame_equal(prepare_dataset(), dataset)
