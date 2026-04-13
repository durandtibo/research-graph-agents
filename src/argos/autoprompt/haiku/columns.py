r"""Contain column name constants for the haiku autoprompt pipeline.

These constants are used as column names in :class:`~polars.DataFrame`
objects throughout the prediction, evaluation, and error-analysis stages.
"""

__all__ = [
    "HAIKU",
    "OVERALL_PREDICTION",
    "OVERALL_REASONING",
    "OVERALL_TARGET",
    "PREDICTION",
    "REASONING",
    "SCORE_PREDICTION",
    "SCORE_REASONING",
    "SCORE_TARGET",
    "STRUCTURE_PREDICTION",
    "STRUCTURE_REASONING",
    "STRUCTURE_TARGET",
    "TARGET",
    "TOPIC",
    "TOPIC_PREDICTION",
    "TOPIC_REASONING",
    "TOPIC_TARGET",
]

HAIKU = "haiku"
PREDICTION = "prediction"
REASONING = "reasoning"
TARGET = "target"
TOPIC = "topic"

OVERALL_PREDICTION = "overall_prediction"
OVERALL_REASONING = "overall_reasoning"
OVERALL_TARGET = "overall_target"

SCORE_PREDICTION = "score_prediction"
SCORE_REASONING = "score_reasoning"
SCORE_TARGET = "score_target"

STRUCTURE_PREDICTION = "structure_prediction"
STRUCTURE_REASONING = "structure_reasoning"
STRUCTURE_TARGET = "structure_target"

TOPIC_PREDICTION = "topic_prediction"
TOPIC_REASONING = "topic_reasoning"
TOPIC_TARGET = "topic_target"
