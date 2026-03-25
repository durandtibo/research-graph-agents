r"""Contain code to generate a haiku dataset."""

from __future__ import annotations

import logging

import polars as pl
from coola.utils.timing import timeblock
from dotenv import load_dotenv

from argos.utils.dataframe import summarize_boolean_columns
from argos.utils.logging import configure_logging

logger = logging.getLogger(__name__)


def generate_examples() -> pl.DataFrame:
    r"""Generate a list of Haiku examples.

    Returns:
        A list of Haiku examples.
    """
    return pl.concat([generate_positive_examples(), generate_negative_examples()], how="vertical")


def generate_positive_examples() -> pl.DataFrame:
    r"""Generate a list of positive examples.

    Returns:
        A list of positive examples.
    """
    return pl.from_dicts(
        [
            # rain
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
                "topic": "rain",
                "haiku": (
                    "Dark clouds fill the sky\n"
                    "Water falls upon the leaves\n"
                    "Washing the world clean."
                ),
                "structure_target": True,
                "topic_target": True,
                "target": True,
            },
            {
                "topic": "rain",
                "haiku": (
                    "Puddles on the ground\nMirrors for the grey heavens\nSplashing as we walk"
                ),
                "structure_target": True,
                "topic_target": True,
                "target": True,
            },
            {
                "topic": "rain",
                "haiku": (
                    "Rhythm on the roof\nGentle tapping through the night\nSinging me to sleep"
                ),
                "structure_target": True,
                "topic_target": True,
                "target": True,
            },
            {
                "topic": "rain",
                "haiku": (
                    "Storm has passed away\n"
                    "Rainbow colors shining bright\n"
                    "Sunlight breaks the clouds"
                ),
                "structure_target": True,
                "topic_target": True,
                "target": True,
            },
            # cat
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
                "topic": "cat",
                "haiku": "Soft fur drinks the light\n"
                "A curled crescent on the rug\n"
                "Warm sleeping tiger",
                "structure_target": True,
                "topic_target": True,
                "target": True,
            },
            {
                "topic": "cat",
                "haiku": (
                    "Two green eyes flash bright\n"
                    "Shadow creeps through midnight grass\n"
                    "A silent, swift pounce"
                ),
                "structure_target": True,
                "topic_target": True,
                "target": True,
            },
            {
                "topic": "cat",
                "haiku": "Shadows on the wall\nJumping high to catch a bug\nLanding on its feet",
                "structure_target": True,
                "topic_target": True,
                "target": True,
            },
            {
                "topic": "cat",
                "haiku": (
                    "Soft paws on the floor\nChasing after sunny beams\nSleeping all the day"
                ),
                "structure_target": True,
                "topic_target": True,
                "target": True,
            },
            {
                "topic": "cat",
                "haiku": "Whiskers in the dark\nSilent hunter strikes so fast\nPurring on my lap",
                "structure_target": True,
                "topic_target": True,
                "target": True,
            },
            # mountain
            {
                "topic": "mountain",
                "haiku": (
                    "Snow upon the peak\nClouds are resting on the stone\nQuiet, cold, and still"
                ),
                "structure_target": True,
                "topic_target": True,
                "target": True,
            },
            {
                "topic": "mountain",
                "haiku": "Ancient rock so high\nReaching up to touch the sky\nShadows fall below",
                "structure_target": True,
                "topic_target": True,
                "target": True,
            },
            {
                "topic": "mountain",
                "haiku": (
                    "Pines upon the slope\nWinds are whispering their song\nEagles taking flight"
                ),
                "structure_target": True,
                "topic_target": True,
                "target": True,
            },
            {
                "topic": "mountain",
                "haiku": (
                    "Granite standing firm\nThrough the winter and the storm\nSleeping giant waits"
                ),
                "structure_target": True,
                "topic_target": True,
                "target": True,
            },
            # deep ocean
            {
                "topic": "ocean",
                "haiku": (
                    "Blue waves softly sigh,\n"
                    "Salt air, a cool, deep embrace,\n"
                    "Sun melts on the deep."
                ),
                "structure_target": True,
                "topic_target": True,
                "target": True,
            },
            {
                "topic": "deep ocean",
                "haiku": "Heavy water waits\nNo sunlight can reach this deep\nSecrets in the dark",
                "structure_target": True,
                "topic_target": True,
                "target": True,
            },
            {
                "topic": "deep ocean",
                "haiku": (
                    "Tiny glowing lights\nFlashing in the midnight zone\nStars beneath the sea"
                ),
                "structure_target": True,
                "topic_target": True,
                "target": True,
            },
            {
                "topic": "deep ocean",
                "haiku": "Giant shadows move\nSinging songs across the miles\nEchoes in the cold",
                "structure_target": True,
                "topic_target": True,
                "target": True,
            },
            {
                "topic": "deep ocean",
                "haiku": (
                    "Silent trench so deep\nWhere the ancient waters rest\nHidden from the world"
                ),
                "structure_target": True,
                "topic_target": True,
                "target": True,
            },
            # train journey
            {
                "topic": "train journey",
                "haiku": ("Iron wheels roll on\nPassing by the green forests\nLeaving home behind"),
                "structure_target": True,
                "topic_target": True,
                "target": True,
            },
            {
                "topic": "train journey",
                "haiku": ("Whistle in the wind\nCities fade into the night\nSleeping in my seat"),
                "structure_target": True,
                "topic_target": True,
                "target": True,
            },
            {
                "topic": "train journey",
                "haiku": ("Silver metal snake\nRhythm beats against the rail\nCarrying me home"),
                "structure_target": True,
                "topic_target": True,
                "target": True,
            },
            {
                "topic": "train journey",
                "haiku": (
                    "Mountains in the glass\nShadows stretch across the floor\nSun is going down"
                ),
                "structure_target": True,
                "topic_target": True,
                "target": True,
            },
            {
                "topic": "train journey",
                "haiku": ("Tickets in my hand\nStrangers sitting in the car\nWaiting for the stop"),
                "structure_target": True,
                "topic_target": True,
                "target": True,
            },
            # morning coffee
            {
                "topic": "morning coffee",
                "haiku": ("Dark and bitter brew\nWaking up my sleepy mind\nWarming up my soul"),
                "structure_target": True,
                "topic_target": True,
                "target": True,
            },
            {
                "topic": "morning coffee",
                "haiku": (
                    "Steam above the cup\nMorning quiet fills the room\nFirst sip starts the day"
                ),
                "structure_target": True,
                "topic_target": True,
                "target": True,
            },
            {
                "topic": "morning coffee",
                "haiku": (
                    "Roasted beans so sweet\nPouring water over grounds\nFragrance in the air"
                ),
                "structure_target": True,
                "topic_target": True,
                "target": True,
            },
            {
                "topic": "morning coffee",
                "haiku": ("Mug into my hands\nDrinking liquid energy\nReady for the sun"),
                "structure_target": True,
                "topic_target": True,
                "target": True,
            },
            {
                "topic": "morning coffee",
                "haiku": (
                    "Sunlight strikes the glass\nDrip by drip the pot will fill\nMorning is alive"
                ),
                "structure_target": True,
                "topic_target": True,
                "target": True,
            },
        ]
    )


def generate_negative_examples() -> pl.DataFrame:
    r"""Generate a list of negative examples.

    Returns:
        A list of negative examples.
    """
    return pl.concat(
        [
            generate_negative_examples_incorrect_topic(),
            generate_negative_examples_incorrect_structure(),
        ],
        how="vertical",
    )


def generate_negative_examples_incorrect_topic() -> pl.DataFrame:
    r"""Generate a list of negative examples.

    Returns:
        A list of negative examples.
    """
    return pl.from_dicts(
        [
            {
                "topic": topic,  # real topic: cat
                "haiku": (
                    "Soft fur, warm light gleam,\n"
                    "Silent paws upon the floor,\n"
                    "Sunbeam, peace descends."
                ),
                "structure_target": True,
                "topic_target": False,
                "target": False,
            }
            for topic in ["rain", "mountain", "ocean", "train journey", "morning coffee"]
        ]
        + [
            {
                "topic": topic,  # real topic: rain
                "haiku": (
                    "Dark clouds fill the sky\n"
                    "Water falls upon the leaves\n"
                    "Washing the world clean."
                ),
                "structure_target": True,
                "topic_target": False,
                "target": False,
            }
            for topic in ["cat", "mountain", "ocean", "train journey", "morning coffee"]
        ]
        + [
            {
                "topic": topic,  # real topic: mountain
                "haiku": (
                    "Snow upon the peak\nClouds are resting on the stone\nQuiet, cold, and still"
                ),
                "structure_target": True,
                "topic_target": False,
                "target": False,
            }
            for topic in ["cat", "rain", "ocean", "train journey", "morning coffee"]
        ]
    )


def generate_negative_examples_incorrect_structure() -> pl.DataFrame:
    r"""Generate a list of negative examples.

    Returns:
        A list of negative examples.
    """
    return pl.from_dicts(
        [
            {
                "topic": "cat",
                "haiku": "meow",
                "structure_target": False,
                "topic_target": True,
                "target": False,
            },
            {
                "topic": "deep ocean",
                "haiku": (
                    "The abyssal weight crushes down in cold, absolute silence.\n"
                    "Yet alien eyes spark like embers in the suffocating dark."
                ),
                "structure_target": False,
                "topic_target": True,
                "target": False,
            },
            {
                "topic": "deep ocean",
                "haiku": (
                    "Centuries drift by like slow-falling snow upon the ocean floor.\n"
                    "Forgotten bones become the architecture of a sunless world."
                ),
                "structure_target": False,
                "topic_target": True,
                "target": False,
            },
            {
                "topic": "deep ocean",
                "haiku": (
                    "The bathypelagic void stretches out in endless chill,\n"
                    "A liquid cosmos lit by fleeting bioluminescent sparks.\n"
                    "Immense and sightless shadows glide completely still,\n"
                    "Through crushing depths where human exploration rarely marks."
                ),
                "structure_target": False,
                "topic_target": True,
                "target": False,
            },
            {
                "topic": "deep ocean",
                "haiku": (
                    "The pressure builds with every sinking league.\n"
                    "Sunlight is a memory, swallowed by the blue,\n"
                    "then the purple, then the absolute black.\n"
                    "Here, the water feels like iron,\n"
                    "heavy, silent, and ancient."
                ),
                "structure_target": False,
                "topic_target": True,
                "target": False,
            },
            {
                "topic": "deep ocean",
                "haiku": (
                    "The giant falls slowly through the water column,\n"
                    "a snowstorm of marine dust settling on bones.\n"
                    "In death, it becomes an oasis,\n"
                    "feeding a thousand tiny mouths in the barren wasteland,\n"
                    "a temporary city built upon a single, mighty ribcage."
                ),
                "structure_target": False,
                "topic_target": True,
                "target": False,
            },
            {
                "topic": "deep ocean",
                "haiku": (
                    "The turquoise surface fades into a deep, impenetrable violet.\n"
                    "Gravity pulls the submersible down into the crushing, breathless deep.\n"
                    "We are merely fragile visitors intruding on the trench's sleep."
                ),
                "structure_target": False,
                "topic_target": True,
                "target": False,
            },
            {
                "topic": "deep ocean",
                "haiku": (
                    "A relentless blizzard of biological dust descends endlessly.\n"
                    "Nourishing the ghostly scavengers that crawl through the abyssal muck.\n"
                    "The cycle of life finding its quiet end at the very bottom of the world."
                ),
                "structure_target": False,
                "topic_target": True,
                "target": False,
            },
            {
                "topic": "cat",
                "haiku": ("Soft paws on the rug\nPurring in her sleep\nDreaming of a little mouse"),
                "structure_target": False,  # structure is 5, 5, 7
                "topic_target": True,
                "target": False,
            },
            {
                "topic": "cat",
                "haiku": (
                    "Eyes of glowing green\nWatching from the dark\nReady for a sudden pounce"
                ),
                "structure_target": False,  # structure is 5, 5, 7
                "topic_target": True,
                "target": False,
            },
            {
                "topic": "cat",
                "haiku": "Sleeping in the sunny spot\nGolden fur so warm\nPurring all the day",
                "structure_target": False,  # structure is 7, 5, 5
                "topic_target": True,
                "target": False,
            },
            {
                "topic": "cat",
                "haiku": "Chasing after flying bugs\nJumping in the air\nLanding on her paws",
                "structure_target": False,  # structure is 7, 5, 5
                "topic_target": True,
                "target": False,
            },
            {
                "topic": "cat",
                "haiku": "Soft paws on the rug\nPurring in her sleep\nDreaming of a mouse",
                "structure_target": False,  # structure is 5, 5, 5
                "topic_target": True,
                "target": False,
            },
            {
                "topic": "cat",
                "haiku": "Tail is standing high\nRubbing on my leg\nBegging for a treat",
                "structure_target": False,  # structure is 5, 5, 5
                "topic_target": True,
                "target": False,
            },
            {
                "topic": "cat",
                "haiku": (
                    "Sleepy kitten in the sun\n"
                    "Stretching out when day is done\n"
                    "Purring softly, having fun"
                ),
                "structure_target": False,  # structure is 7, 7, 7
                "topic_target": True,
                "target": False,
            },
        ]
    )


def main() -> None:
    r"""Define the main function to generate a haiku dataset."""
    with timeblock(message="Example generation time: {time}"):
        examples = generate_examples()
    with pl.Config(tbl_cols=-1, tbl_rows=50):
        logger.info(f"\n{examples}")

    stats = summarize_boolean_columns(
        examples.select(["target", "structure_target", "topic_target"])
    )
    logger.info(f"\n{stats}")


if __name__ == "__main__":
    configure_logging(level=logging.INFO)
    load_dotenv()

    main()
