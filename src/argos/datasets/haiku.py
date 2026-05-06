r"""Contain code to generate a haiku dataset."""

from __future__ import annotations

__all__ = ["generate_haiku_dataset"]


import polars as pl

from argos.autoprompt.haiku import columns


def generate_haiku_dataset() -> pl.DataFrame:
    r"""Generate a labeled dataset of haiku examples.

    The dataset combines positive examples (valid haiku that follow the
    5-7-5 syllable structure and match the topic) with negative
    examples (haiku that fail on structure, topic, or both).

    Returns:
        A :class:`~polars.DataFrame` with the following columns:

        - ``topic`` (``Utf8``): The subject or theme for the haiku.
        - ``haiku`` (``Utf8``): The haiku text, with lines separated
          by newlines.
        - ``structure_target`` (``Boolean``): ``True`` if the haiku
          follows the 5-7-5 syllable structure.
        - ``topic_target`` (``Boolean``): ``True`` if the haiku
          clearly reflects the specified topic.
        - ``overall_target`` (``Boolean``): ``True`` only if both
          ``structure_target`` and ``topic_target`` are ``True``
          (i.e. the overall quality label).

    Example:
        ```pycon
        >>> from argos.datasets import generate_haiku_dataset
        >>> df = generate_haiku_dataset()
        >>> df.columns
        ['topic', 'haiku', 'structure_target', 'topic_target', 'overall_target']

        ```
    """
    return pl.concat([_generate_positive_examples(), _generate_negative_examples()], how="vertical")


def _generate_positive_examples() -> pl.DataFrame:
    r"""Generate positive (valid) haiku examples.

    Each example in the returned DataFrame has a haiku that correctly
    follows the 5-7-5 syllable structure and meaningfully reflects the
    specified topic. All label columns are ``True``.

    Returns:
        A :class:`~polars.DataFrame` with columns ``topic``, ``haiku``,
            ``structure_target``, ``topic_target``, and ``overall_target``.
            All rows have ``structure_target=True``,
            ``topic_target=True``, and ``overall_target=True``.
    """
    return pl.from_dicts(
        [
            # rain
            {
                columns.TOPIC: "rain",
                columns.HAIKU: (
                    "Gray sky descends slow,\n"
                    "Cool drops kiss the thirsty ground,\n"
                    "Silence finds the leaf."
                ),
                columns.STRUCTURE_TARGET: True,
                columns.TOPIC_TARGET: True,
                columns.OVERALL_TARGET: True,
            },
            {
                columns.TOPIC: "rain",
                columns.HAIKU: (
                    "Dark clouds fill the sky\n"
                    "Water falls upon the leaves\n"
                    "Washing the world clean."
                ),
                columns.STRUCTURE_TARGET: True,
                columns.TOPIC_TARGET: True,
                columns.OVERALL_TARGET: True,
            },
            {
                columns.TOPIC: "rain",
                columns.HAIKU: (
                    "Puddles on the ground\nMirrors for the grey heavens\nSplashing as we walk"
                ),
                columns.STRUCTURE_TARGET: True,
                columns.TOPIC_TARGET: True,
                columns.OVERALL_TARGET: True,
            },
            {
                columns.TOPIC: "rain",
                columns.HAIKU: (
                    "Rhythm on the roof\nGentle tapping through the night\nSinging me to sleep"
                ),
                columns.STRUCTURE_TARGET: True,
                columns.TOPIC_TARGET: True,
                columns.OVERALL_TARGET: True,
            },
            {
                columns.TOPIC: "rain",
                columns.HAIKU: (
                    "Storm has passed away\n"
                    "Rainbow colors shining bright\n"
                    "Sunlight breaks the clouds"
                ),
                columns.STRUCTURE_TARGET: True,
                columns.TOPIC_TARGET: True,
                columns.OVERALL_TARGET: True,
            },
            # cat
            {
                columns.TOPIC: "cat",
                columns.HAIKU: (
                    "Soft fur, warm light gleam,\n"
                    "Silent paws upon the floor,\n"
                    "Sunbeam, peace descends."
                ),
                columns.STRUCTURE_TARGET: True,
                columns.TOPIC_TARGET: True,
                columns.OVERALL_TARGET: True,
            },
            {
                columns.TOPIC: "cat",
                columns.HAIKU: "Soft fur drinks the light\n"
                "A curled crescent on the rug\n"
                "Warm sleeping tiger",
                columns.STRUCTURE_TARGET: True,
                columns.TOPIC_TARGET: True,
                columns.OVERALL_TARGET: True,
            },
            {
                columns.TOPIC: "cat",
                columns.HAIKU: (
                    "Two green eyes flash bright\n"
                    "Shadow creeps through midnight grass\n"
                    "A silent, swift pounce"
                ),
                columns.STRUCTURE_TARGET: True,
                columns.TOPIC_TARGET: True,
                columns.OVERALL_TARGET: True,
            },
            {
                columns.TOPIC: "cat",
                columns.HAIKU: "Shadows on the wall\nJumping high to catch a bug\nLanding on its feet",
                columns.STRUCTURE_TARGET: True,
                columns.TOPIC_TARGET: True,
                columns.OVERALL_TARGET: True,
            },
            {
                columns.TOPIC: "cat",
                columns.HAIKU: (
                    "Soft paws on the floor\nChasing after sunny beams\nSleeping all the day"
                ),
                columns.STRUCTURE_TARGET: True,
                columns.TOPIC_TARGET: True,
                columns.OVERALL_TARGET: True,
            },
            {
                columns.TOPIC: "cat",
                columns.HAIKU: "Whiskers in the dark\nSilent hunter strikes so fast\nPurring on my lap",
                columns.STRUCTURE_TARGET: True,
                columns.TOPIC_TARGET: True,
                columns.OVERALL_TARGET: True,
            },
            # mountain
            {
                columns.TOPIC: "mountain",
                columns.HAIKU: (
                    "Snow upon the peak\nClouds are resting on the stone\nQuiet, cold, and still"
                ),
                columns.STRUCTURE_TARGET: True,
                columns.TOPIC_TARGET: True,
                columns.OVERALL_TARGET: True,
            },
            {
                columns.TOPIC: "mountain",
                columns.HAIKU: "Ancient rock so high\nReaching up to touch the sky\nShadows fall below",
                columns.STRUCTURE_TARGET: True,
                columns.TOPIC_TARGET: True,
                columns.OVERALL_TARGET: True,
            },
            {
                columns.TOPIC: "mountain",
                columns.HAIKU: (
                    "Pines upon the slope\nWinds are whispering their song\nEagles taking flight"
                ),
                columns.STRUCTURE_TARGET: True,
                columns.TOPIC_TARGET: True,
                columns.OVERALL_TARGET: True,
            },
            {
                columns.TOPIC: "mountain",
                columns.HAIKU: (
                    "Granite standing firm\nThrough the winter and the storm\nSleeping giant waits"
                ),
                columns.STRUCTURE_TARGET: True,
                columns.TOPIC_TARGET: True,
                columns.OVERALL_TARGET: True,
            },
            # deep ocean
            {
                columns.TOPIC: "ocean",
                columns.HAIKU: (
                    "Blue waves softly sigh,\n"
                    "Salt air, a cool, deep embrace,\n"
                    "Sun melts on the deep."
                ),
                columns.STRUCTURE_TARGET: True,
                columns.TOPIC_TARGET: True,
                columns.OVERALL_TARGET: True,
            },
            {
                columns.TOPIC: "deep ocean",
                columns.HAIKU: "Heavy water waits\nNo sunlight can reach this deep\nSecrets in the dark",
                columns.STRUCTURE_TARGET: True,
                columns.TOPIC_TARGET: True,
                columns.OVERALL_TARGET: True,
            },
            {
                columns.TOPIC: "deep ocean",
                columns.HAIKU: (
                    "Tiny glowing lights\nFlashing in the midnight zone\nStars beneath the sea"
                ),
                columns.STRUCTURE_TARGET: True,
                columns.TOPIC_TARGET: True,
                columns.OVERALL_TARGET: True,
            },
            {
                columns.TOPIC: "deep ocean",
                columns.HAIKU: "Giant shadows move\nSinging songs across the miles\nEchoes in the cold",
                columns.STRUCTURE_TARGET: True,
                columns.TOPIC_TARGET: True,
                columns.OVERALL_TARGET: True,
            },
            {
                columns.TOPIC: "deep ocean",
                columns.HAIKU: (
                    "Silent trench so deep\nWhere the ancient waters rest\nHidden from the world"
                ),
                columns.STRUCTURE_TARGET: True,
                columns.TOPIC_TARGET: True,
                columns.OVERALL_TARGET: True,
            },
            # train journey
            {
                columns.TOPIC: "train journey",
                columns.HAIKU: "Iron wheels roll on\nPassing by the green forests\nLeaving home behind",
                columns.STRUCTURE_TARGET: True,
                columns.TOPIC_TARGET: True,
                columns.OVERALL_TARGET: True,
            },
            {
                columns.TOPIC: "train journey",
                columns.HAIKU: "Whistle in the wind\nCities fade into the night\nSleeping in my seat",
                columns.STRUCTURE_TARGET: True,
                columns.TOPIC_TARGET: True,
                columns.OVERALL_TARGET: True,
            },
            {
                columns.TOPIC: "train journey",
                columns.HAIKU: "Silver metal snake\nRhythm beats against the rail\nCarrying me home",
                columns.STRUCTURE_TARGET: True,
                columns.TOPIC_TARGET: True,
                columns.OVERALL_TARGET: True,
            },
            {
                columns.TOPIC: "train journey",
                columns.HAIKU: (
                    "Mountains in the glass\nShadows stretch across the floor\nSun is going down"
                ),
                columns.STRUCTURE_TARGET: True,
                columns.TOPIC_TARGET: True,
                columns.OVERALL_TARGET: True,
            },
            {
                columns.TOPIC: "train journey",
                columns.HAIKU: "Tickets in my hand\nStrangers sitting in the car\nWaiting for the stop",
                columns.STRUCTURE_TARGET: True,
                columns.TOPIC_TARGET: True,
                columns.OVERALL_TARGET: True,
            },
            # morning coffee
            {
                columns.TOPIC: "morning coffee",
                columns.HAIKU: "Dark and bitter brew\nWaking up my sleepy mind\nWarming up my soul",
                columns.STRUCTURE_TARGET: True,
                columns.TOPIC_TARGET: True,
                columns.OVERALL_TARGET: True,
            },
            {
                columns.TOPIC: "morning coffee",
                columns.HAIKU: (
                    "Steam above the cup\nMorning quiet fills the room\nFirst sip starts the day"
                ),
                columns.STRUCTURE_TARGET: True,
                columns.TOPIC_TARGET: True,
                columns.OVERALL_TARGET: True,
            },
            {
                columns.TOPIC: "morning coffee",
                columns.HAIKU: (
                    "Roasted beans so sweet\nPouring water over grounds\nFragrance in the air"
                ),
                columns.STRUCTURE_TARGET: True,
                columns.TOPIC_TARGET: True,
                columns.OVERALL_TARGET: True,
            },
            {
                columns.TOPIC: "morning coffee",
                columns.HAIKU: "Mug into my hands\nDrinking liquid energy\nReady for the sun",
                columns.STRUCTURE_TARGET: True,
                columns.TOPIC_TARGET: True,
                columns.OVERALL_TARGET: True,
            },
            {
                columns.TOPIC: "morning coffee",
                columns.HAIKU: (
                    "Sunlight strikes the glass\nDrip by drip the pot will fill\nMorning is alive"
                ),
                columns.STRUCTURE_TARGET: True,
                columns.TOPIC_TARGET: True,
                columns.OVERALL_TARGET: True,
            },
            # cherry blossoms
            {
                columns.TOPIC: "cherry blossoms",
                columns.HAIKU: (
                    "Pink buds on the branch\nSoftly waking from the sleep\nSpring has come again"
                ),
                columns.STRUCTURE_TARGET: True,
                columns.TOPIC_TARGET: True,
                columns.OVERALL_TARGET: True,
            },
            {
                columns.TOPIC: "cherry blossoms",
                columns.HAIKU: "Petals on the wind\nDancing through the quiet air\nSnow of rosy light",
                columns.STRUCTURE_TARGET: True,
                columns.TOPIC_TARGET: True,
                columns.OVERALL_TARGET: True,
            },
            {
                columns.TOPIC: "cherry blossoms",
                columns.HAIKU: "Underneath the tree\nFloating on the silver stream\nTime is drifting by",
                columns.STRUCTURE_TARGET: True,
                columns.TOPIC_TARGET: True,
                columns.OVERALL_TARGET: True,
            },
            {
                columns.TOPIC: "cherry blossoms",
                columns.HAIKU: "Brief and lovely bloom\nBeauty fading in the sun\nGone before the night",
                columns.STRUCTURE_TARGET: True,
                columns.TOPIC_TARGET: True,
                columns.OVERALL_TARGET: True,
            },
            {
                columns.TOPIC: "cherry blossoms",
                columns.HAIKU: (
                    "Walking through the park\n"
                    "Clouds of pink above our heads\n"
                    "Earth is breathing sweet"
                ),
                columns.STRUCTURE_TARGET: True,
                columns.TOPIC_TARGET: True,
                columns.OVERALL_TARGET: True,
            },
            # moon
            {
                columns.TOPIC: "moon",
                columns.HAIKU: "Silver orb above\nWatching over sleeping worlds\nSilent light descends",
                columns.STRUCTURE_TARGET: True,
                columns.TOPIC_TARGET: True,
                columns.OVERALL_TARGET: True,
            },
            {
                columns.TOPIC: "moon",
                columns.HAIKU: "Crescent in the dark\nHanging from a velvet sky\nGolden thread of night",
                columns.STRUCTURE_TARGET: True,
                columns.TOPIC_TARGET: True,
                columns.OVERALL_TARGET: True,
            },
            {
                columns.TOPIC: "moon",
                columns.HAIKU: (
                    "Full and bright tonight\nCasting shadows on the snow\nCold and distant glow"
                ),
                columns.STRUCTURE_TARGET: True,
                columns.TOPIC_TARGET: True,
                columns.OVERALL_TARGET: True,
            },
            {
                columns.TOPIC: "moon",
                columns.HAIKU: "Mirror in the stars\nReflecting the hidden sun\nGuardian of dreams",
                columns.STRUCTURE_TARGET: True,
                columns.TOPIC_TARGET: True,
                columns.OVERALL_TARGET: True,
            },
            {
                columns.TOPIC: "moon",
                columns.HAIKU: (
                    "Clouds drift slowly by\nHiding then revealing light\nGhostly face on high"
                ),
                columns.STRUCTURE_TARGET: True,
                columns.TOPIC_TARGET: True,
                columns.OVERALL_TARGET: True,
            },
            # silence
            {
                columns.TOPIC: "silence",
                columns.HAIKU: (
                    "Breath within the dark\nWords are lost in heavy air\nOnly heartbeats stay"
                ),
                columns.STRUCTURE_TARGET: True,
                columns.TOPIC_TARGET: True,
                columns.OVERALL_TARGET: True,
            },
            {
                columns.TOPIC: "silence",
                columns.HAIKU: (
                    "Snow falls on the lake\nMuffling the distant world\nNature holds its breath"
                ),
                columns.STRUCTURE_TARGET: True,
                columns.TOPIC_TARGET: True,
                columns.OVERALL_TARGET: True,
            },
            {
                columns.TOPIC: "silence",
                columns.HAIKU: "Empty room at night\nEchoes of a voice long gone\nQuiet fills the space",
                columns.STRUCTURE_TARGET: True,
                columns.TOPIC_TARGET: True,
                columns.OVERALL_TARGET: True,
            },
            {
                columns.TOPIC: "silence",
                columns.HAIKU: "Stillness in the woods\nEven wind has gone to sleep\nTime begins to fade",
                columns.STRUCTURE_TARGET: True,
                columns.TOPIC_TARGET: True,
                columns.OVERALL_TARGET: True,
            },
            {
                columns.TOPIC: "silence",
                columns.HAIKU: (
                    "Deep beneath the sea\nSunlight fails to reach the floor\nPeace is all there is"
                ),
                columns.STRUCTURE_TARGET: True,
                columns.TOPIC_TARGET: True,
                columns.OVERALL_TARGET: True,
            },
            # colorful leaves
            {
                columns.TOPIC: "colorful leaves",
                columns.HAIKU: (
                    "Crimson, gold, and brown\nFalling to the forest floor\nAutumn's soft descent"
                ),
                columns.STRUCTURE_TARGET: True,
                columns.TOPIC_TARGET: True,
                columns.OVERALL_TARGET: True,
            },
            {
                columns.TOPIC: "colorful leaves",
                columns.HAIKU: (
                    "Fire on the branch\nBurning bright before the cold\nSummer's last goodbye"
                ),
                columns.STRUCTURE_TARGET: True,
                columns.TOPIC_TARGET: True,
                columns.OVERALL_TARGET: True,
            },
            {
                columns.TOPIC: "colorful leaves",
                columns.HAIKU: (
                    "Drifting on the breeze\n"
                    "Painting paths across the grass\n"
                    "Nature's vibrant quilt"
                ),
                columns.STRUCTURE_TARGET: True,
                columns.TOPIC_TARGET: True,
                columns.OVERALL_TARGET: True,
            },
            {
                columns.TOPIC: "colorful leaves",
                columns.HAIKU: (
                    "Crisp beneath my feet\nRustling songs of changing times\nGolden light remains"
                ),
                columns.STRUCTURE_TARGET: True,
                columns.TOPIC_TARGET: True,
                columns.OVERALL_TARGET: True,
            },
            {
                columns.TOPIC: "colorful leaves",
                columns.HAIKU: (
                    "Scarlet maple star\nFloating on the silver pond\nRipple in the glass"
                ),
                columns.STRUCTURE_TARGET: True,
                columns.TOPIC_TARGET: True,
                columns.OVERALL_TARGET: True,
            },
        ]
    )


def _generate_negative_examples() -> pl.DataFrame:
    r"""Generate negative (invalid) haiku examples.

    The negative examples are grouped into three failure modes,
    combined vertically:

    1. Incorrect topic — correct 5-7-5 structure but haiku does not
       match the specified topic.
    2. Incorrect structure — haiku addresses the topic but does not
       follow the 5-7-5 syllable structure.
    3. Incorrect topic and structure — haiku fails on both criteria.

    Returns:
        A :class:`~polars.DataFrame` with columns ``topic``, ``haiku``,
            ``structure_target``, ``topic_target``, and ``target``.
            All rows have ``target=False``.
    """
    return pl.concat(
        [
            _generate_negative_examples_incorrect_topic(),
            _generate_negative_examples_incorrect_structure(),
            _generate_negative_examples_incorrect_topic_and_structure(),
        ],
        how="vertical",
    )


def _generate_negative_examples_incorrect_topic() -> pl.DataFrame:
    r"""Generate negative examples where the topic is incorrect.

    Each example has a haiku with a valid 5-7-5 syllable structure,
    but the haiku was written for a *different* topic than the one
    specified. ``structure_target`` is therefore ``True`` while
    ``topic_target`` and ``target`` are ``False``.

    Returns:
        A :class:`~polars.DataFrame` with columns ``topic``, ``haiku``,
            ``structure_target``, ``topic_target``, and ``target``.
            All rows have ``structure_target=True``,
            ``topic_target=False``, and ``target=False``.
    """
    return pl.from_dicts(
        [
            {
                columns.TOPIC: topic,  # real topic: cat
                columns.HAIKU: (
                    "Soft fur, warm light gleam,\n"
                    "Silent paws upon the floor,\n"
                    "Sunbeam, peace descends."
                ),
                columns.STRUCTURE_TARGET: True,
                columns.TOPIC_TARGET: False,
                columns.OVERALL_TARGET: False,
            }
            for topic in ["rain", "mountain", "ocean", "train journey", "morning coffee"]
        ]
        + [
            {
                columns.TOPIC: topic,  # real topic: rain
                columns.HAIKU: (
                    "Dark clouds fill the sky\n"
                    "Water falls upon the leaves\n"
                    "Washing the world clean."
                ),
                columns.STRUCTURE_TARGET: True,
                columns.TOPIC_TARGET: False,
                columns.OVERALL_TARGET: False,
            }
            for topic in ["cat", "mountain", "ocean", "train journey", "morning coffee"]
        ]
        + [
            {
                columns.TOPIC: topic,  # real topic: mountain
                columns.HAIKU: (
                    "Snow upon the peak\nClouds are resting on the stone\nQuiet, cold, and still"
                ),
                columns.STRUCTURE_TARGET: True,
                columns.TOPIC_TARGET: False,
                columns.OVERALL_TARGET: False,
            }
            for topic in ["cat", "rain", "ocean", "train journey", "morning coffee"]
        ]
        + [
            {
                columns.TOPIC: topic,  # real topic: moon
                columns.HAIKU: (
                    "Silver orb above\nWatching over sleeping worlds\nSilent light descends"
                ),
                columns.STRUCTURE_TARGET: True,
                columns.TOPIC_TARGET: False,
                columns.OVERALL_TARGET: False,
            }
            for topic in ["cat", "rain", "ocean", "train journey", "morning coffee"]
        ]
    )


def _generate_negative_examples_incorrect_structure() -> pl.DataFrame:
    r"""Generate negative examples where the syllable structure is
    incorrect.

    Each example has a haiku that addresses the specified topic but
    does not follow the 5-7-5 syllable structure. ``topic_target`` is
    therefore ``True`` while ``structure_target`` and ``target`` are
    ``False``.

    Returns:
        A :class:`~polars.DataFrame` with columns ``topic``, ``haiku``,
            ``structure_target``, ``topic_target``, and ``target``.
            All rows have ``structure_target=False``,
            ``topic_target=True``, and ``target=False``.
    """
    return pl.from_dicts(
        [
            {
                columns.TOPIC: "cat",
                columns.HAIKU: "meow",
                columns.STRUCTURE_TARGET: False,
                columns.TOPIC_TARGET: True,
                columns.OVERALL_TARGET: False,
            },
            {
                columns.TOPIC: "snow",
                columns.HAIKU: "Footprints in the fresh snow lead directly into the dark woods.",
                columns.STRUCTURE_TARGET: False,
                columns.TOPIC_TARGET: True,
                columns.OVERALL_TARGET: False,
            },
            {
                columns.TOPIC: "deep ocean",
                columns.HAIKU: (
                    "The abyssal weight crushes down in cold, absolute silence.\n"
                    "Yet alien eyes spark like embers in the suffocating dark."
                ),
                columns.STRUCTURE_TARGET: False,
                columns.TOPIC_TARGET: True,
                columns.OVERALL_TARGET: False,
            },
            {
                columns.TOPIC: "deep ocean",
                columns.HAIKU: (
                    "Centuries drift by like slow-falling snow upon the ocean floor.\n"
                    "Forgotten bones become the architecture of a sunless world."
                ),
                columns.STRUCTURE_TARGET: False,
                columns.TOPIC_TARGET: True,
                columns.OVERALL_TARGET: False,
            },
            {
                columns.TOPIC: "deep ocean",
                columns.HAIKU: (
                    "The bathypelagic void stretches out in endless chill,\n"
                    "A liquid cosmos lit by fleeting bioluminescent sparks.\n"
                    "Immense and sightless shadows glide completely still,\n"
                    "Through crushing depths where human exploration rarely marks."
                ),
                columns.STRUCTURE_TARGET: False,
                columns.TOPIC_TARGET: True,
                columns.OVERALL_TARGET: False,
            },
            {
                columns.TOPIC: "deep ocean",
                columns.HAIKU: (
                    "The pressure builds with every sinking league.\n"
                    "Sunlight is a memory, swallowed by the blue,\n"
                    "then the purple, then the absolute black.\n"
                    "Here, the water feels like iron,\n"
                    "heavy, silent, and ancient."
                ),
                columns.STRUCTURE_TARGET: False,
                columns.TOPIC_TARGET: True,
                columns.OVERALL_TARGET: False,
            },
            {
                columns.TOPIC: "deep ocean",
                columns.HAIKU: (
                    "The giant falls slowly through the water column,\n"
                    "a snowstorm of marine dust settling on bones.\n"
                    "In death, it becomes an oasis,\n"
                    "feeding a thousand tiny mouths in the barren wasteland,\n"
                    "a temporary city built upon a single, mighty ribcage."
                ),
                columns.STRUCTURE_TARGET: False,
                columns.TOPIC_TARGET: True,
                columns.OVERALL_TARGET: False,
            },
            {
                columns.TOPIC: "deep ocean",
                columns.HAIKU: (
                    "The turquoise surface fades into a deep, impenetrable violet.\n"
                    "Gravity pulls the submersible down into the crushing, breathless deep.\n"
                    "We are merely fragile visitors intruding on the trench's sleep."
                ),
                columns.STRUCTURE_TARGET: False,
                columns.TOPIC_TARGET: True,
                columns.OVERALL_TARGET: False,
            },
            {
                columns.TOPIC: "deep ocean",
                columns.HAIKU: (
                    "A relentless blizzard of biological dust descends endlessly.\n"
                    "Nourishing the ghostly scavengers that crawl through the abyssal muck.\n"
                    "The cycle of life finding its quiet end at the very bottom of the world."
                ),
                columns.STRUCTURE_TARGET: False,
                columns.TOPIC_TARGET: True,
                columns.OVERALL_TARGET: False,
            },
            {
                columns.TOPIC: "cat",
                columns.HAIKU: "Soft paws on the rug\nPurring in her sleep\nDreaming of a little mouse",
                columns.STRUCTURE_TARGET: False,  # structure is 5, 5, 7
                columns.TOPIC_TARGET: True,
                columns.OVERALL_TARGET: False,
            },
            {
                columns.TOPIC: "cat",
                columns.HAIKU: (
                    "Eyes of glowing green\nWatching from the dark\nReady for a sudden pounce"
                ),
                columns.STRUCTURE_TARGET: False,  # structure is 5, 5, 7
                columns.TOPIC_TARGET: True,
                columns.OVERALL_TARGET: False,
            },
            {
                columns.TOPIC: "cat",
                columns.HAIKU: "Sleeping in the sunny spot\nGolden fur so warm\nPurring all the day",
                columns.STRUCTURE_TARGET: False,  # structure is 7, 5, 5
                columns.TOPIC_TARGET: True,
                columns.OVERALL_TARGET: False,
            },
            {
                columns.TOPIC: "cat",
                columns.HAIKU: "Chasing after flying bugs\nJumping in the air\nLanding on her paws",
                columns.STRUCTURE_TARGET: False,  # structure is 7, 5, 5
                columns.TOPIC_TARGET: True,
                columns.OVERALL_TARGET: False,
            },
            {
                columns.TOPIC: "cat",
                columns.HAIKU: "Soft paws on the rug\nPurring in her sleep\nDreaming of a mouse",
                columns.STRUCTURE_TARGET: False,  # structure is 5, 5, 5
                columns.TOPIC_TARGET: True,
                columns.OVERALL_TARGET: False,
            },
            {
                columns.TOPIC: "cat",
                columns.HAIKU: "Tail is standing high\nRubbing on my leg\nBegging for a treat",
                columns.STRUCTURE_TARGET: False,  # structure is 5, 5, 5
                columns.TOPIC_TARGET: True,
                columns.OVERALL_TARGET: False,
            },
            {
                columns.TOPIC: "cat",
                columns.HAIKU: (
                    "Sleepy kitten in the sun\n"
                    "Stretching out when day is done\n"
                    "Purring softly, having fun"
                ),
                columns.STRUCTURE_TARGET: False,  # structure is 7, 7, 7
                columns.TOPIC_TARGET: True,
                columns.OVERALL_TARGET: False,
            },
            {
                columns.TOPIC: "bakery",
                columns.HAIKU: (
                    "Fresh bread baking in the hot oven\n"
                    "The sweet smell fills the early morning air\n"
                    "Customers line up outside"
                ),
                columns.STRUCTURE_TARGET: False,
                columns.TOPIC_TARGET: True,
                columns.OVERALL_TARGET: False,
            },
            {
                columns.TOPIC: "thunderstorm",
                columns.HAIKU: (
                    "Dark clouds roll over the hills\n"
                    "A sudden flash of lightning strikes the ground\n"
                    "Thunder shakes the window glass"
                ),
                columns.STRUCTURE_TARGET: False,
                columns.TOPIC_TARGET: True,
                columns.OVERALL_TARGET: False,
            },
            {
                columns.TOPIC: "garden",
                columns.HAIKU: (
                    "Weeding the damp soil with bare hands\n"
                    "A bright green tomato still clinging to the vine\n"
                    "Summer is fading away"
                ),
                columns.STRUCTURE_TARGET: False,
                columns.TOPIC_TARGET: True,
                columns.OVERALL_TARGET: False,
            },
            {
                columns.TOPIC: "coffee shop",
                columns.HAIKU: (
                    "The barista calls out a mispronounced name\n"
                    "Espresso machines hiss and steam\n"
                    "Finding an empty corner table"
                ),
                columns.STRUCTURE_TARGET: False,
                columns.TOPIC_TARGET: True,
                columns.OVERALL_TARGET: False,
            },
        ]
    )


def _generate_negative_examples_incorrect_topic_and_structure() -> pl.DataFrame:
    r"""Generate negative examples where both topic and structure are
    incorrect.

    Each example has a haiku that neither follows the 5-7-5 syllable
    structure nor addresses the specified topic. Both
    ``structure_target`` and ``topic_target`` are ``False``.

    Returns:
        A :class:`~polars.DataFrame` with columns ``topic``, ``haiku``,
            ``structure_target``, ``topic_target``, and ``target``.
            All rows have ``structure_target=False``,
            ``topic_target=False``, and ``target=False``.
    """
    return pl.from_dicts(
        [
            {
                columns.TOPIC: "cat",
                columns.HAIKU: (
                    "The abyssal weight crushes down in cold, absolute silence.\n"
                    "Yet alien eyes spark like embers in the suffocating dark."
                ),
                columns.STRUCTURE_TARGET: False,
                columns.TOPIC_TARGET: False,
                columns.OVERALL_TARGET: False,
            },
            {
                columns.TOPIC: "mountain",
                columns.HAIKU: (
                    "The pressure builds with every sinking league.\n"
                    "Sunlight is a memory, swallowed by the blue,\n"
                    "then the purple, then the absolute black.\n"
                    "Here, the water feels like iron,\n"
                    "heavy, silent, and ancient."
                ),
                columns.STRUCTURE_TARGET: False,
                columns.TOPIC_TARGET: False,
                columns.OVERALL_TARGET: False,
            },
            {
                columns.TOPIC: "morning coffee",
                columns.HAIKU: (
                    "The turquoise surface fades into a deep, impenetrable violet.\n"
                    "Gravity pulls the submersible down into the crushing, breathless deep.\n"
                    "We are merely fragile visitors intruding on the trench's sleep."
                ),
                columns.STRUCTURE_TARGET: False,
                columns.TOPIC_TARGET: False,
                columns.OVERALL_TARGET: False,
            },
            {
                columns.TOPIC: "moon",
                columns.HAIKU: "Soft paws on the rug\nPurring in her sleep\nDreaming of a little mouse",
                columns.STRUCTURE_TARGET: False,  # structure is 5, 5, 7
                columns.TOPIC_TARGET: False,
                columns.OVERALL_TARGET: False,
            },
            {
                columns.TOPIC: "train journey",
                columns.HAIKU: "Sleeping in the sunny spot\nGolden fur so warm\nPurring all the day",
                columns.STRUCTURE_TARGET: False,  # structure is 7, 5, 5
                columns.TOPIC_TARGET: False,
                columns.OVERALL_TARGET: False,
            },
            {
                columns.TOPIC: "rain",
                columns.HAIKU: "Tail is standing high\nRubbing on my leg\nBegging for a treat",
                columns.STRUCTURE_TARGET: False,  # structure is 5, 5, 5
                columns.TOPIC_TARGET: False,
                columns.OVERALL_TARGET: False,
            },
            {
                columns.TOPIC: "mountain",
                columns.HAIKU: (
                    "Sleepy kitten in the sun\n"
                    "Stretching out when day is done\n"
                    "Purring softly, having fun"
                ),
                columns.STRUCTURE_TARGET: False,  # structure is 7, 7, 7
                columns.TOPIC_TARGET: False,
                columns.OVERALL_TARGET: False,
            },
            {
                columns.TOPIC: "cat",
                columns.HAIKU: (
                    "Crashing waves against the dark rocks\n"
                    "Salt spray heavy in the air\n"
                    "A lone gull cries out"
                ),
                columns.STRUCTURE_TARGET: False,
                columns.TOPIC_TARGET: False,
                columns.OVERALL_TARGET: False,
            },
            {
                columns.TOPIC: "moon",
                columns.HAIKU: (
                    "The train rattles down the track\n"
                    "Tired faces staring at glowing screens\n"
                    "Rain against the glass"
                ),
                columns.STRUCTURE_TARGET: False,
                columns.TOPIC_TARGET: False,
                columns.OVERALL_TARGET: False,
            },
            {
                columns.TOPIC: "cat",
                columns.HAIKU: (
                    "Snow caps the jagged mountain peaks\n"
                    "Thick clouds roll over the valley below\n"
                    "Perfect silence remains"
                ),
                columns.STRUCTURE_TARGET: False,
                columns.TOPIC_TARGET: False,
                columns.OVERALL_TARGET: False,
            },
        ]
    )
