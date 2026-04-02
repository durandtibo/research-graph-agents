r"""Contain code to generate a haiku dataset."""

from __future__ import annotations

__all__ = ["generate_haiku_dataset"]


import polars as pl


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
        - ``target`` (``Boolean``): ``True`` only if both
          ``structure_target`` and ``topic_target`` are ``True``
          (i.e. the overall quality label).

    Example:
        ```pycon
        >>> from argos.datasets import generate_haiku_dataset
        >>> df = generate_haiku_dataset()
        >>> df.columns
        ['topic', 'haiku', 'structure_target', 'topic_target', 'target']

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
            ``structure_target``, ``topic_target``, and ``target``.
            All rows have ``structure_target=True``,
            ``topic_target=True``, and ``target=True``.
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
                "haiku": "Iron wheels roll on\nPassing by the green forests\nLeaving home behind",
                "structure_target": True,
                "topic_target": True,
                "target": True,
            },
            {
                "topic": "train journey",
                "haiku": "Whistle in the wind\nCities fade into the night\nSleeping in my seat",
                "structure_target": True,
                "topic_target": True,
                "target": True,
            },
            {
                "topic": "train journey",
                "haiku": "Silver metal snake\nRhythm beats against the rail\nCarrying me home",
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
                "haiku": "Tickets in my hand\nStrangers sitting in the car\nWaiting for the stop",
                "structure_target": True,
                "topic_target": True,
                "target": True,
            },
            # morning coffee
            {
                "topic": "morning coffee",
                "haiku": "Dark and bitter brew\nWaking up my sleepy mind\nWarming up my soul",
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
                "haiku": "Mug into my hands\nDrinking liquid energy\nReady for the sun",
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
            # cherry blossoms
            {
                "topic": "cherry blossoms",
                "haiku": (
                    "Pink buds on the branch\nSoftly waking from the sleep\nSpring has come again"
                ),
                "structure_target": True,
                "topic_target": True,
                "target": True,
            },
            {
                "topic": "cherry blossoms",
                "haiku": "Petals on the wind\nDancing through the quiet air\nSnow of rosy light",
                "structure_target": True,
                "topic_target": True,
                "target": True,
            },
            {
                "topic": "cherry blossoms",
                "haiku": "Underneath the tree\nFloating on the silver stream\nTime is drifting by",
                "structure_target": True,
                "topic_target": True,
                "target": True,
            },
            {
                "topic": "cherry blossoms",
                "haiku": "Brief and lovely bloom\nBeauty fading in the sun\nGone before the night",
                "structure_target": True,
                "topic_target": True,
                "target": True,
            },
            {
                "topic": "cherry blossoms",
                "haiku": (
                    "Walking through the park\n"
                    "Clouds of pink above our heads\n"
                    "Earth is breathing sweet"
                ),
                "structure_target": True,
                "topic_target": True,
                "target": True,
            },
            # moon
            {
                "topic": "moon",
                "haiku": "Silver orb above\nWatching over sleeping worlds\nSilent light descends",
                "structure_target": True,
                "topic_target": True,
                "target": True,
            },
            {
                "topic": "moon",
                "haiku": "Crescent in the dark\nHanging from a velvet sky\nGolden thread of night",
                "structure_target": True,
                "topic_target": True,
                "target": True,
            },
            {
                "topic": "moon",
                "haiku": (
                    "Full and bright tonight\nCasting shadows on the snow\nCold and distant glow"
                ),
                "structure_target": True,
                "topic_target": True,
                "target": True,
            },
            {
                "topic": "moon",
                "haiku": "Mirror in the stars\nReflecting the hidden sun\nGuardian of dreams",
                "structure_target": True,
                "topic_target": True,
                "target": True,
            },
            {
                "topic": "moon",
                "haiku": (
                    "Clouds drift slowly by\nHiding then revealing light\nGhostly face on high"
                ),
                "structure_target": True,
                "topic_target": True,
                "target": True,
            },
            # silence
            {
                "topic": "silence",
                "haiku": (
                    "Breath within the dark\nWords are lost in heavy air\nOnly heartbeats stay"
                ),
                "structure_target": True,
                "topic_target": True,
                "target": True,
            },
            {
                "topic": "silence",
                "haiku": (
                    "Snow falls on the lake\nMuffling the distant world\nNature holds its breath"
                ),
                "structure_target": True,
                "topic_target": True,
                "target": True,
            },
            {
                "topic": "silence",
                "haiku": "Empty room at night\nEchoes of a voice long gone\nQuiet fills the space",
                "structure_target": True,
                "topic_target": True,
                "target": True,
            },
            {
                "topic": "silence",
                "haiku": "Stillness in the woods\nEven wind has gone to sleep\nTime begins to fade",
                "structure_target": True,
                "topic_target": True,
                "target": True,
            },
            {
                "topic": "silence",
                "haiku": (
                    "Deep beneath the sea\nSunlight fails to reach the floor\nPeace is all there is"
                ),
                "structure_target": True,
                "topic_target": True,
                "target": True,
            },
            # colorful leaves
            {
                "topic": "colorful leaves",
                "haiku": (
                    "Crimson, gold, and brown\nFalling to the forest floor\nAutumn's soft descent"
                ),
                "structure_target": True,
                "topic_target": True,
                "target": True,
            },
            {
                "topic": "colorful leaves",
                "haiku": (
                    "Fire on the branch\nBurning bright before the cold\nSummer's last goodbye"
                ),
                "structure_target": True,
                "topic_target": True,
                "target": True,
            },
            {
                "topic": "colorful leaves",
                "haiku": (
                    "Drifting on the breeze\n"
                    "Painting paths across the grass\n"
                    "Nature's vibrant quilt"
                ),
                "structure_target": True,
                "topic_target": True,
                "target": True,
            },
            {
                "topic": "colorful leaves",
                "haiku": (
                    "Crisp beneath my feet\nRustling songs of changing times\nGolden light remains"
                ),
                "structure_target": True,
                "topic_target": True,
                "target": True,
            },
            {
                "topic": "colorful leaves",
                "haiku": ("Scarlet maple star\nFloating on the silver pond\nRipple in the glass"),
                "structure_target": True,
                "topic_target": True,
                "target": True,
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
        + [
            {
                "topic": topic,  # real topic: moon
                "haiku": ("Silver orb above\nWatching over sleeping worlds\nSilent light descends"),
                "structure_target": True,
                "topic_target": False,
                "target": False,
            }
            for topic in ["cat", "rain", "ocean", "train journey", "morning coffee"]
        ]
    )


def _generate_negative_examples_incorrect_structure() -> pl.DataFrame:
    r"""Generate negative examples where the syllable structure is incorrect.

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
                "topic": "cat",
                "haiku": "meow",
                "structure_target": False,
                "topic_target": True,
                "target": False,
            },
            {
                "topic": "snow",
                "haiku": "Footprints in the fresh snow lead directly into the dark woods.",
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
                "haiku": "Soft paws on the rug\nPurring in her sleep\nDreaming of a little mouse",
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
            {
                "topic": "bakery",
                "haiku": (
                    "Fresh bread baking in the hot oven\n"
                    "The sweet smell fills the early morning air\n"
                    "Customers line up outside"
                ),
                "structure_target": False,
                "topic_target": True,
                "target": False,
            },
            {
                "topic": "thunderstorm",
                "haiku": (
                    "Dark clouds roll over the hills\n"
                    "A sudden flash of lightning strikes the ground\n"
                    "Thunder shakes the window glass"
                ),
                "structure_target": False,
                "topic_target": True,
                "target": False,
            },
            {
                "topic": "garden",
                "haiku": (
                    "Weeding the damp soil with bare hands\n"
                    "A bright green tomato still clinging to the vine\n"
                    "Summer is fading away"
                ),
                "structure_target": False,
                "topic_target": True,
                "target": False,
            },
            {
                "topic": "coffee shop",
                "haiku": (
                    "The barista calls out a mispronounced name\n"
                    "Espresso machines hiss and steam\n"
                    "Finding an empty corner table"
                ),
                "structure_target": False,
                "topic_target": True,
                "target": False,
            },
        ]
    )


def _generate_negative_examples_incorrect_topic_and_structure() -> pl.DataFrame:
    r"""Generate negative examples where both topic and structure are incorrect.

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
                "topic": "cat",
                "haiku": (
                    "The abyssal weight crushes down in cold, absolute silence.\n"
                    "Yet alien eyes spark like embers in the suffocating dark."
                ),
                "structure_target": False,
                "topic_target": False,
                "target": False,
            },
            {
                "topic": "mountain",
                "haiku": (
                    "The pressure builds with every sinking league.\n"
                    "Sunlight is a memory, swallowed by the blue,\n"
                    "then the purple, then the absolute black.\n"
                    "Here, the water feels like iron,\n"
                    "heavy, silent, and ancient."
                ),
                "structure_target": False,
                "topic_target": False,
                "target": False,
            },
            {
                "topic": "morning coffee",
                "haiku": (
                    "The turquoise surface fades into a deep, impenetrable violet.\n"
                    "Gravity pulls the submersible down into the crushing, breathless deep.\n"
                    "We are merely fragile visitors intruding on the trench's sleep."
                ),
                "structure_target": False,
                "topic_target": False,
                "target": False,
            },
            {
                "topic": "moon",
                "haiku": "Soft paws on the rug\nPurring in her sleep\nDreaming of a little mouse",
                "structure_target": False,  # structure is 5, 5, 7
                "topic_target": False,
                "target": False,
            },
            {
                "topic": "train journey",
                "haiku": "Sleeping in the sunny spot\nGolden fur so warm\nPurring all the day",
                "structure_target": False,  # structure is 7, 5, 5
                "topic_target": False,
                "target": False,
            },
            {
                "topic": "rain",
                "haiku": "Tail is standing high\nRubbing on my leg\nBegging for a treat",
                "structure_target": False,  # structure is 5, 5, 5
                "topic_target": False,
                "target": False,
            },
            {
                "topic": "mountain",
                "haiku": (
                    "Sleepy kitten in the sun\n"
                    "Stretching out when day is done\n"
                    "Purring softly, having fun"
                ),
                "structure_target": False,  # structure is 7, 7, 7
                "topic_target": False,
                "target": False,
            },
            {
                "topic": "cat",
                "haiku": (
                    "Crashing waves against the dark rocks\n"
                    "Salt spray heavy in the air\n"
                    "A lone gull cries out"
                ),
                "structure_target": False,
                "topic_target": False,
                "target": False,
            },
            {
                "topic": "moon",
                "haiku": (
                    "The train rattles down the track\n"
                    "Tired faces staring at glowing screens\n"
                    "Rain against the glass"
                ),
                "structure_target": False,
                "topic_target": False,
                "target": False,
            },
            {
                "topic": "cat",
                "haiku": (
                    "Snow caps the jagged mountain peaks\n"
                    "Thick clouds roll over the valley below\n"
                    "Perfect silence remains"
                ),
                "structure_target": False,
                "topic_target": False,
                "target": False,
            },
        ]
    )
