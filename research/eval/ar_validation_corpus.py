"""Natural-language corpus for the champion AR Validation probe.

The corpus is built from story-local codebooks.  A key phrase such as
``red lantern`` maps to a value word only inside the current story, with
related noise sentences that reuse the same key/value vocabulary without
creating stable global key->value facts.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

AR_VALIDATION_STORY_CORPUS_VERSION = "ar_validation_story_corpus_v1"

DEFAULT_BINDINGS_PER_STORY = 16
DEFAULT_NOISE_SENTENCES_PER_STORY = 48
DEFAULT_QUERIES_PER_STORY = 2
DEFAULT_TRAIN_STORIES = 128
DEFAULT_IN_DIST_EVAL_STORIES = 0
DEFAULT_HELD_KEY_STORIES = 32
DEFAULT_CROSS_STORY_GROUPS = 16

COLORS: tuple[str, ...] = (
    "red",
    "blue",
    "green",
    "white",
    "black",
    "yellow",
    "orange",
    "purple",
    "silver",
    "gold",
    "copper",
    "brass",
    "glass",
    "paper",
    "wooden",
    "iron",
    "scarlet",
    "navy",
    "teal",
    "violet",
    "amber",
    "ivory",
    "crimson",
    "indigo",
    "olive",
    "pearl",
    "coral",
    "bronze",
    "charcoal",
    "cream",
    "rose",
    "jade",
    "ruby",
    "saffron",
    "plum",
    "linen",
    "ashen",
    "cobalt",
    "mint",
    "maroon",
    "tan",
    "khaki",
    "gray",
    "pink",
    "brown",
    "azure",
    "beige",
    "moss",
    "rust",
    "snow",
    "slate",
    "lilac",
    "lemon",
    "ochre",
    "denim",
    "clay",
    "opal",
    "walnut",
    "flax",
    "sky",
    "fern",
    "graphite",
    "honey",
    "wine",
)

OBJECTS: tuple[str, ...] = (
    "lantern",
    "marble",
    "button",
    "ticket",
    "ribbon",
    "candle",
    "pebble",
    "feather",
    "coin",
    "mask",
    "shell",
    "key",
    "ladder",
    "mirror",
    "basket",
    "needle",
    "crown",
    "bell",
    "spoon",
    "wheel",
    "compass",
    "ring",
    "window",
    "book",
    "card",
    "flag",
    "box",
    "cup",
    "stone",
    "leaf",
    "star",
    "flower",
    "anchor",
    "bottle",
    "brush",
    "bucket",
    "chain",
    "clock",
    "drum",
    "flute",
    "glove",
    "hammer",
    "hinge",
    "jacket",
    "kettle",
    "lock",
    "magnet",
    "net",
    "oar",
    "pencil",
    "quill",
    "rope",
    "saddle",
    "tablet",
    "umbrella",
    "vase",
    "whistle",
    "yarn",
    "zipper",
    "badge",
    "bead",
    "bridge",
    "cap",
    "diary",
    "envelope",
    "frame",
    "gate",
    "hook",
    "jar",
    "kite",
    "lamp",
    "map",
    "nest",
    "orb",
    "pin",
    "quilt",
    "rail",
    "scarf",
    "tile",
    "urn",
    "vest",
    "wand",
    "xylophone",
    "yoke",
    "zither",
    "acorn",
    "banner",
    "cloak",
    "dagger",
    "easel",
    "funnel",
    "goblet",
    "helmet",
    "inkwell",
    "journal",
    "knob",
    "lens",
    "mug",
    "notebook",
    "pouch",
    "rudder",
    "satchel",
    "thimble",
    "valve",
    "wallet",
    "amulet",
    "buckle",
    "charm",
    "drawer",
    "emblem",
    "folder",
    "grill",
    "handle",
    "island",
    "jewel",
    "kernel",
    "lever",
    "medal",
    "napkin",
    "parcel",
    "quartz",
    "reel",
    "signal",
    "token",
    "vault",
    "wedge",
    "tablet",
    "marker",
)

VALUE_WORDS: tuple[str, ...] = (
    "cedar",
    "ivy",
    "coral",
    "amber",
    "onyx",
    "violet",
    "maple",
    "pearl",
    "juniper",
    "clover",
    "willow",
    "sapphire",
    "basalt",
    "meadow",
    "silver",
    "harbor",
    "opal",
    "forest",
    "copper",
    "river",
    "laurel",
    "quartz",
    "velvet",
    "granite",
    "orchid",
    "bronze",
    "summit",
    "crystal",
    "linen",
    "hazel",
    "marigold",
    "slate",
    "topaz",
    "elm",
    "brook",
    "jasper",
    "fern",
    "garnet",
    "briar",
    "cotton",
    "flint",
    "moss",
    "agate",
    "pine",
    "lilac",
    "walnut",
    "ruby",
    "birch",
    "lagoon",
    "plum",
    "canyon",
    "oasis",
    "iris",
    "hemlock",
    "gull",
    "ember",
    "lichen",
    "marble",
    "thistle",
    "cobalt",
    "acacia",
    "dune",
    "grove",
    "kelp",
    "mercury",
    "nectar",
    "obsidian",
    "prairie",
    "radish",
    "sequoia",
    "tulip",
    "umber",
    "valley",
    "wicker",
    "yarrow",
    "zephyr",
    "almond",
    "basil",
    "citrine",
    "dahlia",
    "egret",
    "fig",
    "ginger",
    "holly",
    "islet",
    "jasmine",
    "kiwi",
    "lotus",
    "mango",
    "nutmeg",
    "olive",
    "paprika",
    "quince",
    "rosemary",
    "sorrel",
    "teak",
    "upland",
    "vervain",
    "wisteria",
    "xylem",
    "yucca",
    "zinnia",
)

GPT2_SAFE_VALUE_WORDS: tuple[str, ...] = (
    "apple",
    "river",
    "garden",
    "silver",
    "forest",
    "window",
    "flower",
    "cotton",
    "basket",
    "button",
    "ticket",
    "coin",
    "shell",
    "mirror",
    "compass",
    "wheel",
    "book",
    "card",
    "flag",
    "cup",
    "stone",
    "leaf",
    "star",
    "anchor",
    "bottle",
    "brush",
    "bucket",
    "chain",
    "clock",
    "drum",
    "hammer",
    "jacket",
    "lock",
    "tablet",
    "badge",
    "cap",
    "frame",
    "gate",
    "hook",
    "jar",
    "lamp",
    "map",
    "nest",
    "pin",
    "rail",
    "tile",
    "vest",
    "wand",
    "banner",
    "journal",
    "lens",
    "signal",
    "token",
    "marker",
    "bread",
    "cloud",
    "field",
    "grass",
    "hill",
    "island",
    "lemon",
    "metal",
    "morning",
    "paper",
    "planet",
    "water",
    "winter",
    "summer",
    "spring",
    "music",
    "story",
    "table",
    "chair",
    "house",
    "road",
    "train",
    "light",
    "shadow",
    "circle",
    "square",
    "number",
    "letter",
)

BINDING_TEMPLATES: tuple[str, ...] = (
    "In this story, the {key} means {value}.",
    "For this story, the {key} points to {value}.",
    "The code for the {key} is {value}.",
    "Remember that the {key} maps to {value}.",
    "Inside this story, the {key} names {value}.",
)

QUERY_TEMPLATES: tuple[str, ...] = (
    "Question: In Story {story_id}, what does the {key} mean?",
    "Question: For this story, which word belongs to the {key}?",
    "Question: What is the code word for the {key} in Story {story_id}?",
    "Question: Inside this story, the {key} maps to what?",
)

KEY_ONLY_NOISE: tuple[str, ...] = (
    "The {key} was placed near the quiet shelf.",
    "Someone moved the {key} before the final note.",
    "A sketch of the {key} appeared in the margin.",
)

VALUE_ONLY_NOISE: tuple[str, ...] = (
    "{value} appeared on a separate label.",
    "The word {value} was copied into the ledger.",
    "A traveler mentioned {value} during the pause.",
)

KEY_VALUE_DECOY_NOISE: tuple[str, ...] = (
    "The {key} was not connected to {wrong_value}.",
    "A note compared the {key} with {wrong_value}, but gave no code.",
    "The {key} and {wrong_value} appeared in the same picture only by chance.",
)

PAIR_NOISE: tuple[str, ...] = (
    "The {key_a} was listed before the {key_b}.",
    "{value_a} and {value_b} were both written on the page.",
    "The {key_a} was stored beside the {key_b}, away from the answer line.",
)

CROSS_STORY_NOISE: tuple[str, ...] = (
    "In another story, the {key} meant {other_value}.",
    "A different story used the {key} for {other_value}.",
)


@dataclass(frozen=True, slots=True)
class StoryKey:
    color: str
    obj: str

    @property
    def text(self) -> str:
        return f"{self.color} {self.obj}"


@dataclass(frozen=True, slots=True)
class StoryBinding:
    key: StoryKey
    value: str
    template: str

    def sentence(self) -> str:
        return self.template.format(key=self.key.text, value=self.value)


@dataclass(frozen=True, slots=True)
class StoryQuery:
    story_id: int
    split: str
    key: StoryKey
    answer: str
    prompt: str
    choices: tuple[str, ...]

    def answer_line(self) -> str:
        return f"Answer: {self.answer}."


@dataclass(frozen=True, slots=True)
class StoryExample:
    story_id: int
    split: str
    bindings: tuple[StoryBinding, ...]
    noise_sentences: tuple[str, ...]
    queries: tuple[StoryQuery, ...]

    def context_sentences(self) -> tuple[str, ...]:
        return tuple(b.sentence() for b in self.bindings) + self.noise_sentences

    def text(self, *, include_answers: bool = True) -> str:
        lines = [f"Story {self.story_id}.", *self.context_sentences()]
        for query in self.queries:
            lines.append(query.prompt)
            if include_answers:
                lines.append(query.answer_line())
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class StoryCorpusSpec:
    version: str
    seed: int
    train_stories: tuple[StoryExample, ...]
    eval_stories: tuple[StoryExample, ...]
    held_key_phrases: frozenset[str]

    @property
    def stories(self) -> tuple[StoryExample, ...]:
        return self.train_stories + self.eval_stories

    @property
    def train_texts(self) -> tuple[str, ...]:
        return tuple(story.text(include_answers=True) for story in self.train_stories)

    @property
    def eval_queries(self) -> tuple[StoryQuery, ...]:
        return tuple(query for story in self.eval_stories for query in story.queries)


@dataclass(frozen=True, slots=True)
class ARValidationStoryCorpusConfig:
    seed: int = 0
    n_train_stories: int = DEFAULT_TRAIN_STORIES
    n_in_dist_eval_stories: int = DEFAULT_IN_DIST_EVAL_STORIES
    n_held_key_stories: int = DEFAULT_HELD_KEY_STORIES
    n_cross_story_groups: int = DEFAULT_CROSS_STORY_GROUPS
    bindings_per_story: int = DEFAULT_BINDINGS_PER_STORY
    noise_sentences_per_story: int = DEFAULT_NOISE_SENTENCES_PER_STORY
    queries_per_story: int = DEFAULT_QUERIES_PER_STORY
    choices_per_query: int = 4
    n_colors: int = 64
    n_objects: int = 128
    n_values: int = 96


def _key_space(cfg: ARValidationStoryCorpusConfig) -> list[StoryKey]:
    colors = COLORS[: int(cfg.n_colors)]
    objects = OBJECTS[: int(cfg.n_objects)]
    if not colors or not objects:
        raise ValueError("color and object pools must be non-empty")
    return [StoryKey(color, obj) for color in colors for obj in objects]


def _value_space(cfg: ARValidationStoryCorpusConfig) -> list[str]:
    deduped = dict.fromkeys((*GPT2_SAFE_VALUE_WORDS, *VALUE_WORDS))
    values = list(deduped)[: int(cfg.n_values)]
    if len(values) < int(cfg.bindings_per_story):
        raise ValueError("n_values must cover bindings_per_story")
    return values


def _choices(
    answer: str,
    story_values: Sequence[str],
    *,
    rng: random.Random,
    n_choices: int = 4,
) -> tuple[str, ...]:
    if int(n_choices) < 2:
        raise ValueError("choices_per_query must be at least 2")
    if int(n_choices) > len(story_values):
        raise ValueError("choices_per_query cannot exceed bindings_per_story")
    distractors = [value for value in story_values if value != answer]
    rng.shuffle(distractors)
    out = [answer, *distractors[: max(0, int(n_choices) - 1)]]
    rng.shuffle(out)
    return tuple(out)


def _make_bindings(
    keys: Sequence[StoryKey],
    values: Sequence[str],
    *,
    rng: random.Random,
) -> tuple[StoryBinding, ...]:
    if len(keys) != len(values):
        raise ValueError("keys and values must have equal length")
    bindings: list[StoryBinding] = []
    for idx, (key, value) in enumerate(zip(keys, values)):
        template = BINDING_TEMPLATES[idx % len(BINDING_TEMPLATES)]
        bindings.append(StoryBinding(key=key, value=value, template=template))
    rng.shuffle(bindings)
    return tuple(bindings)


def _noise_sentences(
    bindings: Sequence[StoryBinding],
    values: Sequence[str],
    *,
    story_id: int,
    rng: random.Random,
    n_noise: int,
    cross_story_lookup: dict[str, str] | None = None,
) -> tuple[str, ...]:
    out: list[str] = []
    bindings_list = list(bindings)
    value_list = list(values)
    while len(out) < int(n_noise):
        idx = len(out) % 5
        binding = bindings_list[len(out) % len(bindings_list)]
        if idx == 0:
            template = rng.choice(KEY_ONLY_NOISE)
            out.append(template.format(key=binding.key.text))
        elif idx == 1:
            template = rng.choice(VALUE_ONLY_NOISE)
            out.append(template.format(value=binding.value))
        elif idx == 2:
            wrong_values = [value for value in value_list if value != binding.value]
            wrong = rng.choice(wrong_values)
            template = rng.choice(KEY_VALUE_DECOY_NOISE)
            out.append(template.format(key=binding.key.text, wrong_value=wrong))
        elif idx == 3:
            other = bindings_list[(len(out) * 7 + 3) % len(bindings_list)]
            template = rng.choice(PAIR_NOISE)
            out.append(
                template.format(
                    key_a=binding.key.text,
                    key_b=other.key.text,
                    value_a=binding.value,
                    value_b=other.value,
                )
            )
        else:
            other_value = None
            if cross_story_lookup is not None:
                other_value = cross_story_lookup.get(binding.key.text)
            if other_value is None or other_value == binding.value:
                wrong_values = [value for value in value_list if value != binding.value]
                other_value = rng.choice(wrong_values)
            template = rng.choice(CROSS_STORY_NOISE)
            out.append(template.format(key=binding.key.text, other_value=other_value))
    rng.shuffle(out)
    return tuple(out)


def _queries(
    bindings: Sequence[StoryBinding],
    *,
    story_id: int,
    split: str,
    rng: random.Random,
    n_queries: int,
    cfg: ARValidationStoryCorpusConfig,
) -> tuple[StoryQuery, ...]:
    picks = list(bindings)
    rng.shuffle(picks)
    story_values = [binding.value for binding in bindings]
    queries: list[StoryQuery] = []
    for idx, binding in enumerate(picks[: int(n_queries)]):
        template = QUERY_TEMPLATES[idx % len(QUERY_TEMPLATES)]
        prompt = template.format(story_id=story_id, key=binding.key.text)
        queries.append(
            StoryQuery(
                story_id=story_id,
                split=split,
                key=binding.key,
                answer=binding.value,
                prompt=prompt,
                choices=_choices(
                    binding.value,
                    story_values,
                    rng=rng,
                    n_choices=int(cfg.choices_per_query),
                ),
            )
        )
    return tuple(queries)


def _story(
    *,
    story_id: int,
    split: str,
    keys: Sequence[StoryKey],
    values: Sequence[str],
    rng: random.Random,
    cfg: ARValidationStoryCorpusConfig,
    cross_story_lookup: dict[str, str] | None = None,
) -> StoryExample:
    bindings = _make_bindings(keys, values, rng=rng)
    noise = _noise_sentences(
        bindings,
        values,
        story_id=story_id,
        rng=rng,
        n_noise=int(cfg.noise_sentences_per_story),
        cross_story_lookup=cross_story_lookup,
    )
    return StoryExample(
        story_id=story_id,
        split=split,
        bindings=bindings,
        noise_sentences=noise,
        queries=_queries(
            bindings,
            story_id=story_id,
            split=split,
            rng=rng,
            n_queries=int(cfg.queries_per_story),
            cfg=cfg,
        ),
    )


def build_ar_validation_story_corpus(
    cfg: ARValidationStoryCorpusConfig | None = None,
) -> StoryCorpusSpec:
    """Build deterministic train/eval stories for AR Validation v3 calibration."""
    cfg = cfg or ARValidationStoryCorpusConfig()
    if int(cfg.bindings_per_story) <= 0:
        raise ValueError("bindings_per_story must be positive")
    if int(cfg.queries_per_story) <= 0:
        raise ValueError("queries_per_story must be positive")
    if int(cfg.queries_per_story) > int(cfg.bindings_per_story):
        raise ValueError("queries_per_story cannot exceed bindings_per_story")
    if int(cfg.choices_per_query) < 2:
        raise ValueError("choices_per_query must be at least 2")
    if int(cfg.choices_per_query) > int(cfg.bindings_per_story):
        raise ValueError("choices_per_query cannot exceed bindings_per_story")

    rng = random.Random(int(cfg.seed))
    keys = _key_space(cfg)
    rng.shuffle(keys)
    values = _value_space(cfg)
    n_bindings = int(cfg.bindings_per_story)
    n_held_keys = max(n_bindings, int(cfg.n_held_key_stories) * n_bindings)
    if len(keys) < n_held_keys + int(cfg.n_train_stories) * n_bindings:
        raise ValueError("key pool too small for requested train and held stories")

    held_keys = keys[:n_held_keys]
    train_keys = keys[n_held_keys:]
    held_key_phrases = frozenset(key.text for key in held_keys)

    stories: list[StoryExample] = []
    cursor = 0
    for story_idx in range(int(cfg.n_train_stories)):
        story_keys = train_keys[cursor : cursor + n_bindings]
        if len(story_keys) != n_bindings:
            raise ValueError("key pool too small for requested train stories")
        cursor += n_bindings
        story_values = rng.sample(values, n_bindings)
        stories.append(
            _story(
                story_id=story_idx,
                split="train",
                keys=story_keys,
                values=story_values,
                rng=rng,
                cfg=cfg,
            )
        )

    eval_stories: list[StoryExample] = []
    base_eval_id = int(cfg.n_train_stories)
    train_seen_keys = train_keys[:cursor]
    for idx in range(int(cfg.n_in_dist_eval_stories)):
        if len(train_seen_keys) < n_bindings:
            raise ValueError("at least one train story is required for in_dist eval")
        story_keys = rng.sample(train_seen_keys, n_bindings)
        story_values = rng.sample(values, n_bindings)
        eval_stories.append(
            _story(
                story_id=base_eval_id + idx,
                split="in_dist",
                keys=story_keys,
                values=story_values,
                rng=rng,
                cfg=cfg,
            )
        )

    held_cursor = 0
    held_base_id = base_eval_id + int(cfg.n_in_dist_eval_stories)
    for idx in range(int(cfg.n_held_key_stories)):
        story_keys = held_keys[held_cursor : held_cursor + n_bindings]
        held_cursor += n_bindings
        story_values = rng.sample(values, n_bindings)
        eval_stories.append(
            _story(
                story_id=held_base_id + idx,
                split="held_key",
                keys=story_keys,
                values=story_values,
                rng=rng,
                cfg=cfg,
            )
        )

    cross_start = held_base_id + int(cfg.n_held_key_stories)
    for group_idx in range(int(cfg.n_cross_story_groups)):
        story_keys = rng.sample(train_keys[: max(cursor, n_bindings)], n_bindings)
        first_values = rng.sample(values, n_bindings)
        second_values = rng.sample(
            [value for value in values if value not in set(first_values)],
            n_bindings,
        )
        lookup_a = {key.text: value for key, value in zip(story_keys, second_values)}
        lookup_b = {key.text: value for key, value in zip(story_keys, first_values)}
        eval_stories.append(
            _story(
                story_id=cross_start + group_idx * 2,
                split="cross_story",
                keys=story_keys,
                values=first_values,
                rng=rng,
                cfg=cfg,
                cross_story_lookup=lookup_a,
            )
        )
        eval_stories.append(
            _story(
                story_id=cross_start + group_idx * 2 + 1,
                split="cross_story",
                keys=story_keys,
                values=second_values,
                rng=rng,
                cfg=cfg,
                cross_story_lookup=lookup_b,
            )
        )

    return StoryCorpusSpec(
        version=AR_VALIDATION_STORY_CORPUS_VERSION,
        seed=int(cfg.seed),
        train_stories=tuple(stories),
        eval_stories=tuple(eval_stories),
        held_key_phrases=held_key_phrases,
    )


__all__ = [
    "AR_VALIDATION_STORY_CORPUS_VERSION",
    "ARValidationStoryCorpusConfig",
    "StoryBinding",
    "StoryCorpusSpec",
    "StoryExample",
    "StoryKey",
    "StoryQuery",
    "build_ar_validation_story_corpus",
]
