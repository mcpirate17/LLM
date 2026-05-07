from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def _tiny_config():
    from research.eval.small_ar_champion_corpus import SmallARStoryCorpusConfig

    return SmallARStoryCorpusConfig(
        seed=7,
        n_train_stories=4,
        n_in_dist_eval_stories=1,
        n_held_key_stories=2,
        n_cross_story_groups=1,
        bindings_per_story=6,
        noise_sentences_per_story=18,
        queries_per_story=2,
        n_values=32,
    )


def test_story_corpus_builds_natural_language_story_shape():
    from research.eval.small_ar_champion_corpus import (
        SMALL_AR_STORY_CORPUS_VERSION,
        build_small_ar_story_corpus,
    )

    spec = build_small_ar_story_corpus(_tiny_config())
    story = spec.train_stories[0]

    assert spec.version == SMALL_AR_STORY_CORPUS_VERSION
    assert len(story.bindings) == 6
    assert len(story.noise_sentences) == 18
    assert len(story.queries) == 2
    text = story.text()
    assert text.startswith("Story 0.\n")
    assert "Question:" in text
    assert "Answer:" in text
    assert any("In this story" in binding.sentence() for binding in story.bindings)


def test_held_key_split_uses_keys_absent_from_training_stories():
    from research.eval.small_ar_champion_corpus import build_small_ar_story_corpus

    spec = build_small_ar_story_corpus(_tiny_config())
    train_keys = {
        binding.key.text for story in spec.train_stories for binding in story.bindings
    }
    held_key_eval_keys = {
        binding.key.text
        for story in spec.eval_stories
        if story.split == "held_key"
        for binding in story.bindings
    }

    assert held_key_eval_keys
    assert held_key_eval_keys.issubset(spec.held_key_phrases)
    assert train_keys.isdisjoint(held_key_eval_keys)


def test_in_dist_eval_reuses_train_key_surfaces_with_fresh_story_values():
    from research.eval.small_ar_champion_corpus import build_small_ar_story_corpus

    spec = build_small_ar_story_corpus(_tiny_config())
    train_keys = {
        binding.key.text for story in spec.train_stories for binding in story.bindings
    }
    in_dist = [story for story in spec.eval_stories if story.split == "in_dist"]

    assert in_dist
    for story in in_dist:
        assert {binding.key.text for binding in story.bindings}.issubset(train_keys)
        assert all(query.split == "in_dist" for query in story.queries)


def test_story_values_are_episodic_under_cross_story_interference():
    from research.eval.small_ar_champion_corpus import build_small_ar_story_corpus

    spec = build_small_ar_story_corpus(_tiny_config())
    cross = [story for story in spec.eval_stories if story.split == "cross_story"]

    assert len(cross) == 2
    left = {binding.key.text: binding.value for binding in cross[0].bindings}
    right = {binding.key.text: binding.value for binding in cross[1].bindings}
    assert set(left) == set(right)
    assert all(left[key] != right[key] for key in left)
    assert any(
        "another story" in s or "different story" in s for s in cross[0].noise_sentences
    )


def test_noise_reuses_related_words_without_replacing_binding_sentences():
    from research.eval.small_ar_champion_corpus import build_small_ar_story_corpus

    spec = build_small_ar_story_corpus(_tiny_config())
    story = spec.train_stories[0]
    binding_sentences = {binding.sentence() for binding in story.bindings}

    assert binding_sentences.isdisjoint(story.noise_sentences)
    assert any(
        "not connected" in s or "gave no code" in s for s in story.noise_sentences
    )
    assert any("appeared" in s or "written" in s for s in story.noise_sentences)


def test_queries_include_in_episode_choices_and_answer():
    from research.eval.small_ar_champion_corpus import build_small_ar_story_corpus

    spec = build_small_ar_story_corpus(_tiny_config())
    query = spec.eval_queries[0]
    story = next(
        story for story in spec.eval_stories if story.story_id == query.story_id
    )
    story_values = {binding.value for binding in story.bindings}

    assert query.answer in query.choices
    assert set(query.choices).issubset(story_values)
    assert len(query.choices) == 4
    assert query.key.text in query.prompt
    assert query.answer_line() == f"Answer: {query.answer}."


def test_binary_choices_keep_answer_and_one_in_episode_distractor():
    from research.eval.small_ar_champion_corpus import (
        SmallARStoryCorpusConfig,
        build_small_ar_story_corpus,
    )

    spec = build_small_ar_story_corpus(
        SmallARStoryCorpusConfig(
            seed=11,
            n_train_stories=2,
            n_in_dist_eval_stories=1,
            n_held_key_stories=0,
            n_cross_story_groups=0,
            bindings_per_story=6,
            noise_sentences_per_story=0,
            queries_per_story=3,
            choices_per_query=2,
            n_values=24,
        )
    )
    query = spec.eval_queries[0]
    story = next(
        story for story in spec.eval_stories if story.story_id == query.story_id
    )
    story_values = {binding.value for binding in story.bindings}

    assert len(query.choices) == 2
    assert query.answer in query.choices
    assert set(query.choices).issubset(story_values)
    assert len(set(query.choices)) == 2


def test_default_value_pool_stays_distinct_under_gpt2_checkpoint_vocab():
    from research.eval.small_ar_champion_corpus import (
        SmallARStoryCorpusConfig,
        _value_space,
    )
    from research.eval.utils import tokenize_string

    values = _value_space(SmallARStoryCorpusConfig(n_values=32))
    suffixes = {
        tuple(int(i) for i in tokenize_string(f" {value}.", 32_000)) for value in values
    }

    assert len(values) == 32
    assert len(suffixes) == len(values)
    assert all(max(suffix) < 32_000 for suffix in suffixes)
