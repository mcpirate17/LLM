from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_story_calibration_encodes_bpe_items_without_byte_tokenizer():
    from research.eval.small_ar_champion_corpus import (
        SmallARStoryCorpusConfig,
        build_small_ar_story_corpus,
    )
    from research.eval.utils import _get_tiktoken_encoder
    from research.tools.small_ar_story_calibration import (
        TIKTOKEN_ENCODING,
        encode_story_items,
    )

    corpus = build_small_ar_story_corpus(
        SmallARStoryCorpusConfig(
            seed=3,
            n_train_stories=2,
            n_held_key_stories=1,
            n_cross_story_groups=1,
            bindings_per_story=4,
            noise_sentences_per_story=8,
            queries_per_story=1,
            n_values=16,
        )
    )
    enc = _get_tiktoken_encoder(TIKTOKEN_ENCODING)
    items = encode_story_items(corpus.train_stories, enc)

    assert items
    assert TIKTOKEN_ENCODING == "cl100k_base"
    assert all(item.prefix_ids for item in items)
    assert all(item.answer_ids for item in items)
    assert all(max(item.full_ids) < enc.n_vocab for item in items)
    assert "Answer:" in enc.decode(items[0].prefix_ids)


def test_in_story_unqueried_queries_hold_out_some_train_story_bindings():
    from research.eval.small_ar_champion_corpus import (
        SmallARStoryCorpusConfig,
        build_small_ar_story_corpus,
    )
    from research.tools.small_ar_story_calibration import in_story_unqueried_queries

    corpus = build_small_ar_story_corpus(
        SmallARStoryCorpusConfig(
            seed=4,
            n_train_stories=2,
            n_in_dist_eval_stories=0,
            n_held_key_stories=0,
            n_cross_story_groups=0,
            bindings_per_story=6,
            noise_sentences_per_story=0,
            queries_per_story=2,
            n_values=16,
        )
    )

    held_queries = in_story_unqueried_queries(corpus.train_stories)

    assert len(held_queries) == 8
    assert {query.split for query in held_queries} == {"in_story_unqueried"}
    trained_query_keys = {
        query.key.text for story in corpus.train_stories for query in story.queries
    }
    assert all(query.key.text not in trained_query_keys for query in held_queries)

    binary_queries = in_story_unqueried_queries(
        corpus.train_stories,
        choices_per_query=2,
    )
    assert all(len(query.choices) == 2 for query in binary_queries)
    assert any(query.choices[0] != query.answer for query in binary_queries)


def test_choice_batch_marks_correct_candidate_index():
    import torch

    from research.eval.small_ar_champion_corpus import (
        SmallARStoryCorpusConfig,
        build_small_ar_story_corpus,
    )
    from research.eval.utils import _get_tiktoken_encoder
    from research.tools.small_ar_story_calibration import (
        TIKTOKEN_ENCODING,
        _pack_choice_batch,
        encode_story_items,
    )

    corpus = build_small_ar_story_corpus(
        SmallARStoryCorpusConfig(
            seed=5,
            n_train_stories=1,
            n_in_dist_eval_stories=0,
            n_held_key_stories=0,
            n_cross_story_groups=0,
            bindings_per_story=4,
            noise_sentences_per_story=0,
            queries_per_story=2,
            n_values=16,
        )
    )
    enc = _get_tiktoken_encoder(TIKTOKEN_ENCODING)
    items = encode_story_items(corpus.train_stories, enc)

    ids, labels, counts, targets = _pack_choice_batch(
        [items[0]],
        enc,
        device=torch.device("cpu"),
    )

    assert ids.shape[0] == len(items[0].choices)
    assert labels.shape == ids.shape
    assert counts.tolist() == [len(items[0].choices)]
    assert items[0].choices[targets.item()] == items[0].answer


def test_context_trim_keeps_target_binding_visible():
    from research.eval.small_ar_champion_corpus import (
        SmallARStoryCorpusConfig,
        build_small_ar_story_corpus,
    )
    from research.eval.utils import _get_tiktoken_encoder
    from research.tools.small_ar_story_calibration import (
        TIKTOKEN_ENCODING,
        encode_story_items,
    )

    corpus = build_small_ar_story_corpus(
        SmallARStoryCorpusConfig(
            seed=12,
            n_train_stories=1,
            n_in_dist_eval_stories=0,
            n_held_key_stories=0,
            n_cross_story_groups=0,
            bindings_per_story=8,
            noise_sentences_per_story=8,
            queries_per_story=1,
            choices_per_query=2,
            n_values=16,
        )
    )
    enc = _get_tiktoken_encoder(TIKTOKEN_ENCODING)
    full = encode_story_items(corpus.train_stories, enc)[0]
    trimmed = encode_story_items(
        corpus.train_stories,
        enc,
        context_keep_fraction=0.75,
    )[0]
    story = corpus.train_stories[0]
    query = story.queries[0]
    target = next(
        binding.sentence()
        for binding in story.bindings
        if binding.key.text == query.key.text
    )
    trimmed_prefix = enc.decode(trimmed.prefix_ids)

    assert len(trimmed.prefix_ids) < len(full.prefix_ids)
    assert target in trimmed_prefix
    assert query.prompt in trimmed_prefix


def test_dynamic_story_train_items_generate_fresh_bpe_queries():
    import random

    from research.eval.small_ar_champion_corpus import SmallARStoryCorpusConfig
    from research.eval.utils import _get_tiktoken_encoder
    from research.tools.small_ar_story_calibration import (
        TIKTOKEN_ENCODING,
        dynamic_story_train_items,
    )

    cfg = SmallARStoryCorpusConfig(
        seed=6,
        n_train_stories=1,
        n_in_dist_eval_stories=0,
        n_held_key_stories=0,
        n_cross_story_groups=0,
        bindings_per_story=4,
        noise_sentences_per_story=0,
        queries_per_story=1,
        n_values=16,
    )
    enc = _get_tiktoken_encoder(TIKTOKEN_ENCODING)
    rng = random.Random(123)

    first = dynamic_story_train_items(cfg, enc, rng=rng, batch_size=3, step=1)
    second = dynamic_story_train_items(cfg, enc, rng=rng, batch_size=3, step=2)

    assert len(first) == 3
    assert len(second) == 3
    assert {item.story_id for item in first}.isdisjoint(
        {item.story_id for item in second}
    )
    assert any(a.prefix_ids != b.prefix_ids for a, b in zip(first, second))
    assert all(item.split == "dynamic_train" for item in first + second)


def test_micro_retrieval_items_are_canonical_fresh_and_have_context_controls():
    import random

    from research.eval.utils import _get_tiktoken_encoder
    from research.tools.small_ar_story_calibration import (
        TIKTOKEN_ENCODING,
        _preset,
        micro_retrieval_items,
    )

    cfg = _preset("micro_retrieval", seed=13)
    enc = _get_tiktoken_encoder(TIKTOKEN_ENCODING)
    full = micro_retrieval_items(
        cfg,
        enc,
        rng=random.Random(1),
        n_stories=4,
        split="micro_eval",
    )
    missing = micro_retrieval_items(
        cfg,
        enc,
        rng=random.Random(1),
        n_stories=4,
        split="micro_eval",
        context_mode="missing_target",
        split_suffix="_missing_target",
    )
    counterfactual = micro_retrieval_items(
        cfg,
        enc,
        rng=random.Random(1),
        n_stories=4,
        split="micro_eval",
        context_mode="counterfactual_target",
        split_suffix="_counterfactual_target",
    )
    paired_train = micro_retrieval_items(
        cfg,
        enc,
        rng=random.Random(1),
        n_stories=4,
        split="micro_train",
        include_counterfactual_answer=True,
    )

    assert len(full) == 4
    assert {item.story_id for item in full}.isdisjoint(
        {
            item.story_id
            for item in micro_retrieval_items(
                cfg,
                enc,
                rng=random.Random(2),
                n_stories=4,
                split="micro_eval",
                story_id_start=100,
            )
        }
    )
    assert all(len(item.choices) == 2 for item in full)
    assert all(item.answer in item.choices for item in full)
    assert "Question: What does the" in enc.decode(full[0].prefix_ids)
    assert full[0].answer in enc.decode(full[0].prefix_ids)
    assert missing[0].split == "micro_eval_missing_target"
    assert missing[0].answer not in enc.decode(missing[0].prefix_ids)
    wrong = next(choice for choice in full[0].choices if choice != full[0].answer)
    counterfactual_prefix = enc.decode(counterfactual[0].prefix_ids)
    assert counterfactual[0].split == "micro_eval_counterfactual_target"
    assert full[0].answer not in counterfactual_prefix
    assert wrong in counterfactual_prefix
    assert full[0].prompt == counterfactual[0].prompt
    assert len(paired_train) == 8
    paired_counterfactual = [
        item
        for item in paired_train
        if item.split == "micro_train_counterfactual_train"
    ][0]
    assert paired_counterfactual.answer != full[0].answer
    assert paired_counterfactual.answer in full[0].choices


def test_curriculum00_is_minimal_binary_story_curriculum():
    from research.tools.small_ar_story_calibration import _preset

    cfg = _preset("curriculum00", seed=9)

    assert cfg.bindings_per_story == 4
    assert cfg.noise_sentences_per_story == 0
    assert cfg.queries_per_story == 4
    assert cfg.choices_per_query == 2
    assert cfg.n_in_dist_eval_stories > 0
    assert cfg.n_held_key_stories == 0
    assert cfg.n_cross_story_groups == 0


def test_micro_retrieval_preset_is_single_query_binary_curriculum():
    from research.tools.small_ar_story_calibration import _preset

    cfg = _preset("micro_retrieval", seed=9)

    assert cfg.bindings_per_story == 4
    assert cfg.noise_sentences_per_story == 0
    assert cfg.queries_per_story == 1
    assert cfg.choices_per_query == 2
    assert cfg.n_values == 16


def test_micro_retrieval_preset_accepts_binding_and_noise_overrides():
    from research.eval.utils import _get_tiktoken_encoder
    from research.tools.small_ar_story_calibration import (
        TIKTOKEN_ENCODING,
        _preset,
        micro_retrieval_items,
    )

    cfg = _preset(
        "micro_retrieval",
        seed=10,
        bindings_per_story=6,
        noise_sentences_per_story=2,
    )
    enc = _get_tiktoken_encoder(TIKTOKEN_ENCODING)
    item = micro_retrieval_items(
        cfg,
        enc,
        rng=__import__("random").Random(5),
        n_stories=1,
        split="micro_eval",
    )[0]
    prefix = enc.decode(item.prefix_ids)

    assert cfg.bindings_per_story == 6
    assert cfg.noise_sentences_per_story == 2
    assert cfg.n_values >= 24
    assert "quiet shelf" in prefix or "side note" in prefix
