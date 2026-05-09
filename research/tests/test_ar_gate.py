"""Smoke + determinism tests for AR gate-INV (CPU only)."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit]


def test_corpus_determinism_same_seed():
    from research.eval.ar_gate_corpus import build_corpus

    a = build_corpus(seed=42)
    b = build_corpus(seed=42)
    assert a.train_sentences == b.train_sentences
    assert a.test_facts == b.test_facts
    assert a.facts == b.facts


def test_corpus_different_seeds_diverge():
    from research.eval.ar_gate_corpus import build_corpus

    a = build_corpus(seed=1)
    b = build_corpus(seed=2)
    assert a.facts != b.facts


def test_held_out_facts_never_appear_in_training():
    from research.eval.ar_gate_corpus import build_corpus

    spec = build_corpus(seed=7)
    held_sentences = {f.sentence() for f in spec.test_facts if f.held_out}
    assert held_sentences, "expected at least one held-out fact"
    for sent in held_sentences:
        assert sent not in spec.train_sentences, (
            f"held-out fact leaked into training corpus: {sent!r}"
        )


def test_in_dist_facts_repeat_in_training():
    """Each fact-bearing sentence (full corpus, not dedup'd test_facts) repeats."""
    from research.eval.ar_gate_corpus import build_corpus

    spec = build_corpus(seed=11, reps=5)
    in_dist_facts = [f for f in spec.facts if not f.held_out]
    assert in_dist_facts
    for fact in in_dist_facts:
        count = spec.train_sentences.count(fact.sentence())
        assert count == 5, f"{fact} expected 5 reps, got {count}"


def test_test_facts_dedup_by_noun():
    """test_facts has one entry per unique noun (in-dist + held-out)."""
    from research.eval.ar_gate_corpus import (
        DEFAULT_HELD_OUT_NOUNS,
        build_corpus,
    )
    from research.tools.nano_corpus_v0 import NOUNS

    spec = build_corpus(seed=11)
    test_nouns = [f.noun for f in spec.test_facts]
    assert len(test_nouns) == len(NOUNS), (
        f"expected one prompt per noun, got {len(test_nouns)}"
    )
    assert len(set(test_nouns)) == len(test_nouns), "duplicate noun in test_facts"
    held_in_test = {n for n in test_nouns if n in DEFAULT_HELD_OUT_NOUNS}
    assert held_in_test == set(DEFAULT_HELD_OUT_NOUNS)


def test_trained_pairs_lookup_matches_facts():
    """trained_pairs_by_noun lookup contains every in-dist fact pair."""
    from research.eval.ar_gate_corpus import build_corpus

    spec = build_corpus(seed=21, n_pairs_per_noun=3)
    for fact in spec.facts:
        if fact.held_out:
            continue
        accepted = spec.trained_pairs_by_noun.get(fact.noun, frozenset())
        assert (fact.adj, fact.obj) in accepted, (
            f"missing {fact.noun}: ({fact.adj},{fact.obj}) in {accepted}"
        )


def test_held_out_nouns_match_default():
    from research.eval.ar_gate_corpus import (
        DEFAULT_HELD_OUT_NOUNS,
        build_corpus,
    )

    spec = build_corpus(seed=0)
    held_nouns = {f.noun for f in spec.test_facts if f.held_out}
    assert held_nouns == set(DEFAULT_HELD_OUT_NOUNS)


def test_all_corpus_words_are_single_token_under_cl100k_base():
    """All vocab words in the corpus must round-trip as single tokens."""
    import tiktoken

    from research.eval.ar_gate_corpus import OBJECTS
    from research.tools.nano_corpus_v0 import ADJECTIVES, NOUNS

    enc = tiktoken.get_encoding("cl100k_base")
    for w in (*NOUNS, *ADJECTIVES, *OBJECTS):
        ids = enc.encode(" " + w, allowed_special=set())
        assert len(ids) == 1, f"{w!r} → {ids} (expected 1 token)"


def test_query_prompt_format():
    from research.eval.ar_gate_corpus import Fact, query_prompt

    f = Fact(noun="dog", adj="red", obj="apple", held_out=False)
    assert query_prompt(f) == "the dog had a"


def test_facts_are_unique_by_full_tuple():
    from research.eval.ar_gate_corpus import build_corpus

    spec = build_corpus(seed=99)
    triples = [(f.noun, f.adj, f.obj) for f in spec.facts]
    assert len(triples) == len(set(triples)), "duplicate fact tuple detected"


def test_held_out_pairs_dont_collide_with_in_dist():
    """Held-out (adj, object) tuples must not match any in-dist fact's (adj, object)."""
    from research.eval.ar_gate_corpus import build_corpus

    spec = build_corpus(seed=3)
    in_dist_pairs = {(f.adj, f.obj) for f in spec.facts if not f.held_out}
    held_pairs = {(f.adj, f.obj) for f in spec.facts if f.held_out}
    assert not (in_dist_pairs & held_pairs), "held-out pair leaked into in-dist set"


def test_probe_smoke_on_tiny_graph():
    """End-to-end probe on a minimal compiled graph (CPU, ~few seconds)."""
    pytest.importorskip("torch")
    from research.synthesis.graph import ComputationGraph
    from research.synthesis.serializer import graph_to_json

    from research.eval.ar_gate import ARGateConfig, ar_gate

    g = ComputationGraph(model_dim=64)
    inp = g.add_input()
    norm = g.add_op("rmsnorm", [inp])
    attn = g.add_op("softmax_attention", [norm])
    fix = g.add_op("linear_proj", [attn], config={"out_dim": 64})
    out = g.add_op("add", [inp, fix])
    g.set_output(out)
    cfg = ARGateConfig(
        seed=0,
        finetune_steps=20,
        wikitext_warmup_steps=0,  # skip warmup in unit smoke (CPU)
        timeout_s=30.0,
        from_s1=False,
        n_distractors=20,
        n_pairs_per_noun=2,
        reps=2,
    )
    result = ar_gate(graph_json=graph_to_json(g), device="cpu", cfg=cfg)
    assert result.status in ("ok", "timeout"), result.error
    assert 0.0 <= result.in_dist_pair_acc <= 1.0
    assert result.n_in_dist + result.n_held > 0
