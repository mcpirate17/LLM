"""Tests for tokenizer parity: byte vs tiktoken pipeline integration."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from research.training.data_pipeline import (
    ByteTokenizer,
    CorpusConfig,
    CorpusTokenBatcher,
    TiktokenAdapter,
    WhitespaceHashTokenizer,
)


@pytest.mark.unit
class TestTokenizerParity:
    def test_byte_vs_tiktoken_different_sequences(self):
        """Byte tokenizer and TiktokenAdapter(cl100k_base) produce different tokens."""
        text = "The quick brown fox jumps over the lazy dog."
        vocab = 100_277  # cl100k native vocab

        byte_ids = ByteTokenizer().encode(text, vocab)
        tiktoken_ids = TiktokenAdapter("cl100k_base").encode(text, vocab)

        # BPE merges bytes into subwords — fewer tokens, different IDs
        assert byte_ids != tiktoken_ids
        assert len(tiktoken_ids) < len(byte_ids)

    def test_vocab_size_reported_correctly(self):
        """TiktokenAdapter reports native vocab size for each encoding."""
        cl100k = TiktokenAdapter("cl100k_base")
        gpt2 = TiktokenAdapter("gpt2")

        assert cl100k.native_vocab_size == 100_277
        assert gpt2.native_vocab_size == 50_257

    def test_no_modulo_projection_at_native_vocab(self):
        """When vocab_size >= native_vocab_size, no modulo projection occurs."""
        adapter = TiktokenAdapter("cl100k_base")
        text = "Hello world"
        native = adapter.native_vocab_size

        ids_native = adapter.encode(text, native)
        ids_large = adapter.encode(text, native + 1000)
        assert ids_native == ids_large  # no projection in either case

    def test_pipeline_uses_configured_tokenizer(self):
        """CorpusTokenBatcher with tokenizer='tiktoken' uses TiktokenAdapter."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Some sample text for testing the tokenizer pipeline.")
            tmp_path = f.name

        try:
            config = CorpusConfig(
                path=tmp_path,
                tokenizer="tiktoken",
                tiktoken_encoding="cl100k_base",
            )
            batcher = CorpusTokenBatcher(config, vocab_size=100_277)
            assert isinstance(batcher._tokenizer, TiktokenAdapter)
            assert batcher._tokenizer.native_vocab_size == 100_277
            assert batcher.ready
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_corpus_config_default_encoding(self):
        """CorpusConfig defaults tiktoken_encoding to 'gpt2'."""
        config = CorpusConfig(path="/tmp/dummy.txt")
        assert config.tiktoken_encoding == "gpt2"

    def test_whitespace_tokenizer_is_deterministic(self):
        text = "alpha beta alpha"
        vocab = 8192

        first = WhitespaceHashTokenizer().encode(text, vocab)
        second = WhitespaceHashTokenizer().encode(text, vocab)

        assert first == second
        assert len(first) == 3
        assert first[0] == first[2]

    def test_byte_batcher_loads_native_extension(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("abc def ghi")
            tmp_path = f.name

        try:
            config = CorpusConfig(path=tmp_path, tokenizer="byte")
            batcher = CorpusTokenBatcher(config, vocab_size=256)
            assert batcher.ready
            assert batcher._native_ext is None
            batch = batcher.sample_batch(
                batch_size=2,
                seq_len=3,
                generator=torch.Generator().manual_seed(0),
                device=torch.device("cpu"),
            )
            assert batch is not None
            assert batch.shape == (2, 3)
            assert batcher._native_ext is not None
        finally:
            Path(tmp_path).unlink(missing_ok=True)
