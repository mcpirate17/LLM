#!/usr/bin/env python3
"""Run GPT-2 reference on WikiText-103 + tiktoken.

Same settings as Var H's winning run. Stores result as the permanent
anchor for v6 scoring. Adds tokenizer_mode and corpus_path columns
to program_results and leaderboard if missing.

Usage:
    python -m research.tools.run_gpt2_wikitext_reference
"""

from __future__ import annotations

import math
import sqlite3
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Add tokenizer_mode and corpus_path columns if missing."""
    for table in ("program_results", "leaderboard"):
        existing = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
        for col, typ in [("tokenizer_mode", "TEXT"), ("corpus_path", "TEXT")]:
            if col not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")
                print(f"  Added {table}.{col}")
    conn.commit()


def main() -> None:
    from research.synthesis.reference_architectures import build_gpt2_layer
    from research.scientist.native_runner import (
        compile_model_native_first as compile_model,
    )
    from research.training.data_pipeline import CorpusConfig, CorpusTokenBatcher

    # ── Config (matches Var H's winning run) ──
    D_MODEL = 256
    N_LAYERS = 4
    VOCAB_SIZE = 100_277  # tiktoken cl100k_base
    SEQ_LEN = 256
    N_STEPS = 7000
    BATCH_SIZE = 4
    PEAK_LR = 3e-4
    WARMUP_STEPS = 100
    DECAY_START = 1500
    GRAD_CLIP = 1.0
    LOG_EVERY = 50
    CORPUS_PATH = (
        "/home/tim/Projects/LLM/research/corpus/wikitext-103-raw/wiki.train.raw"
    )
    TOKENIZER_MODE = "tiktoken"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Build model ──
    graph = build_gpt2_layer(D_MODEL)
    model = compile_model(
        [graph] * N_LAYERS, vocab_size=VOCAB_SIZE, max_seq_len=SEQ_LEN
    )
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"GPT-2 reference: {n_params:,} params, {N_LAYERS} layers, d={D_MODEL}")

    # ── Load corpus ──
    CORPUS_PATH = "research/corpus/wikitext103_train.npy"
    if not Path(CORPUS_PATH).exists():
        raise FileNotFoundError(f"WikiText-103 not found at {CORPUS_PATH}")

    corpus_cfg = CorpusConfig(
        path=CORPUS_PATH,
        tokenizer="tiktoken",
        tiktoken_encoding="cl100k_base",
        max_chars=500_000,
        train_fraction=0.9,
        val_fraction=0.1,
    )
    batcher = CorpusTokenBatcher(corpus_cfg, VOCAB_SIZE)
    rng = torch.Generator(device=device)
    print(f"Corpus: {CORPUS_PATH}")
    print(f"Tokenizer: {TOKENIZER_MODE} (vocab={VOCAB_SIZE})")
    print(f"Training: {N_STEPS} steps, batch={BATCH_SIZE}, seq_len={SEQ_LEN}")
    print(
        f"Schedule: cosine decay, peak_lr={PEAK_LR}, warmup={WARMUP_STEPS}, decay_start={DECAY_START}"
    )

    # ── Optimizer + schedule ──
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=PEAK_LR, weight_decay=0.01, betas=(0.9, 0.95)
    )

    def get_lr(step: int) -> float:
        if step < WARMUP_STEPS:
            return PEAK_LR * (step + 1) / WARMUP_STEPS
        if step < DECAY_START:
            return PEAK_LR
        progress = (step - DECAY_START) / max(1, N_STEPS - DECAY_START)
        return PEAK_LR * 0.5 * (1.0 + math.cos(math.pi * progress))

    # ── Train ──
    model.train()
    initial_loss = None
    final_loss = None
    min_loss = float("inf")
    loss_curve: list[tuple[int, float]] = []
    val_losses: list[float] = []

    t0 = time.perf_counter()
    for step in range(N_STEPS):
        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        batch = batcher.sample_batch(
            BATCH_SIZE, SEQ_LEN, rng, torch.device(device), split="train"
        )
        if batch is None:
            print(f"  step {step}: no batch available, aborting")
            break
        optimizer.zero_grad(set_to_none=True)

        logits = model(batch)
        sl = logits[:, :-1].contiguous()
        if sl.shape[-1] > VOCAB_SIZE:
            sl = sl[..., :VOCAB_SIZE]
        loss = F.cross_entropy(sl.reshape(-1, sl.shape[-1]), batch[:, 1:].reshape(-1))

        if not torch.isfinite(loss):
            print(f"  step {step}: loss is NaN/Inf, aborting")
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        loss_val = loss.item()
        if initial_loss is None:
            initial_loss = loss_val
        final_loss = loss_val
        min_loss = min(min_loss, loss_val)

        if step % LOG_EVERY == 0:
            elapsed = time.perf_counter() - t0
            loss_curve.append((step, loss_val))
            print(
                f"  step {step:5d}  loss={loss_val:.4f}  lr={lr:.2e}  [{elapsed:.1f}s]"
            )

        # Validation every 500 steps
        if step > 0 and step % 500 == 0:
            model.eval()
            with torch.no_grad():
                vbatch = batcher.sample_batch(
                    BATCH_SIZE, SEQ_LEN, rng, torch.device(device), split="val"
                )
                if vbatch is None:
                    continue
                vlogits = model(vbatch)
                vsl = vlogits[:, :-1].contiguous()
                if vsl.shape[-1] > VOCAB_SIZE:
                    vsl = vsl[..., :VOCAB_SIZE]
                vloss = F.cross_entropy(
                    vsl.reshape(-1, vsl.shape[-1]), vbatch[:, 1:].reshape(-1)
                )
                val_losses.append(vloss.item())
                print(f"         val_loss={vloss.item():.4f}")
            model.train()

    elapsed_total = time.perf_counter() - t0
    print(f"\nDone: {N_STEPS} steps in {elapsed_total:.1f}s")

    if initial_loss is None or final_loss is None:
        print("Training failed — no losses recorded")
        return

    loss_ratio = final_loss / math.log(VOCAB_SIZE)  # normalized
    improvement_rate = (
        (initial_loss - final_loss) / initial_loss if initial_loss > 0 else 0
    )
    print(f"Initial loss: {initial_loss:.4f}")
    print(f"Final loss:   {final_loss:.4f}")
    print(f"Min loss:     {min_loss:.4f}")
    print(f"Loss ratio:   {loss_ratio:.4f}")
    print(f"Improvement:  {improvement_rate:.4f}")

    # ── Store in DB ──
    db_path = "research/lab_notebook.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    _ensure_schema(conn)

    from research.scientist.notebook import LabNotebook
    from research.synthesis.serializer import graph_to_json

    nb = LabNotebook(db_path)
    graph_json_str = graph_to_json(graph)

    result_id = nb.record_program_result(
        experiment_id="gpt2_wikitext103_reference",
        graph_fingerprint=graph.fingerprint(),
        graph_json=graph_json_str,
        stage0_passed=True,
        stage05_passed=True,
        stage1_passed=True,
        loss_ratio=loss_ratio,
        final_loss=final_loss,
        initial_loss=initial_loss,
        min_loss=min_loss,
        loss_improvement_rate=improvement_rate,
        param_count=n_params,
        n_train_steps=N_STEPS,
        model_source="reference",
    )
    print(f"Recorded program_result: {result_id}")

    # Update tokenizer_mode and corpus_path
    conn.execute(
        "UPDATE program_results SET tokenizer_mode = ?, corpus_path = ? WHERE result_id = ?",
        (TOKENIZER_MODE, CORPUS_PATH, result_id),
    )

    # Upsert leaderboard entry
    entry_id = nb.upsert_leaderboard(
        result_id=result_id,
        model_source="reference",
        architecture_desc="GPT-2: GPT-2 transformer (Radford et al. 2019)",
        screening_loss_ratio=loss_ratio,
        screening_passed=True,
        investigation_loss_ratio=loss_ratio,
        investigation_robustness=1.0,
        investigation_passed=True,
        validation_loss_ratio=loss_ratio,
        validation_passed=True,
        tier="validation",
        is_reference=True,
        reference_name="GPT-2-wikitext103",
        tags="reference,gpt2,dense_attention_transformer,tiktoken_native,wikitext103",
        loss_improvement_rate=improvement_rate,
    )
    print(f"Upserted leaderboard: {entry_id}")

    # Store tokenizer_mode and corpus_path on leaderboard too
    conn.execute(
        "UPDATE leaderboard SET tokenizer_mode = ?, corpus_path = ? WHERE entry_id = ?",
        (TOKENIZER_MODE, CORPUS_PATH, entry_id),
    )
    conn.commit()

    nb.flush_writes()
    nb.close()
    conn.close()

    print("\nGPT-2 WikiText-103 reference stored.")
    print(f"  result_id:          {result_id}")
    print(f"  entry_id:           {entry_id}")
    print("  reference_name:     GPT-2-wikitext103")
    print(f"  final_loss:         {final_loss:.4f}")
    print(f"  loss_improvement:   {improvement_rate:.4f}")
    print(f"  params:             {n_params:,}")
    print(f"  tokenizer:          {TOKENIZER_MODE}")
    print(f"  corpus:             {CORPUS_PATH}")


if __name__ == "__main__":
    main()
