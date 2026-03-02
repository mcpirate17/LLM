"""Long-context retrieval probe for architecture robustness.

This module runs a small *train-then-generalize* probe:
1) Briefly train a model on an associative retrieval task at short context.
2) Evaluate retrieval accuracy at longer sequence lengths.

Compared to single-shot passkey guessing, this is more robust because it tests
whether the architecture can learn and preserve long-range information flow.
"""

from __future__ import annotations

import random
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class PasskeyResult:
    seq_len: int
    passed: bool
    accuracy: float
    depth_results: Dict[float, bool]  # depth (0-1) -> success
    error: Optional[str] = None


def _sample_key_value(vocab_size: int) -> tuple[int, int]:
    # Keep clear separation from small marker IDs.
    key = random.randint(128, min(vocab_size - 1, 2048))
    val = random.randint(4096, min(vocab_size - 1, 8192))
    return key, val


def _make_retrieval_batch(
    vocab_size: int,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create synthetic key-value retrieval sequences.

    Pattern (causal LM):
      [noise ... key_marker, K, val_marker, V ... noise ... query_marker, K, ans_marker, <pad>]
    Target:
      predict V at the final answer slot (logits at position -2).
    """
    if seq_len < 16:
        raise ValueError(f"seq_len too small for retrieval probe: {seq_len}")
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    y = torch.zeros((batch_size,), dtype=torch.long, device=device)

    key_marker = 17
    val_marker = 18
    query_marker = 19
    ans_marker = 20

    # Reserve tail: [query_marker, K, ans_marker, PAD]
    tail_start = seq_len - 4
    x[:, tail_start] = query_marker
    x[:, tail_start + 2] = ans_marker
    x[:, tail_start + 3] = 0

    for b in range(batch_size):
        k, v = _sample_key_value(vocab_size)
        # Insert key/value pair somewhere in the body.
        body_start = 8
        body_end = max(body_start + 1, tail_start - 4)
        pair_pos = random.randint(body_start, body_end)
        x[b, pair_pos] = key_marker
        x[b, pair_pos + 1] = k
        x[b, pair_pos + 2] = val_marker
        x[b, pair_pos + 3] = v

        # Query the same key at the tail.
        x[b, tail_start + 1] = k
        y[b] = v

    return x, y


def _make_multi_hop_batch(
    vocab_size: int,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create 2-hop associative retrieval sequences.

    Pattern:
      body contains K -> M and M -> V pairs;
      tail queries K and expects V at answer slot.
    """
    if seq_len < 24:
        raise ValueError(f"seq_len too small for multi-hop probe: {seq_len}")
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    y = torch.zeros((batch_size,), dtype=torch.long, device=device)

    map_marker = 21
    query_marker = 19
    ans_marker = 20

    tail_start = seq_len - 4
    x[:, tail_start] = query_marker
    x[:, tail_start + 2] = ans_marker
    x[:, tail_start + 3] = 0

    for b in range(batch_size):
        k = random.randint(128, min(vocab_size - 1, 2048))
        m = random.randint(2049, min(vocab_size - 1, 4095))
        v = random.randint(4096, min(vocab_size - 1, 8192))

        body_start = 8
        body_end = max(body_start + 5, tail_start - 6)
        pair1 = random.randint(body_start, body_end - 4)
        pair2 = random.randint(pair1 + 3, body_end - 1)

        # K -> M
        x[b, pair1] = map_marker
        x[b, pair1 + 1] = k
        x[b, pair1 + 2] = m

        # M -> V
        x[b, pair2] = map_marker
        x[b, pair2 + 1] = m
        x[b, pair2 + 2] = v

        # Query K, expect V.
        x[b, tail_start + 1] = k
        y[b] = v

    return x, y


def _retrieval_step_loss(
    model: nn.Module,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    logits = model(input_ids)
    # Autoregressive: logits[t] predicts token at t+1.
    answer_logits = logits[:, -2, :]
    return torch.nn.functional.cross_entropy(answer_logits, targets)

def run_passkey_test(
    model: nn.Module,
    vocab_size: int,
    device: torch.device,
    seq_len: int = 512,
    n_trials: int = 10,
) -> PasskeyResult:
    """Run passkey retrieval trials for a given sequence length.
    
    The prompt format is:
    "The passkey is <KEY>. <HAYSTACK> What is the passkey?"
    We check if the model assigns highest probability to the <KEY> token.
    """
    model.eval()
    successes = 0
    depth_results = {}
    
    # Reserve fixed regions:
    # [prefix markers][haystack...][question markers][answer slot]
    prefix_len = 3
    question_len = 3
    answer_len = 1
    prompt_overhead = prefix_len + question_len + answer_len
    haystack_len = seq_len - prompt_overhead
    if haystack_len <= 0:
        return PasskeyResult(seq_len, False, 0.0, {}, error="Sequence length too short for passkey test")

    try:
        with torch.no_grad():
            for i in range(n_trials):
                # 1. Generate random passkey (single token for simplicity in micro-eval)
                # Avoid very low token IDs which might be special tokens
                passkey = random.randint(100, min(vocab_size - 1, 1000))
                
                # 2. Decide depth (0.0 = start, 1.0 = end)
                depth = i / max(1, n_trials - 1)
                insert_idx = int(depth * haystack_len)
                
                # 3. Build sequence:
                # [prefix markers][haystack with key at depth][question markers][answer slot]
                input_ids = torch.randint(0, vocab_size, (1, seq_len), device=device)

                # Marker tokens for "The passkey is" (synthetic marker IDs)
                input_ids[0, 0:prefix_len] = torch.tensor([10, 11, 12], device=device)

                # Insert passkey into the haystack according to depth.
                haystack_start = prefix_len
                haystack_end = seq_len - question_len - answer_len  # exclusive
                insert_pos = haystack_start + min(max(insert_idx, 0), haystack_end - haystack_start - 1)
                input_ids[0, insert_pos] = passkey

                # Marker tokens for "What is the passkey?" at the end
                question_start = seq_len - question_len - answer_len
                input_ids[0, question_start:question_start + question_len] = torch.tensor([13, 14, 15], device=device)

                # Keep answer slot as placeholder token; model predicts it autoregressively.
                answer_pos = seq_len - 1
                input_ids[0, answer_pos] = 0

                logits = model(input_ids)

                # Autoregressive convention: logits[t] predicts token at t+1.
                pred_token = torch.argmax(logits[0, answer_pos - 1, :]).item()
                
                is_correct = (pred_token == passkey)
                if is_correct:
                    successes += 1
                
                depth_results[round(depth, 2)] = is_correct

        accuracy = successes / n_trials
        return PasskeyResult(
            seq_len=seq_len,
            passed=(accuracy >= 0.8),
            accuracy=accuracy,
            depth_results=depth_results
        )

    except Exception as e:
        return PasskeyResult(seq_len, False, 0.0, {}, error=str(e))

def evaluate_long_context_retrieval(
    model: nn.Module,
    vocab_size: int,
    device: torch.device,
    lengths: List[int] = [256, 512, 1024],
    train_seq_len: int = 256,
    train_steps: int = 80,
    train_batch_size: int = 8,
    eval_batches: int = 12,
    eval_batch_size: int = 8,
    lr: float = 3e-4,
) -> Dict:
    """Train short-context retrieval, then evaluate long-context generalization."""
    results = {}
    passkey_results = {}
    train_meta: Dict[str, float | int | str] = {}

    # Primary: 1-hop associative retrieval
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    try:
        for step in range(train_steps):
            input_ids, targets = _make_retrieval_batch(
                vocab_size=vocab_size,
                batch_size=train_batch_size,
                seq_len=train_seq_len,
                device=device,
            )
            loss = _retrieval_step_loss(model, input_ids, targets)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite retrieval loss at step {step}: {float(loss)}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        train_meta["train_steps"] = train_steps
        train_meta["train_seq_len"] = train_seq_len
    except Exception as e:
        return {
            "retrieval_results": {},
            "retrieval_score": 0.0,
            "train_meta": {"error": str(e)[:200], **train_meta},
        }
    finally:
        del optimizer

    model.eval()
    with torch.no_grad():
        for l in lengths:
            correct = 0
            total = 0
            try:
                for _ in range(eval_batches):
                    input_ids, targets = _make_retrieval_batch(
                        vocab_size=vocab_size,
                        batch_size=eval_batch_size,
                        seq_len=l,
                        device=device,
                    )
                    logits = model(input_ids)
                    pred = torch.argmax(logits[:, -2, :], dim=-1)
                    correct += int((pred == targets).sum().item())
                    total += int(targets.numel())
                acc = float(correct) / max(total, 1)
                results[l] = {
                    "accuracy": round(acc, 4),
                    "passed": bool(acc >= 0.5),
                    "depths": {},
                    "error": None,
                }
            except Exception as e:
                results[l] = {
                    "accuracy": 0.0,
                    "passed": False,
                    "depths": {},
                    "error": str(e)[:200],
                }
                # If a length fails (often OOM), stop increasing context.
                break

    # Third benchmark: multi-hop retrieval (train at short context, eval at long).
    multi_hop_results = {}
    try:
        model.train()
        mh_optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        for step in range(max(1, int(train_steps * 0.75))):
            input_ids, targets = _make_multi_hop_batch(
                vocab_size=vocab_size,
                batch_size=train_batch_size,
                seq_len=train_seq_len,
                device=device,
            )
            loss = _retrieval_step_loss(model, input_ids, targets)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite multi-hop loss at step {step}: {float(loss)}")
            mh_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            mh_optimizer.step()
        train_meta["multi_hop_train_steps"] = max(1, int(train_steps * 0.75))
    except Exception as e:
        train_meta["multi_hop_error"] = str(e)[:200]
    finally:
        try:
            del mh_optimizer
        except Exception:
            pass

    model.eval()
    with torch.no_grad():
        for l in lengths:
            correct = 0
            total = 0
            try:
                for _ in range(eval_batches):
                    input_ids, targets = _make_multi_hop_batch(
                        vocab_size=vocab_size,
                        batch_size=eval_batch_size,
                        seq_len=l,
                        device=device,
                    )
                    logits = model(input_ids)
                    pred = torch.argmax(logits[:, -2, :], dim=-1)
                    correct += int((pred == targets).sum().item())
                    total += int(targets.numel())
                acc = float(correct) / max(total, 1)
                multi_hop_results[l] = {
                    "accuracy": round(acc, 4),
                    "passed": bool(acc >= 0.35),
                    "depths": {},
                    "error": None,
                }
            except Exception as e:
                multi_hop_results[l] = {
                    "accuracy": 0.0,
                    "passed": False,
                    "depths": {},
                    "error": str(e)[:200],
                }
                break

    # Secondary benchmark: zero-shot passkey retrieval across the same lengths.
    for l in lengths:
        try:
            pk = run_passkey_test(model, vocab_size, device, seq_len=l, n_trials=8)
            passkey_results[l] = {
                "accuracy": round(float(pk.accuracy), 4),
                "passed": bool(pk.passed),
                "depths": pk.depth_results,
                "error": pk.error,
            }
            # Stop escalating lengths if passkey retrieval fully collapses.
            if pk.accuracy < 0.1:
                break
        except Exception as e:
            passkey_results[l] = {
                "accuracy": 0.0,
                "passed": False,
                "depths": {},
                "error": str(e)[:200],
            }
            break

    # Primary benchmark score: trained associative retrieval.
    assoc_score = sum(float(r["accuracy"]) for r in results.values()) / max(len(results), 1)
    # Third benchmark score: trained multi-hop retrieval.
    multi_hop_score = sum(float(r["accuracy"]) for r in multi_hop_results.values()) / max(len(multi_hop_results), 1)
    # Secondary benchmark score: zero-shot passkey retrieval.
    passkey_score = sum(float(r["accuracy"]) for r in passkey_results.values()) / max(len(passkey_results), 1)
    # Retrieval aggregate used by runner for long-context composition.
    retrieval_aggregate = (assoc_score + passkey_score + multi_hop_score) / 3.0
    return {
        "retrieval_results": results,
        "multi_hop_results": multi_hop_results,
        "passkey_results": passkey_results,
        # Backward-compatible field: keep retrieval_score as primary task score.
        "retrieval_score": round(assoc_score, 4),
        "assoc_retrieval_score": round(assoc_score, 4),
        "multi_hop_score": round(multi_hop_score, 4),
        "passkey_score": round(passkey_score, 4),
        "retrieval_aggregate_score": round(retrieval_aggregate, 4),
        "train_meta": train_meta,
    }
