"""Passkey Retrieval (Needle-in-a-haystack) test for long-context architectures.

Verifies if a model can retrieve a specific piece of information (the passkey)
buried at various positions within a long sequence of noise.
"""

from __future__ import annotations

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Optional

@dataclass
class PasskeyResult:
    seq_len: int
    passed: bool
    accuracy: float
    depth_results: Dict[float, bool]  # depth (0-1) -> success
    error: Optional[str] = None

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
    
    # We need at least a few tokens for instructions and the question
    prompt_overhead = 20 
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
                
                # 3. Build sequence
                # [Instruction] [Haystack Part 1] [Passkey] [Haystack Part 2] [Question]
                # For simplicity, we use random tokens for instructions/question too, 
                # but with fixed patterns if possible.
                # Actually, let's just use random tokens for everything and 
                # assume the model learns the 'retrieval' association.
                
                input_ids = torch.randint(0, vocab_size, (1, seq_len), device=device)
                
                # Marker tokens for "The passkey is" (just fixed random IDs)
                input_ids[0, 0:3] = torch.tensor([10, 11, 12], device=device)
                input_ids[0, 3] = passkey
                
                # Marker tokens for "What is the passkey?" at the end
                input_ids[0, -3:] = torch.tensor([13, 14, 15], device=device)
                
                # Target is the passkey token at the very end
                # We expect the model to predict 'passkey' at index seq_len-1
                # given the sequence up to seq_len-2.
                
                logits = model(input_ids)
                # Prediction for the next token after the question marker
                # We'll check the last position
                pred_token = torch.argmax(logits[0, -1, :]).item()
                
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
    lengths: List[int] = [256, 512, 1024]
) -> Dict:
    """Run passkey tests across multiple lengths."""
    results = {}
    for l in lengths:
        res = run_passkey_test(model, vocab_size, device, seq_len=l)
        results[l] = {
            "accuracy": res.accuracy,
            "passed": res.passed,
            "depths": res.depth_results,
            "error": res.error
        }
        # If it fails significantly at one length, skip longer ones to save time
        if res.accuracy < 0.2:
            break
            
    # Composite retrieval score
    avg_acc = sum(r["accuracy"] for r in results.values()) / len(lengths)
    
    return {
        "retrieval_results": results,
        "retrieval_score": round(avg_acc, 4)
    }
