#!/usr/bin/env python3
"""Standalone test for latent_attention_compressor op and its templates.

Tests:
1. Forward/backward at D=64, 128, 256 — raw op math
2. Full compiled model graph with LAC
3. Both LAC templates: synthesis → compile → forward → backward
"""

import sys
import os

os.chdir(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ".")

import torch
import torch.nn.functional as F

B, S = 2, 64


def _check_numerics(y: torch.Tensor, model=None) -> tuple[bool, bool]:
    """Return (output_ok, grads_ok)."""
    ok = not (torch.isnan(y).any().item() or torch.isinf(y).any().item())
    if model is None:
        return ok, True
    grads_ok = all(
        not torch.isnan(p.grad).any() and not torch.isinf(p.grad).any()
        for p in model.parameters()
        if p.grad is not None
    )
    return ok, grads_ok


def test_raw_op():
    """Test the compiler op at multiple dimensions."""
    print("=" * 60)
    print("TEST 1: Raw latent_attention_compressor forward/backward")
    print("=" * 60)

    for D in [64, 128, 256]:
        x = torch.randn(B, S, D, requires_grad=True)
        latent_dim = max(D // 4, 16)
        kv_compress = torch.nn.Parameter(torch.randn(latent_dim, D) * 0.02)
        kv_up = torch.nn.Parameter(torch.randn(D * 2, latent_dim) * 0.02)

        latent = F.linear(x, kv_compress)
        kv = F.linear(latent, kv_up)
        k, v = kv[..., :D], kv[..., D:]
        y = x + torch.sigmoid(k) * v

        y.sum().backward()

        ok = not (torch.isnan(y).any().item() or torch.isinf(y).any().item())
        grad_ok = x.grad is not None and not torch.isnan(x.grad).any().item()
        status = "PASS" if ok and grad_ok else "FAIL"
        print(
            f"  D={D:3d}: params={kv_compress.numel() + kv_up.numel():,}, "
            f"range=[{y.min():.3f}, {y.max():.3f}], "
            f"grad_norm={x.grad.norm():.4f} → {status}"
        )
    print()


def test_compiled_model():
    """Build a minimal graph, compile, and run."""
    print("=" * 60)
    print("TEST 2: Compiled model with LAC")
    print("=" * 60)

    from research.synthesis.graph import ComputationGraph
    from research.synthesis.compiler import compile_model

    for D in [64, 128, 256]:
        try:
            g = ComputationGraph(model_dim=D)
            inp = g.add_input()
            norm = g.add_op("rmsnorm", [inp])
            proj = g.add_op("linear_proj", [norm], config={"out_dim": D})
            lac = g.add_op("latent_attention_compressor", [proj])
            g.set_output(g.add_op("add", [inp, lac]))

            model = compile_model([g], vocab_size=512, max_seq_len=S)
            x = torch.randint(0, 512, (B, S))
            y = model(x)
            y.sum().backward()

            ok, grads_ok = _check_numerics(y, model)
            params = sum(p.numel() for p in model.parameters())
            status = "PASS" if ok and grads_ok else "FAIL"
            print(f"  D={D:3d}: shape={list(y.shape)}, params={params:,} → {status}")
        except Exception as e:
            print(f"  D={D:3d}: FAIL — {type(e).__name__}: {e}")
    print()


def test_templates():
    """Synthesize graphs from both LAC templates, compile, and run."""
    print("=" * 60)
    print("TEST 3: Template synthesis + compile")
    print("=" * 60)

    import random
    from research.synthesis.graph import ComputationGraph
    from research.synthesis.compiler import compile_model
    from research.synthesis._templates_routing import (
        tpl_latent_compress_block,
        tpl_latent_compress_rwkv,
    )

    D = 128
    for name, tpl_fn in [
        ("latent_compress_block", tpl_latent_compress_block),
        ("latent_compress_rwkv", tpl_latent_compress_rwkv),
    ]:
        print(f"\n  Template: {name}")
        successes = 0
        for trial in range(5):
            rng = random.Random(42 + trial)
            try:
                g = ComputationGraph(model_dim=D)
                inp = g.add_input()
                g.set_output(tpl_fn(g, inp, rng))

                ops = [n.op_name for n in g.nodes.values() if not n.is_input]
                has_lac = "latent_attention_compressor" in ops

                model = compile_model([g], vocab_size=512, max_seq_len=S)
                y = model(torch.randint(0, 512, (B, S)))
                y.sum().backward()

                ok, grads_ok = _check_numerics(y, model)
                if ok and grads_ok:
                    successes += 1
                tag = "has_LAC" if has_lac else "NO_LAC(fallback)"
                status = "PASS" if ok and grads_ok else "FAIL"
                print(f"    trial {trial}: {status} [{tag}] ops={ops}")
            except Exception as e:
                print(f"    trial {trial}: ERROR — {type(e).__name__}: {e}")

        print(f"  → {successes}/5 passed")
    print()


if __name__ == "__main__":
    test_raw_op()
    test_compiled_model()
    test_templates()
    print("All tests complete.")
