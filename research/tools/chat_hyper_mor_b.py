"""Interactive / one-shot text generation for the hyper_mor_b final checkpoint.

Loads the annealed step-125000 checkpoint (the non-QKV hyperbolic MoR bilane) and
generates continuations with temperature + top-k sampling. Tokenizer is
tiktoken cl100k_base (vocab 100277, matching the model). The model was trained at
seq_len 256 with RoPE (max 1024), so generation stays within that window.

Usage:
    python research/tools/chat_hyper_mor_b.py                       # interactive REPL
    python research/tools/chat_hyper_mor_b.py --prompt "Once upon"  # one-shot
    python research/tools/chat_hyper_mor_b.py --temperature 0.7 --top-k 40
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from research.eval.utils import _get_tiktoken_encoder
from research.tools._scaling_lanes import _build_lane_factory
from research.tools.scaling_blimp_study import _build_tinylm

LANE = (
    "hyper_mor_surprise_refine_mlp258_native_semiring_adapt_bilane"
    "_m32_g0_t1_b1_l0_h2_r7_surprise_memory"
)
DEFAULT_CKPT = (
    "research/checkpoints/hyper_mor_b_chin_final/hyper_mor_b_chin_"
    + LANE
    + "_step125000.pt"
)
VOCAB = 100277
MAX_CTX = 256  # training seq_len; keep the context window within it


def load_model(checkpoint: str, device: str) -> torch.nn.Module:
    payload = torch.load(checkpoint, map_location="cpu")  # nosec B614 - local ckpt
    model = _build_tinylm(
        _build_lane_factory(LANE), dim=736, n_blocks=8, vocab_size=VOCAB
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device).eval()
    return model


@torch.no_grad()
def generate(model, enc, prompt, *, device, max_new_tokens, temperature, top_k):
    ids = enc.encode(prompt, allowed_special=set()) if prompt else [enc.eot_token]
    ids = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        ctx = ids[:, -MAX_CTX:]
        out = model(ctx)
        logits = out[0] if isinstance(out, tuple) else out
        logits = logits[:, -1, :].float()
        if temperature <= 0:
            nxt = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k > 0:
                kth = torch.topk(logits, top_k, dim=-1).values[:, -1, None]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, nxt], dim=1)
    return enc.decode(ids[0].tolist())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--prompt", default=None, help="one-shot prompt; omit for interactive REPL"
    )
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    print(f"loading {args.checkpoint} on {args.device} ...", flush=True)
    model = load_model(args.checkpoint, args.device)
    enc = _get_tiktoken_encoder("cl100k_base")
    n = sum(p.numel() for p in model.parameters())
    print(
        f"loaded {n:,} params (non-QKV hyperbolic MoR bilane). "
        f"temp={args.temperature} top_k={args.top_k}\n",
        flush=True,
    )

    def run(prompt):
        return generate(
            model,
            enc,
            prompt,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )

    if args.prompt is not None:
        print(run(args.prompt), flush=True)
        return

    print("Interactive chat (empty line or Ctrl-D to exit).", flush=True)
    while True:
        try:
            prompt = input("\n>>> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt.strip():
            break
        print(run(prompt), flush=True)


if __name__ == "__main__":
    main()
