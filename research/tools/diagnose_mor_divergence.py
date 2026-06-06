"""Find what blows up in the native MoR-refine run.

Reuses the real trainer pipeline (loader, optimizer, ckpt load) and installs
forward hooks that (a) track per-module max|activation| and (b) record the FIRST
module whose output goes non-finite. Runs at the divergence-provoking config
until the first NaN, then prints the culprit module and the magnitude run-up.

Run: python research/tools/diagnose_mor_divergence.py
"""

from __future__ import annotations

import argparse

import torch

from research.tools import native_adaptive_hydra_train as T
from research.tools.scaling_blimp_study import _build_lane_factory, _build_tinylm

_os = __import__("os")
# Default to the step-43000 checkpoint from the unstable back-third of the
# resume_100k run (NaN'd ~43645). Override via DIAG_CKPT. DIAG_NONSTRICT=1 loads
# with a fresh router (for the old native ckpt); default strict (router present).
CKPT = _os.environ.get(
    "DIAG_CKPT",
    "research/reports/native_adaptive_hydra_ckpts/"
    "mor_resume_100k_mor_refine_mlp32_native_semiring_adapt_bilane_m32_g0_t1_b1_l0"
    "_h2_r4_surprise_memory_step043000.pt",
)
LANE = _os.environ.get(
    "DIAG_LANE",
    "mor_refine_mlp32_native_semiring_adapt_bilane_m32_g0_t1_b1_l0_h2_r4_surprise_memory",
)
# Start step controls (a) which deterministic loader batches are replayed and
# (b) the cosine-LR point if DIAG_COSINE is set. Default replays the real
# divergence window (the resume run reloaded 43000 then NaN'd by ~43650).
START_STEP = int(_os.environ.get("DIAG_START", "43001"))


def make_args() -> argparse.Namespace:
    return argparse.Namespace(
        lane=LANE,
        dataset=T.LOCAL_MIX_NAME,
        val_dataset=T.LOCAL_MIX_NAME,
        hydra_root=T.PROJECT_ROOT / "HYDRA",
        dim=512,
        n_blocks=8,
        steps=2000,
        batch=16,
        seq_len=256,
        lr=float(__import__("os").environ.get("DIAG_LR", "3e-4")),
        optimizer=__import__("os").environ.get("DIAG_OPT", "muon"),
        muon_lr=float(__import__("os").environ.get("DIAG_MUON_LR", "0.02")),
        muon_momentum=0.95,
        ns_steps=int(__import__("os").environ.get("DIAG_NS", "5")),
        warmup_steps=0,
        min_lr_frac=1.0,
        weight_decay=0.01,
        grad_clip=1.0,
        seed=0,
        vocab_size=T.VOCAB_SIZE,
        tokenizer="gpt2",
        num_workers=0,
        prefetch_factor=2,
        torch_threads=0,
        device="cuda",
        load_checkpoint=__import__("pathlib").Path(CKPT),
        load_nonstrict=_os.environ.get("DIAG_NONSTRICT", "0") == "1",
        ponder_weight=0.0,
        require_sources=False,
    )


def main() -> None:
    args = make_args()
    torch.manual_seed(args.seed)
    model = _build_tinylm(
        _build_lane_factory(args.lane),
        dim=args.dim,
        n_blocks=args.n_blocks,
        vocab_size=args.vocab_size,
        max_seq_len=max(args.seq_len, 1024),
        use_ffn=True,
    ).to(args.device)
    T._load_checkpoint(model, args)
    from component_fab.generator.mor_bilane import set_ponder_weight

    set_ponder_weight(model, 0.0)
    if __import__("os").environ.get("DIAG_FREEZE"):
        from component_fab.generator.mor_bilane import MoRLaneA

        for mod in model.modules():
            if isinstance(mod, MoRLaneA):
                for p in mod.halt_head.parameters():
                    p.requires_grad_(False)
        print("[DIAG_FREEZE] router frozen")
    opts = T._build_optimizers(model, args)
    base_lrs = [[g["lr"] for g in o.param_groups] for o in opts]
    loader = T._make_loader(args, dataset=args.dataset, seed=args.seed)

    state = {"culprit": None, "maxabs": {}}

    def mk_hook(name):
        def hook(_m, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            if not torch.is_tensor(t):
                return
            mx = t.detach().abs().max().item()
            state["maxabs"][name] = mx
            if state["culprit"] is None and not torch.isfinite(t).all():
                state["culprit"] = name

        return hook

    for name, m in model.named_modules():
        if name:
            m.register_forward_hook(mk_hook(name))

    init_norm = {
        n: p.detach().norm().item() + 1e-9
        for n, p in model.named_parameters()
        if p.ndim >= 2
    }

    def report_growth(tag):
        g = {
            n: p.detach().norm().item() / init_norm[n]
            for n, p in model.named_parameters()
            if n in init_norm
        }
        top = sorted(g.items(), key=lambda kv: -kv[1])[:8]
        print(f"  [{tag}] weight-norm growth (cur/init), top 8:")
        for n, r in top:
            print(f"      {r:8.2f}x  {n}")

    hist = []  # (step, loss, top module maxabs)
    for step in range(START_STEP, START_STEP + args.steps):
        state["culprit"] = None
        state["maxabs"] = {}
        if hasattr(loader, "set_step"):
            loader.set_step(step)
        batch = next(loader)
        ids, labels = T._prepare_batch(
            batch, vocab_size=args.vocab_size, device=args.device
        )
        loss, grad, lr = T._train_step(model, opts, base_lrs, ids, labels, args, step)
        if __import__("os").environ.get("DIAG_PROFILE") and step >= 10004:
            import collections

            items = sorted(state["maxabs"].items(), key=lambda kv: -kv[1])
            print("=== top-20 module |activation| ===")
            for k, v in items[:20]:
                print(f"  {v:10.2e}  {k}")
            comp = collections.defaultdict(float)
            for k, v in state["maxabs"].items():
                key = (
                    "lane_a"
                    if "lane_a" in k
                    else "lane_b"
                    if "lane_b" in k
                    else "attn"
                    if "attn" in k
                    else "ffn"
                    if ("ffn" in k or "mlp" in k or "swiglu" in k)
                    else "embed"
                    if ("embed" in k or "tok" in k or "wte" in k)
                    else "block/other"
                )
                comp[key] = max(comp[key], v)
            print("=== max |activation| by component ===")
            for k, v in sorted(comp.items(), key=lambda kv: -kv[1]):
                print(f"  {v:10.2e}  {k}")
            break
        top = sorted(state["maxabs"].items(), key=lambda kv: -kv[1])[:1]
        topname, topval = top[0] if top else ("-", 0.0)
        hist.append((step, loss, topname, topval))
        if (step - START_STEP) % 200 == 0:
            print(f"step {step} loss {loss:.3f} top|act| {topname}={topval:.2e}")
            report_growth(f"step {step}")
        if not (loss == loss):  # NaN
            print(
                f"\n*** non-finite loss at step {step}; first non-finite module: "
                f"{state['culprit']} ***"
            )
            report_growth(f"divergence@{step}")
            print(
                "magnitude run-up (last 12 steps): step | loss | top-|act| module : value"
            )
            for s, l, n, v in hist[-12:]:
                print(f"  {s}  loss={l:.3f}  {n} : {v:.3e}")
            # which surprise-memory lanes had the biggest activations
            mem = {k: v for k, v in state["maxabs"].items() if "lane" in k}
            big = sorted(mem.items(), key=lambda kv: -kv[1])[:6]
            print("biggest lane activations at divergence:")
            for k, v in big:
                print(f"  {k} : {v:.3e}")
            break
        if step % 50 == 0:
            print(
                f"step {step} loss {loss:.3f} top|act| {topname}={topval:.2e}",
                flush=True,
            )
    else:
        print("no divergence in window")


if __name__ == "__main__":
    main()
