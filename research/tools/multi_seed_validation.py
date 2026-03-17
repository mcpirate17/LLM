"""Multi-seed validation for Var H vs GPT-2 reference.

Usage:
    python -m research.tools.multi_seed_validation
"""

import json
import math
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.synthesis.graph import ComputationGraph
from research.synthesis.compiler import compile_model

# ── Config ──────────────────────────────────────────────────────────────────
VARH_RESULT_ID = "48101a4c-012"
GPT2_RESULT_ID = "ref_gpt2_5b3b2643"
DB_PATH = PROJECT_ROOT / "research" / "lab_notebook.db"
CORPUS_PATH = PROJECT_ROOT / "research" / "corpus" / "wikitext103_train.npy"
SEEDS = [1, 2, 3]
CHECKPOINTS = [2000, 4000, 6000, 7000]

N_LAYERS = 4
MODEL_DIM = 256
VOCAB_SIZE = 100277  # tiktoken cl100k_base
TOTAL_STEPS = 7000
BATCH_SIZE = 4
MAX_SEQ_LEN = 256
PEAK_LR = 3e-4
WARMUP_STEPS = 100
DECAY_START = 1500
MIN_LR = 3e-5
GRAD_CLIP = 1.0
LOG_EVERY = 50


def load_graph_from_db(result_id: str) -> ComputationGraph:
    """Load a ComputationGraph from the database by result_id."""
    conn = sqlite3.connect(str(DB_PATH))
    row = conn.execute(
        "SELECT graph_json FROM program_results WHERE result_id = ?",
        (result_id,),
    ).fetchone()
    conn.close()
    if not row or not row[0]:
        raise ValueError(f"No graph_json for result_id={result_id}")
    return ComputationGraph.from_dict(json.loads(row[0]))


def cosine_lr(step: int) -> float:
    """Cosine decay with warmup. Returns LR for the given step."""
    if step < WARMUP_STEPS:
        return PEAK_LR * step / WARMUP_STEPS
    if step < DECAY_START:
        return PEAK_LR
    progress = (step - DECAY_START) / (TOTAL_STEPS - DECAY_START)
    progress = min(progress, 1.0)
    return MIN_LR + 0.5 * (PEAK_LR - MIN_LR) * (1.0 + math.cos(math.pi * progress))


def build_model(
    graph: ComputationGraph, device: torch.device, use_ir: bool = True
) -> torch.nn.Module:
    """Compile a graph into a model with N_LAYERS layers."""
    layer_graphs = [graph] * N_LAYERS
    model = compile_model(
        layer_graphs,
        vocab_size=VOCAB_SIZE,
        max_seq_len=MAX_SEQ_LEN,
        use_ir=use_ir,
    )
    return model.to(device)


def train_one_seed(
    graph: ComputationGraph,
    seed: int,
    device: torch.device,
    corpus: np.ndarray,
    label: str,
    has_mathspace: bool = False,
) -> dict[int, float]:
    """Train a model for TOTAL_STEPS and return {step: loss} at checkpoints."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # Math-space models need CompiledLayer (has boundary RMSNorm), not IRExecutor
    model = build_model(graph, device, use_ir=not has_mathspace)
    grad_clip = (
        model.recommended_grad_clip
        if hasattr(model, "recommended_grad_clip")
        else GRAD_CLIP
    )
    print(
        f"  grad_clip={grad_clip}, has_mathspace={has_mathspace}, use_ir={not has_mathspace}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=PEAK_LR,
        weight_decay=0.01,
        betas=(0.9, 0.95),
        fused=True,
    )

    corpus_len = len(corpus)
    results: dict[int, float] = {}
    rng = np.random.RandomState(seed)
    nan_count = 0

    t0 = time.time()
    for step in range(1, TOTAL_STEPS + 1):
        # Set LR
        lr = cosine_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Sample batch from corpus
        starts = rng.randint(0, corpus_len - MAX_SEQ_LEN - 1, size=BATCH_SIZE)
        batch = np.stack([corpus[s : s + MAX_SEQ_LEN + 1] for s in starts])
        input_ids = torch.from_numpy(batch[:, :MAX_SEQ_LEN].astype(np.int64)).to(device)
        targets = torch.from_numpy(batch[:, 1 : MAX_SEQ_LEN + 1].astype(np.int64)).to(
            device
        )

        # Forward
        model.train()
        logits = model(input_ids)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
        )

        loss_val = loss.item()

        # NaN guard: skip backward step but keep going
        if not math.isfinite(loss_val):
            nan_count += 1
            if nan_count >= 10:
                print(
                    f"  [{label} seed={seed}] FATAL: 10 consecutive NaNs at step {step}, aborting"
                )
                for cp in CHECKPOINTS:
                    if cp not in results:
                        results[cp] = float("nan")
                break
            optimizer.zero_grad(set_to_none=True)
            if step in CHECKPOINTS:
                results[step] = float("nan")
            continue
        nan_count = 0

        # Backward
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if step % LOG_EVERY == 0:
            elapsed = time.time() - t0
            print(
                f"  [{label} seed={seed}] step {step:5d}/{TOTAL_STEPS}  "
                f"loss={loss_val:.4f}  lr={lr:.6f}  "
                f"elapsed={elapsed:.0f}s"
            )

        if step in CHECKPOINTS:
            results[step] = loss_val

    # Cleanup
    del model, optimizer
    torch.cuda.empty_cache()

    return results


def record_seed_run(
    result_id: str,
    seed: int,
    losses: dict[int, float],
    graph_json: str,
    graph_fingerprint: str,
    parent_result_id: str,
    model_source: str,
    param_count: int,
) -> str:
    """Record a seed run in the database."""
    conn = sqlite3.connect(str(DB_PATH))
    new_id = f"{parent_result_id}_seed{seed}"
    experiment_id = f"multi_seed_{parent_result_id}"
    timestamp = time.time()

    # Create experiment if not exists
    existing = conn.execute(
        "SELECT experiment_id FROM experiments WHERE experiment_id = ?",
        (experiment_id,),
    ).fetchone()
    if not existing:
        conn.execute(
            "INSERT INTO experiments (experiment_id, timestamp, experiment_type, "
            "status, hypothesis, config_json) VALUES (?, ?, ?, ?, ?, ?)",
            (
                experiment_id,
                timestamp,
                "multi_seed_validation",
                "completed",
                f"Multi-seed validation of {parent_result_id}",
                json.dumps(
                    {
                        "seeds": SEEDS,
                        "total_steps": TOTAL_STEPS,
                        "n_layers": N_LAYERS,
                        "model_dim": MODEL_DIM,
                        "parent_result_id": parent_result_id,
                    }
                ),
            ),
        )

    final_loss = losses.get(TOTAL_STEPS, losses.get(max(losses.keys())))
    metadata = json.dumps(
        {
            "variant_type": "multi_seed_validation",
            "parent_result_id": parent_result_id,
            "seed": seed,
            "checkpoint_losses": losses,
        }
    )

    # Check if already recorded
    existing_result = conn.execute(
        "SELECT result_id FROM program_results WHERE result_id = ?",
        (new_id,),
    ).fetchone()
    if existing_result:
        print(f"  Result {new_id} already exists, skipping DB insert")
        conn.close()
        return new_id

    conn.execute(
        "INSERT INTO program_results "
        "(result_id, experiment_id, timestamp, graph_fingerprint, graph_json, "
        " stage0_passed, stage1_passed, final_loss, model_source, param_count, "
        " fingerprint_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            new_id,
            experiment_id,
            timestamp,
            graph_fingerprint,
            graph_json,
            1,
            1,
            final_loss,
            model_source,
            param_count,
            metadata,
        ),
    )

    # Store training curve at checkpoints
    for step_num, loss_val in sorted(losses.items()):
        conn.execute(
            "INSERT OR REPLACE INTO training_curves (result_id, step, loss) "
            "VALUES (?, ?, ?)",
            (new_id, step_num, loss_val),
        )

    conn.commit()
    conn.close()
    return new_id


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load corpus
    corpus = np.load(str(CORPUS_PATH), mmap_mode="r")
    print(f"Corpus: {corpus.shape[0]:,} tokens")

    # Load graphs from DB
    varh_graph = load_graph_from_db(VARH_RESULT_ID)
    gpt2_graph = load_graph_from_db(GPT2_RESULT_ID)
    print(f"Loaded Var H graph: {[n.op_name for n in varh_graph.nodes.values()]}")
    print(f"Loaded GPT-2 graph: {[n.op_name for n in gpt2_graph.nodes.values()]}")

    # Get param counts and graph metadata
    varh_model_tmp = build_model(varh_graph, torch.device("cpu"), use_ir=False)
    gpt2_model_tmp = build_model(gpt2_graph, torch.device("cpu"), use_ir=True)
    varh_params = sum(p.numel() for p in varh_model_tmp.parameters())
    gpt2_params = sum(p.numel() for p in gpt2_model_tmp.parameters())
    print(f"Var H params: {varh_params:,}")
    print(f"GPT-2 params: {gpt2_params:,}")
    del varh_model_tmp, gpt2_model_tmp

    # Get graph JSON strings from DB
    conn = sqlite3.connect(str(DB_PATH))
    varh_gj = conn.execute(
        "SELECT graph_json, graph_fingerprint FROM program_results WHERE result_id = ?",
        (VARH_RESULT_ID,),
    ).fetchone()
    gpt2_gj = conn.execute(
        "SELECT graph_json, graph_fingerprint FROM program_results WHERE result_id = ?",
        (GPT2_RESULT_ID,),
    ).fetchone()
    conn.close()

    # ── Run training ────────────────────────────────────────────────────────
    varh_results: dict[int, dict[int, float]] = {}  # seed -> {step: loss}
    gpt2_results: dict[int, dict[int, float]] = {}

    for seed in SEEDS:
        print(f"\n{'=' * 60}")
        print(f"GPT-2 seed={seed}")
        print(f"{'=' * 60}")
        gpt2_results[seed] = train_one_seed(
            gpt2_graph, seed, device, corpus, "GPT-2", has_mathspace=False
        )

        # Record
        record_seed_run(
            result_id=f"{GPT2_RESULT_ID}_seed{seed}",
            seed=seed,
            losses=gpt2_results[seed],
            graph_json=gpt2_gj[0],
            graph_fingerprint=gpt2_gj[1],
            parent_result_id=GPT2_RESULT_ID,
            model_source="reference_multi_seed",
            param_count=gpt2_params,
        )

        print(f"\n{'=' * 60}")
        print(f"Var H seed={seed}")
        print(f"{'=' * 60}")
        varh_results[seed] = train_one_seed(
            varh_graph, seed, device, corpus, "VarH", has_mathspace=True
        )

        # Record
        record_seed_run(
            result_id=f"{VARH_RESULT_ID}_seed{seed}",
            seed=seed,
            losses=varh_results[seed],
            graph_json=varh_gj[0],
            graph_fingerprint=varh_gj[1],
            parent_result_id=VARH_RESULT_ID,
            model_source="architecture_ablation_var_h_multi_seed",
            param_count=varh_params,
        )

    # ── Results table ───────────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("MULTI-SEED VALIDATION RESULTS")
    print(f"{'=' * 80}\n")

    header = (
        f"| {'Step':>5} | {'GPT2 s1':>8} | {'GPT2 s2':>8} | {'GPT2 s3':>8} | "
        f"{'VarH s1':>8} | {'VarH s2':>8} | {'VarH s3':>8} | "
        f"{'VarH mean':>9} | {'VarH std':>8} | {'Gap mean':>8} |"
    )
    sep = "|" + "|".join(["-" * (len(c)) for c in header.split("|")[1:-1]]) + "|"

    print(header)
    print(sep)

    table_rows = []
    for step in CHECKPOINTS:
        g1 = gpt2_results[1].get(step, float("nan"))
        g2 = gpt2_results[2].get(step, float("nan"))
        g3 = gpt2_results[3].get(step, float("nan"))
        v1 = varh_results[1].get(step, float("nan"))
        v2 = varh_results[2].get(step, float("nan"))
        v3 = varh_results[3].get(step, float("nan"))

        varh_vals = [v1, v2, v3]
        gpt2_vals = [g1, g2, g3]
        varh_mean = np.mean(varh_vals)
        varh_std = np.std(varh_vals)
        gaps = [v - g for v, g in zip(varh_vals, gpt2_vals)]
        gap_mean = np.mean(gaps)

        row = (
            f"| {step:>5} | {g1:>8.4f} | {g2:>8.4f} | {g3:>8.4f} | "
            f"{v1:>8.4f} | {v2:>8.4f} | {v3:>8.4f} | "
            f"{varh_mean:>9.4f} | {varh_std:>8.4f} | {gap_mean:>+8.4f} |"
        )
        print(row)
        table_rows.append(
            {
                "step": step,
                "gpt2": gpt2_vals,
                "varh": varh_vals,
                "varh_mean": varh_mean,
                "varh_std": varh_std,
                "gap_mean": gap_mean,
                "gaps": gaps,
            }
        )

    # ── Verdict ─────────────────────────────────────────────────────────────
    step6000 = next(r for r in table_rows if r["step"] == 6000)
    gap_mean_6k = step6000["gap_mean"]
    gap_std_6k = np.std(step6000["gaps"])

    print("\nStep 6000 analysis:")
    print(f"  Mean gap (VarH - GPT2): {gap_mean_6k:+.4f}")
    print(f"  Gap std: {gap_std_6k:.4f}")

    if gap_mean_6k < 0.0 and gap_std_6k < 0.05:
        verdict = "CONFIRMED"
        verdict_detail = (
            f"Var H beats GPT-2 by {abs(gap_mean_6k):.4f} nats (mean) at step 6000. "
            f"Std={gap_std_6k:.4f} < 0.05 — result is reproducible across 3 seeds."
        )
        tag_action = "replace 'needs_multi_seed' with 'multi_seed_validated'"
        new_tags_note = "multi_seed_validated"
    elif gap_mean_6k < 0.0 and gap_std_6k >= 0.05:
        verdict = "UNSTABLE"
        verdict_detail = (
            f"Var H beats GPT-2 by {abs(gap_mean_6k):.4f} nats (mean) at step 6000, "
            f"but std={gap_std_6k:.4f} >= 0.05 — high variance, architecture sensitive "
            f"to initialization."
        )
        tag_action = "add 'high_variance'"
        new_tags_note = "high_variance"
    else:
        verdict = "NOT CONFIRMED"
        verdict_detail = (
            f"Var H does NOT consistently beat GPT-2 at step 6000. "
            f"Mean gap={gap_mean_6k:+.4f}. Original run was lucky."
        )
        tag_action = "add 'single_seed_advantage'"
        new_tags_note = "single_seed_advantage"

    print(f"\n  VERDICT: {verdict}")
    print(f"  {verdict_detail}")
    print(f"  Tag action: {tag_action}")

    # ── Update leaderboard ──────────────────────────────────────────────────
    conn = sqlite3.connect(str(DB_PATH))

    # Get current tags
    row = conn.execute(
        "SELECT tags, notes FROM leaderboard WHERE result_id = ?",
        (VARH_RESULT_ID,),
    ).fetchone()

    if row:
        current_tags = row[0] or ""
        current_notes = row[1] or ""

        if verdict == "CONFIRMED":
            new_tags = current_tags.replace("needs_multi_seed", "multi_seed_validated")
            # Remove single_seed caveat from notes
            new_notes = current_notes.replace(
                "Single seed — multi-seed validation pending.",
                f"Multi-seed validated (3 seeds, gap={gap_mean_6k:+.4f}, std={gap_std_6k:.4f}).",
            ).replace(
                "Single seed — multi-seed validation pending",
                f"Multi-seed validated (3 seeds, gap={gap_mean_6k:+.4f}, std={gap_std_6k:.4f})",
            )
        elif verdict == "UNSTABLE":
            new_tags = current_tags + ",high_variance"
            new_notes = current_notes + (
                f" high_variance — architecture sensitive to initialization "
                f"(gap std={gap_std_6k:.4f})."
            )
        else:
            new_tags = current_tags.replace("needs_multi_seed", "single_seed_advantage")
            new_notes = current_notes + (
                f" single_seed_advantage — multi-seed does not confirm "
                f"(mean gap={gap_mean_6k:+.4f})."
            )

        conn.execute(
            "UPDATE leaderboard SET tags = ?, notes = ?, "
            "validation_multi_seed_std = ? WHERE result_id = ?",
            (new_tags, new_notes, float(gap_std_6k), VARH_RESULT_ID),
        )
        conn.commit()
        print(f"\n  Leaderboard updated: tags='{new_tags}'")
    else:
        print("\n  WARNING: No leaderboard entry found for Var H")

    conn.close()

    # ── Write audit file ────────────────────────────────────────────────────
    audit_dir = PROJECT_ROOT / "tasks" / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_path = audit_dir / "MULTI_SEED_RESULTS.md"

    md_lines = [
        "# Multi-Seed Validation: Var H vs GPT-2",
        "",
        f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Parent result**: `{VARH_RESULT_ID}`",
        f"**Seeds**: {SEEDS}",
        f"**Steps**: {TOTAL_STEPS} (checkpoints at {CHECKPOINTS})",
        f"**Config**: n_layers={N_LAYERS}, model_dim={MODEL_DIM}, "
        f"batch_size={BATCH_SIZE}, max_seq_len={MAX_SEQ_LEN}",
        f"**LR**: cosine decay, peak={PEAK_LR}, warmup={WARMUP_STEPS}, "
        f"decay_start={DECAY_START}, min={MIN_LR}",
        f"**Corpus**: wikitext103_train.npy ({corpus.shape[0]:,} tokens, tiktoken cl100k_base)",
        f"**Var H params**: {varh_params:,}  |  **GPT-2 params**: {gpt2_params:,}",
        "",
        "## Results",
        "",
        header,
        sep,
    ]

    for step in CHECKPOINTS:
        r = next(x for x in table_rows if x["step"] == step)
        g = r["gpt2"]
        v = r["varh"]
        row_str = (
            f"| {step:>5} | {g[0]:>8.4f} | {g[1]:>8.4f} | {g[2]:>8.4f} | "
            f"{v[0]:>8.4f} | {v[1]:>8.4f} | {v[2]:>8.4f} | "
            f"{r['varh_mean']:>9.4f} | {r['varh_std']:>8.4f} | {r['gap_mean']:>+8.4f} |"
        )
        md_lines.append(row_str)

    md_lines.extend(
        [
            "",
            "## Verdict",
            "",
            f"**{verdict}**: {verdict_detail}",
            "",
            f"Step 6000 mean gap: `{gap_mean_6k:+.4f}` | gap std: `{gap_std_6k:.4f}`",
            f"Leaderboard action: {tag_action}",
        ]
    )

    audit_path.write_text("\n".join(md_lines) + "\n")
    print(f"\n  Audit written to {audit_path}")


if __name__ == "__main__":
    main()
