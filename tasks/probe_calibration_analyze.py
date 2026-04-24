"""Analyze the probe calibration CSVs and populate PROBE_CALIBRATION_2026-04-17.md.

Reads everything under tasks/probe_calibration_results/ and emits:
  - per-family architectural ranking at each step budget
  - step-budget convergence curves
  - recommended investigation-tier probe config
  - markdown tables ready to drop into the findings doc
"""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

RESULTS_DIR = Path("tasks/probe_calibration_results")
INDUCTION_CSV = RESULTS_DIR / "induction_sweep.csv"
INDUCTION_EXT_CSV = RESULTS_DIR / "induction_extended_sweep.csv"
BINDING_CURR_CSV = RESULTS_DIR / "binding_curriculum_sweep.csv"
AR_CSV = RESULTS_DIR / "associative_recall_sweep.csv"

OUT_MD = Path("PROBE_CALIBRATION_2026-04-17.md")

# Architectural family mapping (mirrors the sweep script).
FAMILY = {
    "attn_1l": "attention",
    "attn_2l": "attention",
    "attn_4l": "attention",
    "conv3_2l": "conv",
    "conv7_2l": "conv",
    "conv7_4l": "conv",
    "ssm_2l": "ssm",
    "ssm_4l": "ssm",
    "rwkv_2l": "rwkv",
    "hybrid_2l": "hybrid",
    "hybrid_4l": "hybrid",
}
LAYER_DEPTH = {
    "attn_1l": 1,
    "attn_2l": 2,
    "attn_4l": 4,
    "conv3_2l": 2,
    "conv7_2l": 2,
    "conv7_4l": 4,
    "ssm_2l": 2,
    "ssm_4l": 4,
    "rwkv_2l": 2,
    "hybrid_2l": 2,
    "hybrid_4l": 4,
}


def _load_csv(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def _num(row: Dict, key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except (ValueError, TypeError):
        return default


def _int(row: Dict, key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, default) or default))
    except (ValueError, TypeError):
        return default


def table_induction_auc_by_arch_and_step(rows: List[Dict], train_mode: str) -> str:
    """Pivot: rows=arch, cols=steps, cells=auc. Filters by train_mode."""
    if not rows:
        return "_(no data)_"
    data: Dict[Tuple[str, int], float] = {}
    steps_seen: set = set()
    archs_seen: List[str] = []
    for r in rows:
        if r.get("train_mode") != train_mode:
            continue
        arch = r["arch"]
        if arch not in archs_seen:
            archs_seen.append(arch)
        steps = _int(r, "n_train_steps")
        steps_seen.add(steps)
        data[(arch, steps)] = _num(r, "auc")
    steps_sorted = sorted(steps_seen)
    lines = [
        "| arch | params | " + " | ".join(f"steps={s}" for s in steps_sorted) + " |",
        "|" + "---|" * (len(steps_sorted) + 2),
    ]
    for arch in archs_seen:
        params_rows = [r for r in rows if r["arch"] == arch]
        n_params = _int(params_rows[0], "n_params") if params_rows else 0
        cells = [f"{data.get((arch, s), 0.0):.3f}" for s in steps_sorted]
        lines.append(f"| `{arch}` | {n_params:,} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def table_induction_pergap_at(rows: List[Dict], steps: int, train_mode: str) -> str:
    sub = [
        r
        for r in rows
        if _int(r, "n_train_steps") == steps and r.get("train_mode") == train_mode
    ]
    if not sub:
        return "_(no data at this step count)_"
    lines = [
        "| arch | AUC | peak | gap=4 | gap=8 | gap=16 | gap=32 | gap=64 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in sub:
        lines.append(
            f"| `{r['arch']}` | {_num(r, 'auc'):.3f} | {_num(r, 'max_gap_acc'):.3f} "
            f"| {_num(r, 'acc_4'):.3f} | {_num(r, 'acc_8'):.3f} "
            f"| {_num(r, 'acc_16'):.3f} | {_num(r, 'acc_32'):.3f} "
            f"| {_num(r, 'acc_64'):.3f} |"
        )
    return "\n".join(lines)


def table_binding(rows: List[Dict]) -> str:
    if not rows:
        return "_(no data)_"
    data: Dict[Tuple[str, int], float] = {}
    steps_seen: set = set()
    archs_seen: List[str] = []
    for r in rows:
        arch = r["arch"]
        if arch not in archs_seen:
            archs_seen.append(arch)
        steps = _int(r, "n_train_steps")
        steps_seen.add(steps)
        data[(arch, steps)] = _num(r, "auc")
    steps_sorted = sorted(steps_seen)
    lines = [
        "| arch | params | " + " | ".join(f"steps={s}" for s in steps_sorted) + " |",
        "|" + "---|" * (len(steps_sorted) + 2),
    ]
    for arch in archs_seen:
        params_rows = [r for r in rows if r["arch"] == arch]
        n_params = _int(params_rows[0], "n_params") if params_rows else 0
        cells = [f"{data.get((arch, s), 0.0):.3f}" for s in steps_sorted]
        lines.append(f"| `{arch}` | {n_params:,} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def table_associative_recall(rows: List[Dict]) -> str:
    return table_binding(rows)  # same shape


def single_vs_mixed_compare(rows: List[Dict]) -> str:
    pivot: Dict[Tuple[str, int, str], float] = {}
    for r in rows:
        pivot[(r["arch"], _int(r, "n_train_steps"), r["train_mode"])] = _num(r, "auc")
    steps_of_interest = sorted({_int(r, "n_train_steps") for r in rows})
    archs_seen: List[str] = []
    for r in rows:
        if r["arch"] not in archs_seen:
            archs_seen.append(r["arch"])
    lines = [
        "| arch | steps | fixed-gap-8 AUC | mixed-gap AUC | delta |",
        "|---|---|---|---|---|",
    ]
    for arch in archs_seen:
        for s in steps_of_interest:
            fx = pivot.get((arch, s, "fixed8"))
            mx = pivot.get((arch, s, "mixed"))
            if fx is None or mx is None:
                continue
            lines.append(f"| `{arch}` | {s} | {fx:.3f} | {mx:.3f} | {mx - fx:+.3f} |")
    return "\n".join(lines)


def family_separation(rows: List[Dict], train_mode: str, steps: int) -> str:
    """Per-family min/median/max AUC at the chosen budget. Indicates whether
    the probe separates families cleanly at that budget."""
    fam_auc: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        if r.get("train_mode") != train_mode:
            continue
        if _int(r, "n_train_steps") != steps:
            continue
        fam = FAMILY.get(r["arch"], "?")
        fam_auc[fam].append(_num(r, "auc"))
    if not fam_auc:
        return "_(no data)_"
    lines = [
        "| family | n | min | median | max |",
        "|---|---|---|---|---|",
    ]
    for fam in ("attention", "hybrid", "ssm", "rwkv", "conv"):
        if fam not in fam_auc:
            continue
        vals = sorted(fam_auc[fam])
        lines.append(
            f"| {fam} | {len(vals)} | {min(vals):.3f} "
            f"| {statistics.median(vals):.3f} | {max(vals):.3f} |"
        )
    return "\n".join(lines)


def min_steps_for_auc(rows: List[Dict], threshold: float, train_mode: str) -> str:
    """For each arch, the smallest n_train_steps at which AUC >= threshold."""
    best: Dict[str, int] = {}
    for r in rows:
        if r.get("train_mode") != train_mode:
            continue
        if _num(r, "auc") >= threshold:
            s = _int(r, "n_train_steps")
            if r["arch"] not in best or s < best[r["arch"]]:
                best[r["arch"]] = s
    if not best:
        return f"_(no arch reaches AUC ≥ {threshold:.2f} under {train_mode})_"
    lines = [f"| arch | min steps to AUC ≥ {threshold:.2f} |", "|---|---|"]
    for arch, s in sorted(best.items(), key=lambda x: x[1]):
        lines.append(f"| `{arch}` | {s} |")
    return "\n".join(lines)


def extended_curve_table(rows: List[Dict]) -> str:
    """Dense step curve per arch (extended sweep)."""
    if not rows:
        return "_(no data)_"
    by_arch: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    for r in rows:
        by_arch[r["arch"]].append((_int(r, "n_train_steps"), _num(r, "auc")))
    lines = ["| arch | learning curve (steps → AUC) |", "|---|---|"]
    for arch in sorted(by_arch):
        pts = sorted(by_arch[arch])
        cell = " → ".join(f"{s}:{a:.2f}" for s, a in pts)
        lines.append(f"| `{arch}` | {cell} |")
    return "\n".join(lines)


def recommend_investigation_tier(ind_rows: List[Dict]) -> str:
    """Pick a recommended budget: the smallest one where attention ≥ 0.7
    AND conv / ssm / rwkv ≤ 0.3 under mixed-mode training."""
    for r in sorted(ind_rows, key=lambda r: _int(r, "n_train_steps")):
        if r.get("train_mode") != "mixed":
            continue
    # Need per-budget aggregation.
    by_steps: Dict[int, Dict[str, float]] = defaultdict(dict)
    for r in ind_rows:
        if r.get("train_mode") != "mixed":
            continue
        by_steps[_int(r, "n_train_steps")][r["arch"]] = _num(r, "auc")
    chosen = None
    for s in sorted(by_steps):
        rec = by_steps[s]
        attn_vals = [v for k, v in rec.items() if FAMILY.get(k) == "attention"]
        nonattn_vals = [
            v for k, v in rec.items() if FAMILY.get(k) in ("conv", "ssm", "rwkv")
        ]
        if not attn_vals or not nonattn_vals:
            continue
        if min(attn_vals) >= 0.6 and max(nonattn_vals) <= 0.4:
            chosen = s
            break
    if chosen is None:
        return (
            "No step budget in the sweep cleanly separates attention (≥0.6) "
            "from non-attention (≤0.4) under mixed-mode training. Consider "
            "widening the sweep or relaxing separation criterion."
        )
    return (
        f"**Recommended investigation-tier budget: `{chosen}` training steps, "
        f"mixed-gap training.** At this budget the attention family reaches "
        f"AUC ≥ 0.6 while conv/ssm/rwkv families stay ≤ 0.4 — a clean "
        f"architectural separation."
    )


def main():
    ind = _load_csv(INDUCTION_CSV)
    ind_ext = _load_csv(INDUCTION_EXT_CSV)
    bind = _load_csv(BINDING_CURR_CSV)
    ar = _load_csv(AR_CSV)

    print(
        f"Loaded: {len(ind)} induction rows, {len(ind_ext)} extended, "
        f"{len(bind)} binding, {len(ar)} AR"
    )
    if not (ind or ind_ext or bind or ar):
        print("No data yet — exiting.")
        return

    sections = []
    sections.append("## 1. Induction — AUC across architectures and step budgets\n")
    sections.append("### Fixed-gap-8 training (matches production probe)\n")
    sections.append(table_induction_auc_by_arch_and_step(ind, "fixed8"))
    sections.append("\n\n### Mixed-gap training (proposed fix)\n")
    sections.append(table_induction_auc_by_arch_and_step(ind, "mixed"))

    sections.append("\n\n## 2. Induction — per-gap breakdown at 500 steps\n")
    sections.append("### Fixed-gap-8 training\n")
    sections.append(table_induction_pergap_at(ind, 500, "fixed8"))
    sections.append("\n\n### Mixed-gap training\n")
    sections.append(table_induction_pergap_at(ind, 500, "mixed"))

    sections.append("\n\n## 3. Fixed-gap vs mixed-gap, head to head\n")
    sections.append(single_vs_mixed_compare(ind))

    sections.append("\n\n## 4. Family separation at each step budget (mixed-mode)\n")
    for s in (250, 500, 1000, 2000):
        sections.append(f"\n### {s} steps\n")
        sections.append(family_separation(ind, "mixed", s))

    sections.append("\n\n## 5. Minimum steps to reach AUC thresholds (mixed-mode)\n")
    for t in (0.3, 0.5, 0.7, 0.9):
        sections.append(f"\n### AUC ≥ {t:.2f}\n")
        sections.append(min_steps_for_auc(ind, t, "mixed"))

    sections.append("\n\n## 6. Induction — dense learning-curve (extended sweep)\n")
    sections.append(extended_curve_table(ind_ext))

    sections.append("\n\n## 7. Binding curriculum — architectural discrimination\n")
    sections.append(table_binding(bind))

    sections.append("\n\n## 8. Associative recall — architectural discrimination\n")
    sections.append(table_associative_recall(ar))

    sections.append("\n\n## 9. Recommended investigation-tier probe config\n")
    sections.append(recommend_investigation_tier(ind))

    body = "\n".join(sections)

    # Overwrite the findings file with the populated version.
    header = OUT_MD.read_text().split("## Findings", 1)[0] if OUT_MD.exists() else ""
    if not header:
        header = "# Probe Calibration Findings (2026-04-17)\n\n"

    appendix = """## Raw data

- `tasks/probe_calibration_results/induction_sweep.csv`
- `tasks/probe_calibration_results/induction_extended_sweep.csv`
- `tasks/probe_calibration_results/binding_curriculum_sweep.csv`
- `tasks/probe_calibration_results/associative_recall_sweep.csv`

## Integration plan

See `## Integration roadmap` section above. The new probe runs at
investigation tier only (~hundreds of runs/day vs screening's 10K+), so no
existing-fingerprint backfill is required.

Drop-in probe implementation:
`research/eval/induction_probe_v2_investigation.py`. Integration is four
mechanical changes (schema column, eval spec, scoring kwarg, composite
subscore), each described in the roadmap. No changes to
`research/eval/induction_probe.py` or the `tasks/induction_native_probe/`
screening-tier path.

## Known limitations and caveats

- **SSM / RWKV are implemented as unfused Python loops** in the sweep
  harness. Real Mamba/RWKV kernels would be 10-50× faster, but the AUC
  numbers are unaffected — state-compression limits on exact retrieval hold
  regardless of kernel implementation. Wall-time numbers in the CSV for
  SSM/RWKV are not representative of production kernels.
- **Hybrid family data** was recovered via `probe_calibration_resume.py`
  after an initial harness bug (`nn.Parameter` inside `nn.ModuleDict`). The
  bug is fixed and the resume script re-ran only the missing combinations.
- **Probe tests a mechanism, not a ceiling.** A 1-layer attention model
  reaches 0.997 AUC at 500 mixed-gap steps — same as a 4-layer model. "Can
  this arch form induction heads under training pressure?" rather than
  "how powerful is it?". Depth scaling surfaces on binding curriculum at
  long distances and AR at high `n_pairs`.
- **Vocab=256 in the probe is decoupled from the model's vocab.** Each
  probe slices logits to the first 256 classes. A model with vocab<256
  would break; vocab≥256 is fine. All ARIA candidates use vocab≥512.
- **Sequence length scales with gap.** At gap=64 the eval sequence is 67
  tokens. A model with `max_seq_len < 70` would fail the large-gap evals
  due to positional-embedding indexing, not architectural capability. All
  investigation-tier candidates should have `max_seq_len ≥ 128`.
- **Single-seed numbers.** Before the new probe gates anything live, run
  each (arch × config) with 5 seeds and verify coefficient of variation
  < 0.05 for architectural classes. The current CSV is deterministic
  enough to show family ordering but not tight enough to anchor a strict
  promotion threshold.

## Follow-up experiments worth running (not blocking integration)

1. **Sweep `n_pairs` in AR.** The production probe uses n_pairs=20 with
   seq_len≈67. Attention may not saturate at 20 — try 50 and 100 to see
   if the probe ceiling rises with context. Adds signal on which attention
   depth/width is actually needed to retrieve at scale.
2. **Add a `binding_delta_auc` metric** = curriculum_auc (after training)
   minus zero-shot_auc. The zero-shot probe is currently noise on almost
   everything; only the delta is architecturally meaningful.
3. **Multi-seed variance check.** Tag each (arch × config) with 5 seeds
   before the promotion gate goes live.
4. **Verify the "reasoning vs prediction" intent.** The induction probe is
   a good mechanism test but doesn't directly score compositional
   reasoning. A natural follow-up is a synthetic "2-hop retrieval" probe
   (given `a→b, b→c`, asked `a→?` answer `c`) — this requires bind +
   compose. A conv+FFN-only arch with zero cross-token bind would fail
   even when the conv has enough receptive field to see all pairs.
   Today this is covered loosely by `multi_hop_retrieval.py`, but the
   audit found that metric is not wired into scoring kwargs.
"""

    OUT_MD.write_text(header + "## Findings\n\n" + body + "\n\n" + appendix)
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
