# Million-graph cascade shortlist — candidate novel architectures (FAILSAFE COPY)

`million_cascade_shortlist_clean.jsonl` — **600 rule-clean candidate graphs**, the
shortlist output of a 3M-graph GPU-free screening cascade run on **2026-05-25**.

Kept here (`tasks/`, NOT auto-pruned) as a **failsafe**: the working copy lived in
`research/reports/`, which `session-start.sh` deletes after 14 days. These are the
graphs worth the expensive GPU probe — do not lose them to the pruner.

## How it was produced
1. Predictors rebuilt first (commit `04009e9`): `capability_screener` retrained on ALL
   ~17.9k induction labels (no provenance gate, was capped at 3143), OOD-honest
   (leave-template-family-out + temporal), all model types self-select; `pls_partition_oracle`
   retrained (4 axes). See memory `project_capability_screener_rebuild_ood`.
2. `python -m research.tools.cpu_screening_cascade generate --pool 250000 --max-attempts 3000000`
   — generate → structural gate (mixer-on-path) → data-mined failure rules (`learned_rules`)
   → mechanism score → label-free probe-oracle ML scoring → context-rule backstop.
   ~878k graphs generated/gated → 250k probe-oracle-scored → 600 shortlisted.
3. `cpu_screening_cascade rescreen` → **600/600 clean** (0 context violations, 0 must-check
   failures, 0 high failure-risk). Mean template quality 0.954; 577/600 carry a novel mixer.

## Screening status (2026-05-25)
- A **parallel** 300-graph shortlist (`research/reports/cpu_cascade_million_shortlist_clean.jsonl`,
  a concurrent session's run) is being cheap-probe-funneled (ar_gate → nano_induction_nearest
  → nb → full S1) via `shortlist_cheap_probe_funnel.py` — **~300 have been screened** there.
- This 600-graph file is the **failsafe superset preservation**; the un-probed remainder is the
  forward queue. Nothing here has a stage1 claim in `runs.db`.

## Schema (one JSON object per line)
| field | meaning |
|---|---|
| `fingerprint` | structural graph fingerprint (dedup key, novel vs `runs.db`) |
| `ops` | sorted op-name set |
| `mech_score` | label-free induction-family circuit score (2·mixer_depth + Σmem + n_global + 0.5·n_mix) |
| `novelty` | n_novel_mixers_on_path + algebra_diversity |
| `mixer_depth`, `n_mixers_on_path`, `n_novel_mixers` | routing structure |
| `lit_family` / `lit_model` / `lit_match_type` | closest published arch (exact/family/partial/novel) |
| `template_quality`, `failure_risk` | `learned_rules.score_template_quality` outputs |
| `label_free_probe_score` | probe-oracle multi-axis max ratio (predicted/threshold) |
| `label_free_probe_axes` / `_predictions` | per-axis (ar_gate, nano_induction_nearest, induction, ar_curriculum) |
| `label_free_probe_recommendation` | EXPLORE_PROBE / PREDICT_GOOD / PREDICT_BAD |
| `label_free_probe_novelty_pctile` | kNN distance-novelty percentile |
| `graph` | full graph dict (`model_dim`, `nodes`, `input_node_id`, `output_node_id`, `metadata`) — directly compileable |

## How to consume
```bash
source /home/tim/venvs/llm/bin/activate
# cheap-probe funnel (ar_gate → nano → nb → S1 on the top survivors):
python -m research.tools.shortlist_cheap_probe_funnel --in tasks/cascade_shortlist/million_cascade_shortlist_clean.jsonl
# or GPU-validate top novel-mixer candidates directly:
python -m research.tools.probe_novel_candidates   # reads the cascade shortlist pool
```
Each line's `graph` is a ready-to-compile `ComputationGraph` dict
(`research.synthesis.serializer.graph_from_json`).
