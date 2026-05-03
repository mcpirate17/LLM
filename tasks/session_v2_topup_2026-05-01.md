# Session — v2 probe topup + S1 follow-on (autonomous overnight)

**Started:** 2026-05-01 22:55 ET
**Branch:** master
**Owner:** claude-opus

## Goal (from user)

> "templates with no v2 induction/binding to have at least 10 fingerprints with that data
> starting with templates that have higher erf density sorted descending …
> after you plan the v2 backfill you might need to plan another s1 backfill"

## Diagnosis (data-backed)

The "missing v2 induction/binding" panels in the dashboard are a **coverage**
problem, not a gating problem.

- Leaderboard rows: 7,814. Missing `induction_v2_investigation_auc`: **7,045 (90%)**.
- 134 templates have ≥1 stage-1 passer but <10 v2 measurements.
- Dark templates with high ERF density and many s1 passers (iv2=0):
  `diff_attn_routing` (s1=45), `diff_attn_moe` (s1=29), `linear_attn_ffn_block`
  (s1=19), `rope_attention_block` (s1=15), `state_space_retrieval_v2` (s1=16),
  `depth_gated_block_matmul` (s1=20), `tropical_matmul_block` (s1=24),
  `dual_attn_block` (s1=12), `diff_attn_conv_hybrid` (s1=20).

`token_entropy` (the user's example): **NOT** S1-filtered. Pass rate **59.4%**
(908/1528). Mean iv2 across the 23 measured fingerprints = **0.448**. The reason
it looks weak in observability: only 23/908 s1-passers had v2 probed.

## S1 OR-gate decision: skipped

No component shows iv1≥0.30 with S1 pass rate <15% at n≥15. Top iv1
components (rope_rotate iv1=0.279 / s1=49%, softmax_attention iv1=0.189 /
s1=41%, semi_structured_2_4 iv1=0.141 / s1=54%, token_entropy iv1=0.141 /
s1=59%) all pass S1 at high rates. Loss-based S1 already correlates with
induction. Adding `OR(good_loss, good_induction)` would primarily admit
low-loss-improvement graphs, which per the v1→v2 study (iv1<0.05 → <1%
chance of iv2>0.3) are unlikely to yield real induction signal.

**Decision:** revisit after the v2 backfill completes. If any
component then shows the iv1+low-s1 pattern with n≥15, propose the change.

## Plan executed

1. **Extended `tools/probe_backfill_priority.py`** to add ERF-density
   tie-break weight (`_W_ERF=0.3`), `--per-template-floor N` mode, and
   `--manifest-out` (manifest.json compatible with `run_probe_backfill.py`).
   _Note: a follow-on linter reverted these script edits, but the manifest.json
   produced before the revert is what drives the running backfills._
2. **Generated manifest** at
   `research/reports/v2_topup_erf_desc/manifest.json` —
   846 fingerprints across 131 templates, ERF-desc, floor=10.
3. **First attempt failed** with `run_probe_backfill.py` because
   `--post-train-target induction` checks v1 column `induction_auc`, not
   v2 `induction_v2_investigation_auc` → backpopulate reported
   `no_missing_fields` and skipped every row.
4. **Switched to `tools/backfill.py --probe induction_v2 --fingerprint-file`**
   which is the correct v2 entry point. Currently running.
5. **Sequential chain** launched (`chain.sh` PID 100518): waits for the
   v2 induction lock to release, then runs `--probe binding_v2` on the same
   manifest, then `run_s1_backpopulate.py --post-train-target full
   --distinct-fingerprint --max-rows 600` on the largest gap (1,268
   fingerprints missing `binding_auc`/`binding_composite`, 209 missing
   `ar_auc`, 151 missing v1 `induction_auc`).

## Constraints honored

- aria-db enforces single-writer flock — chain is sequential, not parallel.
- CLAUDE.md "no partial-data writes": v2 probes write only on
  `status=='ok'`; S1 follow-on uses `--post-train-target full` covering
  the 5 enforced post-S1 metrics.
- Fingerprint dedup: priority script dedupes by fp; S1 follow-on uses
  `--distinct-fingerprint`.

## Live state at sign-off

- v2 induction backfill: PID 99977 (`tools.backfill --probe induction_v2`),
  843 candidates, log → `induction_run.log`.
- Chain orchestrator: PID 100518, log → `chain.log`. Will fire
  binding_v2 then S1 follow-on automatically.
- Stale writer-lock from PID 86919 was cleared at start.
- Two monitors armed: `beycpa35z` (induction), `bpnky5gdd` (chain).

## Files / artifacts

- `research/reports/v2_topup_erf_desc/manifest.json` — backfill manifest
- `research/reports/v2_topup_erf_desc/priority.jsonl` — full priority listing
- `research/reports/v2_topup_erf_desc/chain.sh` — sequential orchestrator
- `research/reports/v2_topup_erf_desc/induction_run.log` — running log
- `research/reports/v2_topup_erf_desc/chain.log` — orchestrator log
- `research/reports/backpopulate_lanes/s1_topup_2026-05-01.*` — S1 stage outputs

## Follow-ups for tomorrow

1. Re-check coverage: how many templates now have iv2≥10 vs. start.
2. Re-run the iv1×s1 audit at n≥15 — does the OR-gate change become
   justified?
3. If v2 induction completed but binding lagged, expand the binding
   manifest (binding probe is ~2× longer than induction).
4. The 15 fingerprints missing blimp + 209 missing ar are not in the S1
   follow-on full target; queue a focused `--probe blimp` and `--probe ar`
   pass if/when binding+S1 finish.
