# Codex Handoff: Ablation, Metrics Backfill, and Dashboard Diagnostics

Date: 2026-04-30
Repo: `/home/tim/Projects/LLM`

This handoff summarizes the work completed in this chat and gives instructions for a new Codex session. The user is focused on building a self-improving discovery system, but they are also explicitly concerned about preserving existing leaders, not wasting GPU time, and not creating large incomplete datasets.

## User Requirements To Preserve

- Existing leader fingerprints must not be altered by ablation work.
- Ablation should create/evaluate children and support or deny relationships between components, templates, slots, and ops. It must not mutate the parent graph.
- Do not create duplicate fingerprints unless the run is explicitly marked as an intentional rerun with provenance.
- Never run large GPU campaigns that create rows with empty post-S1 metrics.
- Before DB changes, create a backup under both:
  - `research/db_backups/`
  - `/home/tim/GoogleDrive/Backups/LLM_Research/`
- Live feed/log visibility matters. Long-running tools need clear progress output.
- Loss alone is not enough. Reasoning/understanding probes matter: induction, binding, HellaSwag, BLiMP, AR, WikiText/PPL, ERF/ICLD trajectory data where applicable.

These guardrails were also added to `research/.current_work.md`.

## Major Work Completed

### 1. Diagnosed the original ablation problem

The first large ablation workflow was producing many Stage-1 survivor rows but only persisted loss/basic fields. It did not persist the normal post-S1 metrics needed for useful analysis.

This made the ablation diagnostics misleading because "known good/bad" rules were initially based mostly on loss-only evidence.

### 2. Stopped the bad exhaustive ablation path

The erroneous exhaustive ablation process was killed after the user requested it.

Important IDs:

- Stopped experiment: `cab83594-773`
- Status in DB: `stopped`
- Partial raw child rows were retained, but no finalized causal evidence should be trusted from that run without complete metrics.

### 3. Patched future ablation runs to persist full metrics

File changed:

- `research/scientist/runner/synthesis.py`

The ablation path now persists normal post-S1 metric groups, including:

- WikiText screening fields
- HellaSwag
- BLiMP
- induction probe
- binding probe
- AR probe
- routing fast-lane fields
- trajectory/fingerprint fields including ERF and ICLD
- train/perf/pruning fields where available

It also publishes live runtime events for `ablation_child_completed` with useful metric summaries.

### 4. Created an ablation metric backfill tool

File added:

- `research/tools/backfill_ablation_metrics.py`

Purpose:

- Re-evaluate existing ablation S1 child rows that were missing normal post-S1 metrics.
- Patch the existing `program_results` rows in place.
- Avoid creating new leaderboard rows.
- Mark data as reconstructed with provenance labels, including:
  - `trust_label`
  - `comparability_label`
  - `evaluation_protocol_version`

Backups created before backfill:

- `/home/tim/Projects/LLM/research/db_backups/pre_ablation_metric_backfill_20260429_204544/lab_notebook.db`
- `/home/tim/GoogleDrive/Backups/LLM_Research/pre_ablation_metric_backfill_20260429_204544/lab_notebook.db`
- `/home/tim/Projects/LLM/research/db_backups/pre_ablation_metric_backfill_20260429_204847/lab_notebook.db`
- `/home/tim/GoogleDrive/Backups/LLM_Research/pre_ablation_metric_backfill_20260429_204847/lab_notebook.db`

Backfill runs:

- Smoke run: `f2b7d6a8-87e`
- Full run: `2c877bc2-6f9`

Full backfill result from log:

- Completed: `2026-04-30 03:08:38`
- Total planned rows: `1581`
- Patched rows: `1567`
- Failed rows: `14`
- Elapsed: `22735` seconds, about 6.3 hours

Current DB audit:

- Total ablation rows: `1603`
- S1 ablation rows: `1582`
- S1 ablation rows still missing core metrics: `14`

Log file:

- `research/runtime/ablation_metric_backfill.log`

Monitor command:

```bash
watch -n 20 'tail -80 /home/tim/Projects/LLM/research/runtime/ablation_metric_backfill.log'
```

### 5. Fixed ablation diagnostics to stop treating loss-only evidence as strong evidence

Backend files changed:

- `research/scientist/notebook/notebook_advanced_analytics.py`
- `research/scientist/api_routes/programs_bp.py`

Frontend file added/changed:

- `research/dashboard/src/components/AblationDiagnostics.js`

Diagnostics now include metric-completeness fields such as:

- `metric_observation_count`
- `metric_complete_count`
- `metric_comparable_count`
- `metric_complete_rate`
- metric-specific effects:
  - loss
  - HellaSwag
  - BLiMP
  - induction
  - binding
  - AR
  - WikiText
- `composite_support_effect`

Credibility rules were tightened:

- minimum 3 evidence rows
- minimum 3 child fingerprints
- minimum 3 metric-complete observations
- at least 80% metric coverage

Rows below that threshold should be shown as incomplete leads, not as reliable causal conclusions.

Dashboard build passed:

```bash
cd research/dashboard && npm run build
```

### 6. Added ablation UI and workflow surfaces

The work included adding/surfacing ablation controls and diagnostics areas:

- Program detail ablation entry point
- Bulk/continuous ablation control path
- Diagnostics tab/table for ablation summaries
- Runtime/live feed event plumbing for ablation child completion and backfill progress

Relevant changed files from current git status include:

- `research/dashboard/src/components/app/AppOverlays.jsx`
- `research/dashboard/src/components/app/AppTabContent.jsx`
- `research/dashboard/src/components/app/appConfig.js`
- `research/dashboard/src/components/programDetail/EvalResultsPanel.js`
- `research/dashboard/src/services/apiService.js`
- `research/scientist/api_routes/experiments_bp.py`
- `research/scientist/api_routes/programs_bp.py`
- `research/scientist/notebook/notebook_advanced_analytics.py`
- `research/scientist/runner/synthesis.py`
- `research/scientist/causal_attribution.py`
- `research/tests/test_causal_attribution.py`

### 7. Investigated lost leaders and restored/audited preservation expectations

The user noticed previous best fingerprints were missing from the visible leaderboard. The notebook had backups under:

- `/home/tim/Projects/LLM/research`
- `/home/tim/GoogleDrive/Backups/LLM_Research`

An incident backup exists:

- `research/db_backups/incident_20260429_missing_leaders/lab_notebook.before_restore.db`

The important rule going forward: before any DB mutation, explicitly verify backup coverage and record where backups are.

### 8. Wrote planning docs

Files added:

- `ABLATION_AND_CONSTRUCTION_PLAN_20260429.md`
- `CHAMPION_EXHAUSTIVE_ABLATION_PLAN_20260429.md`

These capture the broader plan around ablation, construction learning, causal attribution, and future continuous improvement.

## Current Measured Findings From Ablation Data

The backfilled ablation data is now much more useful than it was before, but it is still not enough to crown new leaders.

Current notable ablation rows from DB queries:

Best ablation induction:

- result: `005170f1-6b0`
- fingerprint: `217396b322917fbc`
- induction: `0.6380`
- loss ratio: `0.5316`
- WikiText PPL: `714.4`
- HellaSwag: `0.260`
- BLiMP: `0.535`
- binding composite: `0.1933`

Best ablation WikiText/PPL:

- result: `14848ee4-d8d`
- fingerprint: `1b765b41c86d86f7`
- PPL: `76.9`
- loss ratio: `0.3275`
- induction: `0.0020`
- binding composite: `0.0497`
- HellaSwag: `0.260`
- BLiMP: `0.507`

Best ablation HellaSwag:

- result: `e10eccaa-064`
- fingerprint: `7ae32955055f277f`
- HellaSwag: `0.360`
- PPL: `584.9`
- induction: `0.0100`
- binding composite: `0.0074`

Best ablation binding composite:

- result: `005170f1-6b0`
- fingerprint: `217396b322917fbc`
- binding composite: `0.1933`
- induction: `0.6380`
- PPL: `714.4`

Interpretation:

- Some ablation children now show real signal, especially `005170f1-6b0` for induction/binding.
- The best PPL children do not show induction.
- The best HellaSwag child does not show useful induction/binding.
- Current evidence suggests ablation is valuable for causal attribution and negative/positive component evidence, but not yet enough to promote a new champion without validation.

## ERF / ERF Var / ICLD Notes

Definitions:

- `fp_jacobian_erf_density`: effective receptive field density. Higher means broader input positions influence output.
- `fp_jacobian_erf_variance`: variance/structure in the effective receptive field. Very high can be meaningful or pathological.
- `fp_icld_velocity`: in-context learning dynamics slope on synthetic sequence data. More negative is better.
- `fp_icld_delta_loss`: late loss minus early loss. More negative is better.

Observed earlier:

- ERF density often saturates at `1.0`.
- Highest ERF variance values were attached to poor models and should not be treated as automatically good.
- ICLD has been noisy and has not aligned strongly with induction/binding/PPL.
- Use ERF/ICLD as supporting trajectory evidence, not as primary promotion metrics.

## Current Runtime State

At handoff time:

- Dashboard process is running:
  - PID `81042`
  - command: `python -m research --mode=dashboard --port 5000`
- The full ablation metric backfill is complete.
- No `backfill_ablation_metrics.py` process is running.

One stale-looking item:

- Smoke experiment `f2b7d6a8-87e` has `completed_at` populated but still shows `status='running'` in the DB.
- Do not casually patch this without a backup. It is low priority but should be cleaned up during the next DB hygiene pass.

## Current Git State

There are many modified/untracked files. A new Codex should not assume all changes are from one coherent commit. Inspect carefully before editing.

Important untracked/added files include:

- `ABLATION_AND_CONSTRUCTION_PLAN_20260429.md`
- `CHAMPION_EXHAUSTIVE_ABLATION_PLAN_20260429.md`
- `CODEX_HANDOFF_20260430_ABLATION.md`
- `research/dashboard/src/components/AblationDiagnostics.js`
- `research/scientist/causal_attribution.py`
- `research/tests/test_causal_attribution.py`
- `research/tools/backfill_ablation_metrics.py`
- `research/tools/champion_exhaustive_ablation.py`
- `research/runtime/targeted_champion_ablation.py`
- `research/runtime/long_ablation_monitor.sh`

There are also existing file moves into `research/reports/` and tool moves under `research/tools/`.

Start any new session with:

```bash
cd /home/tim/Projects/LLM
git status --short
git diff -- research/scientist/runner/synthesis.py
git diff -- research/scientist/notebook/notebook_advanced_analytics.py
git diff -- research/scientist/api_routes/programs_bp.py
git diff -- research/dashboard/src/components/AblationDiagnostics.js
```

## Handoff Instructions For New Codex

1. Read this file first, then read `research/.current_work.md`.
2. Do not run or modify DB-writing tooling until a fresh backup is made and recorded.
3. Run a current completeness audit before making claims about ablation evidence:

```bash
sqlite3 -header -column research/lab_notebook.db "
select
  count(*) as ablation_rows,
  sum(case when stage1_passed=1 then 1 else 0 end) as s1_rows,
  sum(case when stage1_passed=1 and (
    wikitext_perplexity is null or
    hellaswag_acc is null or
    blimp_overall_accuracy is null or
    induction_auc is null or
    binding_composite is null or
    ar_auc is null
  ) then 1 else 0 end) as s1_missing_core
from program_results
where experiment_id in (
  select experiment_id from experiments where experiment_type like '%ablation%'
);"
```

4. Treat causal rules with small `n` as leads only. Do not present them as proven.
5. If asked to continue ablation, prefer targeted campaigns around current top leaders/breakthrough fingerprints, with dedupe/provenance checks and full post-S1 metrics enabled.
6. If asked to promote anything, require validation reruns and compare against real leaderboard champions using the same protocol/budget.
7. If fixing diagnostics, make the UI distinguish:
   - metric-complete evidence
   - historical evidence
   - executed evidence
   - reconstructed/backfilled evidence
   - incomplete/loss-only evidence
8. Keep the existing leaders intact. Ablation children are evidence, not replacements.

## Suggested Next Steps

Highest ROI next work:

1. Fix the stale smoke experiment status after making a backup.
2. Investigate the remaining `14` S1 ablation rows missing core metrics and either backfill them or mark them explicitly partial/excluded.
3. Open the dashboard and verify the ablation diagnostics tab shows metric coverage, backfill gap, credible rules, and incomplete leads clearly.
4. Add/confirm tests for:
   - ablation row metric persistence
   - causal summary completeness filtering
   - no parent graph mutation during ablation
   - duplicate fingerprint handling/provenance
5. Review `005170f1-6b0` as a potential ablation child worth targeted validation, because it has unusually high induction and binding for an ablation row.
6. Do not run another massive ablation campaign until the diagnostics page and completeness gates are verified end-to-end.

