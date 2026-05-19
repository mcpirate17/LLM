# Repo-Wide Slop Cleanup — Bucket Index

Master plan: `/home/tim/.claude/plans/compressed-doodling-stardust.md` (Claude Code agent local) and the section below.

## Goal

One-shot deep purge of ~10–12GB of stale DB snapshots, ledger rotations, timestamped JSON dumps, and orphan plan docs that have accumulated across the repo. Plus durable guardrails (CLAUDE.md rules, hook auto-prunes, pre-commit gate) so it doesn't reappear.

## Buckets

| # | File | Touches | Frees | Blocked by |
|---|------|---------|-------|------------|
| 1 | `bucket_1_db_purge.md` | research/{db_backups,tmp,perf_artifacts,reports,scientist/perf_audit_artifacts}/, research/*.bak/*.snap_*/*.sql | ~10GB | — |
| 2 | `bucket_2_root_cleanup.md` | repo root .md/.txt/scripts, tasks/audit/, tasks/*.md plan files | <1MB | 1 |
| 3 | `bucket_3_fab_catalog.md` | component_fab/state/ledger.py, component_fab/tools/run_autonomous.py, component_fab/catalog/* | ~100M | 1 |
| 4 | `bucket_4_notes_obsidian.md` | research/notes/*.json (move/delete), .claude/hooks/obsidian_sync.py (sync-notes subcommand) | ~500K | 1, 2 |
| 5 | `bucket_5_guardrails.md` | CLAUDE.md, AGENTS.md, .claude/hooks/session-start.sh, .pre-commit-config.yaml | n/a (prevention) | 1, 2, 3, 4 |

Buckets 2, 3, 4 run after 1. 2 and 3 are independent of each other. 4 wants 2 done (so obsidian_sync.py edits don't clash with file moves). 5 is the cap.

## Live-process exclusion list (DO NOT TOUCH)

- `research/notes/mixer_fingerprint/` — written live by `python -m research.tools.mixer_fingerprint`.
- Open SQLite DBs: `research/lab_notebook.db`, `research/runs.db`, `research/meta_analysis.db`, `research/events.db`, `research/baseline_cache.db`.
- `HYDRA/`, `LA3/`, `personaplex/`, `AbstractMoE/`, `archive/` — gitignored separate projects.
- Any process listed in `ps -ef | grep -E "python|node|cargo"` at start of your session.

## How to pick up a bucket

1. Read `tasks/cleanup/bucket_<N>_*.md` — it's self-contained.
2. Verify the live-process exclusion list is still valid (`ps -ef`).
3. Execute. Report freed bytes / files deleted in your final message.
4. After completion, append a short note to `tasks/cleanup/cleanup_summary.md` (create if missing).

## Decisions locked in (user-confirmed 2026-05-17)

- DB snapshots: keep newest 1 backup per logical DB, delete rest.
- component_fab/catalog rotations: keep last 3 of ledger.jsonl.* and proposals.jsonl.*; last 3 autonomous_run_*.json.
- Root plan docs: active → tasks/, dated/done → DELETE.
- Bucketed parallel execution.
