# LLM Cleanup Audit - 2026-04-26

Scope: `/home/tim/Projects/LLM`, excluding destructive changes to active source/data unless explicitly called out.

## Actions Already Taken

- Condensed `research/perf_artifacts/` to four summary files and folded the later `idcol_backfill_optionA_20260426T095028.log` into `02_ml_gemini_predictor_summary.json` plus `03_logs_digest.log`.
- Left `research/perf_artifacts/idcol_backfill_optionB_20260426T103746.log` in place because it is being actively written by a running `backfill_trajectory_metrics_parallel` process.
- Deleted obvious root test junk:
  - literal `<MagicMock name='mock.db_path' ...>` files and writer-lock companions
  - zero-byte `unused.sqlite`
  - zero-byte `research_lab.db`
- Tightened ignore rules in `.gitignore` and `research/.gitignore` for SQLite snapshots/recovery files:
  - `*.db.snap_*`
  - `*.db.corrupt_*`
  - `*.db.pre_*`
  - `*.db.recover_*`
  - `*.db.recovered*`
  - `*.db?mode=*`
  - `*.sqlite`
  - `*.writer-lock`
- Saved pre-cleanup git status to `cleanup_20260426/git_status_before_cleanup.txt`.
- Removed ignored/generated cleanup targets only:
  - Python bytecode caches under `research/`, `aria_designer/`, and `aria_core/`
  - `.ruff_cache/`
  - `research/.pytest_cache/`
  - frontend `node_modules` / `build` / `dist` directories where ignored
  - CMake/Cython build output directories where ignored
  - runtime logs/test-run artifacts
- Reverted the attempted deletion of tracked Rust `target` files under `research/runtime/native/rust/aria-db/target`; that tree is tracked, so deleting it would create a large source diff.
- Condensed `.current_work.md`, `research/.current_work.md`, and `TOMORROW.md`; originals are archived under `cleanup_20260426/doc_archives/`.
- Left all DB snapshots/backups/corrupt/recovered files untouched.

## Size Summary

Total tree size is about `57G`.

| Area | Size | Assessment |
|---|---:|---|
| `HYDRA/checkpoints/` | `33G` | Biggest cleanup decision. Four 700M checkpoints; likely over-retained. |
| root `.git/` | `14G` | Large model checkpoints are in Git history; worktree cleanup will not shrink this. |
| `research/` | `11G` | Mostly DB snapshots/backups and native/dependency build outputs. |
| `research/runtime/native/rust/*/target/` | `819M` | Regenerable Rust build outputs. |
| `research/dashboard/node_modules/` | `405M` | Regenerable dependency install. |
| `aria_designer/ui/node_modules/` | `182M` | Regenerable dependency install. |
| `.code-review-graph/graph.db` | `105M` | Regenerable local code-search index. |
| `research/corpus/wikitext103_train.npy` | `459M` | Dataset artifact; keep if evals depend on offline corpus. |

## High-Impact Cleanup Decisions

### 1. HYDRA checkpoints

Files:

- `HYDRA/checkpoints/hydra_700m_final.pt` - `9.2G`
- `HYDRA/checkpoints/hydra_700m_step_490000.pt` - `9.2G`
- `HYDRA/checkpoints/hydra_700m_step_489500.pt` - `9.2G`
- `HYDRA/checkpoints/hydra_700m_step_489000.pt` - `4.6G`

Recommendation:

- Keep `hydra_700m_final.pt`.
- Keep `hydra_700m_step_490000.pt` only if exact pre-final resume/debug is still needed.
- Delete `hydra_700m_step_489500.pt` and `hydra_700m_step_489000.pt` after confirming no pending HYDRA analysis references them.

Potential savings:

- Keep final only: about `23G`.
- Keep final + latest step: about `13.8G`.

### 2. Research DB snapshots/backups

Large files in `research/`:

- Current DB: `lab_notebook.db` - `820M`
- Recent pre-sync/snapshots: five files around `819-820M` each
- Corrupt/recovery pair: two files around `859M` each
- Recovered dump: `805M`
- Older April 18 backup: `582M`
- Older April 5 backups: five files around `222-223M` each
- `research/db_backups/20260406_140731/lab_notebook.db` - `245M`

Recommendation:

- Keep current `lab_notebook.db`.
- Keep one latest known-good snapshot from 2026-04-26 and one pre-BPE/pre-eval-sync checkpoint if that is still operationally useful.
- Move the rest to external cold storage or delete after verifying restore confidence.
- Use or extend `research/scientist/snapshot_rotator.py` so this does not recur.

Potential savings:

- Conservative pruning can recover `5-7G`.
- Aggressive pruning can recover over `8G`, but this should be an explicit data-retention decision.

### 3. Git history bloat

`git count-objects -vH` reports:

- loose objects: `403M`
- packed objects: `12.79G`

Largest historical blobs include:

- `research/artifacts/scale_test_4L_50257v/step_186000.pt` - about `1.1G`
- `research/artifacts/scale_test_4L_50257v/step_190000.pt` - about `1.1G`
- `research/artifacts/scale_test_4L_50257v/step_188000.pt` - about `1.1G`
- `research/artifacts/scale_test_4L_50257v/step_192000.pt` - about `1.1G`
- `research/artifacts/scale_test_4L_50257v/latest.pt` - about `1.1G`
- `research/artifacts/scale_test_4L_50257v/step_192031.pt` - about `1.1G`
- `research/artifacts/scale_test_4L_50257v/final.pt` - about `580M`

Recommendation:

- This requires a Git history rewrite with `git filter-repo` or BFG, followed by coordination with any clones/remotes.
- Regular `git gc` will not remove these while reachable from history.
- Do this only as a separate, explicit maintenance operation.

Potential savings:

- About `7G+` from historical checkpoint blobs, likely more after pruning compiled artifacts from history.

## Safe Regenerable Cleanup

These can be deleted with low risk, accepting reinstall/rebuild cost:

| Path | Size | Notes |
|---|---:|---|
| `research/dashboard/node_modules/` | `405M` | Reinstall from `package-lock.json`. |
| `aria_designer/ui/node_modules/` | `182M` | Reinstall from UI package lock. |
| `research/runtime/native/rust/aria-scheduler/target/` | `686M` | Rebuild with Cargo/maturin. |
| `research/runtime/native/rust/aria-db/target/` | `133M` | Rebuild with Cargo/maturin. |
| `research/runtime/native/cython/build/` | `14M` | Regenerable. |
| `research/runtime/native/build/` | `3.8M` | Regenerable CMake build. |
| `research/runtime/native/build_current/` | small | Regenerable CMake build. |
| `aria_designer/runtime/tests/build/` | `1.2M` | Regenerable test build. |
| `aria_designer/ui/dist/` | `2.6M` | Regenerable frontend build. |
| `research/dashboard/build/` | `6.1M` | Regenerable frontend build. |
| `.ruff_cache/` | `1.2M` | Regenerable. |
| `**/__pycache__/`, `.pytest_cache/` | about `15M` | Regenerable. |
| `.code-review-graph/graph.db` | `105M` | Regenerable index; keep only if startup latency matters. |

Potential savings:

- About `1.5G` without touching stateful research data.

## Documentation Cleanup

### Root docs

Current root docs include:

- `.current_work.md` - `1,209` lines
- `README.md` - `396` lines
- `GLOBAL_DEV_PROMPT.md`
- `TOMORROW.md`
- several `*_DO_NOT_DELETE.txt` files

Recommendation:

- Keep `README.md`, `CLAUDE.md`, and the `*_DO_NOT_DELETE.txt` files for now.
- Condense `.current_work.md` into `CURRENT_WORK_SUMMARY.md` plus an archive file, or move old completed entries under a dated archive.
- `TOMORROW.md` appears partly stale now that the root MagicMock artifacts were removed; refresh it or delete it after extracting any still-active tasks.

### `research/` docs

Key docs/journals:

- `research/.current_work.md` - `2,970` lines
- `research/eval/WORKING.archive.md` - `1,241` lines
- `research/eval/WORKING.md` - `40` lines, active pointer
- `research/perf_artifacts/` - now condensed and healthy
- `research/tools/REGISTRY.md` - useful governance doc

Recommendation:

- Keep `research/eval/WORKING.md` active and short.
- Condense `research/.current_work.md` into a short “current state + remaining ROI” summary and move completed benchmark narratives to an archive.
- Leave `research/eval/WORKING.archive.md` unless it is replaced by a dated condensed archive. It is large but intentionally historical.
- `research/docs/` is currently deleted in the worktree; if that was intentional, leave it deleted. If not, restore only the docs still cited by code/tests.

### `HYDRA/`

Docs are coherent but old:

- `HYDRA/README.md` is `1,168` lines.
- `HYDRA/docs/` is only `64K`.

Recommendation:

- No urgent doc cleanup.
- The real issue is checkpoint retention, not docs.

### `aria_designer/`

Docs are small; generated workflow summaries are more numerous:

- `aria_designer/workflows/generated/` has `27` generated JSON/MD artifacts.
- `aria_designer/ui/dist/examples/` duplicates examples from `ui/public/examples/`.

Recommendation:

- Summarize `workflows/generated/` into one manifest if those runs are historical.
- Delete `ui/dist/` with other frontend build outputs when doing safe generated cleanup.

### `LA3/`, `personaplex/`, `.claude/`, `conductor/`

- `LA3/` is small and mostly stable docs/source.
- `personaplex/` has its own `.git`; treat as a separate project.
- `.claude/` is small and useful.
- `conductor/` is currently deleted in Git status; if intentional, no further cleanup.

## Worktree / Git Status Notes

The worktree was already very dirty before this audit:

- Many tracked deletions under `archive/`, `tasks/`, `research/docs/`, `research/scripts/`, and `conductor/`.
- Many active modifications under `research/`, especially dashboard, eval/scoring, runner, notebook, and tools.
- New untracked tools/tests around BPE, v10 scoring, trajectory backfills, and predictor retraining.

I did not restore or revert any of those. Treat them as existing session/user state.

## Recommended Cleanup Order

1. Decide HYDRA checkpoint retention. This is the fastest way to recover `14-23G`.
2. Decide research DB snapshot retention. Use `snapshot_rotator.py` or move old snapshots to cold storage.
3. Run safe generated cleanup for node_modules, Rust targets, CMake/Cython builds, frontend builds, caches, and code-review graph if rebuild cost is acceptable.
4. Condense `.current_work.md` and `research/.current_work.md`.
5. Plan a separate Git history cleanup if the 14G `.git` directory matters.
6. After source work stabilizes, address tracked deletions in `archive/`, `tasks/`, and `research/docs/` as one deliberate commit or restore set.
