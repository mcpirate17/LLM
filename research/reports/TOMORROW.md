# Tomorrow

Condensed on 2026-04-26. Original full note is archived at
`../../cleanup_20260426/doc_archives/TOMORROW_original_20260426.md`.

## Still Relevant

- Add a disposable end-to-end control-flow harness for:
  - start experiment,
  - stop experiment,
  - start autonomous,
  - pause/resume autonomous,
  - stop autonomous.
- Keep that harness isolated from the real notebook. Do not point it at
  `research/lab_notebook.db`.
- Add a regression test around notebook path validation so non-path objects
  cannot silently create junk SQLite files.

## Completed / Cleaned Up

- Root-level MagicMock database artifacts and writer-lock companions were
  removed on 2026-04-26.
- Zero-byte root `unused.sqlite` and `research_lab.db` were removed.
- Ignore rules now cover the SQLite snapshot/recovery naming patterns that were
  polluting status.

## Do Separately

- DB snapshot/backups remain untouched. Treat DB corruption and retention as a
  separate project.
- Dashboard/report follow-up details are preserved in the archived original.
