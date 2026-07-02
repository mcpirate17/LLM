"""Ingest NM-F capability-probe results into runs.db + rolling dashboard note.

Sources: ``research/reports/nm_f_probes/*_nm_f_probes.json`` (auto-pruned at 14d —
this captures every run permanently before the JSON vanishes). Producer:
``research/tools/nm_f_capability_probes.py``; scheduled nightly by
``research/tools/nm_f_probe_nightly.sh`` (systemd user timer ``nm-f-probes.timer``).

Writes ONE new table (idempotent, keyed; safe to re-run any time):
  - nm_f_probe_results  long form: one row per (run_file, task, mixer, seed,
    layout, x, accuracy) where x = n_pairs (binding) or gap (retention).
    SEPARATE namespace from program_results/leaderboard — synthetic capability
    probes, not synthesis programs; does NOT trip S1 completeness enforcement.

Then rewrites ``research/notes/nm_f_probe_dashboard.md``: latest 3-seed medians
per probe, deltas vs the previous run, and REGRESSION flags (median drop > 0.05).
The note is FTS-indexed by ``index_notes.py`` and mirrored to Obsidian, and its
STATUS block is surfaced to every agent session by ``.claude/hooks/session-start.sh``
— this is how probe results reach design decisions without anyone remembering to
look. Query examples:

    SELECT mixer, x, accuracy FROM nm_f_probe_results
      WHERE task='binding' AND run_file=(SELECT MAX(run_file) FROM
      nm_f_probe_results WHERE task='binding') ORDER BY mixer, x;

Run:  python research/tools/ingest_nm_f_probes.py
"""

from __future__ import annotations

import glob
import json
import os
import sqlite3
import statistics
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(REPO, "research", "runs.db")
PROBES_DIR = os.path.join(REPO, "research", "reports", "nm_f_probes")
DASHBOARD = os.path.join(REPO, "research", "notes", "nm_f_probe_dashboard.md")
REGRESSION_DROP = 0.05

_PREAMBLE = """# NM-F probe dashboard (auto-generated — do not hand-edit)

Rolling status of the NM-F operator capability probes (`research/tools/
nm_f_capability_probes.py`), refreshed nightly by `nm-f-probes.timer` and on every
manual ingest. Full history is queryable in `runs.db.nm_f_probe_results`; method
+ interpretation live in `research/notes/nm_f_probe_results_2026-07-02.md` and
`tasks/nm_f_operator_families_2026-07-01.md`.

How to read this for DESIGN decisions:
- **retention / integral (NM-F4)** must stay FLAT to gap 1024 (8× train length).
  A drop = the integral hold path regressed — it is the validated structural fix
  for the p-adic retrieval gap; treat a regression as a release blocker for F4.
- **binding / cdma32 (NM-F9)** is the learned code-addressed binding line. Gap to
  the attn control is addressing accuracy (oracle ceiling 0.999). Rising = the
  multi-slot wall is closing; chips 64/128 rows becoming nonzero = the Welch
  interference curve becomes measurable.
- **attn rows are positive controls**, not baselines-to-adopt: if attn fails a
  probe, the probe recipe (not the ops) is broken that night.
"""


def _ddl(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS nm_f_probe_results (
            run_file TEXT NOT NULL,
            run_ts TEXT NOT NULL,
            task TEXT NOT NULL,
            mixer TEXT NOT NULL,
            seed INTEGER NOT NULL,
            layout TEXT NOT NULL DEFAULT '',
            x INTEGER NOT NULL,
            accuracy REAL NOT NULL,
            config_json TEXT,
            ingested_at REAL NOT NULL,
            PRIMARY KEY (run_file, task, mixer, seed, layout, x)
        );
        CREATE INDEX IF NOT EXISTS idx_nmf_task_mixer
            ON nm_f_probe_results (task, mixer, run_file);
        """
    )


def _rows_from_report(path: str) -> list[tuple]:
    with open(path) as fh:
        blob = json.load(fh)
    run_file = os.path.basename(path)
    run_ts = run_file.split("_nm_f_probes")[0]
    config_json = json.dumps(blob.get("config", {}))
    now = time.time()
    rows: list[tuple] = []
    for task in ("binding", "retention"):
        section = blob.get(task)
        if not section:
            continue
        for mixer, entry in section.items():
            if not isinstance(entry, dict) or "per_seed" not in entry:
                continue  # config keys like train_pairs/seq_len
            for seed, seed_blob in entry["per_seed"].items():
                if "acc_by_layout_pairs" in seed_blob:
                    per_layout = seed_blob["acc_by_layout_pairs"].items()
                elif "acc_by_pairs" in seed_blob:
                    per_layout = [("", seed_blob["acc_by_pairs"])]
                else:
                    per_layout = [("", seed_blob["acc_by_gap"])]
                for layout, accs in per_layout:
                    for x, acc in accs.items():
                        rows.append(
                            (
                                run_file,
                                run_ts,
                                task,
                                mixer,
                                int(seed),
                                layout,
                                int(x),
                                float(acc),
                                config_json,
                                now,
                            )
                        )
    return rows


def _medians(
    conn: sqlite3.Connection, task: str, run_file: str
) -> dict[tuple[str, int], float]:
    """Median over seeds per (mixer, x) for one run (block layout for binding)."""
    layout = "block" if task == "binding" else ""
    cur = conn.execute(
        """SELECT mixer, x, accuracy FROM nm_f_probe_results
           WHERE task=? AND run_file=? AND layout IN (?, '')""",
        (task, run_file, layout),
    )
    acc: dict[tuple[str, int], list[float]] = {}
    for mixer, x, a in cur.fetchall():
        acc.setdefault((mixer, x), []).append(a)
    return {key: statistics.median(vals) for key, vals in acc.items()}


def _config_signature(conn: sqlite3.Connection, task: str, run_file: str) -> tuple:
    """Difficulty/budget signature — regressions are only meaningful between runs
    of the SAME config (n_keys 32 vs 64 is a difficulty change, not a regression)."""
    row = conn.execute(
        "SELECT config_json FROM nm_f_probe_results WHERE task=? AND run_file=? LIMIT 1",
        (task, run_file),
    ).fetchone()
    cfg = json.loads(row[0]) if row and row[0] else {}
    return tuple(
        cfg.get(key)
        for key in ("n_keys", "n_values", "body_len", "steps", "seeds", "lr")
    )


def _multi_seed_runs(conn: sqlite3.Connection, task: str) -> list[str]:
    """Runs with ≥2 seeds, newest first — 1-seed smoke/recipe runs never drive
    the dashboard or regression flags."""
    cur = conn.execute(
        "SELECT run_file FROM nm_f_probe_results WHERE task=? "
        "GROUP BY run_file HAVING COUNT(DISTINCT seed) >= 2 ORDER BY run_file DESC",
        (task,),
    )
    return [row[0] for row in cur.fetchall()]


def _task_section(conn: sqlite3.Connection, task: str) -> tuple[str, list[str]]:
    runs = _multi_seed_runs(conn, task)
    if not runs:
        return f"## {task}\n\n(no multi-seed runs ingested yet)\n", []
    latest = _medians(conn, task, runs[0])
    # Compare only against the most recent PRIOR run with the same config.
    sig = _config_signature(conn, task, runs[0])
    prior = next((r for r in runs[1:] if _config_signature(conn, task, r) == sig), None)
    previous = _medians(conn, task, prior) if prior else {}
    xs = sorted({x for _, x in latest})
    mixers = sorted({m for m, _ in latest})
    header = f"## {task} — latest run `{runs[0]}` (3-seed medians)\n\n"
    header += "| mixer | " + " | ".join(str(x) for x in xs) + " |\n"
    header += "|---|" + "---|" * len(xs) + "\n"
    regressions: list[str] = []
    for mixer in mixers:
        cells = []
        for x in xs:
            val = latest.get((mixer, x))
            cell = "—" if val is None else f"{val:.3f}"
            prev = previous.get((mixer, x))
            if val is not None and prev is not None and prev - val > REGRESSION_DROP:
                cell += f" ⚠(was {prev:.3f})"
                regressions.append(f"{task}/{mixer}@{x}: {prev:.3f} → {val:.3f}")
            cells.append(cell)
        header += f"| {mixer} | " + " | ".join(cells) + " |\n"
    return header + "\n", regressions


def _write_dashboard(conn: sqlite3.Connection) -> str:
    sections, regressions = [], []
    for task in ("retention", "binding"):
        text, regs = _task_section(conn, task)
        sections.append(text)
        regressions.extend(regs)
    stamp = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    if regressions:
        status = "REGRESSION: " + "; ".join(regressions)
    else:
        status = "healthy — no regressions vs previous run"
    status_block = (
        "<!-- STATUS-BEGIN -->\n"
        f"NM-F probes ({stamp}): {status}\n"
        "Query: runs.db `nm_f_probe_results`; interpretation: this note + "
        "nm_f_probe_results_2026-07-02.md\n"
        "<!-- STATUS-END -->\n"
    )
    body = _PREAMBLE + "\n" + status_block + "\n" + "\n".join(sections)
    with open(DASHBOARD, "w") as fh:
        fh.write(body)
    return status


def main() -> None:
    reports = sorted(glob.glob(os.path.join(PROBES_DIR, "*_nm_f_probes.json")))
    conn = sqlite3.connect(DB_PATH, timeout=60)
    try:
        _ddl(conn)
        total = 0
        for path in reports:
            rows = _rows_from_report(path)
            conn.executemany(
                "INSERT OR REPLACE INTO nm_f_probe_results VALUES "
                "(?,?,?,?,?,?,?,?,?,?)",
                rows,
            )
            total += len(rows)
        conn.commit()
        status = _write_dashboard(conn)
        n_runs = conn.execute(
            "SELECT COUNT(DISTINCT run_file) FROM nm_f_probe_results"
        ).fetchone()[0]
        print(
            f"ingested {len(reports)} report file(s), {total} rows "
            f"({n_runs} runs total in db); dashboard: {status}"
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
