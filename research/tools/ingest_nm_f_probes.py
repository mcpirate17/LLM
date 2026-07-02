"""Ingest NM-F capability-probe results into runs.db + rolling dashboard note.

Sources: ``research/reports/nm_f_probes/*_nm_f_probes.json`` (auto-pruned at 14d —
this captures every run permanently before the JSON vanishes). Producer:
``research/tools/nm_f_capability_probes.py``; scheduled nightly by
``research/tools/nm_f_probe_nightly.sh`` (systemd user timer ``nm-f-probes.timer``).

Writes ONE new table (idempotent, keyed; safe to re-run any time):
  - nm_f_probe_results  long form: one row per (run_file, task, mixer, seed,
    layout, x, accuracy). x = n_pairs (binding/overwrite), gap (retention/
    induction), window (anagram), or body length (modcounter); layout carries
    the binding layout ("block"/"scatter") or the split metric ("overwritten"/
    "control", "mod2"/"mod4") and is '' where a task has neither.
    SEPARATE namespace from program_results/leaderboard — synthetic capability
    probes, not synthesis programs; does NOT trip S1 completeness enforcement.

The recognized task list is imported from the probes module (``PROBE_TASKS``)
— one constant owns the contract, and a report containing a task key outside
it RAISES instead of being silently skipped.

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
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from research.tools.nm_f_capability_probes import PROBE_TASKS  # noqa: E402

#: Fail-loud contract: every top-level report key must be a PROBE_TASK or here.
NON_TASK_KEYS = frozenset({"config", "device", "wall_seconds"})
assert not NON_TASK_KEYS & set(PROBE_TASKS), "task/meta key collision"

DB_PATH = os.path.join(REPO, "research", "runs.db")
PROBES_DIR = os.path.join(REPO, "research", "reports", "nm_f_probes")
DASHBOARD = os.path.join(REPO, "research", "notes", "nm_f_probe_dashboard.md")
REGRESSION_DROP = 0.05
#: Dashboard layout filter per task; unlisted tasks show every layout/split.
#: binding's "scatter" layout stays DB-only, as before the 2026-07-02 extension.
DASHBOARD_LAYOUTS = {"binding": ("block",)}

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
- **overwrite / oblique (NM-F2)**: the `[overwritten]` rows must track the
  `[control]` rows — overwritten falling below control = additive-blend leakage
  (stale v1 residue), the exact-replacement law regressing.
- **anagram / lie (NM-F3)**: chance is 0.5; lie above it is the zero-param
  order-sensitivity claim (a commutative mixer CANNOT clear this task).
- **modcounter / phmix (NM-F5)**: read the 1024/4096 columns — the length
  extrapolation lane where the non-QKV mixer must beat the attn control
  OUTRIGHT (attn reported honestly at every length).
- **induction / wavelet (NM-F6)**: gaps ≥256 are out-of-distribution; wavelet
  decaying to chance there = the dyadic dilation transfer failed.
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


def _flatten_acc(acc_blob: dict) -> list[tuple[str, int, float]]:
    """Long-form ``(layout, x, accuracy)`` triples from any per-seed accuracy
    dict: ``{x: acc}`` (retention/anagram/induction); ``{layout: {x: acc}}``
    (binding — non-numeric outer keys); ``{x: {split: acc}}`` (overwrite/
    modcounter — numeric outer keys; the split lands in the layout column)."""
    triples: list[tuple[str, int, float]] = []
    for outer, inner in acc_blob.items():
        if isinstance(inner, dict):
            if str(outer).lstrip("-").isdigit():  # x-major: split dicts per x
                triples.extend(
                    (str(split), int(outer), float(a)) for split, a in inner.items()
                )
            else:  # layout-major: x dicts per layout
                triples.extend((str(outer), int(x), float(a)) for x, a in inner.items())
        else:
            triples.append(("", int(outer), float(inner)))
    return triples


def _rows_from_report(path: str) -> list[tuple]:
    with open(path) as fh:
        blob = json.load(fh)
    run_file = os.path.basename(path)
    unknown = set(blob) - NON_TASK_KEYS - set(PROBE_TASKS)
    if unknown:
        raise ValueError(
            f"{run_file}: unrecognized probe task key(s) {sorted(unknown)} — "
            "a probe exists without ingest coverage; extend PROBE_RUNNERS in "
            "nm_f_capability_probes.py (PROBE_TASKS is derived from it) and "
            "the dashboard preamble here. NEVER skip a task silently."
        )
    run_ts = run_file.split("_nm_f_probes")[0]
    config_json = json.dumps(blob.get("config", {}))
    now = time.time()
    rows: list[tuple] = []
    for task in PROBE_TASKS:
        section = blob.get(task)
        if not section:
            continue
        for mixer, entry in section.items():
            if not isinstance(entry, dict) or "per_seed" not in entry:
                continue  # config keys like train_pairs/seq_len
            for seed, seed_blob in entry["per_seed"].items():
                acc_keys = [k for k in seed_blob if k.startswith("acc_by_")]
                if len(acc_keys) != 1:
                    raise ValueError(
                        f"{run_file}: {task}/{mixer} seed {seed} has accuracy "
                        f"keys {acc_keys}; expected exactly one 'acc_by_*'"
                    )
                rows.extend(
                    (
                        run_file,
                        run_ts,
                        task,
                        mixer,
                        int(seed),
                        layout,
                        x,
                        acc,
                        config_json,
                        now,
                    )
                    for layout, x, acc in _flatten_acc(seed_blob[acc_keys[0]])
                )
    return rows


def _medians(
    conn: sqlite3.Connection, task: str, run_file: str
) -> dict[tuple[str, str, int], float]:
    """Median over seeds per (mixer, layout, x) for one run. Split-metric
    layouts (overwritten/control, mod2/mod4) become separate dashboard rows;
    ``DASHBOARD_LAYOUTS`` restricts tasks that track extra DB-only layouts."""
    query = (
        "SELECT mixer, layout, x, accuracy FROM nm_f_probe_results "
        "WHERE task=? AND run_file=?"
    )
    params: list = [task, run_file]
    layouts = DASHBOARD_LAYOUTS.get(task)
    if layouts:
        query += f" AND layout IN ({','.join('?' * len(layouts))})"
        params.extend(layouts)
    acc: dict[tuple[str, str, int], list[float]] = {}
    for mixer, layout, x, a in conn.execute(query, params):
        acc.setdefault((mixer, layout, x), []).append(a)
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
    xs = sorted({x for _, _, x in latest})
    lines = sorted({(m, lay) for m, lay, _ in latest})
    header = f"## {task} — latest run `{runs[0]}` (3-seed medians)\n\n"
    header += "| mixer | " + " | ".join(str(x) for x in xs) + " |\n"
    header += "|---|" + "---|" * len(xs) + "\n"
    regressions: list[str] = []
    for mixer, layout in lines:
        label = mixer if not layout else f"{mixer}[{layout}]"
        cells = []
        for x in xs:
            val = latest.get((mixer, layout, x))
            cell = "—" if val is None else f"{val:.3f}"
            prev = previous.get((mixer, layout, x))
            if val is not None and prev is not None and prev - val > REGRESSION_DROP:
                cell += f" ⚠(was {prev:.3f})"
                regressions.append(f"{task}/{label}@{x}: {prev:.3f} → {val:.3f}")
            cells.append(cell)
        header += f"| {label} | " + " | ".join(cells) + " |\n"
    return header + "\n", regressions


def _write_dashboard(conn: sqlite3.Connection) -> str:
    sections, regressions = [], []
    for task in PROBE_TASKS:
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
