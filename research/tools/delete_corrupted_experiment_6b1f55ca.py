"""One-shot deletion of corrupted investigation experiment 6b1f55ca-0ac.

Why this experiment is trash:

* On 2026-05-02 13:22:51 it ran an investigation rerun on source_result_id
  ``7cda48ec-9f5`` (fingerprint ``3b42e14e72f0fd95``).
* Training tripped ``inflight_no_progress`` at step 625/2500 and the post-S1
  probe block silently skipped blimp + the v1 induction/binding/ar probes.
* The background recording path tried to INSERT a new ``program_results``
  row claiming ``stage1_passed=True`` but missing 5 of the 7 universal-guard
  required metrics → the universal S1 metric guard correctly blocked the
  write.
* Result: ``program_results`` and ``leaderboard`` are untouched (the original
  source row + leaderboard entry are intact), but the experiment row plus
  10 entries + 17 insights + 1 preregistration record were left in the DB
  with no underlying architecture data.

What this script removes (in a single transaction):

* ``experiments.experiment_id = '6b1f55ca-0ac'``
* ``entries WHERE experiment_id = '6b1f55ca-0ac'`` (~10 rows)
* ``insights WHERE experiment_id = '6b1f55ca-0ac'`` (~17 rows)
* ``hypothesis_preregistrations.preregistration_id = 'fbb0e5d8-7d7'``

It does NOT touch ``program_results`` or ``leaderboard``: the user-visible
fingerprint state for ``7cda48ec-9f5`` / ``3b42e14e72f0fd95`` is preserved
(iv2=1.0, bv2=0.99, composite=382 on the validation tier).

Run with the dashboard / continuous runner / backfill stopped (the writer
flock on lab_notebook.db must be free).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from research.scientist.notebook import LabNotebook  # noqa: E402


CORRUPTED_EXPERIMENT_ID = "6b1f55ca-0ac"
CORRUPTED_PREREGISTRATION_ID = "fbb0e5d8-7d7"
DEFAULT_DB_PATH = "/home/tim/Projects/LLM/research/lab_notebook.db"

# Leaderboard side-effect of the corrupted experiment.  Even though
# record_program_result raised at the universal-S1 guard, the upsert_leaderboard
# call earlier in _record_investigation_result had already succeeded and stamped
# investigation_* fields on the row for fp=3b42e14e72f0fd95.
#
# Pre-corruption snapshot (lab_notebook.db.snap_20260502T011003) had ALL four
# of these as NULL on entry_id 8e601381-c8d, so resetting to NULL restores
# faithfully without losing any pre-existing valid investigation data.
#
# tier (validation) and validation_loss_ratio (0.5712) are NOT touched — those
# came from a separate validation experiment that ran between the snapshot
# (01:10) and the corruption (13:22); they're not corrupted state.
CORRUPTED_LEADERBOARD_ENTRY_ID = "8e601381-c8d"
CORRUPTED_LEADERBOARD_FIELDS_TO_NULL = (
    "investigation_loss_ratio",
    "investigation_robustness",
    "investigation_best_training",
    "investigation_passed",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=DEFAULT_DB_PATH)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete (without this flag, the script reports counts and exits).",
    )
    args = parser.parse_args()

    # Dry run uses a read-only connection so the user can preview while the
    # dashboard or any other writer still holds the lab_notebook.db flock.
    # --apply needs the writer-lock; bail clearly if it's held.
    nb = LabNotebook(args.db, read_only=not args.apply)
    try:
        before = {
            "experiments": nb.conn.execute(
                "SELECT COUNT(*) FROM experiments WHERE experiment_id = ?",
                (CORRUPTED_EXPERIMENT_ID,),
            ).fetchone()[0],
            "entries": nb.conn.execute(
                "SELECT COUNT(*) FROM entries WHERE experiment_id = ?",
                (CORRUPTED_EXPERIMENT_ID,),
            ).fetchone()[0],
            "insights": nb.conn.execute(
                "SELECT COUNT(*) FROM insights WHERE experiment_id = ?",
                (CORRUPTED_EXPERIMENT_ID,),
            ).fetchone()[0],
            "hypothesis_preregistrations": nb.conn.execute(
                "SELECT COUNT(*) FROM hypothesis_preregistrations "
                "WHERE preregistration_id = ?",
                (CORRUPTED_PREREGISTRATION_ID,),
            ).fetchone()[0],
        }
        # Leaderboard side-effect inspection: report current vs. target
        lb_row = nb.conn.execute(
            "SELECT "
            + ", ".join(CORRUPTED_LEADERBOARD_FIELDS_TO_NULL)
            + " FROM leaderboard WHERE entry_id = ?",
            (CORRUPTED_LEADERBOARD_ENTRY_ID,),
        ).fetchone()
        leaderboard_to_null = []
        if lb_row is not None:
            for col in CORRUPTED_LEADERBOARD_FIELDS_TO_NULL:
                if lb_row[col] is not None:
                    leaderboard_to_null.append((col, lb_row[col]))
        # Safety guard: refuse to run if program_results contains rows under
        # this experiment.  The whole premise of "this is corrupted trash"
        # depends on no architecture data being attached.
        n_pr = nb.conn.execute(
            "SELECT COUNT(*) FROM program_results_compat WHERE experiment_id = ?",
            (CORRUPTED_EXPERIMENT_ID,),
        ).fetchone()[0]
        if n_pr:
            print(
                f"ABORT: program_results has {n_pr} rows under "
                f"{CORRUPTED_EXPERIMENT_ID}. Refusing to delete."
            )
            return 2

        print(f"Pre-delete counts for {CORRUPTED_EXPERIMENT_ID}:")
        for tbl, n in before.items():
            print(f"  {tbl:35s}  {n:4d}")
        total = sum(before.values())
        print(f"  {'TOTAL':35s}  {total:4d}")
        if leaderboard_to_null:
            print(
                f"\nLeaderboard side-effect on entry_id {CORRUPTED_LEADERBOARD_ENTRY_ID}"
                f" — these fields will be reset to NULL:"
            )
            for col, val in leaderboard_to_null:
                print(f"  {col:35s}  current={val!r}")
        else:
            print(
                f"\nLeaderboard entry_id {CORRUPTED_LEADERBOARD_ENTRY_ID}: "
                f"no investigation_* fields to reset (already clean)."
            )

        if not args.apply:
            print("\nDry run.  Re-run with --apply to delete + reset.")
            return 0

        nb.conn.execute(
            "DELETE FROM entries WHERE experiment_id = ?",
            (CORRUPTED_EXPERIMENT_ID,),
        )
        nb.conn.execute(
            "DELETE FROM insights WHERE experiment_id = ?",
            (CORRUPTED_EXPERIMENT_ID,),
        )
        nb.conn.execute(
            "DELETE FROM hypothesis_preregistrations WHERE preregistration_id = ?",
            (CORRUPTED_PREREGISTRATION_ID,),
        )
        nb.conn.execute(
            "DELETE FROM experiments WHERE experiment_id = ?",
            (CORRUPTED_EXPERIMENT_ID,),
        )
        if leaderboard_to_null:
            set_clause = ", ".join(
                f"{col} = NULL" for col in CORRUPTED_LEADERBOARD_FIELDS_TO_NULL
            )
            nb.conn.execute(
                f"UPDATE leaderboard SET {set_clause} WHERE entry_id = ?",
                (CORRUPTED_LEADERBOARD_ENTRY_ID,),
            )
        nb.conn.commit()
        nb.flush_writes()

        # Verify
        after = {
            "experiments": nb.conn.execute(
                "SELECT COUNT(*) FROM experiments WHERE experiment_id = ?",
                (CORRUPTED_EXPERIMENT_ID,),
            ).fetchone()[0],
            "entries": nb.conn.execute(
                "SELECT COUNT(*) FROM entries WHERE experiment_id = ?",
                (CORRUPTED_EXPERIMENT_ID,),
            ).fetchone()[0],
            "insights": nb.conn.execute(
                "SELECT COUNT(*) FROM insights WHERE experiment_id = ?",
                (CORRUPTED_EXPERIMENT_ID,),
            ).fetchone()[0],
            "hypothesis_preregistrations": nb.conn.execute(
                "SELECT COUNT(*) FROM hypothesis_preregistrations "
                "WHERE preregistration_id = ?",
                (CORRUPTED_PREREGISTRATION_ID,),
            ).fetchone()[0],
        }
        print("\nPost-delete counts:")
        for tbl, n in after.items():
            print(f"  {tbl:35s}  {n:4d}")
        if any(after.values()):
            print("\nFAIL: residual rows remain.")
            return 1

        # Verify leaderboard reset
        if leaderboard_to_null:
            lb_after = nb.conn.execute(
                "SELECT "
                + ", ".join(CORRUPTED_LEADERBOARD_FIELDS_TO_NULL)
                + " FROM leaderboard WHERE entry_id = ?",
                (CORRUPTED_LEADERBOARD_ENTRY_ID,),
            ).fetchone()
            still_set = (
                [
                    col
                    for col in CORRUPTED_LEADERBOARD_FIELDS_TO_NULL
                    if lb_after is not None and lb_after[col] is not None
                ]
                if lb_after is not None
                else []
            )
            if still_set:
                print(
                    f"\nFAIL: leaderboard fields still set on "
                    f"{CORRUPTED_LEADERBOARD_ENTRY_ID}: {still_set}"
                )
                return 1
            print(
                f"\nLeaderboard {CORRUPTED_LEADERBOARD_ENTRY_ID}: reset "
                f"{len(leaderboard_to_null)} corrupted-investigation fields."
            )

        print(f"\nDeleted {total} rows.  Experiment {CORRUPTED_EXPERIMENT_ID} purged.")
        return 0
    finally:
        nb.close()


if __name__ == "__main__":
    raise SystemExit(main())
