"""Prepare and merge cheap Colab probe backfills.

This helper keeps Colab work focused on the synthetic probes that are cheap
enough to run opportunistically:

* ``binding`` -> ``binding_screening_auc``
* ``induction`` -> ``induction_screening_auc``
* ``ar_gate`` -> ``ar_gate_score``
* ``ar`` -> ``ar_legacy_auc`` (available, but not included in ``all``)

It creates a Google Drive bundle with a consistent SQLite snapshot, a compact
source tarball, candidate TSVs, and a runnable Colab notebook. After Colab
updates the snapshot DB, ``merge`` copies only non-null probe columns back into
the local ``graph_runs`` table. Local non-null values are preserved unless
``--force`` is supplied.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import tarfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from research.scientist.notebook.graph_artifacts import resolve_graph_json_value


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = ROOT / "research" / "runs.db"
DEFAULT_DRIVE_DIR = Path("/home/tim/GoogleDrive/Colab Notebooks/llm_probe_backfill")
SNAPSHOT_DB_NAME = "runs_colab_backfill.db"
SOURCE_TARBALL_NAME = "llm_probe_source.tgz"


@dataclass(frozen=True)
class ProbeSpec:
    name: str
    primary_column: str
    merge_columns: tuple[str, ...]
    default_limit: int
    worker_timeout_seconds: int


PROBES: dict[str, ProbeSpec] = {
    "binding": ProbeSpec(
        name="binding",
        primary_column="binding_screening_auc",
        merge_columns=(
            "binding_screening_auc",
            "binding_distance_accuracies",
            "binding_screening_eval_examples",
            "binding_probe_distances",
            "binding_screening_elapsed_ms",
            "binding_curriculum_auc",
            "binding_distance_accuracies_curriculum",
            "binding_curriculum_steps",
            "binding_curriculum_elapsed_ms",
            "binding_curriculum_protocol_version",
            "binding_screening_composite",
            "data_provenance_json",
        ),
        default_limit=750,
        worker_timeout_seconds=900,
    ),
    "induction": ProbeSpec(
        name="induction",
        primary_column="induction_screening_auc",
        merge_columns=(
            "induction_screening_auc",
            "induction_gap_accuracies",
            "induction_screening_train_steps",
            "induction_screening_eval_examples",
            "induction_screening_batch_size",
            "induction_probe_gaps",
            "induction_screening_elapsed_ms",
            "induction_screening_metric_version",
            "induction_screening_speed_mode",
            "induction_screening_pool_size",
            "binding_screening_composite",
            "data_provenance_json",
        ),
        default_limit=750,
        worker_timeout_seconds=900,
    ),
    "ar": ProbeSpec(
        name="ar",
        primary_column="ar_legacy_auc",
        merge_columns=(
            "ar_legacy_auc",
            "ar_legacy_final_acc",
            "ar_legacy_timed_out",
            "ar_legacy_above_chance",
            "data_provenance_json",
        ),
        default_limit=500,
        worker_timeout_seconds=1200,
    ),
    "ar_gate": ProbeSpec(
        name="ar_gate",
        primary_column="ar_gate_score",
        merge_columns=(
            "ar_gate_metric_version",
            "ar_gate_in_dist_pair_acc",
            "ar_gate_in_dist_class_acc",
            "ar_gate_held_pair_acc",
            "ar_gate_held_class_acc",
            "ar_gate_score",
            "ar_gate_status",
            "ar_gate_elapsed_ms",
            "ar_gate_train_steps_done",
            "ar_gate_no_go",
        ),
        default_limit=0,
        worker_timeout_seconds=240,
    ),
    "nb05": ProbeSpec(
        name="nb05",
        primary_column="language_control_s05_binding_score",
        merge_columns=(
            "language_control_metric_version",
            "language_control_s05_sentence_assoc_score",
            "language_control_s05_binding_order_acc",
            "language_control_s05_binding_score",
        ),
        default_limit=0,
        worker_timeout_seconds=180,
    ),
    "nb10": ProbeSpec(
        name="nb10",
        primary_column="language_control_s10_binding_score",
        merge_columns=(
            "language_control_metric_version",
            "language_control_s10_sentence_assoc_score",
            "language_control_s10_binding_order_acc",
            "language_control_s10_binding_score",
            "language_control_s10_checkpoints_json",
        ),
        default_limit=0,
        worker_timeout_seconds=300,
    ),
    "nano_bind": ProbeSpec(
        name="nano_bind",
        primary_column="failure_op",
        merge_columns=(
            "failure_op",
            "failure_details_json",
        ),
        default_limit=0,
        worker_timeout_seconds=90,
    ),
}


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row["name"]) for row in conn.execute(f"PRAGMA table_info({table})")}


def _selected_probe_specs(raw: str) -> list[ProbeSpec]:
    names = [part.strip().lower() for part in raw.split(",") if part.strip()]
    if not names or names == ["all"]:
        names = ["ar_gate", "nb05", "nb10", "nano_bind"]
    unknown = [name for name in names if name not in PROBES]
    if unknown:
        raise SystemExit(f"unknown probe(s): {', '.join(unknown)}")
    return [PROBES[name] for name in names]


def _candidate_rows(
    conn: sqlite3.Connection,
    spec: ProbeSpec,
    *,
    limit: int | None,
) -> list[dict[str, Any]]:
    sql = f"""
        WITH ranked AS (
            SELECT
                pr.result_id,
                pr.graph_fingerprint,
                pr.experiment_id,
                pr.graph_json,
                pr.timestamp,
                COALESCE(lb.tier, 'off_leaderboard') AS tier,
                COALESCE(lb.composite_score, pr.loss_ratio, 0.0) AS priority_score,
                COALESCE(pr.trust_label, '') AS trust_label,
                COALESCE(pr.comparability_label, '') AS comparability_label,
                COALESCE(pr.graph_n_ops, 0) AS graph_n_ops,
                COALESCE(pr.graph_depth, 0) AS graph_depth,
                ROW_NUMBER() OVER (
                    PARTITION BY pr.graph_fingerprint
                    ORDER BY
                        CASE WHEN lb.entry_id IS NOT NULL THEN 0 ELSE 1 END,
                        COALESCE(lb.composite_score, 0.0) DESC,
                        pr.timestamp DESC,
                        pr.result_id DESC
                ) AS rn
            FROM program_results_compat pr
            LEFT JOIN leaderboard lb ON lb.result_id = pr.result_id
            WHERE pr.graph_fingerprint IS NOT NULL
              AND TRIM(COALESCE(pr.graph_json, '')) <> ''
              AND pr.graph_json <> '{{}}'
              AND length(pr.graph_json) > 10
              AND pr.stage0_passed = 1
              AND pr.stage05_passed = 1
              AND COALESCE(pr.trust_label, '') <> 'reference'
              AND {_missing_clause(spec)}
        )
        SELECT *
        FROM ranked
        WHERE rn = 1
        ORDER BY
            CASE WHEN tier IN ('breakthrough', 'validation', 'investigation') THEN 0 ELSE 1 END,
            priority_score DESC,
            graph_n_ops * graph_depth ASC,
            timestamp DESC
    """
    if limit is not None and limit > 0:
        sql += f" LIMIT {int(limit)}"
    return [dict(row) for row in conn.execute(sql)]


def _missing_clause(spec: ProbeSpec) -> str:
    if spec.name == "nano_bind":
        return "COALESCE(pr.failure_op, '') <> 'nano_bind'"
    return f"pr.{spec.primary_column} IS NULL"


def _write_candidate_tsv(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "result_id",
        "graph_fingerprint",
        "tier",
        "priority_score",
        "graph_n_ops",
        "graph_depth",
        "experiment_id",
        "timestamp",
        "trust_label",
        "comparability_label",
    ]
    count = 0
    with path.open("w", encoding="utf-8") as f:
        f.write("\t".join(headers) + "\n")
        for row in rows:
            count += 1
            values = [
                str(row.get(header, ""))
                .replace("\t", " ")
                .replace("\n", " ")
                .replace("\r", " ")
                for header in headers
            ]
            f.write("\t".join(values) + "\n")
    return count


def _write_candidate_jsonl(
    path: Path,
    rows: Iterable[dict[str, Any]],
    *,
    conn: sqlite3.Connection,
    db_path: Path,
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            payload = dict(row)
            payload.pop("rn", None)
            payload["graph_json"] = resolve_graph_json_value(
                conn,
                db_path,
                payload.get("graph_json"),
            )
            f.write(json.dumps(payload, sort_keys=True) + "\n")
            count += 1
    return count


def _iter_source_files() -> list[Path]:
    cmd = [
        "git",
        "ls-files",
        "-z",
        "--cached",
        "--modified",
        "--others",
        "--exclude-standard",
        "--",
        "research",
        "aria_core",
        "component_fab",
        "pyproject.toml",
        "uv.lock",
        "README.md",
    ]
    raw = subprocess.check_output(cmd, cwd=ROOT)
    files: list[Path] = []
    excluded_parts = {
        "__pycache__",
        ".pytest_cache",
        "node_modules",
        "build",
        "dist",
        ".venv",
    }
    excluded_prefixes = (
        Path("research/artifacts"),
        Path("research/checkpoints"),
        Path("research/corpus"),
        Path("research/data"),
        Path("research/db_backups"),
        Path("research/perf_artifacts"),
        Path("research/reports"),
        Path("research/runtime_events"),
        Path("research/cache"),
        Path("research/profiling"),
        # Keep runtime source code in the bundle, but exclude large generated
        # runtime artifacts/checkpoints. Some graph import paths require
        # ``research.runtime.native`` even when the native extension itself is
        # not built on Colab.
        Path("research/runtime/120m_tropical_gate_pretrain_resume_20260516_211800"),
        Path("research/runtime/ar_validation_fingerprint_sweep"),
        Path("research/runtime/champion_reference_calibration"),
        Path("research/runtime/champion_reference_tests"),
        Path("research/runtime/ar_curriculum_experiment"),
        Path("research/runtime/small_ar_calibration"),
        Path("research/runtime/backfill"),
        Path("research/runtime/native/build"),
        Path("research/runtime/native/rust"),
    )
    excluded_suffixes = {
        ".db",
        ".db-shm",
        ".db-wal",
        ".npy",
        ".npz",
        ".pt",
        ".pth",
        ".safetensors",
        ".jsonl",
        ".ndjson",
        ".log",
    }
    for part in raw.split(b"\0"):
        if not part:
            continue
        rel = Path(os.fsdecode(part))
        path = ROOT / rel
        if not path.is_file():
            continue
        if any(piece in excluded_parts for piece in rel.parts):
            continue
        if any(rel == prefix or prefix in rel.parents for prefix in excluded_prefixes):
            continue
        if path.suffix in excluded_suffixes:
            continue
        files.append(rel)
    return sorted(set(files))


def _write_source_tarball(path: Path) -> int:
    files = _iter_source_files()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as tar:
        for rel in files:
            tar.add(ROOT / rel, arcname=str(rel))
    return len(files)


def _write_colab_script(path: Path, spec: ProbeSpec, limit: int) -> None:
    if spec.name == "ar_gate":
        command_lines = [
            "    sys.executable, '-m', 'research.tools.colab_ar_gate_backfill',",
            "    '--device', 'cuda',",
            "    '--candidates-jsonl', str(candidate),",
            "    '--limit', str(LIMIT),",
            "    '--seeds', '0,1,2',",
            "    '--wikitext-warmup-steps', '0',",
            "    '--finetune-steps', '400',",
            "    '--timeout-s', str(TIMEOUT),",
            "    '--report-jsonl', str(report),",
            "    '--load-processed-from-report', str(report),",
        ]
        report_ext = "jsonl"
    elif spec.name in {"nb05", "nb10"}:
        command_lines = [
            "    sys.executable, '-m', 'research.tools.colab_language_control_nb_backfill',",
            "    '--probe', TARGET,",
            "    '--device', 'cuda',",
            "    '--candidates-jsonl', str(candidate),",
            "    '--limit', str(LIMIT),",
            "    '--report-jsonl', str(report),",
        ]
        report_ext = "jsonl"
    elif spec.name == "nano_bind":
        command_lines = [
            "    sys.executable, '-m', 'research.tools.colab_nano_bind_backfill',",
            "    '--device', 'cuda',",
            "    '--candidates-jsonl', str(candidate),",
            "    '--limit', str(LIMIT),",
            "    '--seed', '0',",
            "    '--report-jsonl', str(report),",
        ]
        report_ext = "jsonl"
    else:
        command_lines = [
            "    sys.executable, '-m', 'research.tools.backpopulate_screening_metrics',",
            "    '--db', str(DB),",
            "    '--device', 'cuda',",
            "    '--from-report', str(candidate),",
            "    '--post-train-target', TARGET,",
            "    '--skip-rapid',",
            "    '--limit', str(LIMIT),",
            "    '--batch-commit', '1',",
            "    '--worker-timeout-seconds', str(TIMEOUT),",
            "    '--max-consecutive-failures', '25',",
            "    '--report', str(report),",
        ]
        report_ext = "tsv"
    lines = [
        "#!/usr/bin/env python3",
        f'"""Run the {spec.name} cheap probe backfill from Drive JSONL candidates."""',
        "",
        "from pathlib import Path",
        "import os",
        "import shutil",
        "import subprocess",
        "import sys",
        "import tarfile",
        "",
        "try:",
        "    from google.colab import drive",
        "",
        "    drive.mount('/content/drive')",
        "except Exception:",
        "    pass",
        "",
        "BUNDLE = Path('/content/drive/MyDrive/Colab Notebooks/llm_probe_backfill')",
        "SRC = Path('/content/llm_probe_source')",
        f"TARGET = {spec.name!r}",
        f"LIMIT = {int(limit)}",
        f"TIMEOUT = {int(spec.worker_timeout_seconds)}",
        "candidate = BUNDLE / f'candidates_{TARGET}.jsonl'",
        f"report = BUNDLE / f'colab_{{TARGET}}_report.{report_ext}'",
        "log = BUNDLE / f'colab_{TARGET}.log'",
        "status = BUNDLE / f'status_{TARGET}.json'",
        "",
        'status.write_text(\'{"state":"setup","step":"install"}\\n\', encoding=\'utf-8\')',
        "subprocess.check_call([",
        "    sys.executable,",
        "    '-m',",
        "    'pip',",
        "    'install',",
        "    '-q',",
        "    'xxhash',",
        "    'zstandard',",
        "    'pyyaml',",
        "    'flask-cors',",
        "    'lightgbm',",
        "    'ninja',",
        "])",
        "",
        'status.write_text(\'{"state":"setup","step":"extract"}\\n\', encoding=\'utf-8\')',
        "shutil.rmtree(SRC, ignore_errors=True)",
        "SRC.mkdir(parents=True, exist_ok=True)",
        "with tarfile.open(BUNDLE / 'llm_probe_source.tgz', 'r:gz') as tar:",
        "    tar.extractall(SRC)",
        "os.environ['PYTHONPATH'] = str(SRC)",
        "cmd = [",
        *command_lines,
        "]",
        "if not candidate.exists() or candidate.stat().st_size == 0:",
        "    raise FileNotFoundError(f'Missing candidate file: {candidate}')",
        "status.write_text('{\"state\":\"running\"}\\n', encoding='utf-8')",
        "print(' '.join(cmd), flush=True)",
        "with log.open('a', encoding='utf-8') as f:",
        "    proc = subprocess.Popen(cmd, cwd=str(SRC), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)",
        "    assert proc.stdout is not None",
        "    for line in proc.stdout:",
        "        print(line, end='')",
        "        f.write(line)",
        "        f.flush()",
        "    rc = proc.wait()",
        'status.write_text(\'{"state":"complete","returncode":%d}\\n\' % rc, encoding=\'utf-8\')',
        "raise SystemExit(rc)",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_readme(
    path: Path, *, probes: list[ProbeSpec], counts: dict[str, int]
) -> None:
    lines = [
        "# LLM cheap probe Colab backfill",
        "",
        "This bundle is for cheap gate/probe runs only.",
        "Default `all` includes AR gate, NB0.5, NB1.0, and NanoBind. Legacy AR is not included.",
        "It does not copy the WikiText corpus and does not run BPE/full evals.",
        "",
        "## Files",
        "",
        f"- `{SOURCE_TARBALL_NAME}`: compact repo source needed by the runner.",
        "- `candidates_<probe>.jsonl`: self-contained candidate rows with graph JSON.",
        "- `candidates_<probe>.tsv`: quick human-readable candidate IDs.",
        "- `run_<probe>_colab.py`: one plain Python Colab command script per probe.",
        "",
        "## Colab use",
        "",
        "Open a blank Colab and paste the contents of the relevant `run_<probe>_colab.py` file.",
        "Each script mounts Drive, installs the small Python deps, unpacks the source tarball, and streams a log.",
        "The scripts do not open the SQLite DB in Colab; they read `candidates_<probe>.jsonl` and write `colab_<probe>_report.jsonl`.",
        "",
        "## Suggested order",
        "",
        "1. `ar_gate`: active AR scoring signal; notebook averages seeds 0,1,2.",
        "2. `nb05`: language-control NanoBind S0.5.",
        "3. `nb10`: language-control NanoBind S1.0.",
        "4. `nano_bind`: hard no-go NanoBind sidecar/report path.",
        "",
        "## Candidate counts",
        "",
    ]
    for spec in probes:
        lines.append(f"- `{spec.name}`: {counts.get(spec.name, 0)} rows")
    lines.extend(
        [
            "",
            "## Merge back locally",
            "",
            "From `/home/tim/Projects/LLM`, after Drive syncs:",
            "",
            "```bash",
            "uv run python -m research.tools.prepare_colab_probe_backfill monitor --probe all",
            "uv run python -m research.tools.prepare_colab_probe_backfill merge --probe ar_gate",
            "uv run python -m research.tools.prepare_colab_probe_backfill merge --probe ar_gate --apply",
            "```",
            "",
            "The merge preserves existing local non-null metrics by default.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def cmd_plan(args: argparse.Namespace) -> None:
    specs = _selected_probe_specs(args.probe)
    with _connect(args.db) as conn:
        summary = {}
        for spec in specs:
            rows = _candidate_rows(conn, spec, limit=None)
            summary[spec.name] = len(rows)
            print(f"{spec.name:<10} missing_candidates={len(rows):>6}")
        print(json.dumps(summary, indent=2, sort_keys=True))


def cmd_prepare(args: argparse.Namespace) -> None:
    specs = _selected_probe_specs(args.probe)
    drive_dir = args.drive_dir
    drive_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    limits: dict[str, int] = {}

    with _connect(args.db) as conn:
        for spec in specs:
            limit = args.limit if args.limit is not None else spec.default_limit
            limits[spec.name] = int(limit)
            rows = _candidate_rows(conn, spec, limit=limit)
            counts[spec.name] = _write_candidate_tsv(
                drive_dir / f"candidates_{spec.name}.tsv", rows
            )
            _write_candidate_jsonl(
                drive_dir / f"candidates_{spec.name}.jsonl",
                rows,
                conn=conn,
                db_path=args.db,
            )

    source_count = _write_source_tarball(drive_dir / SOURCE_TARBALL_NAME)
    for spec in specs:
        _write_colab_script(
            drive_dir / f"run_{spec.name}_colab.py",
            spec,
            limits[spec.name],
        )
    _write_readme(drive_dir / "README.md", probes=specs, counts=counts)
    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_db": str(args.db),
        "snapshot_db": None,
        "source_tarball": str(drive_dir / SOURCE_TARBALL_NAME),
        "source_file_count": source_count,
        "candidate_counts": counts,
        "limits": limits,
    }
    (drive_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


def _make_pre_merge_backup(db_path: Path) -> Path:
    out_dir = ROOT / "research" / "db_backups"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%S")
    dst = out_dir / f"runs_pre_colab_probe_merge_{ts}.db"
    shutil.copy2(db_path, dst)
    return dst


def _merge_probe(
    local_conn: sqlite3.Connection,
    source_conn: sqlite3.Connection,
    spec: ProbeSpec,
    *,
    force: bool,
    apply: bool,
) -> dict[str, int]:
    local_cols = _table_columns(local_conn, "graph_runs")
    source_cols = _table_columns(source_conn, "graph_runs")
    columns = [
        col for col in spec.merge_columns if col in local_cols and col in source_cols
    ]
    if not columns:
        return {"source_rows": 0, "updated_rows": 0, "updated_values": 0}

    selected = ", ".join(["result_id", *columns])
    where = f"{spec.primary_column} IS NOT NULL"
    if spec.name == "nano_bind":
        where = "failure_op = 'nano_bind'"
    rows = source_conn.execute(
        f"""
        SELECT {selected}
        FROM graph_runs
        WHERE {where}
        """
    ).fetchall()
    updated_rows = 0
    updated_values = 0
    for row in rows:
        result_id = row["result_id"]
        target = local_conn.execute(
            f"SELECT {selected} FROM graph_runs WHERE result_id = ?",
            (result_id,),
        ).fetchone()
        if target is None:
            continue
        updates: dict[str, Any] = {}
        for col in columns:
            value = row[col]
            if value is None:
                continue
            if force or target[col] is None:
                updates[col] = value
        if not updates:
            continue
        updated_rows += 1
        updated_values += len(updates)
        if apply:
            set_clause = ", ".join(f"{col} = ?" for col in updates)
            local_conn.execute(
                f"UPDATE graph_runs SET {set_clause} WHERE result_id = ?",
                (*updates.values(), result_id),
            )
    return {
        "source_rows": len(rows),
        "updated_rows": updated_rows,
        "updated_values": updated_values,
    }


def _updates_from_report_record(
    spec: ProbeSpec, record: dict[str, Any]
) -> dict[str, Any]:
    if record.get("status") in {"error", "unparseable"}:
        return {}
    if spec.name == "ar_gate":
        payload = record.get("payload")
        return dict(payload) if isinstance(payload, dict) else {}
    if spec.name in {"nb05", "nb10", "induction", "binding"}:
        # Parallel shard runners emit {result_id, status, updates} jsonl for these
        # (induction/binding extracted from per-shard local DB copies).
        payload = record.get("updates")
        return dict(payload) if isinstance(payload, dict) else {}
    if spec.name == "nano_bind" and record.get("status") == "no_go":
        nb = record.get("nano_bind")
        if not isinstance(nb, dict):
            return {}
        details = {
            "reason": "nano_bind_persistent_zero",
            "scores": nb.get("nano_bind_scores"),
            "metric_version": nb.get("nano_bind_metric_version"),
            "checkpoints": (nb.get("nano_bind_sweep_metadata") or {}).get(
                "checkpoints", []
            )
            if isinstance(nb.get("nano_bind_sweep_metadata"), dict)
            else [],
        }
        return {
            "failure_op": "nano_bind",
            "failure_details_json": json.dumps(
                details, separators=(",", ":"), sort_keys=True
            ),
        }
    return {}


def _merge_probe_report(
    local_conn: sqlite3.Connection,
    report_path: Path,
    spec: ProbeSpec,
    *,
    force: bool,
    apply: bool,
) -> dict[str, int]:
    if not report_path.exists():
        return {"source_rows": 0, "updated_rows": 0, "updated_values": 0}
    local_cols = _table_columns(local_conn, "graph_runs")
    columns = [col for col in spec.merge_columns if col in local_cols]
    source_rows = 0
    updated_rows = 0
    updated_values = 0
    with report_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            source_rows += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            result_id = str(record.get("result_id") or "")
            if not result_id:
                continue
            updates_raw = _updates_from_report_record(spec, record)
            updates_raw = {
                col: value
                for col, value in updates_raw.items()
                if col in columns and value is not None
            }
            if not updates_raw:
                continue
            selected = ", ".join(["result_id", *updates_raw.keys()])
            target = local_conn.execute(
                f"SELECT {selected} FROM graph_runs WHERE result_id = ?",
                (result_id,),
            ).fetchone()
            if target is None:
                continue
            updates = {
                col: value
                for col, value in updates_raw.items()
                if force or target[col] is None
            }
            if not updates:
                continue
            updated_rows += 1
            updated_values += len(updates)
            if apply:
                set_clause = ", ".join(f"{col} = ?" for col in updates)
                local_conn.execute(
                    f"UPDATE graph_runs SET {set_clause} WHERE result_id = ?",
                    (*updates.values(), result_id),
                )
    return {
        "source_rows": source_rows,
        "updated_rows": updated_rows,
        "updated_values": updated_values,
    }


def cmd_merge(args: argparse.Namespace) -> None:
    specs = _selected_probe_specs(args.probe)
    source_db = args.source_db or (args.drive_dir / SNAPSHOT_DB_NAME)
    backup_path = None
    if args.apply:
        backup_path = _make_pre_merge_backup(args.db)
    summary: dict[str, Any] = {
        "apply": bool(args.apply),
        "force": bool(args.force),
        "local_db": str(args.db),
        "source_db": str(source_db),
        "backup": str(backup_path) if backup_path else None,
        "probes": {},
    }
    with _connect(args.db) as local_conn:
        for spec in specs:
            report_path = args.drive_dir / f"colab_{spec.name}_report.jsonl"
            if report_path.exists():
                summary["probes"][spec.name] = _merge_probe_report(
                    local_conn,
                    report_path,
                    spec,
                    force=args.force,
                    apply=args.apply,
                )
            elif source_db.exists():
                with _connect(source_db) as source_conn:
                    summary["probes"][spec.name] = _merge_probe(
                        local_conn,
                        source_conn,
                        spec,
                        force=args.force,
                        apply=args.apply,
                    )
            else:
                summary["probes"][spec.name] = {
                    "source_rows": 0,
                    "updated_rows": 0,
                    "updated_values": 0,
                }
        if args.apply:
            local_conn.commit()
    print(json.dumps(summary, indent=2, sort_keys=True))


def _report_status_counts(path: Path) -> tuple[int, dict[str, int]]:
    counts: Counter[str] = Counter()
    if not path.exists():
        return 0, {}
    if path.suffix == ".jsonl":
        total = 0
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                total += 1
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    counts["unparseable"] += 1
                    continue
                counts[str(payload.get("status") or "unknown")] += 1
        return total, dict(counts)
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return 0, {}
    header = lines[0].split("\t")
    try:
        status_idx = header.index("status")
    except ValueError:
        status_idx = -1
    for line in lines[1:]:
        parts = line.split("\t")
        status = (
            parts[status_idx]
            if status_idx >= 0 and status_idx < len(parts)
            else "unknown"
        )
        counts[status] += 1
    return max(0, len(lines) - 1), dict(counts)


def cmd_monitor(args: argparse.Namespace) -> None:
    specs = _selected_probe_specs(args.probe)
    summary: dict[str, Any] = {}
    for spec in specs:
        status_path = args.drive_dir / f"status_{spec.name}.json"
        log_path = args.drive_dir / f"colab_{spec.name}.log"
        report_jsonl = args.drive_dir / f"colab_{spec.name}_report.jsonl"
        report_tsv = args.drive_dir / f"colab_{spec.name}_report.tsv"
        report_path = (
            report_jsonl
            if spec.name in {"ar_gate", "nb05", "nb10", "nano_bind"}
            else report_tsv
        )
        report_rows, report_counts = _report_status_counts(report_path)
        status_payload: dict[str, Any] = {}
        if status_path.exists():
            try:
                status_payload = json.loads(status_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                status_payload = {"state": "unparseable_status"}
        latest_log = ""
        if log_path.exists():
            try:
                latest_log = log_path.read_text(encoding="utf-8").splitlines()[-1]
            except IndexError:
                latest_log = ""
        summary[spec.name] = {
            "state": status_payload.get("state", "not_started"),
            "updated_at": status_payload.get("updated_at"),
            "candidate_rows": status_payload.get("candidate_rows"),
            "report_rows": report_rows,
            "report_status_counts": report_counts,
            "latest_line": status_payload.get("latest_line") or latest_log,
            "returncode": status_payload.get("returncode"),
            "status_file": str(status_path),
            "report_file": str(report_path),
            "log_file": str(log_path),
        }
    print(json.dumps(summary, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--drive-dir", type=Path, default=DEFAULT_DRIVE_DIR)
    sub = parser.add_subparsers(dest="cmd", required=True)

    plan = sub.add_parser("plan", help="show missing cheap-probe candidates")
    plan.add_argument("--probe", default="all")
    plan.set_defaults(func=cmd_plan)

    prepare = sub.add_parser("prepare", help="write the Drive Colab bundle")
    prepare.add_argument("--probe", default="all")
    prepare.add_argument("--limit", type=int, default=None)
    prepare.set_defaults(func=cmd_prepare)

    merge = sub.add_parser("merge", help="merge Colab snapshot metrics locally")
    merge.add_argument("--probe", default="all")
    merge.add_argument("--source-db", type=Path, default=None)
    merge.add_argument("--apply", action="store_true")
    merge.add_argument("--force", action="store_true")
    merge.set_defaults(func=cmd_merge)

    monitor = sub.add_parser("monitor", help="read Colab status/log/report files")
    monitor.add_argument("--probe", default="all")
    monitor.set_defaults(func=cmd_monitor)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
