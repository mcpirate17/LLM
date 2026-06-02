#!/usr/bin/env python
"""Shard a prepared Colab probe bundle into N parallel jobs (nb10-parallel style).

``prepare_colab_probe_backfill`` writes one ``candidates_<probe>.jsonl`` + one
sequential ``run_<probe>_colab.py``. This splits each candidate file round-robin
into N shards under ``temp_<probe>_parallel/`` and writes ONE driver
``run_<probe>_parallel_colab.py`` that launches N subprocesses concurrently, each
on its own shard writing its own ``colab_<probe>_report_split_K.jsonl``. Resume is
per-shard (each backfill skips already-scored ids in its split report).

Cheap probes (ar_gate / nb05 / nb10) are DB-free: shard → N jsonl reports.
Induction uses ``backpopulate_screening_metrics`` which reads graphs from and
writes metrics to a SQLite DB. To parallelize without copying the 875MB snapshot
N times onto Drive, each induction shard copies the snapshot to Colab-LOCAL disk
(/content), runs there, then extracts its induction columns into the SAME
``{result_id, status, updates}`` jsonl the cheap probes emit — so the local merge
path is uniform and only tiny reports sync back.

Usage:
    python -m research.tools.shard_parallel_backfill --probe ar_gate,nb05,nb10,induction --shards 4
Then locally, after Colab finishes a probe:
    python -m research.tools.shard_parallel_backfill merge-prep --probe ar_gate   # cats splits
    python -m research.tools.prepare_colab_probe_backfill merge --probe ar_gate --apply
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_DRIVE_DIR = Path("/home/tim/GoogleDrive/Colab Notebooks/llm_probe_backfill")
_CHEAP = {"ar_gate", "nb05", "nb10"}
_SUPPORTED = _CHEAP | {"induction"}


def _shard_candidates(src: Path, out_dir: Path, shards: int) -> list[int]:
    """Round-robin split src jsonl into out_dir/candidates_split_{K}.jsonl."""
    lines = [ln for ln in src.read_text(encoding="utf-8").splitlines() if ln.strip()]
    out_dir.mkdir(parents=True, exist_ok=True)
    buckets: list[list[str]] = [[] for _ in range(shards)]
    for i, ln in enumerate(lines):
        buckets[i % shards].append(ln)
    counts = []
    for k in range(shards):
        (out_dir / f"candidates_split_{k}.jsonl").write_text(
            "\n".join(buckets[k]) + ("\n" if buckets[k] else ""), encoding="utf-8"
        )
        counts.append(len(buckets[k]))
    return counts


def _cheap_cmd(probe: str) -> list[str]:
    if probe == "ar_gate":
        return [
            "sys.executable, '-m', 'research.tools.colab_ar_gate_backfill',",
            "'--device', 'cuda', '--candidates-jsonl', str(cand), '--limit', '0',",
            "'--seeds', '0,1,2', '--wikitext-warmup-steps', '0',",
            "'--finetune-steps', '400', '--timeout-s', '240',",
            "'--report-jsonl', str(rep), '--load-processed-from-report', str(rep),",
        ]
    # nb05 / nb10
    return [
        "sys.executable, '-m', 'research.tools.colab_language_control_nb_backfill',",
        f"'--probe', {probe!r}, '--device', 'cuda',",
        "'--candidates-jsonl', str(cand), '--limit', '0', '--report-jsonl', str(rep),",
    ]


def _worker_induction(probe: str) -> str:
    """Induction worker: local DB copy + extract metrics into cheap jsonl format."""
    return f"""
def worker(k):
    cand = TMP / f'candidates_split_{{k}}.jsonl'
    rep_status = TMP / f'colab_{probe}_status_split_{{k}}.tsv'
    rep = TMP / f'colab_{probe}_report_split_{{k}}.jsonl'
    localdb = Path(f'/content/snap_{{k}}.db')
    shutil.copy2(SNAPSHOT, localdb)
    log = TMP / f'colab_{probe}_log_split_{{k}}.log'
    cmd = [
        sys.executable, '-m', 'research.tools.backpopulate_screening_metrics',
        '--db', str(localdb), '--device', 'cuda', '--from-report', str(cand),
        '--post-train-target', 'induction', '--skip-rapid', '--limit', '0',
        '--batch-commit', '1', '--worker-timeout-seconds', '900',
        '--max-consecutive-failures', '25', '--report', str(rep_status),
    ]
    with log.open('a', encoding='utf-8') as f:
        rc = subprocess.call(cmd, cwd=str(SRC), stdout=f, stderr=subprocess.STDOUT)
    # extract induction columns for this shard into the cheap {{result_id,updates}} jsonl
    ids = [json.loads(l)['result_id'] for l in cand.read_text().splitlines() if l.strip()]
    cols = ['induction_screening_auc','induction_gap_accuracies','induction_screening_train_steps',
            'induction_screening_eval_examples','induction_probe_gaps','induction_screening_elapsed_ms',
            'induction_screening_metric_version']
    con = sqlite3.connect(str(localdb)); con.row_factory = sqlite3.Row
    have = {{r[1] for r in con.execute('PRAGMA table_info(graph_runs)')}}
    use = [c for c in cols if c in have]
    sel = ', '.join(['result_id', *use])
    with rep.open('w', encoding='utf-8') as out:
        q = ','.join('?' for _ in ids)
        for r in con.execute(f'SELECT {{sel}} FROM graph_runs WHERE induction_screening_auc IS NOT NULL AND result_id IN ({{q}})', ids):
            upd = {{c: r[c] for c in use if r[c] is not None}}
            if upd:
                out.write(json.dumps({{'result_id': r['result_id'], 'status': 'updated', 'updates': upd}}) + '\\n')
    con.close()
    return k, rc
"""


def _worker_cheap(probe: str) -> str:
    """DB-free worker: shard jsonl in -> split jsonl report out."""
    cmd_lines = "\n        ".join(_cheap_cmd(probe))
    return f"""
def worker(k):
    cand = TMP / f'candidates_split_{{k}}.jsonl'
    rep = TMP / f'colab_{probe}_report_split_{{k}}.jsonl'
    log = TMP / f'colab_{probe}_log_split_{{k}}.log'
    cmd = [
        {cmd_lines}
    ]
    with log.open('a', encoding='utf-8') as f:
        rc = subprocess.call(cmd, cwd=str(SRC), stdout=f, stderr=subprocess.STDOUT)
    return k, rc
"""


def _driver_source(probe: str, shards: int) -> str:
    """Build the parallel Colab driver script text for a probe."""
    is_induction = probe == "induction"
    worker = _worker_induction(probe) if is_induction else _worker_cheap(probe)
    snapshot_line = (
        "SNAPSHOT = BUNDLE / 'runs_colab_backfill.db'" if is_induction else ""
    )
    return f"""#!/usr/bin/env python3
\"\"\"PARALLEL {probe} backfill — {shards} shards. Paste into a blank Colab cell.\"\"\"
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json, os, shutil, sqlite3, subprocess, sys, tarfile

try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception:
    pass

BUNDLE = Path('/content/drive/MyDrive/Colab Notebooks/llm_probe_backfill')
TMP = BUNDLE / 'temp_{probe}_parallel'
SRC = Path('/content/llm_probe_source')
SHARDS = {shards}
{snapshot_line}
STATUS = BUNDLE / 'status_{probe}_parallel.json'

STATUS.write_text('{{\"state\":\"setup\"}}\\n', encoding='utf-8')
subprocess.check_call([sys.executable,'-m','pip','install','-q',
    'xxhash','zstandard','pyyaml','flask-cors','lightgbm','ninja'])
shutil.rmtree(SRC, ignore_errors=True); SRC.mkdir(parents=True, exist_ok=True)
with tarfile.open(BUNDLE / 'llm_probe_source.tgz', 'r:gz') as tar:
    tar.extractall(SRC)
os.environ['PYTHONPATH'] = str(SRC)
{worker}
STATUS.write_text('{{\"state\":\"running_parallel\"}}\\n', encoding='utf-8')
with ThreadPoolExecutor(max_workers=SHARDS) as ex:
    results = list(ex.map(worker, range(SHARDS)))
STATUS.write_text(json.dumps({{'state':'complete','results':results}}) + '\\n', encoding='utf-8')
print('DONE', results)
"""


def cmd_shard(args: argparse.Namespace) -> None:
    probes = [p.strip() for p in args.probe.split(",") if p.strip()]
    unknown = [p for p in probes if p not in _SUPPORTED]
    if unknown:
        raise SystemExit(f"unsupported probe(s) for parallel: {unknown}")
    summary: dict[str, object] = {}
    for probe in probes:
        src = args.drive_dir / f"candidates_{probe}.jsonl"
        if not src.exists():
            raise SystemExit(f"missing {src}; run prepare first")
        tmp = args.drive_dir / f"temp_{probe}_parallel"
        counts = _shard_candidates(src, tmp, args.shards)
        driver = args.drive_dir / f"run_{probe}_parallel_colab.py"
        driver.write_text(_driver_source(probe, args.shards), encoding="utf-8")
        summary[probe] = {"shard_counts": counts, "driver": str(driver)}
    print(json.dumps(summary, indent=2, sort_keys=True))


def cmd_merge_prep(args: argparse.Namespace) -> None:
    """Concatenate split reports into colab_<probe>_report.jsonl for the merge tool."""
    probes = [p.strip() for p in args.probe.split(",") if p.strip()]
    for probe in probes:
        tmp = args.drive_dir / f"temp_{probe}_parallel"
        splits = sorted(tmp.glob(f"colab_{probe}_report_split_*.jsonl"))
        out = args.drive_dir / f"colab_{probe}_report.jsonl"
        n = 0
        with out.open("w", encoding="utf-8") as f:
            for sp in splits:
                for ln in sp.read_text(encoding="utf-8").splitlines():
                    if ln.strip():
                        f.write(ln + "\n")
                        n += 1
        print(f"{probe}: merged {len(splits)} splits -> {out} ({n} rows)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--drive-dir", type=Path, default=DEFAULT_DRIVE_DIR)
    sub = ap.add_subparsers(dest="cmd")
    sh = sub.add_parser("shard", help="split candidates + write parallel drivers")
    sh.add_argument("--probe", default="ar_gate,nb05,nb10,induction")
    sh.add_argument("--shards", type=int, default=4)
    sh.set_defaults(func=cmd_shard)
    mp = sub.add_parser("merge-prep", help="cat split reports for the merge tool")
    mp.add_argument("--probe", default="ar_gate,nb05,nb10,induction")
    mp.set_defaults(func=cmd_merge_prep)
    args = ap.parse_args()
    if not getattr(args, "func", None):
        # default: shard
        args.probe = "ar_gate,nb05,nb10,induction"
        args.shards = 4
        cmd_shard(args)
        return
    args.func(args)


if __name__ == "__main__":
    main()
