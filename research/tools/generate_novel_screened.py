#!/usr/bin/env python
"""Generate NOVEL architectures at scale, each carrying >=1 novel unpublished mixer.

Standalone discovery generator. Streams candidates to JSONL so it scales to millions
on bounded memory. Every emitted graph satisfies three HARD gates:

  1. NOVEL      — structural fingerprint (g.fingerprint(), DB hash space) never seen.
  2. NOVEL MIXER— contains >=1 mixer op classified novel/unpublished in
                  literature_attribution (MIX role, match_type in {novel, partial}):
                  tropical/ultrametric/STDP/clifford/gated-progressive attention, etc.
  3. COMPILES   — generate_layer_graph(validate=True) only returns valid graphs.

Anything else is allowed and encouraged — softmax/SSM/linear attention, branching,
compression, MoE — so a novel mixer sits inside a learnable architecture. Generation
is softly biased toward the novel mixers (so >=1 appears) and the historical learning
scaffold (rope/attention/norm, mined from capable graphs) so the graph can still learn.

The capability screener scores every survivor (predicted induction) and ranks them, but
score is NOT a gate: the point is to explore novel mixers, including ones the screener —
trained on the familiar past — under-rates.

Usage::

    python -m research.tools.generate_novel_screened --target 5000
    python -m research.tools.generate_novel_screened --target 1000000 --workers 8
    python -m research.tools.generate_novel_screened --max-attempts 4000 --no-bias
"""

from __future__ import annotations

import argparse
import heapq
import json
import logging
import multiprocessing as mp
import sqlite3
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np

from research.defaults import RUNS_DB
from research.synthesis.grammar import GrammarConfig, generate_layer_graph
from research.synthesis.op_roles import OpRole, get_role
from research.synthesis.primitives import PRIMITIVE_REGISTRY
from research.tools.capability_screener import featurize_op_sets, load_screener

logger = logging.getLogger(__name__)

_NOVEL_CLASSES = ("novel", "partial")


def _historical_fingerprints(db_path: str) -> Set[str]:
    """Every graph fingerprint ever recorded — the 'already seen' set."""
    con = sqlite3.connect(db_path)
    seen: Set[str] = set()
    for table in ("graph_runs", "program_graph_features", "graphs"):
        try:
            for (fp,) in con.execute(
                f"SELECT DISTINCT graph_fingerprint FROM {table} "  # nosec B608  # nosemgrep: python-sql-string-formatting
                "WHERE graph_fingerprint IS NOT NULL"
            ):
                seen.add(str(fp))
        except sqlite3.OperationalError:
            continue
    con.close()
    return seen


def _novel_mixers(db_path: str, classes: Tuple[str, ...] = _NOVEL_CLASSES) -> Set[str]:
    """MIX-role ops whose literature match_type is novel/unpublished."""
    con = sqlite3.connect(db_path)
    attr = {
        str(k): str(m)
        for k, m in con.execute(
            "SELECT entity_key, match_type FROM literature_attribution "
            "WHERE entity_type='op'"
        )
    }
    con.close()
    want_unattributed = "unattributed" in classes
    out: Set[str] = set()
    for op in PRIMITIVE_REGISTRY:
        if get_role(op) is not OpRole.MIX:
            continue
        m = attr.get(op)
        if (m is None and want_unattributed) or (m in classes):
            out.add(op)
    return out


def _scaffold_boost(
    db_path: str, cap_thr: float, min_enrich: float = 3.0
) -> Dict[str, float]:
    """Learnable-scaffold ops enriched in historically capable graphs (soft bias)."""
    con = sqlite3.connect(db_path)
    cap: Dict[str, List[float]] = {}
    for fp, v in con.execute(
        "SELECT graph_fingerprint, induction_screening_auc FROM graph_runs "
        "WHERE induction_screening_auc IS NOT NULL"
    ):
        cap.setdefault(str(fp), []).append(float(v))
    ops: Dict[str, Set[str]] = {}
    for fp, op in con.execute(
        "SELECT graph_fingerprint, op_name FROM program_graph_ops"
    ):
        if str(fp) in cap:
            ops.setdefault(str(fp), set()).add(str(op))
    con.close()
    capable = [fp for fp, vs in cap.items() if float(np.mean(vs)) > cap_thr]
    incap = [fp for fp, vs in cap.items() if float(np.mean(vs)) <= cap_thr]
    nc, ni = max(len(capable), 1), max(len(incap), 1)
    cc, ic = Counter(), Counter()
    for fp in capable:
        for op in ops.get(fp, set()):
            cc[op] += 1
    for fp in incap:
        for op in ops.get(fp, set()):
            ic[op] += 1
    boost: Dict[str, float] = {}
    for op, n in cc.items():
        enrich = (n / nc) / max(ic[op] / ni, 1e-3)
        if n / nc >= 0.5 and enrich >= min_enrich:
            boost[op] = round(min(enrich, 8.0), 2)
    return boost


def _build_op_weights(
    novel_mixers: Set[str],
    scaffold: Dict[str, float],
    mixer_w: float,
    scaffold_w: float,
) -> Dict[str, float]:
    w: Dict[str, float] = {op: mixer_w for op in novel_mixers}
    for op in scaffold:
        w.setdefault(op, scaffold_w)
    return w


def _graph_features(g: Any) -> Tuple[Set[str], int, int, str]:
    """(op_set, op_count, pair_count, structural_fingerprint) — all free, no forward pass."""
    nodes = g.to_dict()["nodes"].values()
    op_nodes = [n for n in nodes if not n.get("is_input")]
    op_set = {str(n["op_name"]) for n in op_nodes}
    pair_count = sum(len(n.get("input_ids", [])) for n in op_nodes)
    return op_set, len(op_nodes), pair_count, str(g.fingerprint())


def _score_and_write(
    batch: List[Dict[str, Any]],
    booster: Any,
    vocab: List[str],
    fout: Any,
    top: List[Tuple[float, str, Dict[str, Any]]],
    top_k: int,
) -> int:
    """Screen a batch, stream it to JSONL, keep a running top-k heap. Returns batch size."""
    if not batch:
        return 0
    X = featurize_op_sets(
        [set(r["ops"]) for r in batch],
        [r["op_count"] for r in batch],
        [r["pair_count"] for r in batch],
        vocab,
    )
    for r, s in zip(batch, np.asarray(booster.predict(X), dtype=np.float64)):
        r["score"] = round(float(s), 4)
        fout.write(json.dumps(r) + "\n")
        entry = (r["score"], r["fingerprint"], r)
        if len(top) < top_k:
            heapq.heappush(top, entry)
        elif r["score"] > top[0][0]:
            heapq.heapreplace(top, entry)
    n = len(batch)
    batch.clear()
    return n


def _generate_stream(
    *,
    seeds: range,
    op_weights: Dict[str, float],
    novel_mixers: Set[str],
    hist_fps: Set[str],
    out_path: Path,
    target: int,
    top_k: int,
    batch_size: int,
    progress_every: int,
    tag: str,
) -> Tuple[Counter, List[Dict[str, Any]]]:
    """Core loop: generate -> gate (novel fp + novel mixer) -> screen batch -> stream JSONL."""
    cfg = GrammarConfig(op_weights=dict(op_weights))
    booster, meta = load_screener()
    vocab = list(meta["op_vocab"])
    stats: Counter = Counter()
    seen_new: Set[str] = set()
    top: List[Tuple[float, str, Dict[str, Any]]] = []
    batch: List[Dict[str, Any]] = []
    fout = out_path.open("w")
    t0 = time.time()

    for seed in seeds:
        try:
            g = generate_layer_graph(cfg, seed=seed)
        except Exception:
            stats["invalid"] += 1
            continue
        stats["generated"] += 1
        op_set, op_count, pair_count, fp = _graph_features(g)
        if fp in hist_fps or fp in seen_new:
            stats["already_seen"] += 1
            continue
        seen_new.add(fp)
        hits = sorted(op_set & novel_mixers)
        if not hits:
            stats["no_novel_mixer"] += 1
            continue
        batch.append(
            {
                "fingerprint": fp,
                "ops": sorted(op_set),
                "op_count": op_count,
                "pair_count": pair_count,
                "novel_mixers": hits,
            }
        )
        if len(batch) >= batch_size:
            stats["kept"] += _score_and_write(batch, booster, vocab, fout, top, top_k)
        if progress_every and stats["generated"] % progress_every == 0:
            logger.info(
                "[%s] gen=%d kept=%d (%.0f/s)",
                tag,
                stats["generated"],
                stats["kept"] + len(batch),
                stats["generated"] / max(time.time() - t0, 1e-9),
            )
        if stats["kept"] + len(batch) >= target:
            break
    stats["kept"] += _score_and_write(batch, booster, vocab, fout, top, top_k)
    fout.close()
    return stats, [t[2] for t in sorted(top, reverse=True)]


def _worker(payload: Dict[str, Any]) -> Tuple[Dict[str, int], List[Dict[str, Any]]]:
    stats, top = _generate_stream(
        seeds=range(
            payload["seed_start"], payload["seed_start"] + payload["seed_count"]
        ),
        op_weights=payload["op_weights"],
        novel_mixers=set(payload["novel_mixers"]),
        hist_fps=set(payload["hist_fps"]),
        out_path=Path(payload["out_path"]),
        target=payload["target"],
        top_k=payload["top_k"],
        batch_size=payload["batch_size"],
        progress_every=payload["progress_every"],
        tag=payload["tag"],
    )
    return dict(stats), top


def _run_single(
    args: argparse.Namespace,
    op_weights: Dict[str, float],
    novel_mixers: Set[str],
    hist: Set[str],
    out: Path,
) -> Tuple[Counter, List[Dict[str, Any]], List[str]]:
    stats, top = _generate_stream(
        seeds=range(args.seed0, args.seed0 + args.max_attempts),
        op_weights=op_weights,
        novel_mixers=novel_mixers,
        hist_fps=hist,
        out_path=out,
        target=args.target,
        top_k=args.top,
        batch_size=args.batch_size,
        progress_every=args.progress_every,
        tag="main",
    )
    return Counter(stats), top, [out.as_posix()]


def _run_parallel(
    args: argparse.Namespace,
    op_weights: Dict[str, float],
    novel_mixers: Set[str],
    hist: Set[str],
    out: Path,
) -> Tuple[Counter, List[Dict[str, Any]], List[str]]:
    per_worker = max(1, args.max_attempts // args.workers)
    common = {
        "op_weights": op_weights,
        "novel_mixers": sorted(novel_mixers),
        "hist_fps": sorted(hist),
        "target": max(1, args.target // args.workers),
        "top_k": args.top,
        "batch_size": args.batch_size,
        "progress_every": args.progress_every,
    }
    payloads = [
        {
            **common,
            "seed_start": args.seed0 + w * per_worker,
            "seed_count": per_worker,
            "out_path": out.with_suffix(f".w{w}.jsonl").as_posix(),
            "tag": f"w{w}",
        }
        for w in range(args.workers)
    ]
    with mp.Pool(args.workers) as pool:
        results = pool.map(_worker, payloads)
    agg: Counter = Counter()
    merged: List[Dict[str, Any]] = []
    shards: List[str] = []
    for (st, tp), pl in zip(results, payloads):
        agg.update(st)
        merged.extend(tp)
        shards.append(pl["out_path"])
    return agg, sorted(merged, key=lambda r: -r["score"])[: args.top], shards


def run(args: argparse.Namespace) -> Dict[str, Any]:
    hist = _historical_fingerprints(args.db)
    novel_mixers = _novel_mixers(
        args.db,
        _NOVEL_CLASSES + (("unattributed",) if args.include_unattributed else ()),
    )
    scaffold = _scaffold_boost(args.db, args.cap_thr) if args.bias else {}
    op_weights = (
        _build_op_weights(
            novel_mixers, scaffold, args.mixer_weight, args.scaffold_weight
        )
        if args.bias
        else {}
    )
    logger.info("novel mixers (%d): %s", len(novel_mixers), sorted(novel_mixers))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if args.workers <= 1:
        agg, merged_top, shards = _run_single(args, op_weights, novel_mixers, hist, out)
    else:
        agg, merged_top, shards = _run_parallel(
            args, op_weights, novel_mixers, hist, out
        )
    elapsed = time.time() - t0
    return {
        "workers": args.workers,
        "novel_mixer_set": sorted(novel_mixers),
        "shards": shards,
        "elapsed_s": round(elapsed, 1),
        "generated": agg.get("generated", 0),
        "invalid": agg.get("invalid", 0),
        "already_seen": agg.get("already_seen", 0),
        "no_novel_mixer": agg.get("no_novel_mixer", 0),
        "kept_novel_with_mixer": agg.get("kept", 0),
        "gen_per_sec": int(agg.get("generated", 0) / elapsed) if elapsed else None,
        "keep_rate": round(agg.get("kept", 0) / max(agg.get("generated", 1), 1), 3),
        "top_preview": [
            {
                "fingerprint": r["fingerprint"],
                "score": r["score"],
                "novel_mixers": r["novel_mixers"],
                "ops": r["ops"],
            }
            for r in merged_top[:8]
        ],
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=str(RUNS_DB))
    p.add_argument("--target", type=int, default=5000, help="survivors to emit")
    p.add_argument("--max-attempts", type=int, default=200_000, help="generation cap")
    p.add_argument("--seed0", type=int, default=900_000)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--top", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=20_000)
    p.add_argument("--progress-every", type=int, default=20_000)
    p.add_argument("--cap-thr", type=float, default=0.35)
    p.add_argument("--mixer-weight", type=float, default=6.0)
    p.add_argument("--scaffold-weight", type=float, default=2.5)
    p.add_argument(
        "--include-unattributed",
        action="store_true",
        help="also treat unattributed MIX ops as novel (broader, noisier)",
    )
    p.add_argument("--no-bias", dest="bias", action="store_false")
    p.add_argument("--out", default="research/reports/novel_mixer_graphs.jsonl")
    args = p.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
