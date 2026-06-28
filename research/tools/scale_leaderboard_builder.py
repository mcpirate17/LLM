"""Automated scale-run leaderboard + screen-predictivity (Gemini plan P4.1/P4.2).

Replaces the hand-maintained Minimax markdown table (ingested verbatim into
``scale_run_leaderboard``) with a tool that queries ``runs.db`` directly and emits
dynamic param-matched / FLOP-matched rankings, plus a Spearman screen-predictivity
report quantifying how well cheap probes predict expensive capability.

Reads (all written by ``ingest_scale_runs.py``):
  - scale_run_evals          per-(run,seed) config + n_params
  - scale_run_probe_metrics  long/EAV capability metrics (mean over seeds)
  - scale_run_blimp          BLiMP overall
  - scale_run_leaderboard    manual table — mined ONLY for active_m/tokens_m
  - leaderboard              9709 nano-screen rows (cheap screen + outcome same row)

Writes:
  - markdown report (``--out``)
  - ``scale_run_leaderboard_auto`` table in runs.db (queryable; does NOT clobber
    the manual ``scale_run_leaderboard``)

Run:  python research/tools/scale_leaderboard_builder.py --mode both
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import time
from typing import NamedTuple

from scipy.stats import spearmanr

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(REPO, "research", "runs.db")
DEFAULT_OUT = os.path.join(REPO, "research", "notes", "scale_run_leaderboard_auto.md")

# Canonical capability metrics (all higher = better), mean over seeds. Mirrors the
# columns of the manual leaderboard so the auto table is directly comparable.
CAPABILITY_METRICS: dict[str, tuple[str, str]] = {
    "blimp": ("__blimp__", "blimp_overall"),  # special-cased from scale_run_blimp
    "gmqar": ("gmqar", "gmqar_audc"),  # PRIMARY recall (zero-shot, artifact-free)
    "ar_cur": ("ar_curriculum", "ar_curriculum_auc_pair_final"),  # secondary (FT)
    "ar_held": ("ar_validation", "ar_validation_held_pair_acc"),  # secondary (FT)
    "bind": ("binding_curriculum", "binding_screening_auc"),
    "ms_auc": ("binding_multislot", "binding_multislot_auc"),
    "ms_all": ("binding_multislot", "binding_multislot_all_slots_acc"),
    "ind": ("induction_intermediate", "induction_intermediate_auc"),
    "ind_val": ("induction_validation", "induction_validation_auc"),
}

# Screen-predictivity feature/target columns in the nano `leaderboard` table.
# higher_better=False features are inverted before correlating (cost-like).
NANO_SCREEN_FEATURES: dict[str, bool] = {
    "induction_screening_auc": True,
    "binding_screening_auc": True,
    "induction_intermediate_auc": True,
    "binding_intermediate_auc": True,
    "screening_loss_ratio": False,
    "param_efficiency": True,
    "fp_jacobian_spectral_norm": False,
}
NANO_OUTCOME_TARGETS: tuple[str, ...] = (
    "blimp_overall_accuracy",
    "induction_validation_auc",
    "ar_validation_held_pair_acc",
)

# Cheap (fast) vs expensive (slow) scale probes — for the within-scale predictivity
# check. Times from scale_run_probe_metrics._timing averages (seconds).
SCALE_CHEAP: dict[str, tuple[str, str]] = {
    "binding_range_auc": ("binding_range", "binding_screening_auc"),  # ~5s
    "binding_cur_auc": ("binding_curriculum", "binding_screening_auc"),  # ~64s
    "ind_inter_auc": ("induction_intermediate", "induction_intermediate_auc"),  # ~62s
    "ar_cur_pair": ("ar_curriculum", "ar_curriculum_auc_pair_final"),  # ~206s
}
SCALE_EXPENSIVE: dict[str, tuple[str, str]] = {
    "ind_val_auc": ("induction_validation", "induction_validation_auc"),  # ~562s
    "ar_held": ("ar_validation", "ar_validation_held_pair_acc"),  # ~338s
    "blimp": ("__blimp__", "blimp_overall"),
}


class RunRow(NamedTuple):
    run: str
    mixer: str
    n_params_m: float
    active_m: float | None
    tokens_m: float | None
    seq: str | None
    caps: dict[str, float]  # capability metric -> mean value (raw)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _connect(db_path: str) -> sqlite3.Connection:
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"runs.db not found at {db_path}")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _load_capabilities(conn: sqlite3.Connection) -> dict[str, dict[str, float]]:
    """run_name -> {capability_name: mean-over-seeds value}."""
    out: dict[str, dict[str, float]] = {}
    # Probe-metric-backed capabilities (everything except the blimp special case).
    wanted = {
        (fam, key): name
        for name, (fam, key) in CAPABILITY_METRICS.items()
        if fam != "__blimp__"
    }
    rows = conn.execute(
        """
        SELECT run_name, probe_family, metric_key, AVG(value_num) AS v
        FROM scale_run_probe_metrics
        WHERE value_num IS NOT NULL
        GROUP BY run_name, probe_family, metric_key
        """
    ).fetchall()
    for r in rows:
        name = wanted.get((r["probe_family"], r["metric_key"]))
        if name is not None:
            out.setdefault(r["run_name"], {})[name] = float(r["v"])
    # BLiMP from its own table.
    for r in conn.execute(
        "SELECT run_name, blimp_overall FROM scale_run_blimp WHERE blimp_overall IS NOT NULL"
    ).fetchall():
        out.setdefault(r["run_name"], {})["blimp"] = float(r["blimp_overall"])
    return out


def _load_runs(conn: sqlite3.Connection) -> list[RunRow]:
    caps = _load_capabilities(conn)
    # active_m / tokens_m / seq enrichment from the manual table (model == run_name).
    enrich: dict[str, sqlite3.Row] = {
        r["model"]: r
        for r in conn.execute(
            "SELECT model, active_m, tokens_m, seq FROM scale_run_leaderboard"
        ).fetchall()
    }
    cfg = conn.execute(
        """
        SELECT run_name, MAX(mixer) AS mixer, MAX(n_params) AS n_params
        FROM scale_run_evals GROUP BY run_name
        """
    ).fetchall()
    cfg_runs = {r["run_name"]: r for r in cfg}
    runs: list[RunRow] = []
    # Union of runs that have config or capabilities (blimp-only runs included).
    for run in sorted(set(cfg_runs) | set(caps)):
        c = cfg_runs.get(run)
        e = enrich.get(run)
        n_params_m = (float(c["n_params"]) / 1e6) if c and c["n_params"] else 0.0
        runs.append(
            RunRow(
                run=run,
                mixer=(c["mixer"] if c else "") or "",
                n_params_m=round(n_params_m, 2),
                active_m=(float(e["active_m"]) if e and e["active_m"] else None),
                tokens_m=(float(e["tokens_m"]) if e and e["tokens_m"] else None),
                seq=(e["seq"] if e else None),
                caps=caps.get(run, {}),
            )
        )
    return runs


# ---------------------------------------------------------------------------
# Scoring / normalization
# ---------------------------------------------------------------------------


class Scored(NamedTuple):
    run: RunRow
    norm: dict[str, float]  # capability -> min-max normalized [0,1]
    composite: float  # mean of present normalized metrics
    n_metrics: int
    param_eff: float | None  # composite / active_M (or n_params_M fallback)
    compute_eff: float | None  # composite / training-FLOP proxy (tokens known only)


def _minmax(values: list[float]) -> tuple[float, float]:
    lo, hi = min(values), max(values)
    return lo, (hi if hi > lo else lo + 1.0)


def score_runs(runs: list[RunRow]) -> list[Scored]:
    metric_names = list(CAPABILITY_METRICS)
    ranges: dict[str, tuple[float, float]] = {}
    for m in metric_names:
        present = [r.caps[m] for r in runs if m in r.caps]
        if present:
            ranges[m] = _minmax(present)
    scored: list[Scored] = []
    for r in runs:
        norm: dict[str, float] = {}
        for m in metric_names:
            if m in r.caps and m in ranges:
                lo, hi = ranges[m]
                norm[m] = (r.caps[m] - lo) / (hi - lo)
        composite = sum(norm.values()) / len(norm) if norm else 0.0
        param_denom = r.active_m or (r.n_params_m or None)
        param_eff = (composite / param_denom) if param_denom else None
        compute_eff = None
        if r.active_m and r.tokens_m:
            # Training-FLOP proxy = 6 * active_params * tokens (Chinchilla).
            flop = 6.0 * (r.active_m * 1e6) * (r.tokens_m * 1e6)
            compute_eff = composite / (flop / 1e18)  # per 1e18 FLOP
        scored.append(
            Scored(r, norm, round(composite, 4), len(norm), param_eff, compute_eff)
        )
    return scored


def _param_band(n_params_m: float) -> str:
    if n_params_m < 10:
        return "<10M"
    if n_params_m < 50:
        return "10-50M"
    if n_params_m < 110:
        return "50-110M"
    return "110M+"


# ---------------------------------------------------------------------------
# Predictivity (P4.2)
# ---------------------------------------------------------------------------


class Corr(NamedTuple):
    feature: str
    target: str
    rho: float
    pvalue: float
    n: int


def _spearman(xs: list[float], ys: list[float]) -> tuple[float, float]:
    res = spearmanr(xs, ys)
    rho = float(res.statistic)  # type: ignore[attr-defined]
    p = float(res.pvalue)  # type: ignore[attr-defined]
    return (rho, p) if rho == rho else (0.0, 1.0)  # NaN guard


def nano_screen_predictivity(conn: sqlite3.Connection) -> list[Corr]:
    """Spearman(cheap nano screen feature, expensive nano outcome) across the
    9709-row nano leaderboard. This is the high-n predictability signal: both the
    cheap screen and the expensive outcome sit on the SAME row."""
    cols = list(NANO_SCREEN_FEATURES) + list(NANO_OUTCOME_TARGETS)
    have = _existing_columns(conn, "leaderboard", cols)
    sel = ", ".join(f'"{c}"' for c in have)
    rows = conn.execute(f"SELECT {sel} FROM leaderboard").fetchall()
    out: list[Corr] = []
    for feat, higher_better in NANO_SCREEN_FEATURES.items():
        if feat not in have:
            continue
        for tgt in NANO_OUTCOME_TARGETS:
            if tgt not in have:
                continue
            xs, ys = [], []
            for r in rows:
                fv, tv = r[feat], r[tgt]
                if _finite(fv) and _finite(tv):
                    xs.append(fv if higher_better else -fv)
                    ys.append(tv)
            if len(xs) >= 10:
                rho, p = _spearman(xs, ys)
                out.append(Corr(feat, tgt, round(rho, 4), round(p, 6), len(xs)))
    return out


def scale_probe_predictivity(runs: list[RunRow]) -> list[Corr]:
    """Spearman(cheap scale probe, expensive scale probe/BLiMP) across the ~15
    scale runs. Small n — a directional sanity check on the scale side."""
    out: list[Corr] = []
    for fname, fk in SCALE_CHEAP.items():
        for tname, tk in SCALE_EXPENSIVE.items():
            xs, ys = [], []
            for r in runs:
                fv = _cap_for(r, fk)
                tv = _cap_for(r, tk)
                if fv is not None and tv is not None:
                    xs.append(fv)
                    ys.append(tv)
            if len(xs) >= 5:
                rho, p = _spearman(xs, ys)
                out.append(Corr(fname, tname, round(rho, 4), round(p, 6), len(xs)))
    return out


def _cap_for(r: RunRow, spec: tuple[str, str]) -> float | None:
    """Resolve a (family, metric) spec to a run's capability value. Handles both
    canonical capability names and the raw blimp special case."""
    fam, key = spec
    if fam == "__blimp__":
        return r.caps.get("blimp")
    for name, (f, k) in CAPABILITY_METRICS.items():
        if f == fam and k == key:
            return r.caps.get(name)
    return None


def _existing_columns(
    conn: sqlite3.Connection, table: str, cols: list[str]
) -> set[str]:
    have = {
        row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
    }
    return {c for c in cols if c in have}


def _finite(v: object) -> bool:
    return isinstance(v, (int, float)) and v == v and abs(float(v)) != float("inf")


# ---------------------------------------------------------------------------
# Rendering + writeback
# ---------------------------------------------------------------------------


def _fmt(v: float | None, nd: int = 4) -> str:
    return "—" if v is None else f"{v:.{nd}f}"


def _render_composite(scored: list[Scored], min_metrics: int) -> list[str]:
    full = [s for s in scored if s.n_metrics >= min_metrics]
    partial = [s for s in scored if s.n_metrics < min_metrics]
    # Only show metric columns that at least one run actually has — keeps the
    # forward-looking gmqar column out until runs are evaluated with it.
    cols = [m for m in CAPABILITY_METRICS if any(m in s.norm for s in scored)]
    header = (
        "| rank | run | mixer | params(M) | band | composite | n_metrics | "
        + " | ".join(cols)
        + " |"
    )
    sep = "|---|---|---|---|---|---|---|" + "---|" * len(cols)
    lines = [
        f"## Ranking — raw composite capability (≥{min_metrics} metrics)",
        "",
        header,
        sep,
    ]

    def _row(i: int, s: Scored) -> str:
        cells = " | ".join(_fmt(s.norm.get(m), 3) for m in cols)
        return (
            f"| {i} | {s.run.run} | {s.run.mixer[:28]} | {s.run.n_params_m:.1f} | "
            f"{_param_band(s.run.n_params_m)} | {s.composite:.4f} | {s.n_metrics} | {cells} |"
        )

    for i, s in enumerate(sorted(full, key=lambda s: s.composite, reverse=True), 1):
        lines.append(_row(i, s))
    if partial:
        lines += [
            "",
            f"### Partial — fewer than {min_metrics} metrics (composite not comparable)",
            "",
            header,
            sep,
        ]
        for i, s in enumerate(
            sorted(partial, key=lambda s: s.composite, reverse=True), 1
        ):
            lines.append(_row(i, s))
    return lines


def _render_param_eff(scored: list[Scored]) -> list[str]:
    lines = [
        "## Ranking — parameter efficiency (composite / active-M)",
        "",
        "active-M falls back to total params when active is unknown.",
        "",
        "| rank | run | params(M) | active(M) | composite | param_eff |",
        "|---|---|---|---|---|---|",
    ]
    pe = [s for s in scored if s.param_eff is not None]
    for i, s in enumerate(sorted(pe, key=lambda s: s.param_eff or 0, reverse=True), 1):
        lines.append(
            f"| {i} | {s.run.run} | {s.run.n_params_m:.1f} | {_fmt(s.run.active_m, 2)} "
            f"| {s.composite:.4f} | {_fmt(s.param_eff, 5)} |"
        )
    return lines


def _render_compute_eff(scored: list[Scored]) -> list[str]:
    lines = [
        "## Ranking — compute efficiency (composite / 1e18 training FLOP)",
        "",
        "Training-FLOP proxy = 6·active_params·tokens (Chinchilla). Only runs whose "
        "active_m + tokens_m are known (present in the manual table) are rankable here.",
        "",
        "| rank | run | active(M) | tokens(M) | composite | compute_eff |",
        "|---|---|---|---|---|---|",
    ]
    ce = [s for s in scored if s.compute_eff is not None]
    for i, s in enumerate(
        sorted(ce, key=lambda s: s.compute_eff or 0, reverse=True), 1
    ):
        lines.append(
            f"| {i} | {s.run.run} | {_fmt(s.run.active_m, 2)} | {_fmt(s.run.tokens_m, 1)} "
            f"| {s.composite:.4f} | {_fmt(s.compute_eff, 5)} |"
        )
    return lines


def _render_predictivity(nano: list[Corr], scale: list[Corr]) -> list[str]:
    lines = [
        "## Screen predictivity (P4.2)",
        "",
        "### Nano leaderboard — cheap screen → expensive outcome (high n)",
        "Spearman ρ across all nano-screened models; cheap screen and outcome are on "
        "the same row. This is the predictability factor for each cheap probe.",
        "",
        "| feature | target | ρ | p | n |",
        "|---|---|---|---|---|",
    ]
    for c in sorted(nano, key=lambda c: abs(c.rho), reverse=True):
        lines.append(
            f"| {c.feature} | {c.target} | {c.rho:+.4f} | {c.pvalue:.4g} | {c.n} |"
        )
    lines += [
        "",
        "### Scale runs — cheap probe → expensive probe/BLiMP (low n, directional)",
        "",
        "| cheap | expensive | ρ | p | n |",
        "|---|---|---|---|---|",
    ]
    for c in sorted(scale, key=lambda c: abs(c.rho), reverse=True):
        lines.append(
            f"| {c.feature} | {c.target} | {c.rho:+.4f} | {c.pvalue:.4g} | {c.n} |"
        )
    return lines


def render_markdown(
    scored: list[Scored], nano: list[Corr], scale: list[Corr], min_metrics: int = 4
) -> str:
    ts = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    head = [
        "# Scale-Run Leaderboard (auto-generated)",
        "",
        f"Generated {ts} by `research/tools/scale_leaderboard_builder.py` from `runs.db`.",
        "Composite = mean of per-metric min-max-normalized capabilities present for "
        "the run. Capabilities are seed-averaged. **Do not hand-edit — regenerate.**",
        "",
    ]
    caveats = [
        "## Caveats",
        "- active_m/tokens_m exist only for runs in the manual `scale_run_leaderboard`; "
        "compute-efficiency is partial until those are derived for every run.",
        "- No clean per-architecture nano↔scale ladder exists in `runs.db` (the nano "
        "`leaderboard` is fingerprint-keyed, scale runs are named lanes), so P4.2 "
        "reports within-nano and within-scale correlations, not literal dim64→144M pairing.",
        "- A zero on `ind_val`/`ar_held` for a 144M run is usually the known "
        "induction/AR-validation probe TIMEOUT artifact, not a true capability floor — "
        "trust the re-probed values once re-ingested.",
        "",
    ]
    sections = [
        head,
        _render_composite(scored, min_metrics),
        [""],
        _render_param_eff(scored),
        [""],
        _render_compute_eff(scored),
        [""],
        _render_predictivity(nano, scale),
        [""],
        caveats,
    ]
    return "\n".join("\n".join(sec) for sec in sections)


def writeback_table(conn: sqlite3.Connection, scored: list[Scored]) -> None:
    """Refresh the queryable `scale_run_leaderboard_auto` table (separate from the
    manual `scale_run_leaderboard`)."""
    conn.execute("DROP TABLE IF EXISTS scale_run_leaderboard_auto")
    conn.execute(
        """
        CREATE TABLE scale_run_leaderboard_auto (
            rank INTEGER, run TEXT PRIMARY KEY, mixer TEXT, n_params_m REAL,
            param_band TEXT, active_m REAL, tokens_m REAL, composite REAL,
            n_metrics INTEGER, param_eff REAL, compute_eff REAL,
            norm_json TEXT, generated_at REAL NOT NULL
        )
        """
    )
    import json

    now = time.time()
    ordered = sorted(scored, key=lambda s: s.composite, reverse=True)
    conn.executemany(
        """INSERT INTO scale_run_leaderboard_auto VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        [
            (
                i,
                s.run.run,
                s.run.mixer,
                s.run.n_params_m,
                _param_band(s.run.n_params_m),
                s.run.active_m,
                s.run.tokens_m,
                s.composite,
                s.n_metrics,
                s.param_eff,
                s.compute_eff,
                json.dumps(s.norm),
                now,
            )
            for i, s in enumerate(ordered, 1)
        ],
    )
    conn.commit()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument(
        "--mode", choices=("leaderboard", "predictivity", "both"), default="both"
    )
    ap.add_argument(
        "--min-metrics",
        type=int,
        default=4,
        help="runs with fewer capability metrics are listed separately (not ranked)",
    )
    ap.add_argument(
        "--no-writeback", action="store_true", help="skip refreshing the auto table"
    )
    args = ap.parse_args()

    conn = _connect(args.db)
    try:
        runs = _load_runs(conn)
        scored = score_runs(runs)
        nano = (
            nano_screen_predictivity(conn)
            if args.mode in ("predictivity", "both")
            else []
        )
        scale = (
            scale_probe_predictivity(runs)
            if args.mode in ("predictivity", "both")
            else []
        )
        md = render_markdown(scored, nano, scale, min_metrics=args.min_metrics)
        with open(args.out, "w") as f:
            f.write(md)
        if not args.no_writeback and args.mode in ("leaderboard", "both"):
            writeback_table(conn, scored)
        print(f"[scale_leaderboard_builder] {len(runs)} runs scored -> {args.out}")
        if nano:
            top = max(nano, key=lambda c: abs(c.rho))
            print(
                f"  best nano screen predictor: {top.feature} -> {top.target} "
                f"rho={top.rho:+.3f} (n={top.n})"
            )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
