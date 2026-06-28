#!/usr/bin/env python3
"""Autonomous, model-agnostic, goal-driven audit orchestrator.

The contract:
  * AUDITS fan out, read-only, across ALL models (claude, codex, agy, minimax) for diverse
    opinions and to spend cheap tokens on broad sweeps. The auditor is never the fixer.
  * TRIAGE is done by claude (the orchestrator brain): it merges every model's findings into
    one deduplicated, ROI-ordered fix plan.
  * FIXES are applied one at a time, each on its own git branch behind a regression gate
    (smoke tests + fixed-seed scoring diff). easy/medium -> codex, hard -> claude.
  * The LOOP repeats audit -> triage -> fix and re-measures a DETERMINISTIC violation vector
    each round. It stops when a round stops reducing that vector (no remaining ROI), so a
    god file that was never actually split keeps coming back until it is.

Run unattended:
    python orchestrate.py loop            # the whole thing
Other commands:
    python orchestrate.py doctor          # validate config, CLIs, gate (do this first)
    python orchestrate.py capture-baseline# freeze the fixed-seed scoring baseline
    python orchestrate.py measure         # print the current violation vector
    python orchestrate.py audit [--round N]   # one read-only fan-out round only
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tomllib
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import adapters
import detectors
import gate

HERE = Path(__file__).resolve().parent
STATE = HERE / "state"
FINDINGS_ROOT = HERE.parent / "findings"  # audit/findings/


# --- config / io -----------------------------------------------------------


def cfg() -> dict:
    return tomllib.loads((HERE / "config.toml").read_text())


def book() -> dict:
    return json.loads((HERE / "playbook.json").read_text())


def repo_root(c: dict) -> Path:
    return (HERE / c["repo"]["root"]).resolve()


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ledger(entry: dict) -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / "ledger.jsonl").open("a") as f:
        f.write(json.dumps(entry) + "\n")


def sh(cmd: str, cwd: Path) -> int:
    return subprocess.run(cmd, cwd=cwd, shell=True, text=True).returncode


def sh_out(cmd: str, cwd: Path) -> str:
    return subprocess.run(
        cmd, cwd=cwd, shell=True, text=True, capture_output=True
    ).stdout.strip()


# --- audit phase -----------------------------------------------------------


def _fill(text: str, scope: list[str]) -> str:
    targets = ", ".join(scope) or "the whole repo"
    return text.replace("{targets}", targets)


def _routed_scope(model: str, pass_difficulty: str, c: dict) -> list[str]:
    """Targets `model` should audit for a pass of this difficulty, per capability routing.
    Empty list => this model is not eligible for this pass (skip the job)."""
    routing = c["audit"].get("routing", {})
    rule = routing.get(model)
    targets = c["loop"]["targets"]
    if rule is None:  # unrouted model → full scope (back-compat)
        return targets
    if pass_difficulty not in rule.get("difficulties", []):
        return []
    sizes = c["audit"].get("target_sizes", {})
    ok_sizes = set(rule.get("sizes", []))
    return [t for t in targets if sizes.get(t, "medium") in ok_sizes]


def _run_one_audit(
    model_name: str,
    p: dict,
    scope: list[str],
    preamble_tmpl: str,
    repo: Path,
    c: dict,
    round_dir: Path,
) -> tuple[str, str, adapters.Result]:
    ad = adapters.get(model_name)
    preamble = _fill(preamble_tmpl, scope)
    prompt = (
        preamble + "\n\nAUDIT PASS: " + p["title"] + "\n\n" + _fill(p["prompt"], scope)
    )
    res = ad.run(prompt, repo, fix=False, timeout=c["audit"]["timeout"])
    out = round_dir / f"{p['id']}__{model_name}.md"
    header = (
        f"# {p['id']} — {p['title']}\n\nmodel: {model_name} · severity: {p['severity']}"
        f" · difficulty: {p.get('difficulty', '?')} · scope: {', '.join(scope)}"
        f" · {_stamp()}\n\n---\n\n"
    )
    out.write_text(header + (res.text or "(no output)") + "\n")
    return p["id"], model_name, res


def _tracked_dirty(repo: Path) -> set[str]:
    """Set of tracked files with uncommitted modifications right now."""
    out = sh_out("git status --porcelain --untracked-files=no", repo)
    return {ln[3:] for ln in out.splitlines() if ln}


def _assert_readonly(repo: Path, pre: set[str], rnd: int) -> None:
    """Audits must not mutate tracked files. Anything newly dirty is a read-only
    violation — log it and revert it (safe in the isolated worktree this runs in)."""
    new = _tracked_dirty(repo) - pre
    if not new:
        print("    read-only integrity: OK (no tracked files mutated by audits)")
        return
    print(
        f"    READ-ONLY VIOLATION: audits mutated {len(new)} tracked file(s); reverting"
    )
    for f in sorted(new):
        print(f"      ! {f}")
    sh("git checkout -- " + " ".join(f'"{f}"' for f in new), repo)
    _ledger(
        {
            "ts": _stamp(),
            "round": rnd,
            "phase": "readonly-violation",
            "files": sorted(new),
        }
    )


def run_audit_round(
    c: dict,
    b: dict,
    repo: Path,
    rnd: int,
    only_models: list[str] | None = None,
    only_passes: list[str] | None = None,
) -> Path:
    round_dir = FINDINGS_ROOT / f"round-{rnd:02d}"
    round_dir.mkdir(parents=True, exist_ok=True)
    pool = only_models or c["audit"]["models"]
    models = [m for m in pool if adapters.available(m)]
    skipped = [m for m in pool if m not in models]
    if skipped:
        print(f"  (skipping unavailable CLIs: {', '.join(skipped)})")
    audit_passes = [
        p for p in b["audit_passes"] if not only_passes or p["id"] in only_passes
    ]
    # Capability routing: a (model, pass) job exists only if the model is eligible for the
    # pass difficulty AND has a non-empty target scope at its allowed sizes.
    jobs = []
    for p in audit_passes:
        for m in models:
            scope = _routed_scope(m, p.get("difficulty", "medium"), c)
            if scope:
                jobs.append((m, p, scope))
    routed_out = [
        (m, p["id"])
        for p in audit_passes
        for m in models
        if not _routed_scope(m, p.get("difficulty", "medium"), c)
    ]
    print(
        f"  round {rnd}: {len(audit_passes)} passes × {len(models)} models "
        f"-> {len(jobs)} routed audits (parallel={c['audit']['max_parallel']})"
    )
    if routed_out:
        print(f"  (routing skipped {len(routed_out)} model/pass combos by capability)")
    preamble_tmpl = b["preamble"]
    pre = _tracked_dirty(repo)  # read-only integrity snapshot
    total = 0.0
    with ThreadPoolExecutor(max_workers=c["audit"]["max_parallel"]) as ex:
        futs = {
            ex.submit(_run_one_audit, m, p, scope, preamble_tmpl, repo, c, round_dir): (
                p["id"],
                m,
            )
            for (m, p, scope) in jobs
        }
        for fut in as_completed(futs):
            pid, mdl, res = fut.result()
            total += res.cost
            mark = "ok " if res.ok else "ERR"
            print(f"    [{mark}] {pid:<16} {mdl:<8} ${res.cost:.3f}")
    _assert_readonly(repo, pre, rnd)
    print(f"  audit findings -> {round_dir}  (${total:.2f})")
    _ledger(
        {
            "ts": _stamp(),
            "round": rnd,
            "phase": "audit",
            "jobs": len(jobs),
            "cost": total,
        }
    )
    return round_dir


# --- triage ----------------------------------------------------------------


def _parse_tasks(text: str) -> list[dict]:
    """Extract the JSON task plan from a possibly-chatty triage response."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    tasks = data.get("tasks", [])
    return [t for t in tasks if t.get("instruction") and t.get("files")]


def triage(
    c: dict, b: dict, repo: Path, round_dir: Path, rnd: int, reuse: bool = False
) -> list[dict]:
    cache = round_dir / "_triage.json"
    if reuse and cache.exists():
        tasks = _parse_tasks(cache.read_text())[: c["fix"]["max_tasks_per_round"]]
        if tasks:
            print(f"  triage: reusing cached plan ({len(tasks)} tasks) — no re-spend")
            return tasks
    findings = "\n\n".join(
        f"### {f.name}\n{f.read_text()[:8000]}" for f in sorted(round_dir.glob("*.md"))
    )
    prompt = (
        b["triage"]["prompt"]
        .replace("{models}", ", ".join(c["audit"]["models"]))
        .replace("{round}", str(rnd))
        .replace("{max_tasks}", str(c["fix"]["max_tasks_per_round"]))
        .replace("{findings}", findings)
    )
    ad = adapters.get(c["triage"]["model"])
    res = ad.run(prompt, repo, fix=False, timeout=c["triage"]["timeout"])
    (round_dir / "_triage.json").write_text(res.text)
    tasks = _parse_tasks(res.text)[: c["fix"]["max_tasks_per_round"]]
    print(f"  triage ({c['triage']['model']}): {len(tasks)} fix tasks")
    _ledger({"ts": _stamp(), "round": rnd, "phase": "triage", "tasks": len(tasks)})
    return tasks


# --- fix phase -------------------------------------------------------------


def _route(task: dict, c: dict) -> str:
    if (
        task.get("difficulty") == "hard"
        or task.get("severity") in c["fix"]["hard_severity"]
    ):
        return c["fix"]["hard_model"]
    return c["fix"]["easy_model"]


def _apply_one_fix(
    task: dict, c: dict, b: dict, repo: Path, rnd: int, idx: int
) -> bool:
    slug = re.sub(r"[^a-z0-9-]+", "-", task.get("slug", f"task{idx}").lower()).strip(
        "-"
    )
    branch = f"audit/fix-r{rnd:02d}-{slug}"[:60]
    model = _route(task, c)
    if not adapters.get(model).can_fix or not adapters.available(model):
        print(f"    SKIP {slug}: fixer {model} unavailable")
        return False
    print(
        f"    fix {slug} via {model} ({task.get('difficulty', '?')}/{task.get('severity', '?')})"
    )
    if sh(f"git checkout -b {branch}", repo) != 0:
        print(f"      could not create branch {branch}; skipping")
        return False
    prompt = (
        b["fix"]["prompt"]
        .replace("{title}", task.get("title", slug))
        .replace("{severity}", task.get("severity", "?"))
        .replace("{files}", ", ".join(task.get("files", [])))
        .replace("{instruction}", task["instruction"])
    )
    before = set(sh_out("git status --porcelain", repo).splitlines())
    res = adapters.get(model).run(prompt, repo, fix=True, timeout=c["fix"]["timeout"])
    print((res.text or "").strip()[:1200])
    # Stage ONLY what this fix changed — never sweep in pre-existing/concurrent edits.
    changed = [
        ln[3:]
        for ln in sh_out("git status --porcelain", repo).splitlines()
        if ln and ln not in before
    ]
    for f in changed:
        sh(f'git add -- "{f}"', repo)
    if sh_out("git diff --cached --name-only", repo) == "":
        print("      no changes produced; abandoning branch")
        sh(f"git checkout {c['merge']['base_branch']} && git branch -D {branch}", repo)
        _ledger(
            {
                "ts": _stamp(),
                "round": rnd,
                "task": slug,
                "model": model,
                "result": "no-op",
                "cost": res.cost,
            }
        )
        return False
    sh(
        f'git commit -q --no-verify -m "fix(audit): {task.get("title", slug)[:60]}\n\n'
        f"Auto-applied by audit orchestrator (round {rnd}, model {model}).\n\n"
        f'Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"',
        repo,
    )
    passed = gate.passes(c, repo)
    _finish_fix(passed, c, repo, branch, slug, model, rnd, res.cost)
    return passed


def _finish_fix(
    passed: bool,
    c: dict,
    repo: Path,
    branch: str,
    slug: str,
    model: str,
    rnd: int,
    cost: float,
) -> None:
    base = c["merge"]["base_branch"]
    mode = c["merge"]["mode"]
    if passed and mode == "auto":
        sh(
            f"git checkout {base} && git merge --no-ff --no-edit {branch} && git branch -d {branch}",
            repo,
        )
        print(f"      GATE PASS -> merged to {base}")
    elif passed and mode == "accumulate":
        sh(f"git checkout {base}", repo)  # leave commit on branch for one sweep merge
        print(f"      GATE PASS -> kept on {branch} (accumulate mode)")
    elif passed:  # branch mode
        sh(f"git checkout {base}", repo)
        print(f"      GATE PASS -> left on {branch} (branch mode; merge manually)")
    else:
        sh(f"git checkout {base} && git branch -D {branch}", repo)
        print(f"      GATE FAIL -> reverted (branch {branch} deleted)")
    _ledger(
        {
            "ts": _stamp(),
            "round": rnd,
            "task": slug,
            "model": model,
            "result": "merged"
            if (passed and mode == "auto")
            else ("kept" if passed else "reverted"),
            "cost": cost,
        }
    )


def run_fix_round(tasks: list[dict], c: dict, b: dict, repo: Path, rnd: int) -> int:
    applied = 0
    for idx, task in enumerate(tasks):
        if sh_out("git status --porcelain", repo):
            print("    working tree dirty; committing strays to keep gate honest")
            sh("git stash -u", repo)
        if _apply_one_fix(task, c, b, repo, rnd, idx):
            applied += 1
    return applied


# --- the loop --------------------------------------------------------------


def cmd_loop(args, c: dict) -> None:
    b = book()
    repo = repo_root(c)
    _require_clean_tree(repo)
    _ensure_baseline(c, repo)
    prev = detectors.measure(repo, c["loop"]["targets"], set(c["loop"]["exclude"]))
    print(f"start violation vector: {prev.as_dict()}")
    _ledger({"ts": _stamp(), "phase": "start", "metrics": prev.as_dict()})

    for rnd in range(1, c["loop"]["max_rounds"] + 1):
        print(f"\n========== ROUND {rnd} ==========")
        round_dir = run_audit_round(c, b, repo, rnd)
        tasks = triage(c, b, repo, round_dir, rnd)
        if not tasks:
            print(
                "triage produced no actionable tasks — codebase is at its ROI floor. Done."
            )
            break
        applied = run_fix_round(tasks, c, b, repo, rnd)
        cur = detectors.measure(repo, c["loop"]["targets"], set(c["loop"]["exclude"]))
        gain = prev.total - cur.total
        print(
            f"round {rnd}: applied {applied} fixes · violation total {prev.total} -> "
            f"{cur.total} (Δ{gain:+d})"
        )
        _ledger(
            {
                "ts": _stamp(),
                "round": rnd,
                "phase": "round-end",
                "applied": applied,
                "metrics": cur.as_dict(),
                "gain": gain,
            }
        )
        if gain < c["loop"]["roi_min_improvement"]:
            print(
                f"round gain {gain} < roi_min_improvement "
                f"{c['loop']['roi_min_improvement']} — no remaining ROI. Done."
            )
            break
        prev = cur
    else:
        print(f"\nreached max_rounds={c['loop']['max_rounds']}.")

    final = detectors.measure(repo, c["loop"]["targets"], set(c["loop"]["exclude"]))
    print(f"\nfinal violation vector: {final.as_dict()}")
    _write_summary(final)


def _resolve_one(
    task: dict, c: dict, b: dict, repo: Path, idx: int
) -> tuple[bool, float]:
    """Apply one fix ON THE CURRENT BRANCH (no per-fix branch). Commit if it passes the
    gate; hard-reset the commit if it fails. Returns (kept, cost)."""
    slug = re.sub(r"[^a-z0-9-]+", "-", task.get("slug", f"task{idx}").lower()).strip(
        "-"
    )
    model = _route(task, c)
    if not adapters.get(model).can_fix or not adapters.available(model):
        print(f"    SKIP {slug}: fixer {model} unavailable")
        return False, 0.0
    print(
        f"    fix {slug} via {model} "
        f"({task.get('difficulty', '?')}/{task.get('severity', '?')})"
    )
    prompt = (
        b["fix"]["prompt"]
        .replace("{title}", task.get("title", slug))
        .replace("{severity}", task.get("severity", "?"))
        .replace("{files}", ", ".join(task.get("files", [])))
        .replace("{instruction}", task["instruction"])
    )
    before = set(sh_out("git status --porcelain", repo).splitlines())
    res = adapters.get(model).run(prompt, repo, fix=True, timeout=c["fix"]["timeout"])
    print((res.text or "").strip()[:800])
    changed = [
        ln[3:]
        for ln in sh_out("git status --porcelain", repo).splitlines()
        if ln and ln not in before
    ]
    if not changed:
        print("      no changes produced; skipping")
        return False, res.cost
    for f in changed:
        sh(f'git add -- "{f}"', repo)
    sh(
        f'git commit -q --no-verify -m "fix(audit): {task.get("title", slug)[:60]}"',
        repo,
    )
    if gate.passes(c, repo):
        print("      GATE PASS -> kept on resolve branch")
        return True, res.cost
    print("      GATE FAIL -> dropping this fix (git reset --hard HEAD~1)")
    sh("git reset --hard HEAD~1", repo)
    return False, res.cost


def cmd_resolve(args, c: dict) -> None:
    """Resolution phase: reuse EXISTING audit findings (no costly re-audit), triage them,
    apply gated fixes that accumulate on one branch, and re-measure the deterministic
    violation vector after each — so we SEE each fix actually lower the count."""
    b = book()
    repo = repo_root(c)
    _require_clean_tree(repo)
    _ensure_baseline(c, repo)
    round_dir = FINDINGS_ROOT / f"round-{args.round:02d}"
    if not round_dir.exists():
        sys.exit(f"no findings at {round_dir}; run an audit round first")
    branch = f"audit/resolve-r{args.round:02d}"
    if (
        sh(f"git checkout -B {branch}", repo) != 0
    ):  # -B: create or reset (idempotent re-runs)
        sys.exit(f"could not create resolve branch {branch}")
    print(f"resolving on branch {branch} (master untouched)")

    tasks = triage(c, b, repo, round_dir, args.round, reuse=True)
    if args.difficulties:
        tasks = [t for t in tasks if t.get("difficulty", "medium") in args.difficulties]
    if args.max_fixes:
        tasks = tasks[: args.max_fixes]
    print(
        f"resolving {len(tasks)} task(s) "
        f"(difficulty filter: {args.difficulties or 'all'}, cap: {args.max_fixes or 'none'})"
    )

    start = detectors.measure(repo, c["loop"]["targets"], set(c["loop"]["exclude"]))
    prev = start
    kept = 0
    total_cost = 0.0
    for idx, task in enumerate(tasks):
        ok, cost = _resolve_one(task, c, b, repo, idx)
        total_cost += cost
        if ok:
            kept += 1
            cur = detectors.measure(
                repo, c["loop"]["targets"], set(c["loop"]["exclude"])
            )
            print(
                f"      violation total {prev.total} -> {cur.total} (Δ{prev.total - cur.total:+d})"
            )
            _ledger(
                {
                    "ts": _stamp(),
                    "phase": "resolve",
                    "task": task.get("slug"),
                    "kept": True,
                    "metrics": cur.as_dict(),
                    "cost": cost,
                }
            )
            prev = cur
    final = detectors.measure(repo, c["loop"]["targets"], set(c["loop"]["exclude"]))
    print(
        f"\nresolve done: kept {kept}/{len(tasks)} fixes on {branch} · "
        f"violation total {start.total} -> {final.total} (Δ{start.total - final.total:+d}) · "
        f"${total_cost:.2f}"
    )
    print(
        f"review & merge:  git diff master...{branch}  &&  git checkout master && git merge {branch}"
    )
    _ledger(
        {
            "ts": _stamp(),
            "phase": "resolve-end",
            "kept": kept,
            "start": start.as_dict(),
            "final": final.as_dict(),
            "cost": total_cost,
        }
    )


def _write_summary(final: detectors.Metrics) -> None:
    out = (
        HERE.parent
        / f"orchestrator_run_summary_{datetime.now(timezone.utc):%Y-%m-%d}.md"
    )
    lines = [
        f"# Audit orchestrator run summary — {_stamp()}\n",
        "## Final deterministic violation vector\n",
        "| metric | count |",
        "|---|---|",
    ]
    for k, v in final.as_dict().items():
        if k != "detail":
            lines.append(f"| {k} | {v} |")
    lines.append(
        "\nSee `findings/round-*/` for per-model reports and `state/ledger.jsonl` "
        "for the full audit→fix→gate trail."
    )
    out.write_text("\n".join(lines) + "\n")
    print(f"summary -> {out}")


# --- guards / subcommands --------------------------------------------------


def _require_clean_tree(repo: Path) -> None:
    # Only TRACKED modifications matter — the gate diffs committed state, and untracked
    # tooling/findings (this orchestrator, findings/) don't affect it.
    if sh_out("git status --porcelain --untracked-files=no", repo):
        sys.exit(
            "tracked files modified — commit/stash before an autonomous run "
            "(the gate diffs against committed state)"
        )
    branch = sh_out("git rev-parse --abbrev-ref HEAD", repo)
    print(f"on branch {branch}, clean tracked tree.")


def _ensure_baseline(c: dict, repo: Path) -> None:
    base = repo / c["gate"]["reference_baseline"]
    if base.exists():
        print(f"scoring baseline present: {base}")
        return
    print("no scoring baseline yet — capturing one from current HEAD before any fix...")
    gate.capture_baseline(c, repo)


def cmd_capture_baseline(args, c: dict) -> None:
    gate.capture_baseline(c, repo_root(c))


def cmd_measure(args, c: dict) -> None:
    repo = repo_root(c)
    m = detectors.measure(repo, c["loop"]["targets"], set(c["loop"]["exclude"]))
    print(json.dumps(m.as_dict(), indent=2))


def cmd_audit(args, c: dict) -> None:
    run_audit_round(
        c,
        book(),
        repo_root(c),
        args.round,
        only_models=args.models or None,
        only_passes=args.passes or None,
    )


def cmd_doctor(args, c: dict) -> None:
    repo = repo_root(c)
    print(f"repo root: {repo}  (exists={repo.exists()})")
    print("\nCLI availability:")
    for m in c["audit"]["models"] + [c["fix"]["easy_model"], c["fix"]["hard_model"]]:
        print(f"  {m:<8} {'found' if adapters.available(m) else 'MISSING'}")
    print("\nfixers (must be codex/claude only):")
    for m in {c["fix"]["easy_model"], c["fix"]["hard_model"]}:
        print(f"  {m:<8} can_fix={adapters.get(m).can_fix}")
    print("\ngate config:")
    print(f"  smoke_cmd     : {c['gate']['smoke_cmd'][:80]}...")
    print(f"  reference_cmd : {c['gate']['reference_cmd'][:80]}...")
    print(
        f"  baseline      : {'present' if (repo / c['gate']['reference_baseline']).exists() else 'NOT captured'}"
    )
    print(f"\nmerge mode: {c['merge']['mode']} -> {c['merge']['base_branch']}")
    print("\nmeasuring current violation vector (deterministic oracle)...")
    m = detectors.measure(repo, c["loop"]["targets"], set(c["loop"]["exclude"]))
    print(json.dumps({k: v for k, v in m.as_dict().items() if k != "detail"}, indent=2))
    print("\ndoctor OK — run `capture-baseline` then `loop`.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("loop")
    sub.add_parser("doctor")
    sub.add_parser("capture-baseline")
    sub.add_parser("measure")
    a = sub.add_parser("audit")
    a.add_argument("--round", type=int, default=0)
    a.add_argument(
        "--models",
        nargs="*",
        default=[],
        help="subset of the audit pool (default: all configured)",
    )
    a.add_argument(
        "--passes",
        nargs="*",
        default=[],
        help="subset of audit-pass ids (default: all)",
    )
    rs = sub.add_parser("resolve")
    rs.add_argument(
        "--round",
        type=int,
        required=True,
        help="findings round to resolve (reused, not re-audited)",
    )
    rs.add_argument(
        "--max-fixes", type=int, default=0, help="cap number of fixes (0=all)"
    )
    rs.add_argument(
        "--difficulties",
        nargs="*",
        default=[],
        help="only fix tasks of these difficulties (e.g. easy medium)",
    )
    args = ap.parse_args()
    c = cfg()
    {
        "loop": cmd_loop,
        "resolve": cmd_resolve,
        "doctor": cmd_doctor,
        "capture-baseline": cmd_capture_baseline,
        "measure": cmd_measure,
        "audit": cmd_audit,
    }[args.cmd](args, c)


if __name__ == "__main__":
    main()
