# component_fab Colab runner
# Paste this whole file into one Google Colab cell, or run it as a script after cloning.
# Default mode is a safe smoke run. Change MODE below for longer work.

from __future__ import annotations

from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import time

try:
    from google.colab import drive  # type: ignore
    drive.mount('/content/drive')
except Exception:
    pass

# ---- User knobs ----
REPO_URL = 'https://github.com/mcpirate17/LLM.git'
BRANCH = 'fix/component-fab-colab'
MODE = os.environ.get('FAB_MODE', 'smoke')
# Modes: smoke, autonomous_screen, deep_probe_dry, fidelity, surrogate, invention_dry

DRIVE_ROOT = Path('/content/drive/MyDrive/component_fab_colab')
WORK_ROOT = Path('/content/component_fab_work')
REPO_DIR = WORK_ROOT / 'LLM'
STATUS = DRIVE_ROOT / 'status.json'
LOG = DRIVE_ROOT / f'{MODE}.log'
REPORT_DIR = DRIVE_ROOT / 'reports'
LEDGER = DRIVE_ROOT / 'ledger.jsonl'


def now() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def write_status(state: str, **extra) -> None:
    DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
    payload = {'state': state, 'mode': MODE, 'updated_at': now(), **extra}
    tmp = STATUS.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    tmp.replace(STATUS)


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    print(' '.join(cmd), flush=True)
    with LOG.open('a', encoding='utf-8') as log:
        log.write(f'\n=== {now()} :: ' + ' '.join(cmd) + ' ===\n')
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd or REPO_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        latest = ''
        last_status = time.monotonic()
        for line in proc.stdout:
            latest = line.rstrip()
            print(line, end='')
            log.write(line)
            log.flush()
            if time.monotonic() - last_status >= 10:
                write_status('running', latest_line=latest)
                last_status = time.monotonic()
        rc = proc.wait()
        write_status('complete' if rc == 0 else 'failed', returncode=rc, latest_line=latest)
        if rc != 0:
            raise SystemExit(rc)


def setup_repo() -> None:
    write_status('setup', step='clone')
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    if REPO_DIR.exists():
        shutil.rmtree(REPO_DIR)
    subprocess.check_call(['git', 'clone', '--depth', '1', '--branch', BRANCH, REPO_URL, str(REPO_DIR)])
    os.environ['PYTHONPATH'] = str(REPO_DIR)

    write_status('setup', step='install')
    run([
        sys.executable,
        '-m',
        'pip',
        'install',
        '-q',
        '-e',
        str(REPO_DIR),
        'torch',
        'numpy',
        'xxhash',
        'zstandard',
        'pyyaml',
        'flask-cors',
        'lightgbm',
        'ninja',
    ], cwd=REPO_DIR)


def run_mode() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    LEDGER.parent.mkdir(parents=True, exist_ok=True)

    if MODE == 'smoke':
        run([
            sys.executable,
            '-m',
            'component_fab.tools.run_invention',
            '--dry-run',
            '--max-specs',
            '2',
            '--ledger',
            str(LEDGER),
        ])
        return

    if MODE == 'invention_dry':
        run([
            sys.executable,
            '-m',
            'component_fab.tools.run_invention',
            '--dry-run',
            '--max-specs',
            '8',
            '--ledger',
            str(LEDGER),
        ])
        return

    if MODE == 'autonomous_screen':
        # Safe screening mode: cheap, no paired promotion evidence, output summary to Drive.
        run([
            sys.executable,
            '-m',
            'component_fab.tools.run_autonomous',
            '--cycles',
            '2',
            '--max-graded-per-cycle',
            '8',
            '--max-nas-specs',
            '2',
            '--probe-steps',
            '40',
            '--emit-run-summary',
            '--quiet',
        ])
        run(['bash', '-lc', f'cp -av component_fab/catalog/*.json {REPORT_DIR}/ || true'], cwd=REPO_DIR)
        run(['bash', '-lc', f'cp -av component_fab/catalog/*.jsonl {DRIVE_ROOT}/ || true'], cwd=REPO_DIR)
        return

    if MODE == 'deep_probe_dry':
        run([
            sys.executable,
            '-m',
            'component_fab.tools.run_deep_probe',
            '--top-k',
            '6',
            '--steps',
            '1000',
            '--seed-count',
            '3',
            '--ledger-path',
            str(LEDGER),
            '--output',
            str(REPORT_DIR / 'deep_probe.json'),
        ])
        return

    if MODE == 'fidelity':
        run([
            sys.executable,
            '-m',
            'component_fab.tools.run_fidelity',
            '--ledger',
            str(LEDGER),
            '--max-candidates',
            '4',
            '--r1-steps',
            '300',
            '--store',
            str(DRIVE_ROOT / 'fidelity_scores.jsonl'),
            '--out',
            str(REPORT_DIR / 'fidelity_report.json'),
        ])
        return

    if MODE == 'surrogate':
        run([
            sys.executable,
            '-m',
            'component_fab.tools.run_surrogate',
            '--ledger',
            str(LEDGER),
            '--out',
            str(REPORT_DIR / 'surrogate_report.json'),
        ])
        return

    raise SystemExit(f'Unknown FAB_MODE/MODE: {MODE}')


write_status('starting')
setup_repo()
write_status('running')
run_mode()
write_status('complete', report_dir=str(REPORT_DIR), log=str(LOG), ledger=str(LEDGER))
print(f'\nDone. Drive output: {DRIVE_ROOT}')
