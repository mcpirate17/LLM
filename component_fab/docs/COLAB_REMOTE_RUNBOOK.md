# component_fab Remote Colab Runbook

This is the no-computer path. It assumes only an iPad/browser, GitHub, Google Colab, and Google Drive.

## One-cell bootstrap

Open a blank Colab notebook and paste this cell:

```python
from google.colab import drive
drive.mount('/content/drive')

!rm -rf /content/LLM
!git clone --branch fix/component-fab-colab https://github.com/mcpirate17/LLM.git /content/LLM
%cd /content/LLM

!python -m pip install -q xxhash zstandard pyyaml flask-cors lightgbm ninja
!python -m component_fab.tools.colab_worker --mode smoke
```

If smoke passes, the remote path is usable.

## Where output goes

Default Drive folder:

```text
/content/drive/MyDrive/Colab Notebooks/component_fab/
```

Inside it:

```text
reports/   JSON outputs
logs/      streamed stdout/stderr
status/    live status JSON per mode
ledger.jsonl  remote component_fab ledger
```

## Recommended runs from London

### 1. Smoke check

```python
%cd /content/LLM
!python -m component_fab.tools.colab_worker --mode smoke
```

### 2. Surrogate report

```python
%cd /content/LLM
!python -m component_fab.tools.colab_worker --mode surrogate
```

Use this to check whether surrogate selection is trustworthy yet.

### 3. Deep probe — highest value run

```python
%cd /content/LLM
!python -m component_fab.tools.colab_worker --mode deep_probe -- --top-k 12 --steps 3000 --seed-count 3 --statuses promoted+pending --output {report_dir}/deep_probe_top12.json
```

This is the best near-term use of Colab. It asks whether promoted/pending fab candidates actually beat frontier baselines.

### 4. Fidelity ladder

```python
%cd /content/LLM
!python -m component_fab.tools.colab_worker --mode fidelity -- --max-candidates 8 --r1-steps 500 --store {report_dir}/fidelity_scores.jsonl --out {report_dir}/fidelity_report.json
```

This checks whether nano-scale ranking predicts deeper results.

### 5. Probe cost benchmark

```python
%cd /content/LLM
!python -m component_fab.tools.colab_worker --mode probe_bench
```

Use this before expanding probe suites.

### 6. Small autonomous screen

Do not start with a giant run. Use a small screened cycle:

```python
%cd /content/LLM
!python -m component_fab.tools.colab_worker --mode autonomous -- --cycles 1 --max-graded-per-cycle 8 --paired-seeds 3 --emit-run-summary
```

## What not to do first

Do not run huge autonomous/NAS sweeps from Colab until deep-probe and fidelity reports show that the ranking signals are meaningful.

The project already generates many candidates. The bottleneck is credible evidence.

## Best weekly operating cadence

1. Run `deep_probe` on promoted+pending.
2. Run `fidelity` to measure nano-to-R1 correlation.
3. Run `surrogate` after enough ledger evidence accumulates.
4. Run one small autonomous cycle only when the evidence says selection is improving.
5. Review reports in Drive before scaling anything.

## If Colab disconnects

Check:

```text
Colab Notebooks/component_fab/status/<mode>.json
Colab Notebooks/component_fab/logs/<mode>.log
```

If the status is `failed`, open the log first. Do not rerun blindly; failed imports or missing ledger paths usually mean the setup cell was not run from `/content/LLM`.
