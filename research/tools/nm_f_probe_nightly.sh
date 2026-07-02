#!/usr/bin/env bash
# Nightly NM-F capability probe run + runs.db ingest + dashboard refresh.
# Scheduled by the systemd user timer nm-f-probes.timer (unit copies versioned in
# research/tools/systemd/; install: cp research/tools/systemd/nm-f-probes.* \
#   ~/.config/systemd/user/ && systemctl --user daemon-reload && \
#   systemctl --user enable --now nm-f-probes.timer).
# Logs: journalctl --user -u nm-f-probes.service
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="/home/tim/venvs/llm/bin/activate"
GPU_BUSY_MIB=8000  # a real training run owns the GPU; probes must never contend

cd "$REPO"
# shellcheck disable=SC1090
source "$VENV"

used_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [ "${used_mib:-0}" -gt "$GPU_BUSY_MIB" ]; then
    echo "SKIP: GPU busy (${used_mib} MiB > ${GPU_BUSY_MIB} MiB) — a training run" \
         "is active; one run at a time. Will retry next scheduled window." >&2
    exit 0
fi

echo "=== NM-F nightly probes $(date -u +%FT%TZ) ==="
python research/tools/nm_f_capability_probes.py \
    --probe all --steps 6000 --seeds 3 --lr 1e-3 --body-len 48

python research/tools/ingest_nm_f_probes.py
python .claude/hooks/obsidian_sync.py sync-notes || true
echo "=== done $(date -u +%FT%TZ) ==="
