"""Single source of truth for shared configuration defaults.

All projects (research, aria_designer, aria_core) import from here
instead of hardcoding values.  Keep this module dependency-free
(stdlib only) so it can be imported anywhere without side-effects.
"""

from __future__ import annotations

from pathlib import Path

# ── Project root (absolute, CWD-independent) ─────────────────────────
# Anchored on this file's location: research/defaults.py → parent.parent = LLM/.
# Use the *_ABS Path constants below (not the legacy relative strings) when
# constructing filesystem paths in code that may run with cwd=research/.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# ── Service ports ─────────────────────────────────────────────────────
DASHBOARD_PORT: int = 5000
DESIGNER_API_PORT: int = 8091
DESIGNER_UI_PORT: int = 5174
OLLAMA_PORT: int = 11434

# ── Service URLs (derived from ports) ────────────────────────────────
RESEARCH_API_BASE: str = f"http://127.0.0.1:{DASHBOARD_PORT}"
DESIGNER_API_BASE: str = f"http://127.0.0.1:{DESIGNER_API_PORT}"
DESIGNER_UI_BASE: str = f"http://127.0.0.1:{DESIGNER_UI_PORT}"
DESIGNER_API_HEALTH: str = f"{DESIGNER_API_BASE}/health"
OLLAMA_BASE: str = f"http://localhost:{OLLAMA_PORT}"

# ── Database paths (relative to project root) ────────────────────────
# Legacy relative strings — kept for back-compat with consumers that already
# anchor on PROJECT_ROOT externally. New code should prefer the *_ABS Paths.
LAB_NOTEBOOK_DB: str = "research/lab_notebook.db"
RUNS_DB: str = "research/runs.db"
EVENTS_DB: str = "research/events.db"
RUNTIME_EVENTS_DIR: str = "research/runtime_events"
NOTEBOOK_ARTIFACTS_DIR: str = "research/artifacts/notebook"

# Absolute Paths — use these when the consumer can't guarantee its CWD.
# (CWD=research/ + relative "research/foo" silently writes to research/research/foo.)
RUNTIME_EVENTS_DIR_ABS: Path = PROJECT_ROOT / "research" / "runtime_events"
RUNTIME_DIR_ABS: Path = PROJECT_ROOT / "research" / "runtime"

# ── Model architecture defaults ──────────────────────────────────────
MODEL_DIM: int = 256
VOCAB_SIZE: int = 100277  # tiktoken cl100k_base
MAX_SEQ_LEN: int = 256  # stage-1 / screening
VALIDATION_SEQ_LEN: int = 512  # investigation + validation
N_LAYERS: int = 6
N_HEADS: int = 8
N_KV_HEADS: int = 4

# ── Training defaults (stage budgets) ────────────────────────────────
STAGE1_STEPS: int = 750
STAGE1_LR: float = 3e-4
STAGE1_BATCH_SIZE: int = 4
INVESTIGATION_STEPS: int = 2500
INVESTIGATION_BATCH_SIZE: int = 4
VALIDATION_STEPS: int = 10000
VALIDATION_BATCH_SIZE: int = 8
SCALE_UP_STEPS: int = 5000
SCALE_UP_BATCH_SIZE: int = 8
SCALE_UP_SEQ_LEN: int = 512

# ── Positional encoding ──────────────────────────────────────────────
ROPE_THETA_BASE: float = 10000.0

# ── Timeouts & retries ───────────────────────────────────────────────
DESIGNER_PROXY_TIMEOUT: float = 10.0
DESIGNER_BOOT_TIMEOUT: float = 90.0
DESIGNER_IDLE_TIMEOUT: float = 900.0
LINEAGE_SYNC_TIMEOUT: float = 3.0
SQLITE_BUSY_TIMEOUT_MS: int = 30000
