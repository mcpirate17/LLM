# Task Plan

## 2026-04-12 — codex — P2 cleanup and API route splitting

1. Audit claimed deletion targets against live imports, route wiring, and recent git history; only delete modules that are truly unwired.
2. Refactor the large API route modules by extracting nested handlers into top-level functions and reducing each `register_*_routes` function to route wiring.
3. Run targeted import/syntax verification, then summarize what was deleted versus what could not be safely removed.
