# Aria Observability Plan — Infrastructure + Analytics

## Problem Statement

The current "Observability" tab is component health only (TF-IDF blame, gradient stability, pass rates). What's missing is **infrastructure observability** — the plumbing view: errors, crashes, pipeline throughput, resource utilization, failure patterns, and deep analytical views that let you see whether the system is working or slowly breaking.

This plan splits observability into two dashboard tabs and adds the missing data collection.

---

## Tab Structure

### Tab 1: Infrastructure (new)
Pipeline health, errors, process lifecycle, throughput, resource usage.
This is the "is the system broken?" tab.

### Tab 2: Component Analytics (existing, expanded)
Component health grid, op pair analysis, loss distributions, grammar evolution.
This is the "what's working and what's not?" tab.

---

## Tab 1: Infrastructure

### 1.1 Error Log Viewer

**What exists:** `aria_dashboard.log` (rotating, 2MB), error_type/error_message in program_results, Flask 4xx/5xx post-response logger.

**What to build:**
- `/api/observability/errors` — query recent errors from program_results + learning_log
  - Filter by: time range, error_type, severity, experiment_id
  - Return: `[{timestamp, error_type, error_message, experiment_id, result_id, stage}]`
  - Aggregate: error_type histogram over last 1h/6h/24h
- React: Scrollable error log with severity coloring, filterable by type
- React: Error rate sparkline (errors/hour over last 24h)

**Data source:** `program_results.error_type`, `program_results.error_message`, `program_results.stage0_error`

**Schema addition:** None needed — data already exists but isn't queryable via API.

### 1.2 Experiment Lifecycle Monitor

**What exists:** experiments table with status column, but orphaned "running" experiments require manual cleanup.

**What to build:**
- `/api/observability/experiments/lifecycle` — experiment state machine view
  - Shows: all experiments by status (running/completed/failed), with duration
  - Flags: orphaned (running but no process), stale (no results in >1h)
  - Actions: mark-failed, delete (with FK cascade)
- React: State machine diagram showing experiment flow
- React: Table of recent experiments with status badges and duration

**Recovery automation:**
- On dashboard load, detect orphaned experiments (status=running, no PID alive)
- Offer one-click cleanup button
- Signal handler addition: runner registers `atexit` + `SIGTERM` handler to mark experiments as `interrupted` on crash

**Insertion point:** `research/scientist/runner/core.py` — add `atexit.register(_cleanup_running)` in `__init__`

### 1.3 Pipeline Throughput

**What exists:** timestamp on every program_result, but no aggregate throughput metric.

**What to build:**
- `/api/observability/throughput` — pipeline velocity metrics
  - `programs_per_hour`: COUNT(program_results) in last 1h
  - `experiments_per_hour`: COUNT(experiments completed) in last 1h
  - `s1_per_hour`: COUNT(stage1_passed=1) in last 1h
  - `stage_latency`: {s0_median_ms, s1_median_ms, total_median_ms} from timing columns
  - `queue_depth`: number of experiments in "running" state
  - Time series: hourly counts over last 24h for trend line
- React: Throughput gauge cards + 24h trend sparklines
- React: Stage latency percentile bars (p50, p90, p99)

**Data source:** `program_results.timestamp`, `program_results.compile_time_ms`, `program_results.total_train_time_ms`, `experiments.timestamp`

### 1.4 Resource Utilization

**What exists:** `peak_memory_mb`, `avg_step_time_ms`, `throughput_tok_s` per program. GPU starvation JSON blobs.

**What to build:**
- `/api/observability/resources` — aggregate resource view
  - GPU memory: p50/p90/max of peak_memory_mb over recent runs
  - Training speed: p50/p90 of avg_step_time_ms
  - Throughput: p50/p90 of throughput_tok_s
  - CUDA status: available, device name, free/total VRAM (from system_bp)
  - Starvation events: count of GPU starvation reports in last 24h
- React: Memory usage distribution chart
- React: CUDA status card with live VRAM bar

**Data source:** `program_results.peak_memory_mb`, `program_results.avg_step_time_ms`, `program_results.throughput_tok_s`, `program_results.gpu_starvation_json`

### 1.5 API Health

**What exists:** Flask error handlers log 4xx/5xx, but no aggregate tracking.

**What to build:**
- Middleware: increment in-memory counter per status code bucket (2xx/4xx/5xx)
- `/api/observability/api-health` — API request stats
  - `requests_total`, `errors_total`, `error_rate`
  - Per-endpoint: top 5 slowest, top 5 most-errored
  - Uptime: seconds since last restart
- React: API health card with error rate gauge

**Insertion point:** `research/scientist/api.py` `@app.after_request` handler (already logs 4xx/5xx — add counter)

### 1.6 SSE Connection Status

**What exists:** SSE endpoints exist but client reconnection is not implemented.

**What to build:**
- Frontend: `EventSource` wrapper with exponential backoff reconnection
- React: Connection status indicator (connected/reconnecting/disconnected)
- Heartbeat tracking: if no keepalive in 10s, show "stream stale"

### 1.7 Database Health

**What exists:** WAL mode, writer thread with queue serialization.

**What to build:**
- `/api/observability/db-health` — database status
  - `size_mb`: file size of lab_notebook.db
  - `wal_size_mb`: WAL file size (indicates write backlog)
  - `table_counts`: {experiments, program_results, leaderboard, insights, ...}
  - `write_queue_depth`: length of writer thread queue
  - `last_write_ts`: timestamp of last successful write
- React: DB health card with size, queue depth, last write indicator

---

## Tab 2: Component Analytics (expanded)

### 2.1 Component Health Grid (exists — keep as-is)
TF-IDF blame, gradient stability, NaN/Inf detection, pass rates.

### 2.2 Op Pair Interaction Map (new)

**What exists:** `pair_profiles` in profiling DB (composition metrics), `failure_signatures` table (op-pair bigrams with fail rates), `mine_op_pairs.py` tool.

**What to build:**
- `/api/observability/op-pairs` — op pair analytics
  - Query `failure_signatures` for toxic pairs (high fail rate)
  - Query `pair_profiles` from profiling DB for composition health
  - Compute co-occurrence matrix from program_results graph_json
  - Return: `[{op_a, op_b, n_cooccur, s1_rate_when_both_present, avg_loss_ratio, toxic}]`
- React: Heatmap matrix showing op pair success rates
- React: Top 10 toxic pairs table, top 10 synergistic pairs table
- Click on cell → shows pair-specific metrics (stability delta, grad health)

**Data sources:**
- `failure_signatures` table: `signature, n_failures, n_successes`
- `profiling/component_profiles.db::pair_profiles`: `stability_delta, distribution_shift, speed_overhead`
- `program_results.graph_json`: parse and extract op co-occurrence

### 2.3 Loss Distribution by Op (new)

**What exists:** `avg_loss_ratio` per op in op_success_rates, but no distribution data.

**What to build:**
- `/api/observability/op-loss-distribution` — per-op loss percentiles
  - For each op: compute p10, p25, p50, p75, p90 of loss_ratio from programs containing that op
  - Compare to global loss distribution (baseline)
  - Flag ops where p25 is worse than global p75 (consistently harmful)
  - Flag ops where p75 is better than global p25 (consistently helpful)
- React: Box-and-whisker chart for top 30 ops by usage
- React: Sortable table with percentile columns + "vs global" delta
- Color coding: green if op's median < global median, red if worse

**Query pattern:**
```sql
SELECT op_name,
  -- need to join program_results with op presence
  -- parse graph_json to extract ops per program, then GROUP BY op
```

**Implementation note:** This requires parsing `graph_json` for each program to extract ops, which is expensive. Cache the result with a 5-minute TTL. Pre-compute on the notebook side as a periodic background task, or compute once per experiment completion.

### 2.4 Grammar Weight Evolution (new)

**What exists:** `learning_log` table with `grammar_weights_applied` events containing old/new weights as JSON.

**What to build:**
- `/api/observability/grammar-evolution` — grammar weight time series
  - Extract all `grammar_weights_applied` events from learning_log
  - Return time series per category: `[{timestamp, category, old_weight, new_weight}]`
  - Compute: weight stability (std of changes), direction (trending up/down), magnitude
- React: Multi-line chart showing weight evolution per category over time
- React: "Grammar health" summary: stable/volatile/converging

**Data source:** `learning_log WHERE event_type = 'grammar_weights_applied'`, fields: `old_weights`, `new_weights`, `timestamp`

### 2.5 Failure Pattern Clustering (new)

**What exists:** `error_type` classification (5 stage-0 types, 5 inflight types), `failure_signatures` table.

**What to build:**
- `/api/observability/failure-patterns` — structured failure analysis
  - Group failures by: error_type × top contributing ops
  - Return: `[{error_type, count, pct, top_ops, example_graph_fingerprint}]`
  - Trend: failure type distribution over last 7 days (shifting failure modes)
- React: Stacked bar chart of error types over time
- React: Drilldown: click error type → see which ops are most associated

**Data source:** `program_results.error_type`, `program_results.graph_json` (parse for op extraction)

### 2.6 Leaderboard Dynamics (new)

**What exists:** leaderboard table with tier + composite_score + timestamp, but no transition tracking.

**What to build:**
- `/api/observability/leaderboard-dynamics` — search velocity
  - `new_entries_per_day`: screening entries added per day (last 14 days)
  - `promotions_per_day`: tier upgrades per day
  - `best_score_trajectory`: best composite_score over time
  - `tier_distribution`: {screening: N, investigation: N, validation: N}
  - `staleness`: days since last promotion to validation tier
- React: Tier funnel visualization (screening → investigation → validation)
- React: Score trajectory line chart

### 2.7 Insight Effectiveness (new)

**What exists:** `selection_insight_trials` with reward tracking, `insights` table with Bayesian alpha/beta.

**What to build:**
- `/api/observability/insight-effectiveness` — insight ROI
  - Per insight: `{insight_id, category, n_trials, mean_reward, confidence_interval}`
  - Top 5 most effective insights, bottom 5 least effective
  - Interaction analysis: which insight pairs have highest mean_reward
- React: Insight leaderboard with confidence intervals
- React: Interaction heatmap (insight × insight → mean reward)

---

## Data Collection Additions

### New columns needed: None
All data already exists in the DB. The gap is **aggregation and API exposure**, not storage.

### New background computations needed:

1. **Op co-occurrence matrix** — compute once per experiment completion
   - Parse `graph_json` for all stage1-passing programs
   - Build `{(op_a, op_b): {count, avg_loss_ratio, s1_rate}}` dict
   - Cache in memory with 5-minute TTL
   - ~50ms for 500 programs (parse JSON + count pairs)

2. **Op loss percentiles** — compute once per experiment completion
   - For each op, collect all loss_ratios from programs containing it
   - Compute p10/p25/p50/p75/p90
   - Cache with 5-minute TTL

3. **Hourly throughput buckets** — increment in-memory counters
   - Count programs evaluated, s1 passes, experiments completed per hour
   - Rolling 24h window (array of 24 hourly buckets)

4. **API request counters** — in-memory dict
   - Increment per request in `@app.after_request`
   - Bucket by: endpoint, status_code, 5-minute window

### Signal handler addition:
```python
# research/scientist/runner/core.py __init__
import atexit, signal
def _cleanup():
    for exp_id in self._active_experiment_ids:
        self.notebook.conn.execute(
            "UPDATE experiments SET status='interrupted' WHERE experiment_id=?",
            (exp_id,))
    self.notebook.conn.commit()
atexit.register(_cleanup)
signal.signal(signal.SIGTERM, lambda *_: _cleanup())
```

---

## Implementation Order

| Phase | What | LOC (est) | Impact |
|-------|------|-----------|--------|
| **P0** | Error log viewer API + React | ~120 | See what's failing right now |
| **P0** | Experiment lifecycle monitor + orphan cleanup | ~100 | Stop ghost experiments |
| **P0** | Pipeline throughput metrics | ~80 | Know if search is productive |
| **P1** | Op pair heatmap (from failure_signatures + graph_json) | ~150 | See toxic/synergistic combos |
| **P1** | Loss distribution by op (percentile box plots) | ~120 | Know which ops help vs hurt |
| **P1** | Resource utilization (GPU memory, training speed) | ~80 | Capacity planning |
| **P2** | Grammar weight evolution chart | ~80 | Track if learning is converging |
| **P2** | Failure pattern clustering | ~100 | Understand recurring failures |
| **P2** | Leaderboard dynamics (tier funnel + velocity) | ~80 | Search progress tracking |
| **P2** | SSE reconnection + connection status | ~40 | Reliable live updates |
| **P3** | API health counters | ~60 | Dashboard API reliability |
| **P3** | DB health monitor | ~50 | Storage health |
| **P3** | Insight effectiveness leaderboard | ~100 | ROI on hypothesis system |
| **P3** | Signal handler for clean shutdown | ~20 | Prevent orphaned experiments |

**Total: ~1,180 LOC** across Python API + React components.

---

## API Endpoint Summary

### Infrastructure Tab (new)
```
GET /api/observability/errors              Error log with filters
GET /api/observability/experiments/lifecycle  Experiment state machine
GET /api/observability/throughput           Pipeline velocity metrics
GET /api/observability/resources            GPU/memory/training speed
GET /api/observability/api-health           Request counters + error rates
GET /api/observability/db-health            Database size, WAL, queue depth
```

### Component Analytics Tab (expanded)
```
GET /api/observability/health              (exists) TF-IDF component health
GET /api/observability/op-pairs            Op pair interaction heatmap
GET /api/observability/op-loss-distribution Per-op loss percentiles
GET /api/observability/grammar-evolution    Grammar weight time series
GET /api/observability/failure-patterns     Error type clustering
GET /api/observability/leaderboard-dynamics Tier funnel + velocity
GET /api/observability/insight-effectiveness Insight ROI leaderboard
```

### Shared
```
GET  /api/observability/monitor            (exists) Compact CLI summary
GET  /api/observability/alerts             (exists) Threshold alerts
GET  /api/observability/stream             (exists) SSE training stream
POST /api/observability/health/refresh     (exists) Force refresh cache
```

---

## React Component Structure

```
dashboard/src/components/
├── ObservabilityDashboard.js      (exists — rename to ComponentHealth.js)
├── InfrastructureDashboard.js     (new — Tab 1)
│   ├── ErrorLogPanel              Error log viewer with filters
│   ├── ExperimentLifecycle        State machine + orphan cleanup
│   ├── ThroughputGauges           Pipeline velocity cards + sparklines
│   ├── ResourceUtilization        GPU memory + training speed
│   ├── ApiHealthCard              Request counters + error rate
│   └── DbHealthCard              Database size + WAL + queue depth
└── ComponentAnalytics.js          (new — Tab 2, wraps existing + new)
    ├── ComponentGrid              (exists — from ObservabilityDashboard.js)
    ├── OpPairHeatmap              Co-occurrence + success rate matrix
    ├── LossDistributionChart      Per-op box-and-whisker
    ├── GrammarEvolutionChart      Weight time series
    ├── FailurePatternView         Error type stacked bars
    ├── LeaderboardDynamics        Tier funnel + score trajectory
    └── InsightEffectiveness       Insight leaderboard + interactions
```

---

## Key Design Principles

1. **No vanity metrics** — every panel must answer "is something broken?" or "what should I change?"
2. **Actionable alerts** — every red/yellow indicator links to the data that explains it
3. **CLI-first** — every endpoint works with `curl | jq` for automated monitoring
4. **Cached computation** — expensive queries (graph_json parsing) cached with TTL, not computed per-request
5. **Incremental rollout** — each panel is independent; deploy one at a time
6. **No new tables** — all data already exists; the gap is aggregation and API exposure
