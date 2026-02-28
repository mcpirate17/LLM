import React, { useEffect, useState, useMemo, useCallback } from 'react';
import { formatTime, formatDuration, scoreColor } from '../utils/format';
import { noveltyColor, reliabilityColor } from '../utils/colors';
import { experimentScore, experimentScoreBreakdown } from '../utils/scoringEngine';
import { filterRowsByQuery } from '../utils/tableFiltering';
import useCopyToClipboard from '../hooks/useCopyToClipboard';

/** Color-code experiment outcome: green (good), amber (ok), red (bad) */
function experimentRating(exp) {
  if (exp?.status === 'running' && exp?.experiment_type === 'validation') {
    return {
      color: 'var(--accent-blue)',
      label: 'Validating',
      tip: 'Validation run in progress — rating shown after aggregate seed results arrive',
    };
  }

  const n = exp.n_programs_generated || 0;
  const s1 = exp.n_stage1_passed || 0;
  const rate = n > 0 ? s1 / n : 0;

  if (s1 > 2 || rate > 0.05) return { color: 'var(--accent-green)', label: 'Strong', tip: 'Multiple architectures learned — productive experiment' };
  if (s1 > 0) return { color: 'var(--accent-yellow)', label: 'Some', tip: 'At least one learnable architecture found' };
  if ((exp.n_stage0_passed || 0) > n * 0.3) return { color: 'var(--accent-orange, #f0883e)', label: 'Weak', tip: 'Programs compiled but none learned — need better op combinations' };
  return { color: 'var(--accent-red)', label: 'Poor', tip: 'Most programs failed to compile — grammar too aggressive' };
}

function metricText(value, fallbackReason, formatter) {
  if (value == null) return fallbackReason;
  return formatter(value);
}

function parseExperimentTime(exp) {
  const raw = exp?.timestamp || exp?.created_at || '';
  const t = raw ? new Date(raw).getTime() : Number.NaN;
  return Number.isFinite(t) ? t : 0;
}

function MiniSparkline({ values, color }) {
  const finite = (Array.isArray(values) ? values : []).filter((v) => Number.isFinite(v));
  if (finite.length < 2) {
    return <div style={{ height: 26, borderRadius: 4, background: 'var(--bg-tertiary)', border: '1px solid var(--border)' }} />;
  }

  const W = 130;
  const H = 26;
  const pad = 2;
  const min = Math.min(...finite);
  const max = Math.max(...finite);
  const range = max - min || 1;
  const points = finite.map((value, idx) => {
    const x = pad + (idx / Math.max(finite.length - 1, 1)) * (W - pad * 2);
    const y = H - pad - ((value - min) / range) * (H - pad * 2);
    return `${x},${y}`;
  }).join(' ');
  return (
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 26 }}>
      <polyline points={points} fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function ExperimentKpiStrip({ experiments }) {
  const series = useMemo(() => {
    const rows = (Array.isArray(experiments) ? experiments : [])
      .slice()
      .sort((a, b) => parseExperimentTime(a) - parseExperimentTime(b))
      .slice(-20);

    const passRate = rows.map((exp) => {
      const generated = exp.n_programs_generated || 0;
      return generated > 0 ? ((exp.n_stage1_passed || 0) / generated) * 100 : 0;
    });
    const bestLoss = rows.map((exp) => (exp.best_loss_ratio == null ? null : exp.best_loss_ratio));
    const novelty = rows.map((exp) => (exp.best_novelty_score == null ? null : exp.best_novelty_score));
    const failureRate = rows.map((exp) => {
      const generated = exp.n_programs_generated || 0;
      const compiled = exp.n_stage0_passed || 0;
      return generated > 0 ? (1 - (compiled / generated)) * 100 : 0;
    });
    return { passRate, bestLoss, novelty, failureRate };
  }, [experiments]);

  const kpis = [
    {
      key: 'passRate',
      label: 'S1 pass',
      values: series.passRate,
      color: 'var(--accent-green)',
      higherBetter: true,
      format: (v) => `${v.toFixed(1)}%`,
    },
    {
      key: 'bestLoss',
      label: 'Best loss',
      values: series.bestLoss,
      color: 'var(--accent-blue)',
      higherBetter: false,
      format: (v) => v.toFixed(3),
    },
    {
      key: 'novelty',
      label: 'Novelty',
      values: series.novelty,
      color: 'var(--accent-purple)',
      higherBetter: true,
      format: (v) => v.toFixed(3),
    },
    {
      key: 'failureRate',
      label: 'Compile failure',
      values: series.failureRate,
      color: 'var(--accent-red)',
      higherBetter: false,
      format: (v) => `${v.toFixed(1)}%`,
    },
  ];

  return (
    <div style={{
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
      gap: 10,
      marginBottom: 12,
    }}>
      {kpis.map((kpi) => {
        const finite = kpi.values.filter((v) => Number.isFinite(v));
        const current = finite.length > 0 ? finite[finite.length - 1] : null;
        const previous = finite.length > 1 ? finite[finite.length - 2] : null;
        const delta = current != null && previous != null ? current - previous : null;
        const positiveDelta = delta != null && ((kpi.higherBetter && delta >= 0) || (!kpi.higherBetter && delta <= 0));
        return (
          <div key={kpi.key} style={{ border: '1px solid var(--border)', borderRadius: 6, background: 'var(--bg-tertiary)', padding: '8px 10px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 4, gap: 8 }}>
              <span style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase' }}>{kpi.label}</span>
              <span style={{ fontSize: 12, color: kpi.color, fontWeight: 700 }}>
                {current == null ? '--' : kpi.format(current)}
              </span>
            </div>
            <MiniSparkline values={kpi.values} color={kpi.color} />
            <div style={{ marginTop: 3, fontSize: 10, color: delta == null ? 'var(--text-muted)' : (positiveDelta ? 'var(--accent-green)' : 'var(--accent-red)') }}>
              {delta == null ? 'No prior point' : `${delta > 0 ? '+' : ''}${kpi.format(delta)}`}
            </div>
          </div>
        );
      })}
    </div>
  );
}


function experimentMetricChips(exp) {
  const nPrograms = exp.n_programs_generated || 0;
  const s1 = exp.n_stage1_passed || 0;
  const evidenceReliability = nPrograms >= 100 ? 'high' : nPrograms >= 30 ? 'medium' : 'low';
  return [
    {
      label: 'Loss',
      source: exp.best_loss_ratio != null ? 'measured' : 'not-evaluated',
      reliability: exp.best_loss_ratio != null ? evidenceReliability : 'low',
    },
    {
      label: 'Novelty',
      source: exp.best_novelty_score != null ? 'heuristic' : 'insufficient-data',
      reliability: s1 > 0 ? evidenceReliability : 'low',
    },
    {
      label: 'Baseline',
      source: 'not-available',
      reliability: 'low',
    },
  ];
}

const COLUMNS = [
  { key: 'score', label: 'Score' },
  { key: 'rating', label: 'Rating' },
  { key: 'experiment_id', label: 'ID' },
  { key: 'experiment_type', label: 'Type' },
  { key: 'hypothesis', label: 'Hypothesis' },
  { key: 'status', label: 'Status' },
  { key: 'stage_funnel', label: 'Funnel Progress' },
  { key: 'top_discoveries', label: 'Top Discoveries' },
  { key: 'n_stage1_passed', label: 'S1 Pass' },
  { key: 'best_loss_ratio', label: 'Best Loss' },
  { key: 'best_novelty_score', label: 'Best Novelty' },
  { key: 'timestamp', label: 'Time' },
];

/** Mini stage funnel: generated -> compiled -> stage0.5 -> S1 */
function StageFunnel({ generated, s0, s05, s1 }) {
  if (!generated) return <span style={{ color: 'var(--text-muted)' }}>--</span>;
  
  const stages = [
    { label: 'S0', value: s0 || 0, color: 'var(--accent-blue)', total: generated },
    { label: 'S0.5', value: s05 || 0, color: 'var(--accent-yellow)', total: generated },
    { label: 'S1', value: s1 || 0, color: 'var(--accent-green)', total: generated },
  ];

  return (
    <div style={{ width: 100, display: 'flex', flexDirection: 'column', gap: 2 }}>
      <div style={{ display: 'flex', height: 6, borderRadius: 2, overflow: 'hidden', background: 'var(--bg-tertiary)' }}>
        {stages.map((s, i) => (
          <div 
            key={i} 
            title={`${s.label}: ${s.value}/${s.total}`}
            style={{ 
              width: `${(s.value / generated) * 100}%`, 
              height: '100%', 
              background: s.color,
              opacity: 0.8,
              transition: 'width 0.3s ease'
            }} 
          />
        ))}
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 9, color: 'var(--text-muted)' }}>
        <span>{generated} gen</span>
        <span>{s1} pass</span>
      </div>
    </div>
  );
}

const EXPERIMENT_LIST_SORT_PREFS_KEY = 'dashboard.experiment-list.sort.v1';
const EXPERIMENT_LIST_EXPERT_KEY = 'dashboard.experiment-list.expert.v1';

function ExperimentList({
  experiments,
  onSelectExperiment,
  onRefresh,
  onLoadMore,
  hasMore = false,
  loadingMore = false,
  pageSize = 200,
  onPageSizeChange,
}) {
  const [showExpertColumns, setShowExpertColumns] = useState(() => {
    try {
      return localStorage.getItem(EXPERIMENT_LIST_EXPERT_KEY) === 'true';
    } catch {
      return false;
    }
  });
  const [hideLowScore, setHideLowScore] = useState(true);

  const [sortKey, setSortKey] = useState(() => {
    try {
      const stored = JSON.parse(localStorage.getItem(EXPERIMENT_LIST_SORT_PREFS_KEY) || '{}');
      if (typeof stored.sortKey === 'string' && COLUMNS.some((column) => column.key === stored.sortKey)) {
        return stored.sortKey;
      }
    } catch {}
    return 'score';
  });
  const [sortDesc, setSortDesc] = useState(() => {
    try {
      const stored = JSON.parse(localStorage.getItem(EXPERIMENT_LIST_SORT_PREFS_KEY) || '{}');
      if (typeof stored.sortDesc === 'boolean') {
        return stored.sortDesc;
      }
    } catch {}
    return true;
  });
  const [filterQuery, setFilterQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');
  const [typeFilter, setTypeFilter] = useState('all');
  const [outcomeFilter, setOutcomeFilter] = useState('all');
  const [copiedValue, copyText] = useCopyToClipboard();
  const [cancellingId, setCancellingId] = useState(null);
  const [rerunningId, setRerunningId] = useState(null);
  const [confirmingAction, setConfirmingAction] = useState(null); // { id, type }
  const [inlineError, setInlineError] = useState(null); // { id, message }
  const [selectedIds, setSelectedIds] = useState(new Set());
  const [batchRerunActive, setBatchRerunActive] = useState(false);
  const [batchRerunStatus, setBatchRerunStatus] = useState(null);

  const handleCancel = async (e, experimentId) => {
    e.stopPropagation();
    if (!confirmingAction || confirmingAction.id !== experimentId || confirmingAction.type !== 'cancel') {
      setConfirmingAction({ id: experimentId, type: 'cancel' });
      return;
    }
    setConfirmingAction(null);
    setCancellingId(experimentId);
    try {
      const res = await fetch(`/api/experiments/${experimentId}/cancel`, { method: 'POST' });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        setInlineError({ id: experimentId, message: data.error || 'Failed to cancel experiment' });
      } else if (onRefresh) {
        onRefresh();
      }
    } catch (err) {
      setInlineError({ id: experimentId, message: 'Network error cancelling experiment' });
    } finally {
      setCancellingId(null);
    }
  };

  const handleRerun = async (e, experimentId) => {
    e.stopPropagation();
    if (!confirmingAction || confirmingAction.id !== experimentId || confirmingAction.type !== 'rerun') {
      setConfirmingAction({ id: experimentId, type: 'rerun' });
      return;
    }
    setConfirmingAction(null);
    setRerunningId(experimentId);
    try {
      const res = await fetch(`/api/experiments/${experimentId}/rerun`, { method: 'POST' });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        setInlineError({ id: experimentId, message: data.error || 'Failed to rerun experiment' });
      } else if (onRefresh) {
        onRefresh();
      }
    } catch (err) {
      setInlineError({ id: experimentId, message: 'Network error rerunning experiment' });
    } finally {
      setRerunningId(null);
    }
  };

  useEffect(() => {
    localStorage.setItem(EXPERIMENT_LIST_SORT_PREFS_KEY, JSON.stringify({ sortKey, sortDesc }));
  }, [sortKey, sortDesc]);

  useEffect(() => {
    localStorage.setItem(EXPERIMENT_LIST_EXPERT_KEY, String(showExpertColumns));
  }, [showExpertColumns]);

  const signalKeys = new Set(['score', 'n_stage1_passed', 'best_loss_ratio', 'best_novelty_score', 'status', 'timestamp', 'experiment_id']);
  const visibleColumns = COLUMNS.filter(col => showExpertColumns || signalKeys.has(col.key));

  const handleSort = (key) => {
    if (sortKey === key) {
      setSortDesc(!sortDesc);
    } else {
      setSortKey(key);
      setSortDesc(true);
    }
  };

  // Augment experiments with computed score
  const augmented = useMemo(() => {
    if (!experiments) return [];
    return experiments.map(exp => ({
      ...exp,
      _score: experimentScore(exp),
      _rating: experimentRating(exp),
    }));
  }, [experiments]);

  const experimentTypes = useMemo(() => {
    const unique = Array.from(new Set(
      augmented
        .map((exp) => exp?.experiment_type)
        .filter((value) => typeof value === 'string' && value.trim().length > 0)
    ));
    unique.sort((a, b) => a.localeCompare(b));
    return unique;
  }, [augmented]);

  const queryFiltered = useMemo(() => (
    filterRowsByQuery(augmented, filterQuery, [
      'experiment_id',
      'experiment_type',
      'hypothesis',
      'status',
      'aria_summary',
    ])
  ), [augmented, filterQuery]);

  const filtered = useMemo(() => (
    queryFiltered.filter((exp) => {
      if (hideLowScore && exp._score < 5 && exp.status !== 'running') return false;
      if (statusFilter !== 'all' && exp.status !== statusFilter) return false;
      if (typeFilter !== 'all' && exp.experiment_type !== typeFilter) return false;
      if (outcomeFilter === 'has_s1' && (exp.n_stage1_passed || 0) <= 0) return false;
      if (outcomeFilter === 'no_s1' && (exp.n_stage1_passed || 0) > 0) return false;
      if (outcomeFilter === 'unevaluated' && !(exp.status === 'completed' && (exp.n_stage1_passed || 0) === 0 && exp.best_loss_ratio == null)) return false;
      return true;
    })
  ), [queryFiltered, statusFilter, typeFilter, outcomeFilter, hideLowScore]);

  const hasActiveFilters = (
    filterQuery.trim().length > 0 ||
    statusFilter !== 'all' ||
    typeFilter !== 'all' ||
    outcomeFilter !== 'all'
  );

  const clearFilters = () => {
    setFilterQuery('');
    setStatusFilter('all');
    setTypeFilter('all');
    setOutcomeFilter('all');
  };

  const sorted = useMemo(() => {
    const arr = [...filtered];
    arr.sort((a, b) => {
      let va, vb;
      if (sortKey === 'score') {
        va = a._score; vb = b._score;
      } else if (sortKey === 'rating') {
        // Map label to numeric for sorting
        const order = { Strong: 4, Some: 3, Validating: 2, Weak: 1, Poor: 0 };
        va = order[a._rating.label] ?? -1;
        vb = order[b._rating.label] ?? -1;
      } else if (sortKey === 'stage_funnel') {
        // Sort by compilation rate (stage0/generated)
        va = (a.n_programs_generated || 0) > 0 ? (a.n_stage0_passed || 0) / a.n_programs_generated : 0;
        vb = (b.n_programs_generated || 0) > 0 ? (b.n_stage0_passed || 0) / b.n_programs_generated : 0;
      } else {
        va = a[sortKey]; vb = b[sortKey];
      }
      // Nulls/undefined sort to bottom
      if (va == null && vb == null) return 0;
      if (va == null) return 1;
      if (vb == null) return -1;
      if (typeof va === 'string') {
        return sortDesc ? vb.localeCompare(va) : va.localeCompare(vb);
      }
      return sortDesc ? vb - va : va - vb;
    });
    return arr;
  }, [filtered, sortKey, sortDesc]);

  const toggleSelected = useCallback((id, e) => {
    if (e) e.stopPropagation();
    setSelectedIds(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  }, []);

  const toggleSelectAll = useCallback(() => {
    setSelectedIds(prev => {
      if (prev.size > 0 && sorted.every(exp => prev.has(exp.experiment_id))) {
        return new Set();
      }
      return new Set(sorted.map(exp => exp.experiment_id));
    });
  }, [sorted]);

  const handleBatchRerun = useCallback(async () => {
    if (selectedIds.size === 0) return;
    if (!window.confirm(`Rerun ${selectedIds.size} selected experiment(s)? They will be queued and run sequentially.`)) return;
    setBatchRerunActive(true);
    try {
      const res = await fetch('/api/experiments/batch-rerun', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ experiment_ids: Array.from(selectedIds) }),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        alert(data.error || 'Failed to start batch rerun');
        setBatchRerunActive(false);
        return;
      }
      setSelectedIds(new Set());
      const poll = setInterval(async () => {
        try {
          const sr = await fetch('/api/experiments/batch-rerun/status');
          const st = await sr.json();
          setBatchRerunStatus(st);
          if (!st.active) {
            clearInterval(poll);
            setBatchRerunActive(false);
            setBatchRerunStatus(null);
            if (onRefresh) onRefresh();
          }
        } catch { /* ignore poll errors */ }
      }, 5000);
    } catch (err) {
      alert('Network error starting batch rerun');
      setBatchRerunActive(false);
    }
  }, [selectedIds, onRefresh]);

  if (!experiments || experiments.length === 0) {
    return (
      <div className="card">
        <div className="card-title">Experiments</div>
        <p style={{ color: 'var(--text-muted)', fontSize: 13 }}>
          No experiments yet. Run: python -m research --mode=synthesize --n 100
        </p>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-title" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12 }}>
        <span>Experiments</span>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <input
            value={filterQuery}
            onChange={(e) => setFilterQuery(e.target.value)}
            placeholder="Filter experiments"
            style={{
              fontSize: 11,
              padding: '4px 8px',
              borderRadius: 4,
              border: '1px solid var(--border)',
              background: 'var(--bg-tertiary)',
              color: 'var(--text-primary)',
              minWidth: 180,
            }}
          />
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            style={{ fontSize: 11, padding: '4px 8px', borderRadius: 4, border: '1px solid var(--border)', background: 'var(--bg-tertiary)', color: 'var(--text-primary)' }}
            aria-label="Filter by status"
          >
            <option value="all">All status</option>
            <option value="completed">Completed</option>
            <option value="running">Running</option>
            <option value="failed">Failed</option>
          </select>
          <select
            value={typeFilter}
            onChange={(e) => setTypeFilter(e.target.value)}
            style={{ fontSize: 11, padding: '4px 8px', borderRadius: 4, border: '1px solid var(--border)', background: 'var(--bg-tertiary)', color: 'var(--text-primary)' }}
            aria-label="Filter by experiment type"
          >
            <option value="all">All types</option>
            {experimentTypes.map((type) => (
              <option key={type} value={type}>{type}</option>
            ))}
          </select>
          <select
            value={outcomeFilter}
            onChange={(e) => setOutcomeFilter(e.target.value)}
            style={{ fontSize: 11, padding: '4px 8px', borderRadius: 4, border: '1px solid var(--border)', background: 'var(--bg-tertiary)', color: 'var(--text-primary)' }}
            aria-label="Filter by outcome"
          >
            <option value="all">All outcomes</option>
            <option value="has_s1">Has S1 pass</option>
            <option value="no_s1">No S1 pass</option>
            <option value="unevaluated">Unevaluated</option>
          </select>
          {onPageSizeChange && (
            <select
              value={String(pageSize)}
              onChange={(e) => onPageSizeChange(e.target.value)}
              style={{ fontSize: 11, padding: '4px 8px', borderRadius: 4, border: '1px solid var(--border)', background: 'var(--bg-tertiary)', color: 'var(--text-primary)' }}
              aria-label="Experiments page size"
            >
              <option value="100">100 / page</option>
              <option value="200">200 / page</option>
              <option value="500">500 / page</option>
            </select>
          )}
          <button
            className="refresh-btn"
            style={{ fontSize: 11, padding: '3px 10px' }}
            onClick={clearFilters}
            disabled={!hasActiveFilters}
          >
            Clear filters
          </button>
          <button
            className="refresh-btn"
            style={{ fontSize: 11, padding: '3px 10px' }}
            onClick={() => setShowExpertColumns(!showExpertColumns)}
          >
            {showExpertColumns ? 'Hide noise' : 'Show expert columns'}
          </button>
          <button
            className="refresh-btn"
            style={{ fontSize: 11, padding: '3px 10px', color: hideLowScore ? 'var(--text-muted)' : 'var(--accent-yellow)' }}
            onClick={() => setHideLowScore(!hideLowScore)}
            title="Filter out experiments with very low scores (failed synthesis or no learning)"
          >
            {hideLowScore ? 'Show low-signal' : 'Hide low-signal'}
          </button>
          {selectedIds.size > 0 && (
            <button
              className="refresh-btn"
              style={{
                fontSize: 11, padding: '3px 10px',
                background: 'var(--accent-blue)', color: '#fff', border: 'none',
                opacity: batchRerunActive ? 0.6 : 1,
              }}
              disabled={batchRerunActive}
              onClick={handleBatchRerun}
            >
              {batchRerunActive ? 'Rerunning...' : `Rerun Selected (${selectedIds.size})`}
            </button>
          )}
        </div>
      </div>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
        Each experiment generates a batch of random computation graphs, then tests whether they can
        function as LLM layers. S1 Pass = architectures that actually learned from data.
        The system formulates a hypothesis before each experiment and adjusts strategy based on outcomes.
        Click any row for the full breakdown.
      </p>
      {batchRerunStatus && batchRerunStatus.active && (
        <div style={{ fontSize: 12, padding: '6px 10px', marginBottom: 8, borderRadius: 4, background: 'var(--accent-blue)22', border: '1px solid var(--accent-blue)44', color: 'var(--text-primary)' }}>
          Batch rerun: {batchRerunStatus.completed}/{batchRerunStatus.total} done
          {batchRerunStatus.current && <span> — running <code style={{ fontSize: 11 }}>{batchRerunStatus.current.slice(0, 8)}</code></span>}
          {batchRerunStatus.remaining.length > 0 && <span>, {batchRerunStatus.remaining.length} queued</span>}
        </div>
      )}
      <ExperimentKpiStrip experiments={sorted} />
      <table className="data-table">
        <thead>
          <tr>
            <th style={{ width: 30, textAlign: 'center', padding: '4px 2px' }}>
              <input
                type="checkbox"
                checked={sorted.length > 0 && sorted.every(exp => selectedIds.has(exp.experiment_id))}
                onChange={toggleSelectAll}
                title="Select all visible"
                style={{ cursor: 'pointer' }}
              />
            </th>
            {visibleColumns.map(col => (
              <th
                key={col.key}
                onClick={() => handleSort(col.key)}
                aria-label={`Sort op success table by ${col.label}${sortKey === col.key ? `, currently ${sortDesc ? 'descending' : 'ascending'}` : ''}`}
                style={{ cursor: 'pointer', userSelect: 'none', whiteSpace: 'nowrap' }}
              >
                {col.label}
                {sortKey === col.key && (
                  <span style={{ marginLeft: 4, fontSize: 10 }}>
                    {sortDesc ? '\u25BC' : '\u25B2'}
                  </span>
                )}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map(exp => {
            const rating = exp._rating;
            const score = exp._score;
            const isActiveValidation = exp.status === 'running' && exp.experiment_type === 'validation';
            const nUsed = exp.n_programs_generated || 0;
            const s1Count = exp.n_stage1_passed || 0;
            const chips = experimentMetricChips(exp);

            return (
              <tr key={exp.experiment_id}
                style={{ cursor: onSelectExperiment ? 'pointer' : 'default' }}
                onClick={() => onSelectExperiment && onSelectExperiment(exp.experiment_id)}>
                <td style={{ width: 30, textAlign: 'center', padding: '4px 2px' }} onClick={(e) => e.stopPropagation()}>
                  <input
                    type="checkbox"
                    checked={selectedIds.has(exp.experiment_id)}
                    onChange={(e) => toggleSelected(exp.experiment_id, e)}
                    style={{ cursor: 'pointer' }}
                  />
                </td>
                {visibleColumns.map(col => {
                  if (col.key === 'score') {
                    return (
                      <td key="score" style={{ fontWeight: 600, color: isActiveValidation ? 'var(--accent-blue)' : scoreColor(score) }}>
                        {isActiveValidation ? (
                          'running validation'
                        ) : (
                          <span title={`S1 rate ${(experimentScoreBreakdown(exp).passRate || 0).toFixed(1)}/40 | Loss ${(experimentScoreBreakdown(exp).loss || 0).toFixed(1)}/30 | Novelty ${(experimentScoreBreakdown(exp).novelty || 0).toFixed(1)}/20 | Completion ${(experimentScoreBreakdown(exp).completion || 0).toFixed(1)}/10`}>
                            {score}
                          </span>
                        )}
                      </td>
                    );
                  }
                  if (col.key === 'rating') {
                    return (
                      <td key="rating" title={rating.tip}>
                        <span style={{
                          display: 'inline-block', width: 10, height: 10, borderRadius: '50%',
                          background: rating.color, marginRight: 6,
                        }} />
                        <span style={{ fontSize: 11, color: rating.color }}>{rating.label}</span>
                      </td>
                    );
                  }
                  if (col.key === 'experiment_id') {
                    return (
                      <td key="id" style={{ fontFamily: 'monospace', fontSize: 12, color: 'var(--accent-blue)' }}>
                        <span title={exp.experiment_id}>{exp.experiment_id?.slice(0, 8)}...</span>
                        {exp.experiment_id && (
                          <button
                            className="refresh-btn"
                            style={{ fontSize: 10, padding: '1px 5px', marginLeft: 6 }}
                            onClick={(e) => {
                              e.stopPropagation();
                              copyText(exp.experiment_id);
                            }}
                            aria-label={`Copy experiment id ${exp.experiment_id}`}
                          >
                            {copiedValue === exp.experiment_id ? 'Copied' : 'Copy'}
                          </button>
                        )}
                      </td>
                    );
                  }
                  if (col.key === 'top_discoveries') {
                    return (
                      <td key="top_discoveries">
                        <div style={{ display: 'flex', gap: 4 }}>
                          {exp.best_loss_ratio != null && (
                            <span 
                              title={`Best Loss: ${exp.best_loss_ratio.toFixed(4)}`}
                              style={{ fontSize: 10, padding: '1px 4px', borderRadius: 3, background: 'rgba(63, 185, 80, 0.15)', color: 'var(--accent-green)', border: '1px solid rgba(63, 185, 80, 0.3)' }}
                            >
                              Loss: {exp.best_loss_ratio.toFixed(2)}
                            </span>
                          )}
                          {exp.best_novelty_score != null && (
                            <span 
                              title={`Best Novelty: ${exp.best_novelty_score.toFixed(3)}`}
                              style={{ fontSize: 10, padding: '1px 4px', borderRadius: 3, background: 'rgba(188, 140, 255, 0.15)', color: 'var(--accent-purple)', border: '1px solid rgba(188, 140, 255, 0.3)' }}
                            >
                              Nov: {exp.best_novelty_score.toFixed(2)}
                            </span>
                          )}
                          {exp.n_stage1_passed > 0 && !exp.best_loss_ratio && (
                            <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>awaiting eval</span>
                          )}
                          {exp.n_stage1_passed === 0 && exp.status === 'completed' && (
                            <span style={{ fontSize: 10, color: 'var(--accent-red)', opacity: 0.6 }}>no survivors</span>
                          )}
                        </div>
                      </td>
                    );
                  }
                  if (col.key === 'experiment_type') {
                    return <td key="type">{exp.experiment_type}</td>;
                  }
                  if (col.key === 'hypothesis') {
                    return (
                      <td key="hypothesis" style={{ maxWidth: 220, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', fontSize: 12, color: 'var(--text-secondary)' }}
                          title={exp.hypothesis || 'No hypothesis'}>
                        {exp.hypothesis
                          ? (exp.hypothesis.length > 60 ? exp.hypothesis.slice(0, 60) + '...' : exp.hypothesis)
                          : <span style={{ color: 'var(--text-muted)', fontStyle: 'italic' }}>none</span>
                        }
                      </td>
                    );
                  }
                  if (col.key === 'status') {
                    return (
                      <td key="status">
                        <span className={`badge ${exp.status === 'completed' ? 'pass' :
                          exp.status === 'running' ? 'running' : 'fail'}`}>
                          {exp.status}
                        </span>
                        {exp.status === 'running' && (
                          confirmingAction?.id === exp.experiment_id && confirmingAction?.type === 'cancel' ? (
                            <span style={{ fontSize: 10, marginLeft: 6 }}>
                              <span style={{ color: 'var(--accent-yellow)' }}>Cancel?</span>
                              <button className="refresh-btn" style={{ fontSize: 10, padding: '1px 5px', marginLeft: 4, color: 'var(--accent-red)', borderColor: 'var(--accent-red)' }} onClick={(e) => handleCancel(e, exp.experiment_id)}>Yes</button>
                              <button className="refresh-btn" style={{ fontSize: 10, padding: '1px 5px', marginLeft: 2 }} onClick={(e) => { e.stopPropagation(); setConfirmingAction(null); }}>No</button>
                            </span>
                          ) : (
                            <button
                              className="refresh-btn"
                              style={{
                                fontSize: 10, padding: '1px 5px', marginLeft: 6,
                                color: 'var(--accent-red)', borderColor: 'var(--accent-red)',
                              }}
                              disabled={cancellingId === exp.experiment_id}
                              onClick={(e) => handleCancel(e, exp.experiment_id)}
                              aria-label="Cancel experiment"
                            >
                              {cancellingId === exp.experiment_id ? '...' : 'Cancel'}
                            </button>
                          )
                        )}
                        {(exp.status === 'running' || exp.status === 'failed' || exp.status === 'completed') && (
                          confirmingAction?.id === exp.experiment_id && confirmingAction?.type === 'rerun' ? (
                            <span style={{ fontSize: 10, marginLeft: 6 }}>
                              <span style={{ color: 'var(--accent-yellow)' }}>Rerun?</span>
                              <button className="refresh-btn" style={{ fontSize: 10, padding: '1px 5px', marginLeft: 4 }} onClick={(e) => handleRerun(e, exp.experiment_id)}>Yes</button>
                              <button className="refresh-btn" style={{ fontSize: 10, padding: '1px 5px', marginLeft: 2 }} onClick={(e) => { e.stopPropagation(); setConfirmingAction(null); }}>No</button>
                            </span>
                          ) : (
                            <button
                              className="refresh-btn"
                              style={{ fontSize: 10, padding: '1px 5px', marginLeft: 6 }}
                              disabled={rerunningId === exp.experiment_id}
                              onClick={(e) => handleRerun(e, exp.experiment_id)}
                              aria-label="Rerun experiment"
                            >
                              {rerunningId === exp.experiment_id ? '...' : 'Rerun'}
                            </button>
                          )
                        )}
                        {inlineError?.id === exp.experiment_id && (
                          <span style={{ fontSize: 10, marginLeft: 6, color: 'var(--accent-red)' }}>
                            {inlineError.message}
                            <button className="refresh-btn" style={{ fontSize: 9, padding: '0 4px', marginLeft: 4 }} onClick={(e) => { e.stopPropagation(); setInlineError(null); }}>&times;</button>
                          </span>
                        )}
                      </td>
                    );
                  }
                  if (col.key === 'stage_funnel') {
                    return (
                      <td key="funnel" title={`${exp.n_programs_generated || 0} generated \u2192 ${exp.n_stage0_passed ?? '?'} compiled \u2192 ${exp.n_stage05_passed ?? '?'} stage0.5 \u2192 ${exp.n_stage1_passed || 0} S1`}>
                        <StageFunnel
                          generated={exp.n_programs_generated || 0}
                          s0={exp.n_stage0_passed}
                          s05={exp.n_stage05_passed}
                          s1={exp.n_stage1_passed || 0}
                        />
                      </td>
                    );
                  }
                  if (col.key === 'n_stage1_passed') {
                    return (
                      <td key="s1" style={{ color: s1Count > 0 ? 'var(--accent-green)' : 'var(--text-muted)' }}>
                        {s1Count}
                        <span style={{ marginLeft: 4, fontSize: 11, color: 'var(--text-muted)' }}>
                          / {nUsed}
                          {nUsed > 0 && ` (${((s1Count / nUsed) * 100).toFixed(1)}%)`}
                        </span>
                      </td>
                    );
                  }
                  if (col.key === 'best_loss_ratio') {
                    return (
                      <td key="loss" style={{
                        color: exp.best_loss_ratio != null
                          ? (exp.best_loss_ratio < 0.5 ? 'var(--accent-green)' : exp.best_loss_ratio < 0.8 ? 'var(--accent-yellow)' : 'var(--text-muted)')
                          : 'var(--text-muted)'
                      }}>
                        {metricText(
                          exp.best_loss_ratio,
                          (exp.n_stage1_passed || 0) > 0 ? 'not computed' : 'not yet evaluated',
                          (v) => v.toFixed(3),
                        )}
                      </td>
                    );
                  }
                  if (col.key === 'best_novelty_score') {
                    return (
                      <td key="novelty" style={{ color: noveltyColor(exp.best_novelty_score) }}>
                        {metricText(
                          exp.best_novelty_score,
                          (exp.n_stage1_passed || 0) > 0 ? 'not computed' : 'insufficient data',
                          (v) => v.toFixed(3),
                        )}
                        <div style={{ marginTop: 4, display: 'flex', gap: 4, flexWrap: 'wrap', maxWidth: 220 }}>
                          {chips.map(chip => (
                            <span
                              key={`${exp.experiment_id}-${chip.label}`}
                              title={`${chip.label}: ${chip.source}, ${chip.reliability} reliability`}
                              style={{
                                fontSize: 10,
                                padding: '1px 5px',
                                borderRadius: 4,
                                border: `1px solid ${reliabilityColor(chip.reliability)}55`,
                                color: reliabilityColor(chip.reliability),
                                background: `${reliabilityColor(chip.reliability)}22`,
                                whiteSpace: 'nowrap',
                              }}
                            >
                              {chip.label}: {chip.source}
                            </span>
                          ))}
                        </div>
                      </td>
                    );
                  }
                  if (col.key === 'aria_summary') {
                    return (
                      <td key="outcome" style={{ maxWidth: 240, fontSize: 12, color: 'var(--text-secondary)' }}
                          title={exp.aria_summary || exp.research_question || ''}>
                        {exp.aria_summary
                          ? (exp.aria_summary.length > 80
                              ? exp.aria_summary.slice(0, 80) + '...'
                              : exp.aria_summary)
                          : exp.research_question
                            ? <span style={{ fontStyle: 'italic', color: 'var(--text-muted)' }}>
                                {exp.research_question.length > 60
                                  ? exp.research_question.slice(0, 60) + '...'
                                  : exp.research_question}
                              </span>
                            : <span style={{ color: 'var(--text-muted)', fontStyle: 'italic' }}>--</span>
                        }
                      </td>
                    );
                  }
                  if (col.key === 'duration_seconds') {
                    return <td key="duration">{formatDuration(exp.duration_seconds)}</td>;
                  }
                  if (col.key === 'timestamp') {
                    return (
                      <td key="time" style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                        {formatTime(exp.timestamp)}
                      </td>
                    );
                  }
                  return <td key={col.key}>--</td>;
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
      <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 8, display: 'flex', gap: 16 }}>
        <span><span style={{ color: 'var(--accent-green)' }}>Green</span> = good results (learnable architectures found)</span>
        <span><span style={{ color: 'var(--accent-yellow)' }}>Amber</span> = some results (limited learning)</span>
        <span><span style={{ color: 'var(--accent-red)' }}>Red</span> = no learning (grammar needs adjustment)</span>
        {onSelectExperiment && <span style={{ marginLeft: 'auto' }}>Click a row for details</span>}
      </div>
      <div style={{ marginTop: 10, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
          Showing {sorted.length} experiment{sorted.length === 1 ? '' : 's'}
        </span>
        {onLoadMore && (
          <button
            className="refresh-btn"
            onClick={onLoadMore}
            disabled={!hasMore || loadingMore}
            style={{ fontSize: 11, padding: '4px 10px' }}
          >
            {loadingMore ? 'Loading…' : hasMore ? 'Load more' : 'All loaded'}
          </button>
        )}
      </div>
    </div>
  );
}

export default ExperimentList;
