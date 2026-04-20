import React, { useState, useEffect, useMemo } from 'react';
import { formatTime, formatDuration, scoreColor } from '../../utils/format';
import { lossColor, noveltyColor } from '../../utils/colors';
import { trendScore, trendScoreBreakdown } from '../../utils/dashboardHeuristics';
import { metricText } from '../../utils/metricText';
import useCopyToClipboard from '../../hooks/useCopyToClipboard';
import apiService from '../../services/apiService';
import useInteractiveTable from '../shared/useInteractiveTable';
import SortIndicator from '../shared/SortIndicator';
import { useAriaData } from '../../hooks/useAriaData';

const DATA_SORT_PREFS_KEY = 'aria_trend_data_sort_v1';

const DATA_COLUMNS = [
  { key: '_score', label: 'Score' },
  { key: 'experiment_id', label: 'ID' },
  { key: 's1_pass_rate', label: 'S1 Rate (per-exp)' },
  { key: 'trend_confidence', label: 'Confidence' },
  { key: 'best_loss_ratio', label: 'Best Loss' },
  { key: 'best_novelty_score', label: 'Best Novelty' },
  {
    key: 'avg_throughput_tok_s',
    label: 'Avg Throughput',
    tooltip: 'Average per-program throughput (tok/s). Falls back to perf report if available.'
  },
  {
    key: 'avg_routing_token_retention',
    label: 'Routing Retention',
    tooltip: 'Share of tokens processed by routing modules (higher is better).'
  },
  {
    key: 'avg_routing_utilization_entropy',
    label: 'Routing Entropy',
    tooltip: 'Load-balance entropy across experts (higher = more balanced).'
  },
  {
    key: 'avg_depth_savings_ratio',
    label: 'Depth Savings',
    tooltip: 'MoD savings vs full depth (higher = more compute saved).'
  },
  {
    key: 'avg_recursion_savings_ratio',
    label: 'Recursion Savings',
    tooltip: 'MoR savings vs max recursion (higher = more compute saved).'
  },
  { key: 'n_programs_generated', label: 'Programs' },
  { key: 'n_stage1_passed', label: 'S1 Pass' },
  { key: 'duration_seconds', label: 'Duration' },
  { key: 'timestamp', label: 'Time' },
  { key: '_actions', label: 'Actions', sortable: false },
];

function hasGaps(d) {
  if (d.status === 'running') return false;
  return d.best_loss_ratio == null || d.best_novelty_score == null;
}

export function ExperimentDataTab({ onSelectExperiment, onRerunExperiment, onFillGapsExperiment, onStartExperiment }) {
  const [trends, setTrends] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [statusFilter, setStatusFilter] = useState('all');
  const [typeFilter, setTypeFilter] = useState('all');
  const [outcomeFilter, setOutcomeFilter] = useState('all');
  const [rerunningIds, setRerunningIds] = useState(new Set());
  const [copiedValue, copyText] = useCopyToClipboard();
  const { slowPollTick } = useAriaData();

  useEffect(() => {
    let active = true;
    const fetchData = async () => {
      try {
        const payload = await apiService.getTrends();
        if (!active) return;
        setTrends(Array.isArray(payload?.trends) ? payload.trends : []);
        setError(null);
      } catch (e) {
        if (active) setError('Failed to load experiment data: ' + e.message);
      } finally {
        if (active) setLoading(false);
      }
    };
    fetchData();
    return () => { active = false; };
  }, [slowPollTick]);

  const handleRerun = async (experimentId) => {
    if (!experimentId || !onRerunExperiment) return; 
    setRerunningIds(prev => new Set(prev).add(experimentId));
    try {
      await onRerunExperiment(experimentId);
    } finally {
      setRerunningIds(prev => {
        const next = new Set(prev);
        next.delete(experimentId);
        return next;
      });
    }
  };

  const handleFillGaps = async (experimentId) => {
    if (!experimentId || !onFillGapsExperiment) return;
    setRerunningIds(prev => new Set(prev).add(experimentId));
    try {
      await onFillGapsExperiment(experimentId);
    } finally {
      setRerunningIds(prev => {
        const next = new Set(prev);
        next.delete(experimentId);
        return next;
      });
    }
  };

  const augmented = useMemo(() => {
    if (!trends) return [];
    return trends.map(d => ({ ...d, _score: trendScore(d) }));
  }, [trends]);

  const experimentTypes = useMemo(() => {
    const unique = Array.from(new Set(
      augmented.map((r) => r?.experiment_type).filter((v) => typeof v === 'string' && v.trim().length > 0)
    ));
    unique.sort((a, b) => a.localeCompare(b));
    return unique;
  }, [augmented]);

  const statusTypeOutcomeFiltered = useMemo(() => (
    augmented.filter((row) => {
      if (statusFilter !== 'all' && row.status !== statusFilter) return false;
      if (typeFilter !== 'all' && row.experiment_type !== typeFilter) return false;
      if (outcomeFilter === 'has_s1' && (row.n_stage1_passed || 0) <= 0) return false;
      if (outcomeFilter === 'no_s1' && (row.n_stage1_passed || 0) > 0) return false;
      return true;
    })
  ), [augmented, statusFilter, typeFilter, outcomeFilter]);

  const {
    sortKey, sortDesc, filterQuery, setFilterQuery, sortedRows: sorted, handleSort: _handleSort,
  } = useInteractiveTable({
    rows: statusTypeOutcomeFiltered,
    filterFields: ['experiment_id', 'hypothesis', 'experiment_type', 'status'],
    initialSortKey: '_score',
    initialSortDesc: true,
    storageKey: DATA_SORT_PREFS_KEY,
  });

  const handleSort = (key) => {
    if (key === '_actions') return;
    _handleSort(key);
  };

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

  if (loading) {
    return (
      <div className="card">
        <div className="ux-state ux-state-loading">
          <span className="ux-spinner" />
          <div className="ux-stack">
            <span className="ux-state-title">Loading experiment data</span>
            <span className="ux-state-subtle">Preparing sortable history with tier and score data.</span>
          </div>
        </div>
      </div>
    );
  }
  if (error) return <div className="card"><p style={{ color: 'var(--accent-red)' }}>{error}</p></div>;
  if (!trends || trends.length === 0) {
    return (
      <div className="card">
        <div className="card-title">Experiment Data</div>
        <p className="ux-state ux-state-empty" style={{ fontSize: 13, marginBottom: 10 }}>
          No experiments yet. Run some experiments to populate this table with results.
        </p>
        {onStartExperiment && (
          <button
            className="refresh-btn"
            style={{ fontSize: 12, padding: '5px 14px' }}
            onClick={() => onStartExperiment({
              mode: 'continuous', n_cycles: 5, source: 'data_tab',
              auto_harden: true, preflight_override: true, enforce_preflight: true,
            })}
          >
            Run 5 Continuous Experiments
          </button>
        )}
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-title">Experiment Data</div>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
        Full experiment table with all metrics. Rows with missing key metrics (loss, novelty, throughput) are highlighted — use "Fill gaps" to re-evaluate them.
      </p>
      <div className="table-toolbar">
        <select
          value={statusFilter}
          onChange={(e) => setStatusFilter(e.target.value)}
          style={{ fontSize: 11, padding: '4px 8px', borderRadius: 4, border: '1px solid var(--border)', background: 'var(--bg-tertiary)', color: 'var(--text-primary)' }}
          aria-label="Data filter by status"
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
          aria-label="Data filter by type"
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
          aria-label="Data filter by outcome"
        >
          <option value="all">All outcomes</option>
          <option value="has_s1">Has S1 pass</option>
          <option value="no_s1">No S1 pass</option>
        </select>
        <input
          value={filterQuery}
          onChange={(e) => setFilterQuery(e.target.value)}
          placeholder="Search experiments..."
          className="filter-input"
        />
        <button
          className="refresh-btn"
          style={{ fontSize: 11, padding: '3px 10px' }}
          onClick={clearFilters}
          disabled={!hasActiveFilters}
        >
          Clear filters
        </button>
      </div>
      <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8 }}>
        Showing {sorted.length} of {augmented.length} experiments.
      </div>
      <div style={{ maxHeight: 600, overflowY: 'auto', border: '1px solid var(--border)', borderRadius: 6 }}>
        <table className="data-table" style={{ marginBottom: 0 }}>
          <thead>
            <tr>
              {DATA_COLUMNS.map(col => (
                <th
                  key={col.key}
                  onClick={() => handleSort(col.key)}
                  title={col.tooltip}
                  style={{
                    cursor: col.sortable === false ? 'default' : 'pointer',
                    userSelect: 'none',
                    whiteSpace: 'nowrap',
                    position: 'sticky',
                    top: 0,
                    background: 'var(--bg-secondary)',
                    zIndex: 1,
                  }}
                >
                  {col.label}
                  {col.sortable !== false && <SortIndicator active={sortKey === col.key} desc={sortDesc} />}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sorted.map((d, i) => {
              const gaps = hasGaps(d);
              const isRunning = d.status === 'running';
              const isRerunning = rerunningIds.has(d.experiment_id);
              return (
                <tr
                  key={d.experiment_id || i}
                  style={gaps ? { borderLeft: '2px solid var(--accent-yellow)' } : undefined}
                >
                  <td style={{ fontWeight: 600, color: scoreColor(d._score) }}>
                    <span title={`S1 rate ${(trendScoreBreakdown(d).passRate || 0).toFixed(1)}/35 | Loss ${(trendScoreBreakdown(d).loss || 0).toFixed(1)}/30 | Novelty ${(trendScoreBreakdown(d).novelty || 0).toFixed(1)}/25 | Efficiency ${(trendScoreBreakdown(d).efficiency || 0).toFixed(1)}/10`}>
                      {d._score}
                    </span>
                  </td>
                  <td style={{ fontFamily: 'monospace', fontSize: 12 }}>
                    <button
                      className="refresh-btn"
                      style={{ fontSize: 11, padding: '2px 6px', marginRight: 6 }}
                      onClick={() => onSelectExperiment && d.experiment_id && onSelectExperiment(d.experiment_id)}
                      disabled={!onSelectExperiment || !d.experiment_id}
                      aria-label={`Open experiment ${(d.experiment_id || '').slice(0, 12)}`}
                    >
                      {(d.experiment_id || '').slice(0, 12)}
                    </button>
                    {d.experiment_id && (
                      <button
                        className="refresh-btn"
                        style={{ fontSize: 10, padding: '1px 5px' }}
                        onClick={() => copyText(d.experiment_id)}
                        aria-label={`Copy experiment id ${d.experiment_id}`}
                      >
                        {copiedValue === d.experiment_id ? 'Copied' : 'Copy'}
                      </button>
                    )}
                  </td>
                  <td style={{
                    color: (d.s1_pass_rate || 0) > 0.05 ? 'var(--accent-green)' : 'var(--text-muted)',
                  }}>
                    {d.adjusted_s1_pass_rate != null
                      ? `${(d.adjusted_s1_pass_rate * 100).toFixed(1)}% adj`
                      : d.s1_pass_rate != null
                        ? `${(d.s1_pass_rate * 100).toFixed(1)}%`
                        : 'insufficient data'}
                    {d.s1_pass_rate != null && (
                      <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                        raw {(d.s1_pass_rate * 100).toFixed(1)}% ({d.n_stage1_passed || 0}/{d.n_programs_generated || 0})
                      </div>
                    )}
                  </td>
                  <td>
                    <span style={{
                      color: d.trend_confidence === 'high'
                        ? 'var(--accent-green)'
                        : d.trend_confidence === 'medium'
                          ? 'var(--accent-yellow)'
                          : 'var(--accent-red)',
                      fontWeight: 600,
                      textTransform: 'uppercase',
                      fontSize: 10,
                    }}>
                      {d.trend_confidence || 'low'}
                    </span>
                    {d.s1_confidence_halfwidth != null && (
                      <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                        ±{(d.s1_confidence_halfwidth * 100).toFixed(1)}%
                      </div>
                    )}
                    {d.trend_weight != null && (
                      <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                        weight {(d.trend_weight * 100).toFixed(0)}%
                      </div>
                    )}
                  </td>
                  <td style={{ color: lossColor(d.best_loss_ratio) }}>
                    {metricText(d.best_loss_ratio, 'not computed', (v) => v.toFixed(4))}
                  </td>
                  <td style={{ color: noveltyColor(d.best_novelty_score) }}>
                    {metricText(d.best_novelty_score, 'not computed', (v) => v.toFixed(3))}
                  </td>
                  <td style={{ color: 'var(--text-secondary)' }}>
                    {d.avg_throughput_tok_s != null
                      ? `${Math.round(d.avg_throughput_tok_s).toLocaleString()} tok/s`
                      : '—'}
                  </td>
                  <td style={{ color: 'var(--text-secondary)' }}>
                    {d.avg_routing_token_retention != null
                      ? `${(d.avg_routing_token_retention * 100).toFixed(1)}%`
                      : '—'}
                  </td>
                  <td style={{ color: 'var(--text-secondary)' }}>
                    {d.avg_routing_utilization_entropy != null
                      ? d.avg_routing_utilization_entropy.toFixed(3)
                      : '—'}
                  </td>
                  <td style={{ color: 'var(--text-secondary)' }}>
                    {d.avg_depth_savings_ratio != null
                      ? `${(d.avg_depth_savings_ratio * 100).toFixed(1)}%`
                      : '—'}
                  </td>
                  <td style={{ color: 'var(--text-secondary)' }}>
                    {d.avg_recursion_savings_ratio != null
                      ? `${(d.avg_recursion_savings_ratio * 100).toFixed(1)}%`
                      : '—'}
                  </td>
                  <td>{d.n_programs_generated || 0}</td>
                  <td style={{ color: (d.n_stage1_passed || 0) > 0 ? 'var(--accent-green)' : 'var(--text-muted)' }}>
                    {d.n_stage1_passed || 0}
                  </td>
                  <td>{formatDuration(d.duration_seconds)}</td>
                  <td style={{ fontSize: 12, color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>
                    {formatTime(d.timestamp)}
                  </td>
                  <td>
                    <button
                      className="refresh-btn"
                      style={{
                        fontSize: 10,
                        padding: '2px 8px',
                        whiteSpace: 'nowrap',
                        color: gaps ? 'var(--accent-yellow)' : undefined,
                        borderColor: gaps ? 'var(--accent-yellow)' : undefined,
                      }}
                      onClick={() => (gaps ? handleFillGaps(d.experiment_id) : handleRerun(d.experiment_id))}
                      disabled={isRunning || isRerunning || !d.experiment_id}
                    >
                      {isRerunning ? (gaps ? 'Filling...' : 'Starting...') : gaps ? 'Fill gaps' : 'Rerun'}
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default ExperimentDataTab;
