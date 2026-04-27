import { apiCall, postJson } from "../services/apiService";
import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { SCORE_MAX, scoreColor } from '../utils/format';
import { bestLoss, percentOfReference, TIER_ORDER, TIER_COLORS, TIER_LABELS } from '../utils/scoringEngine';
import { compressionSummary } from './report/reportUtils';
import { useAriaData } from '../hooks/useAriaData';
import { LEADERBOARD_PREFS_KEY, COLUMNS } from './leaderboard/leaderboardConfig';
import { candidateEligibility, toRetentionPercent } from './leaderboard/leaderboardUtils';
import { capabilityQualityRank, capabilityQualityStatus } from '../utils/discoveryStatus';
import LeaderboardRow from './leaderboard/LeaderboardRow';
import RerunAutoModal from './leaderboard/RerunAutoModal';
import SortIndicator from './shared/SortIndicator';
import useResizableColumns from './shared/useResizableColumns';

const thStyle = {
  padding: '6px 8px',
  textAlign: 'left',
  fontSize: 11,
  color: 'var(--text-muted)',
  fontWeight: 600,
  textTransform: 'uppercase',
  whiteSpace: 'nowrap',
};

/**
 * Leaderboard — Technical ranking of all candidate architectures.
 * Features tiered filtering, client-side sorting, and detailed drill-downs.
 */
function Leaderboard({
  onSelectProgram,
  onInvestigate,
  onValidate,
  onConfirm,
  highlightResultId,
  onHighlightClear,
  onQueueAdd,
  onQueueRemove,
  queuedResultIds,
  eligibilityByResultId,
  onOpenInDesigner,
}) {
  const leaderboardPrefs = (() => {
    try {
      if (typeof window === 'undefined') return {};
      const stored = window.localStorage.getItem(LEADERBOARD_PREFS_KEY);
      return stored ? JSON.parse(stored) : {};
    } catch {
      return {};
    }
  })();

  const {
    leaderboardEntries: rawEntries,
    loading,
    error,
    lastUpdated,
    refreshSharedData: fetchLeaderboard,
  } = useAriaData() || {};

  const [activeTier, setActiveTier] = useState(() => {
    const tier = leaderboardPrefs?.activeTier;
    return ['all', 'screening', 'investigation', 'validation', 'breakthrough'].includes(tier) ? tier : 'all';
  });
  const [sortKey, setSortKey] = useState(() => {
    return typeof leaderboardPrefs?.sortKey === 'string' ? leaderboardPrefs.sortKey : 'composite_score';
  });
  const [sortDesc, setSortDesc] = useState(() => {
    return typeof leaderboardPrefs?.sortDesc === 'boolean' ? leaderboardPrefs.sortDesc : true;
  });
  const [actionError, setActionError] = useState(null);
  const [expandedRowId, setExpandedRowId] = useState(null);
  const [searchQuery, setSearchQuery] = useState(() => {
    return typeof leaderboardPrefs?.searchQuery === 'string' ? leaderboardPrefs.searchQuery : '';
  });
  const [showReferences, setShowReferences] = useState(() => {
    return typeof leaderboardPrefs?.showReferences === 'boolean' ? leaderboardPrefs.showReferences : true;
  });
  const [onlyRobust, setOnlyRobust] = useState(() => {
    return typeof leaderboardPrefs?.onlyRobust === 'boolean' ? leaderboardPrefs.onlyRobust : false;
  });
  const [visibleColumns, setVisibleColumns] = useState(() => {
    const baseline = ['_score', 'tier', '_verified', '_rate', '_gap', 'architecture_family', '_composition', 'composite_score', 'screening_loss_ratio', 'validation_loss_ratio', 'induction_v2_investigation_auc', 'binding_v2_investigation_auc', 'hellaswag_acc', 'blimp_overall_accuracy', 'ar_auc', 'fp_jacobian_erf_density', 'fp_id_collapse_rate', 'fp_jacobian_erf_decay_slope', '_actions'];
    const saved = Array.isArray(leaderboardPrefs?.visibleColumns) ? [...leaderboardPrefs.visibleColumns] : baseline;
    for (const key of ['induction_v2_investigation_auc', 'binding_v2_investigation_auc', 'hellaswag_acc', 'blimp_overall_accuracy', 'ar_auc', 'fp_jacobian_erf_density', 'fp_id_collapse_rate', 'fp_jacobian_erf_decay_slope']) {
      const actionIndex = saved.indexOf('_actions');
      if (!saved.includes(key)) saved.splice(actionIndex >= 0 ? actionIndex : saved.length, 0, key);
    }
    return saved;
  });
  const [capabilityFilter, setCapabilityFilter] = useState(() => {
    const value = leaderboardPrefs?.capabilityFilter;
    return ['all', 'qualified', 'training_only', 'pending'].includes(value) ? value : 'all';
  });
  const [highlightId, setHighlightId] = useState(null);
  const [showColumnPicker, setShowColumnPicker] = useState(false);
  const [showAutoRerunModal, setShowAutoRerunModal] = useState(false);
  const { columnWidths, onResizeStart } = useResizableColumns('aria_leaderboard_col_widths');
  const queuedSet = useMemo(() => new Set(queuedResultIds || []), [queuedResultIds]);

  useEffect(() => {
    try {
      window.localStorage.setItem(LEADERBOARD_PREFS_KEY, JSON.stringify({
        activeTier, sortKey, sortDesc, searchQuery, showReferences, onlyRobust, visibleColumns, capabilityFilter,
      }));
    } catch { /* ignore */ }
  }, [activeTier, sortKey, sortDesc, searchQuery, showReferences, onlyRobust, visibleColumns, capabilityFilter]);

  useEffect(() => {
    if (highlightResultId) {
      setHighlightId(highlightResultId);
      const timer = setTimeout(() => {
        setHighlightId(null);
        if (onHighlightClear) onHighlightClear();
      }, 3000);
      return () => clearTimeout(timer);
    }
  }, [highlightResultId, onHighlightClear]);

  const handleSort = useCallback((key) => {
    if (key === '_actions') return;
    setSortKey(prev => {
      if (prev === key) { setSortDesc(d => !d); return prev; }
      setSortDesc(true);
      return key;
    });
  }, []);

  const handleInvestigate = useCallback((resultIds) => {
    if (onInvestigate) { setActionError(null); onInvestigate(resultIds); }
    else {
      postJson('/api/experiments/start', { mode: 'investigation', result_ids: resultIds })
        .then(() => fetchLeaderboard())
        .catch((e) => setActionError('Failed: ' + e.message));
    }
  }, [onInvestigate, fetchLeaderboard]);

  const handleValidate = useCallback((resultIds) => {
    if (onValidate) { setActionError(null); onValidate(resultIds); }
    else {
      postJson('/api/experiments/start', { mode: 'validation', result_ids: resultIds })
        .then(() => fetchLeaderboard())
        .catch((e) => setActionError('Failed: ' + e.message));
    }
  }, [onValidate, fetchLeaderboard]);

  const togglePin = useCallback(async (entryId, currentPinned) => {
    try {
      const res = await postJson('/api/leaderboard/pin', { entry_id: entryId, pinned: !currentPinned });
      if (res.ok) fetchLeaderboard();
    } catch (e) { console.error(e); }
  }, [fetchLeaderboard]);

  const handleDelete = useCallback(async (entryId) => {
    try {
      const res = await apiCall(`/api/leaderboard/${entryId}`, { method: 'DELETE' });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        setActionError(`Delete failed: ${err.error || res.statusText}`);
        return;
      }
      fetchLeaderboard();
    } catch (e) {
      setActionError('Delete failed: ' + e.message);
    }
  }, [fetchLeaderboard]);

  const referenceEntries = useMemo(() => rawEntries.filter(e => e.is_reference), [rawEntries]);
  const referenceByFamily = useMemo(() => {
    const mapping = {};
    for (const ref of referenceEntries) {
      const family = String(ref?.architecture_family || '').trim();
      if (family && !mapping[family]) mapping[family] = ref;
    }
    return mapping;
  }, [referenceEntries]);
  const primaryReference = referenceEntries[0] || null;

  const tierCounts = {};
  for (const entry of rawEntries) {
    const t = entry.tier || 'screening';
    tierCounts[t] = (tierCounts[t] || 0) + 1;
  }

  const sorted = useMemo(() => {
    const augmented = rawEntries.map(e => {
      const compression = compressionSummary(e);
      const matchedRef = referenceByFamily[e?.architecture_family] || primaryReference;
      return {
        ...e,
        _score: e._score ?? 0, // Fallback if not in API yet
        _compression_ratio: compression.ratio,
        _compression_summary: compression,
        _vs_reference: e.is_reference ? null : percentOfReference(bestLoss(e), bestLoss(matchedRef)),
        _matched_reference: matchedRef?.reference_name || matchedRef?.architecture_desc || null,
        _quant_retention_pct: toRetentionPercent(e?.quant_int8_retention),
      };
    });
    augmented.sort((a, b) => {
      if (Boolean(a.is_pinned) !== Boolean(b.is_pinned)) return b.is_pinned ? 1 : -1;
      if (Boolean(a.is_reference) !== Boolean(b.is_reference)) return b.is_reference ? 1 : -1;
      let va = a[sortKey], vb = b[sortKey];
      if (sortKey === 'tier') { va = TIER_ORDER[a.tier] || 0; vb = TIER_ORDER[b.tier] || 0; }
      if (sortKey === '_capability_quality') { va = capabilityQualityRank(a); vb = capabilityQualityRank(b); }
      if (va == null && vb == null) return 0;
      if (va == null) return 1;
      if (vb == null) return -1;
      return sortDesc ? (vb > va ? 1 : -1) : (va > vb ? 1 : -1);
    });
    return augmented;
  }, [rawEntries, sortKey, sortDesc, referenceByFamily, primaryReference]);

  const filtered = useMemo(() => {
    let entries = sorted;
    if (!showReferences) entries = entries.filter(e => !e.is_reference);
    if (capabilityFilter !== 'all') {
      entries = entries.filter((e) => {
        const status = capabilityQualityStatus(e);
        if (capabilityFilter === 'qualified') return status === 'qualified' || status === 'breakthrough';
        if (capabilityFilter === 'training_only') return status === 'training_only';
        if (capabilityFilter === 'pending') return status === 'pending';
        return true;
      });
    }
    if (onlyRobust) entries = entries.filter(e => (e.robustness_noise_score || 1) < 0.3 && (e._quant_retention_pct || 0) > 80);
    if (!searchQuery.trim()) return entries;
    const q = searchQuery.toLowerCase();
    return entries.filter(e => 
      (e.result_id?.toLowerCase().includes(q)) || 
      (e.architecture_desc?.toLowerCase().includes(q)) ||
      (e.architecture_family?.toLowerCase().includes(q))
    );
  }, [sorted, showReferences, capabilityFilter, onlyRobust, searchQuery]);

  const highlightRef = useRef(null);
  useEffect(() => {
    if (highlightId && highlightRef.current) highlightRef.current.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }, [highlightId, filtered]);

  // Derive sort direction label for aria-sort
  const ariaSortAttr = (colKey) => {
    if (sortKey !== colKey) return 'none';
    return sortDesc ? 'descending' : 'ascending';
  };

  return (
    <div className="card" style={{ padding: 16 }}>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 10, marginBottom: 4 }}>
        <div className="card-title" style={{ marginBottom: 0 }}>Qualified Models</div>
        <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>
          {rawEntries.length} total
          {lastUpdated ? ` · updated ${lastUpdated.toLocaleTimeString()}` : ''}
        </span>
      </div>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 14 }}>
        Canonical v10 score scale: colors use fixed 15/30/45/60/75/90% bands of the{' '}
        {SCORE_MAX}-point rubric ceiling.
      </p>

      {/* Tier filter tabs */}
      <div style={{ display: 'flex', gap: 6, marginBottom: 12, flexWrap: 'wrap', alignItems: 'center' }}>
        {['all', 'screening', 'investigation', 'validation', 'breakthrough'].map(t => (
          <button
            key={t}
            onClick={() => setActiveTier(t)}
            className={`step-btn ${activeTier === t ? 'active' : ''}`}
            aria-pressed={activeTier === t}
          >
            {t === 'all' ? 'All' : TIER_LABELS[t]}
            {tierCounts[t] ? <span style={{ marginLeft: 4, opacity: 0.75 }}>({tierCounts[t]})</span> : null}
          </button>
        ))}
        <button
          onClick={() => setShowColumnPicker(!showColumnPicker)}
          className="refresh-btn"
          style={{ fontSize: 11, marginLeft: 'auto' }}
          aria-expanded={showColumnPicker}
        >
          Columns
        </button>
        <button onClick={fetchLeaderboard} className="refresh-btn" style={{ fontSize: 11 }}>
          Refresh
        </button>
        <button
          onClick={() => setShowAutoRerunModal(true)}
          className="refresh-btn"
          style={{ fontSize: 11, borderColor: 'rgba(88, 166, 255, 0.45)', color: 'var(--accent-blue)' }}
          title="Preview & queue score-stability reruns for fingerprints in striking distance of the top-N boundary"
        >
          Auto Reruns…
        </button>
        <label style={{ display: 'inline-flex', alignItems: 'center', gap: 6, fontSize: 11, color: 'var(--text-secondary)' }}>
          <span style={{ color: 'var(--text-muted)' }}>Capability</span>
          <select
            value={capabilityFilter}
            onChange={(e) => setCapabilityFilter(e.target.value)}
            style={{ fontSize: 11, border: '1px solid var(--border)', borderRadius: 4, background: 'var(--bg-secondary)', color: 'var(--text-primary)' }}
          >
            <option value="all">All</option>
            <option value="qualified">Capability-Qualified</option>
            <option value="training_only">Training-Only</option>
            <option value="pending">Validation Pending</option>
          </select>
        </label>
      </div>

      {/* Search */}
      <div style={{ marginBottom: 14 }}>
        <input
          type="search"
          placeholder="Search by ID, family, or description..."
          value={searchQuery}
          onChange={e => setSearchQuery(e.target.value)}
          aria-label="Search models"
          style={{
            width: '100%',
            maxWidth: 420,
            padding: '7px 10px',
            fontSize: 12,
            border: '1px solid var(--border)',
            borderRadius: 6,
            background: 'var(--bg-secondary)',
            color: 'var(--text-primary)',
          }}
        />
      </div>

      {/* Action error */}
      {actionError && (
        <div className="error-banner" style={{ marginBottom: 12 }}>{actionError}</div>
      )}

      {/* Loading state */}
      {loading && (
        <div className="ux-state ux-state-loading" style={{ marginBottom: 12 }}>
          <span className="ux-spinner" />
          <div className="ux-stack">
            <span className="ux-state-title">Loading leaderboard</span>
            <span className="ux-state-subtle">Fetching ranked candidates from the server...</span>
          </div>
        </div>
      )}

      {/* Error state */}
      {error && !loading && (
        <div className="ux-state ux-state-error" style={{ marginBottom: 12 }}>
          <span style={{ fontSize: 18 }}>!</span>
          <div className="ux-stack">
            <span className="ux-state-title">Failed to load leaderboard</span>
            <span className="ux-state-subtle">{error}</span>
          </div>
        </div>
      )}

      {/* Table */}
      {!loading && !error && (
        <div style={{ overflow: 'auto', maxHeight: 'calc(100vh - 300px)' }}>
          <table className="data-table table-wide" role="grid" aria-label="Model leaderboard" style={{ tableLayout: Object.keys(columnWidths).length > 0 ? 'fixed' : 'auto' }}>
            <thead style={{ position: 'sticky', top: 0, zIndex: 2, background: 'var(--bg-secondary)' }}>
              <tr>
                <th style={thStyle} scope="col">#</th>
                {COLUMNS.filter(c => visibleColumns.includes(c.key)).map(c => (
                  <th
                    key={c.key}
                    style={{
                      ...thStyle,
                      position: 'relative',
                      ...(columnWidths[c.key] ? { width: columnWidths[c.key], minWidth: columnWidths[c.key] } : {}),
                    }}
                    scope="col"
                    className={c.key !== '_actions' ? 'th-sortable' : undefined}
                    onClick={() => handleSort(c.key)}
                    aria-sort={ariaSortAttr(c.key)}
                  >
                    {c.label}
                    {c.key !== '_actions' && <SortIndicator active={sortKey === c.key} desc={sortDesc} />}
                    <span
                      onMouseDown={(e) => onResizeStart(e, c.key)}
                      style={{
                        position: 'absolute',
                        right: 0,
                        top: 0,
                        bottom: 0,
                        width: 5,
                        cursor: 'col-resize',
                        zIndex: 3,
                      }}
                    />
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {filtered.length === 0 ? (
                <tr>
                  <td colSpan={visibleColumns.length + 1}>
                    <div className="empty-state">
                      <div className="empty-state-icon">&#x2205;</div>
                      <div className="empty-state-title">No models match your filters</div>
                      <p className="empty-state-hint">
                        {searchQuery
                          ? `No results for "${searchQuery}". Try a different search term or clear the filter.`
                          : 'No candidates have reached this tier yet. Run an experiment to populate the leaderboard.'}
                      </p>
                    </div>
                  </td>
                </tr>
              ) : (
                filtered.map((entry, idx) => (
                  <LeaderboardRow
                    key={entry.result_id || idx}
                    entry={entry}
                    index={idx}
                    visibleColumns={visibleColumns}
                    isHighlighted={highlightId === entry.result_id}
                    highlightRef={highlightId === entry.result_id ? highlightRef : null}
                    isExpanded={expandedRowId === (entry.entry_id || entry.result_id)}
                    onSelect={onSelectProgram}
                    onTogglePin={togglePin}
                    onToggleExpand={(id) => setExpandedRowId(expandedRowId === id ? null : id)}
                    onInvestigate={handleInvestigate}
                    onValidate={handleValidate}
                    onConfirm={onConfirm}
                    onDelete={handleDelete}
                    onOpenInDesigner={onOpenInDesigner}
                    onQueueAdd={onQueueAdd}
                    onQueueRemove={onQueueRemove}
                    eligibilityFromParent={eligibilityByResultId?.[entry.result_id]}
                  />
                ))
              )}
            </tbody>
          </table>
        </div>
      )}
      <RerunAutoModal
        open={showAutoRerunModal}
        onClose={() => setShowAutoRerunModal(false)}
      />
    </div>
  );
}

export default Leaderboard;
