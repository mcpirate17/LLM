import { apiCall } from "../services/apiService";
import React, { useState, useEffect, useCallback, useMemo, useRef, useLayoutEffect } from 'react';
import { scoreColor } from '../utils/format';
import { bestLoss, percentOfReference, TIER_ORDER, TIER_COLORS, TIER_LABELS } from '../utils/scoringEngine';
import { compressionSummary } from './report/reportUtils';

import { LEADERBOARD_PREFS_KEY, COLUMNS } from './leaderboard/leaderboardConfig';
import { candidateEligibility, toRetentionPercent } from './leaderboard/leaderboardUtils';
import LeaderboardRow from './leaderboard/LeaderboardRow';

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

  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTier, setActiveTier] = useState(() => {
    const tier = leaderboardPrefs?.activeTier;
    return ['all', 'screening', 'investigation', 'validation', 'breakthrough'].includes(tier) ? tier : 'all';
  });
  const [sortKey, setSortKey] = useState(() => {
    return typeof leaderboardPrefs?.sortKey === 'string' ? leaderboardPrefs.sortKey : '_score';
  });
  const [sortDesc, setSortDesc] = useState(() => {
    return typeof leaderboardPrefs?.sortDesc === 'boolean' ? leaderboardPrefs.sortDesc : true;
  });
  const [actionError, setActionError] = useState(null);
  const [expandedRowId, setExpandedRowId] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);
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
    const baseline = ['_score', 'tier', 'architecture_family', '_composition', 'composite_score', 'discovery_loss_ratio', 'validation_loss_ratio', 'moe_routing_efficiency', 'arch_quality_score', 'screening_loss_ratio', 'screening_novelty', 'investigation_loss_ratio', 'validation_baseline_ratio', '_actions'];
    const saved = Array.isArray(leaderboardPrefs?.visibleColumns) ? leaderboardPrefs.visibleColumns : baseline;
    return saved;
  });
  const [highlightId, setHighlightId] = useState(null);
  const [showColumnPicker, setShowColumnPicker] = useState(false);
  const [columnWidths, setColumnWidths] = useState(() => {
    try {
      const saved = window.localStorage.getItem('aria_leaderboard_col_widths');
      return saved ? JSON.parse(saved) : {};
    } catch { return {}; }
  });
  const resizingRef = useRef(null);
  const queuedSet = useMemo(() => new Set(queuedResultIds || []), [queuedResultIds]);

  useEffect(() => {
    try {
      window.localStorage.setItem(LEADERBOARD_PREFS_KEY, JSON.stringify({
        activeTier, sortKey, sortDesc, searchQuery, showReferences, onlyRobust, visibleColumns,
      }));
    } catch { /* ignore */ }
  }, [activeTier, sortKey, sortDesc, searchQuery, showReferences, onlyRobust, visibleColumns]);

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

  const lastDataRef = useRef(null);

  const fetchLeaderboard = useCallback(async (isBackground = false) => {
    if (!isBackground) { setLoading(true); setError(null); }
    try {
      const params = new URLSearchParams({ sort: 'composite_score', limit: '100' });
      if (activeTier !== 'all') params.set('tier', activeTier);
      const res = await apiCall(`/api/leaderboard?${params}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = await res.json();
      const jsonStr = JSON.stringify(json);
      if (jsonStr !== lastDataRef.current) {
        lastDataRef.current = jsonStr;
        setData(json);
        setLastUpdated(new Date());
      }
      setError(null);
    } catch (e) {
      if (!isBackground) setError('Failed to load leaderboard: ' + e.message);
    }
    if (!isBackground) setLoading(false);
  }, [activeTier]);

  useEffect(() => {
    fetchLeaderboard(false);
    const interval = setInterval(() => fetchLeaderboard(true), 60000);
    return () => clearInterval(interval);
  }, [fetchLeaderboard]);

  const handleSort = (key) => {
    if (key === '_actions') return;
    if (sortKey === key) setSortDesc(!sortDesc);
    else { setSortKey(key); setSortDesc(true); }
  };

  const handleInvestigate = (resultIds) => {
    if (onInvestigate) { setActionError(null); onInvestigate(resultIds); } 
    else {
      apiCall(`/api/experiments/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: 'investigation', result_ids: resultIds }),
      }).then(() => fetchLeaderboard()).catch(e => setActionError('Failed: ' + e.message));
    }
  };

  const handleValidate = (resultIds) => {
    if (onValidate) { setActionError(null); onValidate(resultIds); }
    else {
      apiCall(`/api/experiments/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: 'validation', result_ids: resultIds }),
      }).then(() => fetchLeaderboard()).catch(e => setActionError('Failed: ' + e.message));
    }
  };

  const togglePin = async (entryId, currentPinned) => {
    try {
      const res = await apiCall(`/api/leaderboard/pin`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ entry_id: entryId, pinned: !currentPinned }),
      });
      if (res.ok) fetchLeaderboard();
    } catch (e) { console.error(e); }
  };

  const handleDelete = async (entryId) => {
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
  };

  const rawEntries = data?.entries || [];
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
    if (onlyRobust) entries = entries.filter(e => (e.robustness_noise_score || 1) < 0.3 && (e._quant_retention_pct || 0) > 80);
    if (!searchQuery.trim()) return entries;
    const q = searchQuery.toLowerCase();
    return entries.filter(e => 
      (e.result_id?.toLowerCase().includes(q)) || 
      (e.architecture_desc?.toLowerCase().includes(q)) ||
      (e.architecture_family?.toLowerCase().includes(q))
    );
  }, [sorted, showReferences, onlyRobust, searchQuery]);

  const highlightRef = useRef(null);
  useEffect(() => {
    if (highlightId && highlightRef.current) highlightRef.current.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }, [highlightId, filtered]);

  // Persist column widths
  useEffect(() => {
    try {
      window.localStorage.setItem('aria_leaderboard_col_widths', JSON.stringify(columnWidths));
    } catch { /* ignore */ }
  }, [columnWidths]);

  const onResizeStart = useCallback((e, colKey) => {
    e.preventDefault();
    e.stopPropagation();
    const startX = e.clientX;
    const th = e.target.parentElement;
    const startWidth = th.offsetWidth;
    resizingRef.current = colKey;

    const onMouseMove = (moveE) => {
      const diff = moveE.clientX - startX;
      const newWidth = Math.max(40, startWidth + diff);
      setColumnWidths(prev => ({ ...prev, [colKey]: newWidth }));
    };
    const onMouseUp = () => {
      resizingRef.current = null;
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseup', onMouseUp);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };
    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('mouseup', onMouseUp);
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
  }, []);

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
        Ranked candidates with tiered evidence. Click a row to open details.
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
                    {sortKey === c.key && c.key !== '_actions' && (
                      <span className="th-sort-icon" aria-hidden="true">
                        {sortDesc ? '\u25BC' : '\u25B2'}
                      </span>
                    )}
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
    </div>
  );
}

export default Leaderboard;
