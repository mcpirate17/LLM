import { apiCall } from "../services/apiService";
import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { scoreColor } from '../utils/format';
import { lossColor, noveltyColor, reliabilityColor } from '../utils/colors';
import { useAriaData } from '../hooks/useAriaData';
import { candidateScore, TIER_COLORS, TIER_LABELS, TIER_ORDER, bestLoss, percentOfReference } from '../utils/scoringEngine';
import {
  ExpandedDetail,
  FingerprintLeaderboardChart,
  ScoreCell,
  StatusBadge,
  SummaryBar,
} from './discoveries/DiscoveryUiBits';
import SortIndicator from './shared/SortIndicator';

const DISCOVERIES_PREFS_KEY = 'aria_discoveries_prefs_v1';
const QUALITY_FLOOR_THRESHOLD = 0.8;

function provenanceBucket(entry) {
  const cohort = String(entry?.result_cohort || '').trim().toLowerCase();
  const experimentType = String(entry?.experiment_type || '').trim().toLowerCase();
  const trustLabel = String(entry?.trust_label || '').trim().toLowerCase();
  const comparabilityLabel = String(entry?.comparability_label || '').trim().toLowerCase();

  if (experimentType === 'exact_graph_replay' || cohort === 'exact_graph_replay') {
    return 'replay';
  }
  if (
    cohort === 'backfill'
    || trustLabel === 'backfill_observation'
    || comparabilityLabel === 'reconstructed_init_variant'
  ) {
    return 'backfill';
  }
  if (['candidate_screening', 'candidate_grade', 'reference'].includes(trustLabel)) {
    return 'trusted';
  }
  return 'untrusted';
}

function discoveryLossDisplay(entry) {
  if (entry?.discovery_loss_ratio != null) return Number(entry.discovery_loss_ratio);
  if (entry?.screening_loss_ratio != null) return Number(entry.screening_loss_ratio);
  if (entry?.loss_ratio != null) return Number(entry.loss_ratio);
  return null;
}

function validationLossDisplay(entry) {
  if (entry?.validation_loss_ratio != null) return Number(entry.validation_loss_ratio);
  const tier = String(entry?.tier || '').toLowerCase();
  if (tier === 'validation' || tier === 'breakthrough') {
    if (entry?.investigation_loss_ratio != null) return Number(entry.investigation_loss_ratio);
  }
  return null;
}

function finitePositiveOrNull(value) {
  if (value == null) return null;
  const num = Number(value);
  if (!Number.isFinite(num) || num <= 0) return null;
  return num;
}

// ── Main Component ─────────────────────────────────────────────────

const COLUMNS = [
  { key: '_score', label: 'Discovery Score', width: 124, title: 'Internal ranking score based on novelty and performance.' },
  { key: 'display_name', label: 'Architecture', width: 240, title: 'Human-readable name or fingerprint of the model topology.' },
  { key: 'architecture_family', label: 'Family', width: 120, title: 'The architectural category (e.g., Attention, SSM, Hybrid).' },
  { key: 'discovery_loss_ratio', label: 'Disc Loss', width: 92, title: 'Loss ratio on random tokens (fast triage).' },
  { key: 'validation_loss_ratio', label: 'Val Loss', width: 92, title: 'Loss ratio on real micro-corpus (true causal performance).' },
  { key: '_best_loss', label: 'Best', width: 84, title: 'The lowest loss ratio achieved by this architecture across all tests.' },
  { key: '_vs_ref', label: 'vs Ref', width: 84, title: 'How this model compares to the GPT-2 baseline (lower % is better).' },
  { key: '_novelty', label: 'Novelty', width: 78, title: 'Measures how unique this model is compared to existing designs.' },
  { key: 'param_efficiency', label: 'P Eff', width: 78, title: 'Parameter efficiency: FLOPs per parameter (higher = parameters are used more efficiently).' },
  { key: 'sample_efficiency', label: 'S Eff', width: 78, title: 'How quickly the model converges to 25% of initial loss (1.0 = instant, 0.0 = never).' },
  { key: 'investigation_robustness', label: 'Robust', width: 82, title: 'Consistency across different training recipes (higher is more stable).' },
  { key: 'robustness_long_ctx_score', label: 'LongCtx', width: 82, title: 'Combined long-context score used in final evaluation.' },
  { key: 'robustness_long_ctx_scaling_score', label: 'LC-Scale', width: 82, title: 'Long-context scaling component score.' },
  { key: 'robustness_long_ctx_assoc_score', label: 'LC-Assoc', width: 82, title: 'Associative retrieval benchmark score.' },
  { key: 'robustness_long_ctx_multi_hop_score', label: 'LC-MHop', width: 82, title: 'Multi-hop retrieval benchmark score.' },
  { key: 'robustness_long_ctx_passkey_score', label: 'LC-Key', width: 82, title: 'Zero-shot passkey retrieval benchmark score.' },
  { key: 'robustness_long_ctx_retrieval_aggregate', label: 'LC-Retr', width: 82, title: 'Aggregate retrieval score across long-context benchmarks.' },
  { key: 'max_viable_seq_len', label: 'MaxLen', width: 86, title: 'Maximum viable sequence length from long-context scaling sweep.' },
  { key: 'jacobian_spectral_norm', label: 'Spectral', width: 82, title: 'Jacobian Spectral Norm: stability of gradient propagation (lower is better).' },
  { key: 'init_sensitivity_std', label: 'InitStd', width: 82, title: 'Sensitivity to weight initialization (lower means more predictable).' },
  { key: 'tier', label: 'Status', width: 128, title: 'Current research phase of this architecture.' },
  { key: '_details', label: 'View', width: 72, title: 'Open the detailed fingerprint side panel for this architecture.' },
  { key: '_compare', label: 'Cmp', width: 72, title: 'Add architecture to side-by-side comparison.' },
  { key: '_designer', label: 'UI', width: 72, title: 'Open architecture in the visual designer.' },
];

const CORE_VISIBLE_COLUMNS = [
  '_score',
  'display_name',
  'architecture_family',
  'discovery_loss_ratio',
  'validation_loss_ratio',
  '_best_loss',
  '_novelty',
  'jacobian_spectral_norm',
  'tier',
  '_details',
  '_compare',
  '_designer',
];

const RESEARCH_VISIBLE_COLUMNS = [
  ...CORE_VISIBLE_COLUMNS,
  '_vs_ref',
  'param_efficiency',
  'sample_efficiency',
  'investigation_robustness',
  'robustness_long_ctx_score',
  'max_viable_seq_len',
  'init_sensitivity_std',
];

function Discoveries({
  onSelectProgram,
  onAddToComparison,
  onRescreen,
  onPromoteScreening,
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
  const { slowPollTick } = useAriaData();

  const isPinnedReferenceRow = useCallback((entry) => (
    Boolean(entry?.is_reference)
    || String(entry?.model_source || '').toLowerCase() === 'reference'
    || Boolean(entry?.reference_name)
  ), []);

  const prefs = (() => {
    try {
      if (typeof window === 'undefined') return {};
      const stored = window.localStorage.getItem(DISCOVERIES_PREFS_KEY);
      return stored ? JSON.parse(stored) : {};
    } catch { return {}; }
  })();

  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTier, setActiveTier] = useState(() =>
    ['all', 'screening', 'investigation', 'validation', 'breakthrough'].includes(prefs?.activeTier) ? prefs.activeTier : 'all'
  );
  const [sortKey, setSortKey] = useState(() => typeof prefs?.sortKey === 'string' ? prefs.sortKey : '_score');
  const [sortDesc, setSortDesc] = useState(() => typeof prefs?.sortDesc === 'boolean' ? prefs.sortDesc : true);
  const [searchQuery, setSearchQuery] = useState(() => typeof prefs?.searchQuery === 'string' ? prefs.searchQuery : '');
  const [debouncedSearchQuery, setDebouncedSearchQuery] = useState(() => typeof prefs?.searchQuery === 'string' ? prefs.searchQuery : '');
  const [expandedRowId, setExpandedRowId] = useState(null);
  const [highlightId, setHighlightId] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [statusDrafts, setStatusDrafts] = useState({});
  const [savingStatusRowId, setSavingStatusRowId] = useState(null);
  const [statusError, setStatusError] = useState(null);
  const [showChart, setShowChart] = useState(true);
  const [showReferences, setShowReferences] = useState(() =>
    typeof prefs?.showReferences === 'boolean' ? prefs.showReferences : true
  );
  const [hideFailed, setHideFailed] = useState(() =>
    typeof prefs?.hideFailed === 'boolean' ? prefs.hideFailed : true
  );
  const [qualityFloorEnabled, setQualityFloorEnabled] = useState(() =>
    typeof prefs?.qualityFloorEnabled === 'boolean' ? prefs.qualityFloorEnabled : true
  );
  const [sourceFilter, setSourceFilter] = useState(() =>
    ['trusted', 'all', 'untrusted', 'backfill', 'replay'].includes(prefs?.sourceFilter)
      ? prefs.sourceFilter
      : (typeof prefs?.trustedOnly === 'boolean' ? (prefs.trustedOnly ? 'trusted' : 'all') : 'trusted')
  );
  const [visibleColumns, setVisibleColumns] = useState(() =>
    {
      const requiredLongCtx = [
        'robustness_long_ctx_score',
        'robustness_long_ctx_scaling_score',
        'robustness_long_ctx_assoc_score',
        'robustness_long_ctx_multi_hop_score',
        'robustness_long_ctx_passkey_score',
        'robustness_long_ctx_retrieval_aggregate',
        'max_viable_seq_len',
      ];
      const validKeys = new Set(COLUMNS.map(c => c.key));
      const saved = Array.isArray(prefs?.visibleColumns)
        ? prefs.visibleColumns.filter((key) => validKeys.has(key))
        : null;

      // Respect saved user choices exactly. Only inject long-context defaults
      // for first-time users with no saved column preferences.
      if (saved && saved.length > 0) {
        return saved;
      }

      const defaults = [...COLUMNS.map(c => c.key)];
      for (const key of requiredLongCtx) {
        if (!defaults.includes(key) && validKeys.has(key)) defaults.push(key);
      }
      return CORE_VISIBLE_COLUMNS.filter((key) => validKeys.has(key));
    }
  );
  const [showColumnPicker, setShowColumnPicker] = useState(false);
  const queuedSet = useMemo(() => new Set(queuedResultIds || []), [queuedResultIds]);
  const highlightRef = useRef(null);
  const visibleTableColumns = useMemo(
    () => COLUMNS.filter((col) => visibleColumns.includes(col.key)),
    [visibleColumns]
  );
  const applyColumnPreset = useCallback((keys) => {
    const valid = new Set(COLUMNS.map((col) => col.key));
    setVisibleColumns(keys.filter((key) => valid.has(key)));
  }, []);

  // Persist preferences
  useEffect(() => {
    try {
      if (typeof window === 'undefined') return;
      window.localStorage.setItem(DISCOVERIES_PREFS_KEY, JSON.stringify({
        activeTier, sortKey, sortDesc, searchQuery, showChart, showReferences,
        qualityFloorEnabled, visibleColumns, hideFailed, sourceFilter,
      }));
    } catch {}
  }, [activeTier, sortKey, sortDesc, searchQuery, showChart, showReferences, qualityFloorEnabled, visibleColumns, hideFailed, sourceFilter]);

  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedSearchQuery(searchQuery);
    }, 250);
    return () => clearTimeout(timer);
  }, [searchQuery]);

  // Handle external highlight
  useEffect(() => {
    if (highlightResultId) {
      setHighlightId(highlightResultId);
      const timer = setTimeout(() => {
        setHighlightId(null);
        onHighlightClear?.();
      }, 3000);
      return () => clearTimeout(timer);
    }
  }, [highlightResultId, onHighlightClear]);

  // Scroll to highlighted row
  useEffect(() => {
    if (highlightId && highlightRef.current) {
      highlightRef.current.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }, [highlightId]);

  const lastDataRef = useRef(null);

  const fetchData = useCallback(async (isBackground = false) => {
    if (!isBackground) {
      setLoading(true);
      setError(null);
    }
    try {
      const limit = sourceFilter === 'trusted' ? '200' : '2500';
      const params = new URLSearchParams({
        sort: 'composite_score',
        limit,
        view: 'ranked',
        trusted_only: sourceFilter === 'trusted' ? '1' : '0',
      });
      if (activeTier !== 'all') params.set('tier', activeTier);
      const q = debouncedSearchQuery.trim();
      if (q) {
        params.set('q', q);
        params.set('scope', 'all');
      }
      const res = await apiCall(`/api/discoveries?${params}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = await res.json();
      // Only update state if data actually changed — prevents scroll reset
      const entries = json?.entries || [];
      const fingerprint = `${entries.length}:${entries[0]?.entry_id || ''}:${entries[entries.length - 1]?.entry_id || ''}:${entries[0]?.composite_score ?? ''}`;
      if (fingerprint !== lastDataRef.current) {
        lastDataRef.current = fingerprint;
        setData(json);
        setLastUpdated(new Date());
      }
      setError(null);
    } catch (e) {
      if (!isBackground) setError('Failed to load discoveries: ' + e.message);
    } finally {
      if (!isBackground) setLoading(false);
    }
  }, [activeTier, debouncedSearchQuery, sourceFilter]);

  useEffect(() => {
    fetchData(slowPollTick > 0);
  }, [fetchData, slowPollTick]);

  const handleSort = (key) => {
    if (key === '_details' || key === '_compare' || key === '_designer') return;
    if (sortKey === key) setSortDesc(!sortDesc);
    else { setSortKey(key); setSortDesc(true); }
  };

  const handleStatusDraftChange = useCallback((rowId, tier) => {
    setStatusDrafts(prev => ({ ...prev, [rowId]: tier }));
  }, []);

  const handleSaveStatus = useCallback(async (entry) => {
    const rowId = entry.entry_id || entry.result_id;
    if (!rowId) return;
    const nextTier = statusDrafts[rowId] || entry.tier;
    if (!nextTier || nextTier === entry.tier) return;

    setSavingStatusRowId(rowId);
    setStatusError(null);
    try {
      const res = await apiCall(`/api/leaderboard/status`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          entry_id: entry.entry_id,
          result_id: entry.result_id,
          tier: nextTier,
        }),
      });
      if (!res.ok) {
        let payload = null;
        try { payload = await res.json(); } catch {}
        throw new Error(payload?.error || `HTTP ${res.status}`);
      }

      setData(prev => {
        if (!prev?.entries) return prev;
        return {
          ...prev,
          entries: prev.entries.map(item => (
            (item.entry_id && item.entry_id === entry.entry_id)
              || (item.result_id && item.result_id === entry.result_id)
              ? { ...item, tier: nextTier }
              : item
          )),
        };
      });
    } catch (e) {
      setStatusError(`Failed to save status: ${e.message}`);
    } finally {
      setSavingStatusRowId(null);
    }
  }, [statusDrafts]);

  const handleDelete = useCallback(async (entryId) => {
    try {
      const res = await apiCall(`/api/leaderboard/${entryId}`, { method: 'DELETE' });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        setStatusError(`Delete failed: ${err.error || res.statusText}`);
        return;
      }
      fetchData();
    } catch (e) {
      setStatusError('Delete failed: ' + e.message);
    }
  }, [fetchData]);

  // Sort & augment discovery entries
  const sorted = useMemo(() => {
    const entries = data?.entries || [];
    const refs = data?.references || [];
    
    const augmented = entries.map(e => {
      const entryBestLoss = bestLoss(e);
      
      // Find best reference in same family or paradigm for comparison
      let vsRef = null;
      if (entryBestLoss != null && !e.is_reference) {
        // 1. Try same family
        let bestRefLoss = null;
        const familyRefs = refs.filter(r => r.architecture_family === e.architecture_family && bestLoss(r) != null);
        if (familyRefs.length > 0) {
          bestRefLoss = Math.min(...familyRefs.map(r => bestLoss(r)));
        } else {
          // 2. Fallback to GPT-2 Small as the "universal" baseline
          const gpt2 = refs.find(r => r.reference_name === 'GPT-2 Small' || r.reference_name === 'GPT-2');
          bestRefLoss = bestLoss(gpt2);
        }
        
        if (bestRefLoss != null) {
          vsRef = percentOfReference(entryBestLoss, bestRefLoss);
        }
      }

      return {
        ...e,
        discovery_loss_ratio: discoveryLossDisplay(e),
        validation_loss_ratio: validationLossDisplay(e),
        // Keep Discoveries score aligned with backend leaderboard composite when present.
        _score: (e.composite_score != null ? Number(e.composite_score) : candidateScore(e, TIER_ORDER)),
        _best_loss: entryBestLoss,
        _vs_ref: vsRef,
        _novelty: e.screening_novelty ?? e.novelty_score ?? null,
      };
    });
    augmented.sort((a, b) => {
      let va, vb;
      if (sortKey === 'tier') {
        va = TIER_ORDER[a.tier] || 0;
        vb = TIER_ORDER[b.tier] || 0;
      } else if (sortKey === 'display_name' || sortKey === 'architecture_family') {
        va = a[sortKey] || '';
        vb = b[sortKey] || '';
        return sortDesc ? vb.localeCompare(va) : va.localeCompare(vb);
      } else {
        va = a[sortKey]; vb = b[sortKey];
      }
      if (va == null && vb == null) return 0;
      if (va == null) return 1;
      if (vb == null) return -1;
      return sortDesc ? vb - va : va - vb;
    });
    return augmented;
  }, [data?.entries, sortKey, sortDesc]);

  const references = useMemo(() => {
    const refs = (data?.references || []).map((e) => ({
      ...e,
      _score: (e.composite_score != null ? Number(e.composite_score) : candidateScore(e, TIER_ORDER)),
      _best_loss: bestLoss(e),
      _novelty: e.screening_novelty ?? e.novelty_score ?? null,
    }));
    refs.sort((a, b) => {
      const aScore = Number(a._score || 0);
      const bScore = Number(b._score || 0);
      return bScore - aScore;
    });
    return refs;
  }, [data?.references]);

  const sourceFiltered = useMemo(() => {
    return sorted.filter((entry) => {
      const bucket = provenanceBucket(entry);
      if (sourceFilter === 'all') return true;
      if (sourceFilter === 'trusted') return bucket === 'trusted';
      if (sourceFilter === 'untrusted') return bucket !== 'trusted';
      return bucket === sourceFilter;
    });
  }, [sorted, sourceFilter]);

  const effectiveQualityFloorEnabled = useMemo(() => {
    if (!qualityFloorEnabled) return false;
    return sourceFilter === 'trusted' || sourceFilter === 'all';
  }, [qualityFloorEnabled, sourceFilter]);

  const failedFiltered = useMemo(() => {
    if (!hideFailed) return sourceFiltered;
    return sourceFiltered.filter(e => {
      if (e.is_reference) return true;
      const tier = String(e.tier || '').toLowerCase();
      // Tier-based failures
      if (tier === 'screened_out' || tier === 'failed' || tier === 'rejected') return false;
      // Explicit flag failures
      if (e.screening_passed === false || e.investigation_passed === false || e.validation_passed === false) return false;
      // Derived failures (mirrors DiscoveryUiBits logic)
      if (tier === 'investigation' && e.investigation_robustness != null && !e.investigation_passed) return false;
      if (tier === 'validation' && e.validation_baseline_ratio != null && !e.validation_passed) return false;
      return true;
    });
  }, [sourceFiltered, hideFailed]);

  const failedHiddenCount = useMemo(() => {
    if (!hideFailed) return 0;
    return Math.max(0, (sourceFiltered?.length || 0) - (failedFiltered?.length || 0));
  }, [hideFailed, sourceFiltered, failedFiltered]);

  const qualityFiltered = useMemo(() => {
    if (!effectiveQualityFloorEnabled) return failedFiltered;
    return failedFiltered.filter((e) => {
      if (e?.is_reference) return true;
      const score = e?.composite_score;
      return score != null && (Number(score) / 100.0) >= QUALITY_FLOOR_THRESHOLD;
    });
  }, [failedFiltered, effectiveQualityFloorEnabled]);

  const qualityHiddenCount = useMemo(() => {
    if (!effectiveQualityFloorEnabled) return 0;
    return Math.max(0, (failedFiltered?.length || 0) - (qualityFiltered?.length || 0));
  }, [effectiveQualityFloorEnabled, failedFiltered, qualityFiltered]);

  const filtered = qualityFiltered;

  const counts = data?.counts || data?.tier_counts || {};
  const summaryCounts = useMemo(() => {
    const base = { ...(counts || {}) };
    const entries = Array.isArray(data?.entries) ? data.entries : [];
    let backfill = 0;
    let replay = 0;
    for (const entry of entries) {
      const bucket = provenanceBucket(entry);
      if (bucket === 'replay') {
        replay += 1;
        continue;
      }
      if (bucket === 'backfill') {
        backfill += 1;
      }
    }
    base.backfill = backfill;
    base.replay = replay;
    return base;
  }, [counts, data?.entries]);
  const tiers = ['all', 'screening', 'investigation', 'validation', 'breakthrough'];
  const hasLoadedData = Boolean(
    data && (Array.isArray(data.entries) || Array.isArray(data.references))
  );

  return (
    <div className="card" style={{ padding: 16 }}>
      <div className="card-title" style={{ marginBottom: 8 }}>
        Discoveries
        <span style={{ fontSize: 12, color: 'var(--text-muted)', marginLeft: 8 }}>
          {lastUpdated ? `Updated ${lastUpdated.toLocaleTimeString()}` : 'Loading...'}
        </span>
      </div>

      {/* Summary bar */}
      <div style={{ display: 'flex', gap: 12, alignItems: 'flex-start', marginBottom: 12 }}>
        <div style={{ flex: 1 }}>
          <SummaryBar tierCounts={summaryCounts} />
        </div>
        <button
          className={`refresh-btn ${showChart ? 'active' : ''}`}
          style={{ padding: '8px 12px', fontSize: 12 }}
          onClick={() => setShowChart(!showChart)}
          title={showChart ? 'Hide performance chart' : 'Show performance chart'}
        >
          {showChart ? 'Hide Chart' : 'Show Chart'}
        </button>
      </div>

      {showChart && filtered.length > 0 && (
        <FingerprintLeaderboardChart entries={filtered} />
      )}

      {/* Tier filter tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 12, flexWrap: 'wrap' }}>
        {tiers.map(tier => {
          const count = tier === 'all'
            ? (counts.all || 0)
            : (counts[tier] || 0);
          return (
            <button
              key={tier}
              onClick={() => setActiveTier(tier)}
              aria-label={`Filter by ${tier === 'all' ? 'all tiers' : `${TIER_LABELS[tier]} tier`}`}
              style={{
                padding: '5px 14px', borderRadius: 4,
                border: `1px solid ${activeTier === tier ? 'var(--accent-blue)' : 'var(--border)'}`,
                background: activeTier === tier ? 'rgba(88, 166, 255, 0.15)' : 'transparent',
                color: activeTier === tier ? 'var(--accent-blue)' : 'var(--text-secondary)',
                cursor: 'pointer', fontSize: 12, fontWeight: activeTier === tier ? 600 : 400,
              }}
            >
              {tier === 'all' ? 'All' : TIER_LABELS[tier]}
              {count > 0 && (
                <span style={{
                  marginLeft: 5, fontSize: 10,
                  color: tier === 'all' ? 'var(--text-muted)' : (TIER_COLORS[tier] || 'var(--text-muted)'),
                }}>
                  ({count})
                </span>
              )}
            </button>
          );
        })}
        <button
          onClick={() => setShowReferences(v => !v)}
          aria-label={showReferences ? 'Hide references' : 'Show references'}
          style={{
            fontSize: 11, padding: '5px 12px', cursor: 'pointer',
            background: showReferences ? 'rgba(188, 140, 255, 0.12)' : 'transparent',
            border: `1px solid ${showReferences ? 'var(--accent-purple)' : 'var(--border)'}`,
            borderRadius: 4, color: showReferences ? 'var(--accent-purple)' : 'var(--text-secondary)',
          }}
        >
          {showReferences ? `Hide references` : `Show references`}
          {Number(counts.references || 0) > 0 && (
            <span style={{ marginLeft: 5, fontSize: 10, color: 'var(--accent-purple)' }}>
              ({counts.references})
            </span>
          )}
        </button>
        <label
          title="Filter by provenance source"
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: 6,
            fontSize: 11,
            padding: '0 10px',
            border: '1px solid var(--border)',
            borderRadius: 4,
            color: 'var(--text-secondary)',
            background: 'transparent',
            minHeight: 28,
          }}
        >
          <span style={{ color: 'var(--text-muted)' }}>Source</span>
          <select
            value={sourceFilter}
            onChange={(e) => setSourceFilter(e.target.value)}
            aria-label="Filter by source provenance"
            style={{
              fontSize: 11,
              border: '1px solid var(--border)',
              borderRadius: 4,
              background: 'var(--bg-secondary)',
              color: 'var(--text-primary)',
              outline: 'none',
              cursor: 'pointer',
              padding: '3px 22px 3px 8px',
              appearance: 'auto',
            }}
          >
            <option value="trusted">Trusted</option>
            <option value="all">All</option>
            <option value="untrusted">Untrusted</option>
            <option value="backfill">Backfill</option>
            <option value="replay">Replay</option>
          </select>
        </label>
        <button
          onClick={() => setHideFailed(v => !v)}
          aria-label={hideFailed ? 'Show failed' : 'Hide failed'}
          style={{
            fontSize: 11, padding: '5px 12px', cursor: 'pointer',
            background: hideFailed ? 'rgba(248, 81, 73, 0.12)' : 'transparent',
            border: `1px solid ${hideFailed ? 'var(--accent-red)' : 'var(--border)'}`,
            borderRadius: 4, color: hideFailed ? 'var(--accent-red)' : 'var(--text-secondary)',
          }}
        >
          {hideFailed ? 'Show failed' : 'Hide failed'}
        </button>
        <button
          onClick={() => setQualityFloorEnabled(v => !v)}
          aria-label={qualityFloorEnabled ? 'Disable quality floor' : 'Enable quality floor'}
          style={{
            fontSize: 11, padding: '5px 12px', cursor: 'pointer',
            background: qualityFloorEnabled ? 'rgba(63, 185, 80, 0.14)' : 'transparent',
            border: `1px solid ${qualityFloorEnabled ? 'var(--accent-green)' : 'var(--border)'}`,
            borderRadius: 4, color: qualityFloorEnabled ? 'var(--accent-green)' : 'var(--text-secondary)',
          }}
          title={`Hide entries with composite score < ${(QUALITY_FLOOR_THRESHOLD * 100).toFixed(0)}`}
        >
          {qualityFloorEnabled ? `Quality floor ≥ ${(QUALITY_FLOOR_THRESHOLD * 100).toFixed(0)}` : 'Show all quality'}
        </button>
        <button
          onClick={() => setShowColumnPicker(!showColumnPicker)}
          style={{
            fontSize: 11, padding: '5px 12px', cursor: 'pointer',
            border: `1px solid ${showColumnPicker ? 'var(--accent-blue)' : 'var(--border)'}`, 
            borderRadius: 4,
            background: showColumnPicker ? 'rgba(88, 166, 255, 0.12)' : 'transparent', 
            color: showColumnPicker ? 'var(--accent-blue)' : 'var(--text-secondary)',
          }}
        >
          Columns
        </button>
        <button
          onClick={() => applyColumnPreset(CORE_VISIBLE_COLUMNS)}
          style={{
            fontSize: 11, padding: '5px 10px', cursor: 'pointer',
            border: '1px solid var(--border)',
            borderRadius: 4,
            background: 'transparent',
            color: 'var(--text-secondary)',
          }}
          title="Show the core discovery columns"
        >
          Core
        </button>
        <button
          onClick={() => applyColumnPreset(RESEARCH_VISIBLE_COLUMNS)}
          style={{
            fontSize: 11, padding: '5px 10px', cursor: 'pointer',
            border: '1px solid var(--border)',
            borderRadius: 4,
            background: 'transparent',
            color: 'var(--text-secondary)',
          }}
          title="Show an expanded research-oriented column set"
        >
          Research
        </button>
        <button
          onClick={() => applyColumnPreset(COLUMNS.map((col) => col.key))}
          style={{
            fontSize: 11, padding: '5px 10px', cursor: 'pointer',
            border: '1px solid var(--border)',
            borderRadius: 4,
            background: 'transparent',
            color: 'var(--text-secondary)',
          }}
          title="Show all available columns"
        >
          All
        </button>
        <button
          onClick={fetchData}
          disabled={loading}
          aria-label="Refresh discoveries"
          style={{ marginLeft: 'auto', fontSize: 11, padding: '5px 12px', cursor: loading ? 'not-allowed' : 'pointer', background: 'transparent', border: '1px solid var(--border)', borderRadius: 4, color: 'var(--text-secondary)', opacity: loading ? 0.6 : 1 }}
        >
          {loading ? 'Refreshing...' : 'Refresh'}
        </button>
      </div>

      {showColumnPicker && (
        <div style={{
          marginBottom: 12, padding: 12, background: 'var(--bg-secondary)', 
          border: '1px solid var(--border)', borderRadius: 6,
          display: 'flex', gap: 12, flexWrap: 'wrap'
        }}>
          {COLUMNS.map(col => (
            <label key={col.key} style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, color: 'var(--text-primary)', cursor: 'pointer' }}>
              <input 
                type="checkbox" 
                checked={visibleColumns.includes(col.key)}
                onChange={(e) => {
                  if (e.target.checked) {
                    setVisibleColumns([...visibleColumns, col.key]);
                  } else {
                    setVisibleColumns(visibleColumns.filter(k => k !== col.key));
                  }
                }}
              />
              {col.label}
            </label>
          ))}
        </div>
      )}

      {/* Search */}
      <div style={{ marginBottom: 12 }}>
        <input
          type="text"
          placeholder="Search by name, family, fingerprint, or ID..."
          value={searchQuery}
          onChange={e => setSearchQuery(e.target.value)}
          aria-label="Search discoveries"
          style={{
            width: '100%', maxWidth: 400, padding: '6px 10px', fontSize: 12,
            border: '1px solid var(--border)', borderRadius: 4,
            background: 'var(--bg-secondary)', color: 'var(--text-primary)',
          }}
        />
        {searchQuery && (
          <span style={{ marginLeft: 8, fontSize: 11, color: 'var(--text-muted)' }}>
            {loading ? 'Searching full DB...' : `${filtered.length} matches`}
          </span>
        )}
        {qualityFloorEnabled && qualityHiddenCount > 0 && (
          <span style={{ marginLeft: 8, fontSize: 11, color: 'var(--accent-yellow)' }}>
            {qualityHiddenCount} low-quality hidden
          </span>
        )}
        {qualityFloorEnabled && !effectiveQualityFloorEnabled && (
          <span style={{ marginLeft: 8, fontSize: 11, color: 'var(--text-muted)' }}>
            quality floor bypassed for {sourceFilter}
          </span>
        )}
        {hideFailed && failedHiddenCount > 0 && (
          <span style={{ marginLeft: 8, fontSize: 11, color: 'var(--accent-red)' }}>
            {failedHiddenCount} failed hidden
          </span>
        )}
      </div>

      {/* Reference Baselines Banner */}
      {showReferences && references.length > 0 && (
        <div style={{
          marginBottom: 14, padding: '10px 14px',
          background: 'rgba(188, 140, 255, 0.06)',
          border: '1px solid rgba(188, 140, 255, 0.25)',
          borderRadius: 6,
        }}>
          <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--accent-purple)', marginBottom: 8 }}>
            Reference Baselines
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 10 }}>
            Baselines stay visible independently of discovery stage filters.
          </div>
          <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
            {references.map(ref => (
              <div key={ref.entry_id || ref.result_id} style={{
                padding: '6px 12px', borderRadius: 5,
                background: 'rgba(188, 140, 255, 0.10)',
                border: '1px solid rgba(188, 140, 255, 0.18)',
                fontSize: 11, lineHeight: 1.5, minWidth: 150,
              }}>
                <div style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
                  {ref.reference_name || ref.display_name || ref.architecture_desc || 'Reference'}
                </div>
                <div style={{ color: 'var(--text-muted)' }}>
                  {ref.architecture_family || '--'}
                  {ref._best_loss != null && <span style={{ marginLeft: 8 }}>Loss: {ref._best_loss.toFixed(4)}</span>}
                </div>
                {ref.param_count != null && (
                  <div style={{ color: 'var(--text-muted)' }}>
                    {(ref.param_count / 1e6).toFixed(1)}M params
                  </div>
                )}
                {ref.result_id && (
                  <div style={{ marginTop: 8, display: 'flex', gap: 6 }}>
                    <button
                      onClick={() => onSelectProgram?.(ref.result_id)}
                      style={{
                        ...actionBtnStyle,
                        borderColor: 'var(--accent-purple)',
                        color: 'var(--accent-purple)',
                        background: 'rgba(188, 140, 255, 0.12)',
                      }}
                      title="Open full reference detail"
                    >
                      Details
                    </button>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {error && <p style={{ color: 'var(--accent-red)', fontSize: 13, marginBottom: 8 }}>{error}</p>}
      {statusError && <p style={{ color: 'var(--accent-red)', fontSize: 12, marginBottom: 8 }}>{statusError}</p>}

      {loading && !hasLoadedData ? (
        <p style={{ color: 'var(--text-muted)' }}>Loading discoveries...</p>
      ) : filtered.length === 0 && !error ? (
        <div style={{ color: 'var(--text-muted)', fontSize: 13, lineHeight: 1.6 }}>
          {searchQuery.trim() ? (
            <p>No discoveries match "{searchQuery}" in the full notebook.</p>
          ) : activeTier === 'all' ? (
            <p>No discoveries yet. Run experiments to generate candidates.</p>
          ) : (
            <p>No entries in {TIER_LABELS[activeTier]} tier yet.</p>
          )}
        </div>
      ) : (
        <div>
          {loading && hasLoadedData && (
            <p style={{ color: 'var(--text-muted)', fontSize: 12, marginBottom: 8 }}>
              Refreshing discoveries...
            </p>
          )}
          <div style={{ overflowX: 'auto', overflowY: 'auto', maxHeight: 'calc(100vh - 280px)' }}>
          <table className="data-table table-wide" style={{ tableLayout: 'fixed' }}>
            <colgroup>
              <col style={{ width: 26 }} />
              <col style={{ width: 44 }} />
              {visibleTableColumns.map((col) => (
                <col key={col.key} style={{ width: col.width ? `${col.width}px` : '104px' }} />
              ))}
            </colgroup>
            <thead style={{ position: 'sticky', top: 0, zIndex: 2, background: 'var(--bg-card, #1a1a2e)' }}>
              <tr style={{ borderBottom: '1px solid var(--border)' }}>
                <th style={{ ...thStyle, width: 26, position: 'sticky', top: 0, background: 'inherit' }} aria-label="Pinned marker" />
                <th style={{ ...thStyle, position: 'sticky', top: 0, background: 'inherit' }}>#</th>
                {visibleTableColumns.map(col => (
                  <th
                    key={col.key}
                    onClick={() => handleSort(col.key)}
                    title={col.title}
                    style={{
                      ...thStyle,
                      position: 'sticky',
                      top: 0,
                      background: 'inherit',
                      width: col.width ? `${col.width}px` : undefined,
                      cursor: (col.key === '_details' || col.key === '_compare' || col.key === '_designer') ? 'default' : 'pointer',
                      userSelect: 'none',
                    }}
                  >
                    {col.label}
                    <SortIndicator active={sortKey === col.key} desc={sortDesc} />
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {filtered.map((entry, i) => {
                const rowId = entry.entry_id || entry.result_id || i;
                const isExpanded = expandedRowId === rowId;
                const isHighlighted = highlightId && entry.result_id === highlightId;
                const isQueued = !!entry.result_id && queuedSet.has(entry.result_id);
                const isPinnedReference = isPinnedReferenceRow(entry);
                const eligibility = eligibilityByResultId?.[entry.result_id] || null;
                const displayName = entry.display_name || entry.architecture_desc || entry.graph_fingerprint?.slice(0, 10) || '--';
                return (
                  <DiscoveryRow
                    key={rowId}
                    entry={entry}
                    i={i}
                    rowId={rowId}
                    isExpanded={isExpanded}
                    isHighlighted={isHighlighted}
                    isQueued={isQueued}
                    isPinnedReference={isPinnedReference}
                    eligibility={eligibility}
                    displayName={displayName}
                    highlightRef={highlightRef}
                    onSelectProgram={onSelectProgram}
                    onAddToComparison={onAddToComparison}
                    tdStyle={tdStyle}
                    COLUMNS={COLUMNS}
                    visibleColumns={visibleTableColumns.map((col) => col.key)}
                    onOpenInDesigner={onOpenInDesigner}
                    setExpandedRowId={setExpandedRowId}
                    actionBtnStyle={actionBtnStyle}
                    handleDelete={handleDelete}
                    onRescreen={onRescreen}
                    onPromoteScreening={onPromoteScreening}
                    onInvestigate={onInvestigate}
                    onValidate={onValidate}
                    onQueueAdd={onQueueAdd}
                    onQueueRemove={onQueueRemove}
                    statusDrafts={statusDrafts}
                    handleStatusDraftChange={handleStatusDraftChange}
                    handleSaveStatus={handleSaveStatus}
                    savingStatusRowId={savingStatusRowId}
                  />
                );
              })}
            </tbody>
          </table>
        </div>
        </div>
      )}
    </div>
  );
}

const thStyle = {
  padding: '6px 12px', textAlign: 'left', fontSize: 11,
  color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', whiteSpace: 'nowrap',
};

const tdStyle = {
  padding: '6px 12px', whiteSpace: 'nowrap',
};

const actionBtnStyle = {
  padding: '4px 8px', fontSize: 10,
  border: '1px solid rgba(88, 166, 255, 0.4)', borderRadius: 4,
  background: 'rgba(88, 166, 255, 0.12)', color: 'var(--accent-blue)', cursor: 'pointer',
};


const DiscoveryRow = React.memo(function DiscoveryRow({
  entry, 
  i, 
  rowId, 
  isExpanded, 
  isHighlighted, 
  isQueued, 
  isPinnedReference, 
  eligibility, 
  displayName,
  highlightRef,
  onSelectProgram,
  onAddToComparison,
  tdStyle,
  COLUMNS,
  visibleColumns,
  onOpenInDesigner,
  setExpandedRowId,
  actionBtnStyle,
  handleDelete,
  onRescreen,
  onPromoteScreening,
  onInvestigate,
  onValidate,
  onQueueAdd,
  onQueueRemove,
  statusDrafts,
  handleStatusDraftChange,
  handleSaveStatus,
  savingStatusRowId
}) {
  const canDelete = !entry.is_reference && (entry.tier === 'screening' || entry.tier === 'failed' || entry.tier === 'rejected' || entry.screening_passed === false || entry.investigation_passed === false || entry.validation_passed === false);

  return (
    <React.Fragment>
      <tr
        ref={isHighlighted ? highlightRef : undefined}
        style={{
          borderBottom: '1px solid var(--border)',
          cursor: 'pointer',
          background: isHighlighted
            ? 'rgba(88, 166, 255, 0.2)'
            : isPinnedReference
              ? 'rgba(188, 140, 255, 0.14)'
              : entry.tier === 'breakthrough' ? 'rgba(63, 185, 80, 0.08)' : undefined,
          animation: isHighlighted ? 'leaderboard-pulse 1.5s ease-in-out 2' : undefined,
        }}
        onClick={() => setExpandedRowId(isExpanded ? null : rowId)}
      >
        <td style={{ ...tdStyle, width: 26, textAlign: 'center', paddingLeft: 4, paddingRight: 4 }}>
          {isPinnedReference ? (
            <span title="Pinned reference" style={{ color: 'var(--accent-purple)', fontSize: 12, fontWeight: 700 }}>
              ★
            </span>
          ) : null}
        </td>
        <td style={tdStyle}>{i + 1}</td>
        {visibleColumns.map((colKey) => {
          const col = COLUMNS.find((item) => item.key === colKey);
          if (!col) return null;
          switch (col.key) {
            case '_score':
              return <td key={col.key} style={tdStyle}><ScoreCell entry={entry} /></td>;
            case 'display_name':
              return (
                <td key={col.key} style={{ ...tdStyle, maxWidth: 260, whiteSpace: 'normal', overflowWrap: 'anywhere', lineHeight: 1.35 }}>
                  <div style={{ fontWeight: 500 }}>{displayName}</div>
                  {entry.graph_fingerprint && (
                    <div style={{ fontSize: 10, color: 'var(--text-muted)', fontFamily: 'monospace', whiteSpace: 'normal', overflowWrap: 'anywhere' }}>
                      {entry.graph_fingerprint}
                    </div>
                  )}
                </td>
              );
            case 'architecture_family':
              return (
                <td key={col.key} style={{ ...tdStyle, whiteSpace: 'normal' }}>
                  <span style={{
                    fontSize: 11, padding: '1px 6px', borderRadius: 3,
                    background: 'var(--bg-tertiary)', color: 'var(--text-secondary)',
                  }}>
                    {entry.architecture_family || '--'}
                  </span>
                </td>
              );
            case 'discovery_loss_ratio':
              const discoveryDisplay = discoveryLossDisplay(entry);
              return (
                <td key={col.key} style={{ ...tdStyle, textAlign: 'right', color: lossColor(discoveryDisplay), fontFamily: 'monospace' }}>
                  {discoveryDisplay != null ? Number(discoveryDisplay).toFixed(4) : '--'}
                </td>
              );
            case 'validation_loss_ratio':
              const validationDisplay = validationLossDisplay(entry);
              return (
                <td key={col.key} style={{ ...tdStyle, textAlign: 'right', color: lossColor(validationDisplay), fontFamily: 'monospace' }}>
                  {validationDisplay != null ? Number(validationDisplay).toFixed(4) : '--'}
                </td>
              );
            case '_best_loss':
              return (
                <td key={col.key} style={{ ...tdStyle, textAlign: 'right', color: lossColor(entry._best_loss), fontFamily: 'monospace' }}>
                  {entry._best_loss != null ? (Number(entry._best_loss) !== 0 && Math.abs(Number(entry._best_loss)) < 0.0001 ? Number(entry._best_loss).toExponential(2) : Number(entry._best_loss).toFixed(4)) : '--'}
                </td>
              );
            case '_vs_ref':
              return (
                <td key={col.key} style={{ ...tdStyle, textAlign: 'right', fontFamily: 'monospace', color: entry._vs_ref != null ? (entry._vs_ref <= 100 ? 'var(--accent-green)' : 'var(--accent-red)') : 'var(--text-muted)' }}>
                  {entry._vs_ref != null ? `${entry._vs_ref.toFixed(1)}%` : '--'}
                </td>
              );
            case '_novelty':
              return (
                <td key={col.key} style={{ ...tdStyle, textAlign: 'right', color: noveltyColor(entry._novelty), fontFamily: 'monospace' }}>
                  {entry._novelty != null ? Number(entry._novelty).toFixed(3) : '--'}
                </td>
              );
            case 'param_efficiency':
              return (
                <td key={col.key} style={{ ...tdStyle, textAlign: 'right', fontFamily: 'monospace' }}>{entry.param_efficiency != null ? Number(entry.param_efficiency).toFixed(3) : '--'}</td>
              );
            case 'sample_efficiency':
              return <td key={col.key} style={{ ...tdStyle, textAlign: 'right', fontFamily: 'monospace' }}>{entry.sample_efficiency != null ? Number(entry.sample_efficiency).toFixed(3) : '--'}</td>;
            case 'investigation_robustness':
              return <td key={col.key} style={{ ...tdStyle, textAlign: 'right', fontFamily: 'monospace' }}>{entry.investigation_robustness != null ? Number(entry.investigation_robustness).toFixed(3) : '--'}</td>;
            case 'robustness_long_ctx_score':
              return <td key={col.key} style={{ ...tdStyle, textAlign: 'right', fontFamily: 'monospace' }}>{entry.robustness_long_ctx_score != null ? Number(entry.robustness_long_ctx_score).toFixed(3) : '--'}</td>;
            case 'robustness_long_ctx_scaling_score':
              return <td key={col.key} style={{ ...tdStyle, textAlign: 'right', fontFamily: 'monospace' }}>{entry.robustness_long_ctx_scaling_score != null ? Number(entry.robustness_long_ctx_scaling_score).toFixed(3) : '--'}</td>;
            case 'robustness_long_ctx_assoc_score':
              return <td key={col.key} style={{ ...tdStyle, textAlign: 'right', fontFamily: 'monospace' }}>{entry.robustness_long_ctx_assoc_score != null ? Number(entry.robustness_long_ctx_assoc_score).toFixed(3) : '--'}</td>;
            case 'robustness_long_ctx_multi_hop_score':
              return <td key={col.key} style={{ ...tdStyle, textAlign: 'right', fontFamily: 'monospace' }}>{entry.robustness_long_ctx_multi_hop_score != null ? Number(entry.robustness_long_ctx_multi_hop_score).toFixed(3) : '--'}</td>;
            case 'robustness_long_ctx_passkey_score':
              return <td key={col.key} style={{ ...tdStyle, textAlign: 'right', fontFamily: 'monospace' }}>{entry.robustness_long_ctx_passkey_score != null ? Number(entry.robustness_long_ctx_passkey_score).toFixed(3) : '--'}</td>;
            case 'robustness_long_ctx_retrieval_aggregate':
              return <td key={col.key} style={{ ...tdStyle, textAlign: 'right', fontFamily: 'monospace' }}>{entry.robustness_long_ctx_retrieval_aggregate != null ? Number(entry.robustness_long_ctx_retrieval_aggregate).toFixed(3) : '--'}</td>;
            case 'max_viable_seq_len':
              return <td key={col.key} style={{ ...tdStyle, textAlign: 'right', fontFamily: 'monospace' }}>{entry.max_viable_seq_len != null ? Number(entry.max_viable_seq_len).toFixed(0) : '--'}</td>;
            case 'jacobian_spectral_norm':
              const specVal = finitePositiveOrNull(entry.jacobian_spectral_norm ?? entry.fp_jacobian_spectral_norm);
              return <td key={col.key} style={{ ...tdStyle, textAlign: 'right', fontFamily: 'monospace' }}>{specVal != null ? Number(specVal).toFixed(4) : '--'}</td>;
            case 'init_sensitivity_std':
              return <td key={col.key} style={{ ...tdStyle, textAlign: 'right', fontFamily: 'monospace' }}>{entry.init_sensitivity_std != null ? Number(entry.init_sensitivity_std).toFixed(4) : '--'}</td>;
            case 'tier':
              return (
                <td key={col.key} style={tdStyle}>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                    <StatusBadge entry={entry} />
                  </div>
                </td>
              );
            case '_details':
              return (
                <td key={col.key} style={tdStyle} onClick={e => e.stopPropagation()}>
                  <button
                    onClick={() => {
                      if (entry.result_id) onSelectProgram?.(entry.result_id);
                    }}
                    style={{
                      ...actionBtnStyle,
                      width: '100%',
                      borderColor: 'var(--accent-blue)',
                      color: 'var(--accent-blue)',
                      background: 'transparent',
                    }}
                    title={entry.result_id ? 'Open detailed fingerprint side panel' : 'Detail view unavailable: missing result ID'}
                  >
                    View
                  </button>
                </td>
              );
            case '_compare':
              return (
                <td key={col.key} style={tdStyle} onClick={e => e.stopPropagation()}>
                  {onAddToComparison ? (
                    <button
                      onClick={() => {
                        if (entry.result_id) onAddToComparison(entry.result_id);
                      }}
                      disabled={!entry.result_id}
                      style={{
                        ...actionBtnStyle,
                        width: '100%',
                        borderColor: 'var(--accent-green)',
                        color: 'var(--accent-green)',
                        opacity: entry.result_id ? 1 : 0.5,
                        cursor: entry.result_id ? 'pointer' : 'not-allowed',
                      }}
                      title={entry.result_id ? 'Add architecture to side-by-side comparison' : 'Comparison unavailable: missing result ID'}
                    >
                      Cmp
                    </button>
                  ) : <span style={{ color: 'var(--text-muted)' }}>--</span>}
                </td>
              );
            case '_designer':
              return (
                <td key={col.key} style={tdStyle} onClick={e => e.stopPropagation()}>
                  {onOpenInDesigner ? (
                    <button
                      onClick={() => {
                        if (entry.result_id) onOpenInDesigner(entry.result_id)
                      }}
                      disabled={!entry.result_id}
                      style={{
                        ...actionBtnStyle,
                        width: '100%',
                        borderColor: 'var(--accent-purple)',
                        color: 'var(--accent-purple)',
                        opacity: entry.result_id ? 1 : 0.5,
                        cursor: entry.result_id ? 'pointer' : 'not-allowed',
                      }}
                      title={entry.result_id ? 'Open architecture in visual designer' : 'Designer unavailable: missing result ID'}
                    >
                      Open
                    </button>
                  ) : <span style={{ color: 'var(--text-muted)' }}>--</span>}
                </td>
              );
            default:
              return null;
          }
        })}
      </tr>
      {isExpanded && (
        <ExpandedDetail
          entry={entry}
          onRescreen={onRescreen}
          onPromoteScreening={onPromoteScreening}
          onInvestigate={onInvestigate}
          onValidate={onValidate}
          onQueueAdd={onQueueAdd}
          onQueueRemove={onQueueRemove}
          onDelete={handleDelete}
          isQueued={isQueued}
          eligibility={eligibility}
          statusDraft={statusDrafts[rowId] || entry.tier}
          onStatusDraftChange={(tier) => handleStatusDraftChange(rowId, tier)}
          onSaveStatus={() => handleSaveStatus(entry)}
          savingStatus={savingStatusRowId === rowId}
          actionBtnStyle={actionBtnStyle}
        />
      )}
    </React.Fragment>
  );
});

export default Discoveries;
