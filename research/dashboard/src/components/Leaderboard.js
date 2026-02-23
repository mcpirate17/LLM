import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { scoreColor } from '../utils/format';
import { reliabilityColor } from '../utils/colors';
import { qkvUsageDescriptor, detectQkvFree } from '../utils/architecture';
import { candidateScore, candidateScoreBreakdown, promotionEvidence } from '../utils/scoringEngine';

const API_BASE = process.env.REACT_APP_API_URL || '';
const LEADERBOARD_PREFS_KEY = 'aria_leaderboard_prefs_v1';

const TIER_COLORS = {
  screening: 'var(--accent-blue)',
  investigation: 'var(--accent-yellow)',
  validation: 'var(--accent-purple)',
  breakthrough: 'var(--accent-green)',
};

const TIER_LABELS = {
  screening: 'Screening',
  investigation: 'Investigation',
  validation: 'Validation',
  breakthrough: 'Breakthrough',
};

const TIER_ORDER = { breakthrough: 4, validation: 3, investigation: 2, screening: 1 };

const COMPRESSION_FACTORS = {
  low_rank: 0.55,
  shared_basis: 0.5,
  hash_trick: 0.35,
  structured_sparse: 0.4,
  kronecker: 0.5,
  polynomial: 0.6,
  residual_quantized: 0.3,
  compressed_attention: 0.7,
};


function parseArchSpec(value) {
  if (!value || typeof value !== 'string') return null;
  try {
    const parsed = JSON.parse(value);
    return parsed && typeof parsed === 'object' ? parsed : null;
  } catch {
    return null;
  }
}

function compressionSummary(entry) {
  const spec = parseArchSpec(entry.arch_spec_json);
  const compressionKey = spec?.choices?.weight_storage || spec?.choices?.token_representation;
  const factor = COMPRESSION_FACTORS[compressionKey] || 1.0;
  const rawParams = entry.param_count || entry.graph_n_params_estimate || null;
  const compressedParams = rawParams != null ? Math.max(1, Math.round(rawParams * factor)) : null;
  const ratio = rawParams != null && compressedParams != null
    ? Math.max(0.01, Math.min(1.0, compressedParams / rawParams))
    : null;
  const memoryMb = compressedParams != null
    ? (compressedParams * 4) / (1024 * 1024)
    : null;
  const qualityRetention = entry.validation_baseline_ratio != null
    ? Math.max(0, Math.min(1, 1.25 - entry.validation_baseline_ratio))
    : entry.investigation_loss_ratio != null
      ? Math.max(0, Math.min(1, 1.1 - entry.investigation_loss_ratio))
      : entry.screening_loss_ratio != null
        ? Math.max(0, Math.min(1, 1.0 - entry.screening_loss_ratio))
        : null;
  return {
    label: compressionKey || 'dense',
    ratio,
    memoryMb,
    qualityRetention,
  };
}

function metricChips(entry) {
  const chips = [];
  chips.push({
    label: 'Loss',
    source: 'measured',
    reliability: entry.validation_loss_ratio != null ? 'high' : entry.investigation_loss_ratio != null ? 'medium' : 'low',
  });
  chips.push({
    label: 'Novelty',
    source: entry.cka_source === 'artifact' ? 'artifact-backed' : 'heuristic',
    reliability: entry.novelty_confidence != null
      ? (entry.novelty_confidence >= 0.7 ? 'high' : entry.novelty_confidence >= 0.4 ? 'medium' : 'low')
      : 'low',
  });
  chips.push({
    label: 'Baseline',
    source: entry.validation_baseline_ratio != null ? 'baseline-run' : 'not-available',
    reliability: entry.validation_multi_seed_std != null
      ? (entry.validation_multi_seed_std <= 0.12 ? 'high' : 'medium')
      : 'low',
  });
  if (entry.routing_confidence_mean != null) {
    chips.push({
      label: 'Routing',
      source: 'telemetry',
      reliability: entry.routing_confidence_mean >= 0.7 ? 'high' : entry.routing_confidence_mean >= 0.4 ? 'medium' : 'low',
    });
  }
  return chips;
}

function qualityFlags(entry) {
  const flags = [];
  if (entry.cka_source === 'artifact') {
    flags.push({ label: 'CKA artifact-backed', tone: 'high' });
  } else {
    flags.push({ label: 'CKA fallback heuristic', tone: 'low' });
  }
  if (entry.validation_baseline_ratio != null) {
    flags.push({ label: 'Baseline measured', tone: 'medium' });
  } else {
    flags.push({ label: 'Baseline unavailable', tone: 'low' });
  }
  if (entry.routing_confidence_mean != null) {
    flags.push({ label: 'Routing telemetry', tone: 'medium' });
  }
  const qkv = qkvUsageDescriptor(entry);
  flags.push({ label: qkv.label, tone: qkv.tone, detail: qkv.detail });
  return flags;
}

function decisionGate(entry) {
  const checks = {
    screeningEvidence: entry.screening_loss_ratio != null && entry.screening_novelty != null,
    investigationEvidence: entry.investigation_loss_ratio != null && entry.investigation_robustness != null,
    robustnessFloor: entry.investigation_robustness != null && entry.investigation_robustness >= 0.5,
    validationEvidence: entry.validation_loss_ratio != null
      && entry.validation_baseline_ratio != null
      && entry.validation_multi_seed_std != null,
    baselineBeatsReference: entry.validation_baseline_ratio != null && entry.validation_baseline_ratio < 1.0,
    consistencyBounded: entry.validation_multi_seed_std != null && entry.validation_multi_seed_std <= 0.12,
  };
  const decisionReady = Object.values(checks).every(Boolean);
  const missing = Object.entries(checks)
    .filter(([, ok]) => !ok)
    .map(([name]) => name);
  return {
    decisionReady,
    label: decisionReady ? 'Decision-Ready' : 'Exploratory',
    color: decisionReady ? 'var(--accent-green)' : 'var(--accent-yellow)',
    missing,
    checks,
  };
}

function candidateEligibility(entry) {
  const tier = typeof entry?.tier === 'string' ? entry.tier.toLowerCase() : '';
  const hasInvestigationEvidence = entry?.investigation_loss_ratio != null;
  const hasValidationEvidence = entry?.validation_loss_ratio != null || Boolean(entry?.validation_passed);

  const investigationEligible = tier === 'screening' && !hasInvestigationEvidence;
  const validationEligible = tier === 'investigation' && Boolean(entry?.investigation_passed) && !hasValidationEvidence;

  let queueReason = null;
  if (!investigationEligible && !validationEligible) {
    if (tier === 'screening' && hasInvestigationEvidence) {
      queueReason = 'already_investigated_unchanged';
    } else if (tier === 'investigation' && !entry?.investigation_passed) {
      queueReason = 'not_investigation_passed';
    } else if (tier === 'validation' || tier === 'breakthrough') {
      queueReason = 'already_promoted';
    } else {
      queueReason = 'not_progression_eligible';
    }
  }

  return {
    investigationEligible,
    validationEligible,
    queueEligible: investigationEligible || validationEligible,
    queueReason,
  };
}

function reproducibilityPacketStatus(entry) {
  const spec = parseArchSpec(entry?.arch_spec_json);
  const checks = [
    { label: 'result_id', ok: !!entry?.result_id },
    { label: 'graph_fingerprint', ok: !!entry?.graph_fingerprint },
    { label: 'arch_spec', ok: !!spec },
    { label: 'baseline_ratio', ok: entry?.validation_baseline_ratio != null },
    { label: 'multi_seed_std', ok: entry?.validation_multi_seed_std != null },
    { label: 'cka_artifact', ok: entry?.cka_source === 'artifact' },
  ];
  const readyCount = checks.filter(check => check.ok).length;
  const totalChecks = checks.length;
  const label = readyCount === totalChecks ? 'Ready' : readyCount >= 4 ? 'Partial' : 'Sparse';
  const color = readyCount === totalChecks
    ? 'var(--accent-green)'
    : readyCount >= 4
      ? 'var(--accent-yellow)'
      : 'var(--accent-red)';
  return {
    label,
    color,
    readyCount,
    totalChecks,
    missing: checks.filter(check => !check.ok).map(check => check.label),
  };
}

function TierBadge({ tier, entry }) {
  if (!tier) return null;

  const gate = decisionGate(entry || {});
  const checkLabels = {
    screeningEvidence: 'Screening evidence',
    investigationEvidence: 'Investigation evidence',
    robustnessFloor: 'Robustness \u2265 0.50',
    validationEvidence: 'Validation evidence',
    baselineBeatsReference: 'Baseline < 1.0',
    consistencyBounded: 'Multi-seed std \u2264 0.12',
  };

  const tooltipLines = ['Promotion criteria:'];
  Object.entries(gate.checks).forEach(([name, ok]) => {
    tooltipLines.push(`${ok ? '\u2713' : '\u2717'} ${checkLabels[name] || name}`);
  });

  if (tier !== 'breakthrough' && gate.missing.length > 0) {
    tooltipLines.push('');
    tooltipLines.push(`Missing for breakthrough: ${gate.missing.map(m => checkLabels[m] || m).join(', ')}`);
  }

  const tooltip = tooltipLines.join('\n');

  return (
    <span
      title={tooltip}
      style={{
        padding: '2px 8px',
        borderRadius: 4,
        fontSize: 11,
        fontWeight: 600,
        color: TIER_COLORS[tier] || 'var(--text-muted)',
        background: `${TIER_COLORS[tier] || 'var(--text-muted)'}22`,
        border: `1px solid ${TIER_COLORS[tier] || 'var(--border)'}`,
        textTransform: 'uppercase',
        cursor: 'help',
      }}
    >
      {TIER_LABELS[tier] || tier}
    </span>
  );
}

function ScoreBreakdown({ entry }) {
  const [show, setShow] = useState(false);
  const breakdown = candidateScoreBreakdown(entry, TIER_ORDER);
  const score = candidateScore(entry, TIER_ORDER);

  const keyMap = {
    sLoss: { label: 'Screening Loss', color: 'var(--accent-blue)' },
    iLoss: { label: 'Investigation Loss', color: '#1f6feb' },
    loss: { label: 'Loss', color: 'var(--accent-blue)' },
    novelty: { label: 'Novelty', color: 'var(--accent-purple)' },
    vBase: { label: 'Baseline', color: 'var(--accent-green)' },
    baseline: { label: 'Baseline', color: 'var(--accent-green)' },
    robust: { label: 'Robustness', color: 'var(--accent-yellow)' },
    consistency: { label: 'Consistency', color: '#d29922' },
    tierBonus: { label: 'Tier Bonus', color: 'var(--accent-orange)' },
    throughput: { label: 'Throughput', color: 'var(--text-muted)' },
    efficiencyBonus: { label: 'Efficiency', color: '#58a6ff' },
    routingBonus: { label: 'Routing', color: '#3fb950' },
    adaptiveBonus: { label: 'Adaptive Compute', color: '#c77dff' },
  };

  const components = Object.entries(breakdown)
    .filter(([, weight]) => weight > 0)
    .map(([key, weight]) => ({
      key,
      weight,
      ...(keyMap[key] || { label: key, color: 'var(--border)' })
    }));

  const total = components.reduce((acc, c) => acc + (Number(c.weight) || 0), 0) || 1;

  return (
    <div
      style={{ minWidth: 80, position: 'relative', display: 'inline-block' }}
      onMouseEnter={() => setShow(true)}
      onMouseLeave={() => setShow(false)}
    >
      <div style={{ fontWeight: 600, color: scoreColor(score), marginBottom: 4 }}>
        {score}
      </div>
      <div style={{ display: 'flex', height: 4, borderRadius: 2, overflow: 'hidden', background: 'var(--bg-tertiary)' }}>
        {components.map(c => (
          <div
            key={c.key}
            style={{
              width: `${c.weight}%`,
              background: c.color,
              height: '100%'
            }}
          />
        ))}
      </div>
      {show && (
        <div style={{
          position: 'absolute',
          top: '100%',
          left: '50%',
          transform: 'translateX(-50%)',
          marginTop: 8,
          padding: '10px 12px',
          background: '#161b22',
          border: '1px solid var(--border)',
          borderRadius: 6,
          boxShadow: '0 6px 16px rgba(0,0,0,0.45)',
          zIndex: 1000,
          minWidth: 220,
          fontSize: 11,
          color: 'var(--text-primary)',
        }}>
          <div style={{ fontWeight: 600, marginBottom: 6 }}>Score Breakdown</div>
          {components.map(c => (
            <div key={`break-${c.key}`} style={{ marginBottom: 6 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
                <span>{c.label}</span>
                <span>{Number(c.weight).toFixed(1)}</span>
              </div>
              <div style={{ height: 4, background: 'var(--bg-tertiary)', borderRadius: 2, overflow: 'hidden' }}>
                <div style={{ width: `${(c.weight / total) * 100}%`, height: '100%', background: c.color }} />
              </div>
            </div>
          ))}
          <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>Internal composite only.</div>
        </div>
      )}
    </div>
  );
}

const COLUMNS = [
  {
    key: '_score',
    label: 'Score',
    title: 'Internal 0-100 composite for relative ranking only; not a claim of GPT-level replacement or external benchmark parity.',
  },
  { key: 'tier', label: 'Tier' },
  { key: '_stability', label: 'Stability' },
  { key: 'model_source', label: 'Source' },
  { key: 'architecture_family', label: 'Family' },
  { key: 'architecture_desc', label: 'Description' },
  { key: 'composite_score', label: 'Composite' },
  { key: 'screening_loss_ratio', label: 'S.Loss' },
  { key: 'screening_novelty', label: 'Novelty' },
  { key: 'investigation_loss_ratio', label: 'I.Loss' },
  { key: 'investigation_robustness', label: 'Robust' },
  { key: 'validation_loss_ratio', label: 'V.Loss' },
  { key: 'validation_baseline_ratio', label: 'V.Base' },
  { key: '_compression_ratio', label: 'Compression' },
  { key: '_metric_quality', label: 'Metric Quality' },
  { key: '_actions', label: 'Actions' },
];

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
  const [highlightId, setHighlightId] = useState(null);
  const queuedSet = useMemo(() => new Set(queuedResultIds || []), [queuedResultIds]);

  useEffect(() => {
    try {
      if (typeof window === 'undefined') return;
      window.localStorage.setItem(LEADERBOARD_PREFS_KEY, JSON.stringify({
        activeTier,
        sortKey,
        sortDesc,
        searchQuery,
      }));
    } catch {
      // Ignore localStorage failures.
    }
  }, [activeTier, sortKey, sortDesc, searchQuery]);

  // Accept external highlight request
  useEffect(() => {
    if (highlightResultId) {
      setHighlightId(highlightResultId);
      // Clear highlight after 3s animation
      const timer = setTimeout(() => {
        setHighlightId(null);
        if (onHighlightClear) onHighlightClear();
      }, 3000);
      return () => clearTimeout(timer);
    }
  }, [highlightResultId, onHighlightClear]);

  const fetchLeaderboard = useCallback(async () => {
    try {
      const params = new URLSearchParams({ sort: 'composite_score', limit: '100' });
      if (activeTier !== 'all') params.set('tier', activeTier);
      const res = await fetch(`${API_BASE}/api/leaderboard?${params}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = await res.json();
      setData(json);
      setLastUpdated(new Date());
      setError(null);
    } catch (e) {
      setError('Failed to load leaderboard: ' + e.message);
    }
    setLoading(false);
  }, [activeTier]);

  useEffect(() => {
    fetchLeaderboard();
    const interval = setInterval(fetchLeaderboard, 15000);
    return () => clearInterval(interval);
  }, [fetchLeaderboard]);

  const handleSort = (key) => {
    if (key === '_actions') return;
    if (sortKey === key) {
      setSortDesc(!sortDesc);
    } else {
      setSortKey(key);
      setSortDesc(true);
    }
  };

  const tiers = ['all', 'screening', 'investigation', 'validation', 'breakthrough'];

  const fmt = (v, d = 4) => v != null ? Number(v).toFixed(d) : '--';

  const handleInvestigate = (resultIds) => {
    if (onInvestigate) {
      setActionError(null);
      onInvestigate(resultIds);
    } else {
      fetch(`${API_BASE}/api/experiments/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: 'investigation', result_ids: resultIds }),
      })
        .then(r => r.ok ? r.json() : Promise.reject(r))
        .then(() => {
          setActionError(null);
          fetchLeaderboard();
        })
        .catch(e => setActionError('Failed to start investigation: ' + (e?.message || String(e))));
    }
  };

  const handleValidate = (resultIds) => {
    if (onValidate) {
      setActionError(null);
      onValidate(resultIds);
    } else {
      fetch(`${API_BASE}/api/experiments/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: 'validation', result_ids: resultIds }),
      })
        .then(r => r.ok ? r.json() : Promise.reject(r))
        .then(() => {
          setActionError(null);
          fetchLeaderboard();
        })
        .catch(e => setActionError('Failed to start validation: ' + (e?.message || String(e))));
    }
  };

  const rawEntries = data?.entries || [];
  const stabilitySummary = data?.cross_run_stability_summary || {};
  const stabilityWindow = data?.cross_run_stability_window || 0;

  // Count by tier for tab badges (from raw unfiltered data)
  const tierCounts = {};
  for (const entry of rawEntries) {
    const t = entry.tier || 'screening';
    tierCounts[t] = (tierCounts[t] || 0) + 1;
  }

  // Augment with computed score and sort client-side
  const sorted = useMemo(() => {
    const augmented = rawEntries.map(e => {
      const compression = compressionSummary(e);
      return {
        ...e,
        _score: candidateScore(e, TIER_ORDER),
        _compression_ratio: compression.ratio,
        _compression_summary: compression,
      };
    });
    augmented.sort((a, b) => {
      let va, vb;
      if (sortKey === 'tier') {
        va = TIER_ORDER[a.tier] || 0;
        vb = TIER_ORDER[b.tier] || 0;
      } else if (
        sortKey === 'model_source'
        || sortKey === 'architecture_desc'
        || sortKey === 'architecture_family'
      ) {
        va = a[sortKey] || '';
        vb = b[sortKey] || '';
        return sortDesc ? vb.localeCompare(va) : va.localeCompare(vb);
      } else {
        va = a[sortKey];
        vb = b[sortKey];
      }
      if (va == null && vb == null) return 0;
      if (va == null) return 1;
      if (vb == null) return -1;
      return sortDesc ? vb - va : va - vb;
    });
    return augmented;
  }, [rawEntries, sortKey, sortDesc]);

  // Apply search filter
  const filtered = useMemo(() => {
    if (!searchQuery.trim()) return sorted;
    const q = searchQuery.trim().toLowerCase();
    return sorted.filter(e =>
      (e.result_id && e.result_id.toLowerCase().includes(q)) ||
      (e.graph_fingerprint && e.graph_fingerprint.toLowerCase().includes(q)) ||
      (e.architecture_desc && e.architecture_desc.toLowerCase().includes(q)) ||
      (e.architecture_family && e.architecture_family.toLowerCase().includes(q))
    );
  }, [sorted, searchQuery]);

  // Scroll to highlighted row
  const highlightRef = useRef(null);
  useEffect(() => {
    if (highlightId && highlightRef.current) {
      highlightRef.current.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }, [highlightId, filtered]);

  return (
    <div className="card" style={{ padding: 16 }}>
      <div className="card-title" style={{ marginBottom: 12 }}>
        Qualified Models
        <span style={{ fontSize: 12, color: 'var(--text-muted)', marginLeft: 8 }}>
          {rawEntries.length} entries
        </span>
      </div>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 8, lineHeight: 1.5 }}>
        Ranked candidates with tiered evidence — click any row for details.
        For broad survivor browsing, use the <span style={{ color: 'var(--accent-blue)', textDecoration: 'underline', cursor: 'pointer' }} onClick={() => onSelectProgram && onSelectProgram('_CANDIDATES_TAB_')}>Candidates (All)</span> tab.
        <span style={{ marginLeft: 8, fontSize: 11 }}>
          Updated: {lastUpdated ? lastUpdated.toLocaleTimeString() : 'loading'}
          {' · '}Stability (window {stabilityWindow}): {stabilitySummary.stable || 0} stable, {stabilitySummary.up || 0} up, {stabilitySummary.down || 0} down, {stabilitySummary.new || 0} new
        </span>
      </p>
      {!!queuedSet.size && (
        <p style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8 }}>
          Progression queue: {queuedSet.size} candidate{queuedSet.size === 1 ? '' : 's'} pinned.
        </p>
      )}
      <details style={{ marginBottom: 10, fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.5 }}>
        <summary style={{ cursor: 'pointer', color: 'var(--text-secondary)' }}>Glossary &amp; notes</summary>
        <div style={{ marginTop: 6, paddingLeft: 8, borderLeft: '2px solid var(--border)' }}>
          <p style={{ margin: '4px 0' }}>Use this tab for promote/investigate/validate decisions. For broad survivor browsing, use Candidates (All).</p>
          <p style={{ margin: '4px 0' }}>S.Loss = screening loss ratio, I.Loss = investigation loss ratio, V.Loss = validation loss ratio, V.Base {'<'} 1 means better than baseline.</p>
          <p style={{ margin: '4px 0' }}>Decision gate: rows are <strong>Decision-Ready</strong> only when screening+investigation+validation metrics are present, robustness ≥ 0.50, baseline ratio {'<'} 1.00, and multi-seed std ≤ 0.12.</p>
          <p style={{ margin: '4px 0' }}>Metric quality chips show source and reliability: <strong>artifact-backed</strong> vs <strong>heuristic</strong>, with reliability bands from available validation depth and confidence.</p>
          <p style={{ margin: '4px 0' }}>External early-research benchmark: Open LLM Leaderboard (MMLU, ARC, HellaSwag, TruthfulQA, Winogrande, GSM8K) via lm-eval harness.</p>
        </div>
      </details>

      {/* Tier tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 12, flexWrap: 'wrap' }}>
        {tiers.map(tier => (
          <button
            key={tier}
            onClick={() => setActiveTier(tier)}
            aria-label={`Filter leaderboard by ${tier === 'all' ? 'all tiers' : `${TIER_LABELS[tier]} tier`}`}
            style={{
              padding: '4px 12px',
              borderRadius: 4,
              border: `1px solid ${activeTier === tier ? 'var(--accent-blue)' : 'var(--border)'}`,
              background: activeTier === tier ? 'rgba(88, 166, 255, 0.15)' : 'transparent',
              color: activeTier === tier ? 'var(--accent-blue)' : 'var(--text-secondary)',
              cursor: 'pointer',
              fontSize: 12,
              fontWeight: activeTier === tier ? 600 : 400,
            }}
          >
            {tier === 'all' ? 'All' : TIER_LABELS[tier]}
            {tier !== 'all' && tierCounts[tier] > 0 && (
              <span style={{
                marginLeft: 4, fontSize: 10,
                color: TIER_COLORS[tier],
              }}>
                ({tierCounts[tier]})
              </span>
            )}
          </button>
        ))}
        <button
          onClick={fetchLeaderboard}
          aria-label="Refresh leaderboard"
          style={{
            marginLeft: 'auto', fontSize: 11, padding: '4px 10px', cursor: 'pointer',
            border: '1px solid var(--border)', borderRadius: 4,
            background: 'transparent', color: 'var(--text-secondary)',
          }}
        >
          Refresh
        </button>
      </div>

      {/* Search filter */}
      <div style={{ marginBottom: 12 }}>
        <input
          type="text"
          placeholder="Search by fingerprint, result ID, family, or description..."
          value={searchQuery}
          onChange={e => setSearchQuery(e.target.value)}
          aria-label="Search leaderboard entries"
          style={{
            width: '100%',
            maxWidth: 400,
            padding: '6px 10px',
            fontSize: 12,
            border: '1px solid var(--border)',
            borderRadius: 4,
            background: 'var(--bg-secondary)',
            color: 'var(--text-primary)',
          }}
        />
        {searchQuery && (
          <span style={{ marginLeft: 8, fontSize: 11, color: 'var(--text-muted)' }}>
            {filtered.length} of {sorted.length} entries
          </span>
        )}
      </div>

      {error && (
        <p style={{ color: 'var(--accent-red)', fontSize: 13, marginBottom: 8 }}>{error}</p>
      )}
      {actionError && (
        <p style={{ color: 'var(--accent-red)', fontSize: 13, marginBottom: 8 }}>{actionError}</p>
      )}

      {loading ? (
        <p style={{ color: 'var(--text-muted)' }}>Loading leaderboard...</p>
      ) : filtered.length === 0 && !error ? (
        <div style={{ color: 'var(--text-muted)', fontSize: 13, lineHeight: 1.6 }}>
          {searchQuery.trim() ? (
            <p style={{ margin: 0 }}>
              No entries match "{searchQuery}". Try a different search term.
            </p>
          ) : activeTier === 'all' ? (
            <>
              <p style={{ margin: 0 }}>
                No leaderboard entries yet.
              </p>
              <p style={{ margin: '6px 0 0' }}>
                Start a screening experiment from Overview to generate candidates, then return here to review and promote top results.
              </p>
            </>
          ) : (
            <>
              <p style={{ margin: 0 }}>
                No entries in {TIER_LABELS[activeTier]} yet.
              </p>
              <p style={{ margin: '6px 0 0' }}>
                Advance candidates from lower tiers using Investigate/Validate actions to populate this tier.
              </p>
            </>
          )}
        </div>
      ) : (
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ borderBottom: '1px solid var(--border)' }}>
                <th style={thStyle}>#</th>
                {COLUMNS.map(col => (
                  <th
                    key={col.key}
                    onClick={() => handleSort(col.key)}
                    aria-label={col.key === '_actions'
                      ? 'Actions column'
                      : `Sort leaderboard by ${col.label}${sortKey === col.key ? `, currently ${sortDesc ? 'descending' : 'ascending'}` : ''}`}
                    title={col.title}
                    style={{
                      ...thStyle,
                      cursor: col.key === '_actions' ? 'default' : 'pointer',
                      userSelect: 'none',
                    }}
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
              {filtered.map((entry, i) => {
                const gate = decisionGate(entry);
                const compression = entry._compression_summary || compressionSummary(entry);
                const chips = metricChips(entry);
                const flags = qualityFlags(entry);
                const promotion = promotionEvidence(entry);
                const reproPacket = reproducibilityPacketStatus(entry);
                const isHighlighted = highlightId && entry.result_id === highlightId;
                const isQueued = !!entry.result_id && queuedSet.has(entry.result_id);
                const eligibility = eligibilityByResultId?.[entry.result_id] || candidateEligibility(entry);
                const queueIntent = eligibility.validationEligible
                  ? 'validation'
                  : eligibility.investigationEligible
                    ? 'investigation'
                    : null;
                const queueAddLabel = queueIntent === 'validation' ? 'Queue Validate' : 'Queue Investigate';
                const queueAddTitle = queueIntent === 'validation'
                  ? 'Add to validation queue'
                  : 'Add to investigation queue';
                const rowId = entry.entry_id || entry.result_id || i;
                const isExpanded = expandedRowId === rowId;
                return (
                <tr
                  key={rowId}
                  ref={isHighlighted ? highlightRef : undefined}
                  style={{
                    borderBottom: '1px solid var(--border)',
                    cursor: 'pointer',
                    background: isHighlighted
                      ? 'rgba(88, 166, 255, 0.2)'
                      : entry.tier === 'breakthrough' ? 'rgba(63, 185, 80, 0.08)' : undefined,
                    animation: isHighlighted ? 'leaderboard-pulse 1.5s ease-in-out 2' : undefined,
                  }}
                  onClick={() => onSelectProgram && onSelectProgram(entry.result_id)}
                >
                  <td style={tdStyle}>{i + 1}</td>
                  <td style={tdStyle}><ScoreBreakdown entry={entry} /></td>
                  <td style={tdStyle}><TierBadge tier={entry.tier} entry={entry} /></td>
                  <td style={tdStyle}>
                    {(() => {
                      const s = entry.cross_run_stability || {};
                      const trend = s.trend || 'unknown';
                      const color = trend === 'up'
                        ? 'var(--accent-green)'
                        : trend === 'down'
                          ? 'var(--accent-red)'
                          : trend === 'stable'
                            ? 'var(--accent-yellow)'
                            : 'var(--text-muted)';
                      return (
                        <span
                          title={`Trend ${trend}; seen runs ${s.seen_runs ?? 0}; latest rank ${s.latest_rank ?? '--'}; previous rank ${s.previous_rank ?? '--'}`}
                          style={{
                            fontSize: 10,
                            fontWeight: 600,
                            textTransform: 'uppercase',
                            padding: '2px 6px',
                            borderRadius: 4,
                            color,
                            background: `${color}22`,
                            border: `1px solid ${color}55`,
                          }}
                        >
                          {trend}
                        </span>
                      );
                    })()}
                  </td>
                  <td style={tdStyle}>
                    <span style={{
                      fontSize: 10,
                      color: entry.model_source === 'morphological_box'
                        ? 'var(--accent-purple)' : 'var(--accent-blue)',
                    }}>
                      {entry.model_source === 'morphological_box' ? 'MORPH' : 'GRAPH'}
                    </span>
                  </td>
                  <td style={tdStyle}>{entry.architecture_family || '--'}</td>
                  <td
                    style={{ ...tdStyle, maxWidth: 150, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
                    title={entry.architecture_desc || entry.result_id || 'not available'}
                  >
                    {entry.architecture_desc || entry.result_id?.slice(0, 12)}
                  </td>
                  <td style={{ ...tdStyle, color: 'var(--accent-green)' }}>
                    {fmt(entry.composite_score, 3)}
                  </td>
                  <td style={tdStyle}>{fmt(entry.screening_loss_ratio)}</td>
                  <td style={tdStyle}>{fmt(entry.screening_novelty, 3)}</td>
                  <td style={tdStyle}>{fmt(entry.investigation_loss_ratio)}</td>
                  <td style={tdStyle}>
                    {entry.investigation_robustness != null
                      ? <span style={{
                          color: entry.investigation_robustness >= 0.5
                            ? 'var(--accent-green)' : 'var(--accent-red)',
                        }}>
                          {fmt(entry.investigation_robustness, 2)}
                        </span>
                      : '--'}
                  </td>
                  <td style={tdStyle}>{fmt(entry.validation_loss_ratio)}</td>
                  <td style={tdStyle}>
                    {entry.validation_baseline_ratio != null
                      ? <span style={{
                          color: entry.validation_baseline_ratio < 1
                            ? 'var(--accent-green)' : 'var(--accent-red)',
                        }}>
                          {fmt(entry.validation_baseline_ratio)}
                        </span>
                      : '--'}
                  </td>
                  <td style={tdStyle}>
                    <div style={{ fontSize: 11, color: 'var(--text-secondary)' }}>
                      {compression.ratio != null ? `${(compression.ratio * 100).toFixed(0)}%` : '--'}
                    </div>
                    <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                      {compression.memoryMb != null ? `${compression.memoryMb.toFixed(2)} MB` : 'n/a'} · {compression.label}
                    </div>
                    <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                      retention {compression.qualityRetention != null ? `${(compression.qualityRetention * 100).toFixed(0)}%` : 'n/a'}
                    </div>
                  </td>
                  <td style={tdStyle}>
                    <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                      Quality: {promotion.label} · Repro: {reproPacket.label}
                    </div>
                    {isExpanded && (
                      <>
                        <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', maxWidth: 220, marginTop: 4 }}>
                          {chips.map(chip => (
                            <span
                              key={`${rowId}-${chip.label}`}
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
                        <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', maxWidth: 220, marginTop: 4 }}>
                          {flags.map(flag => (
                            <span
                              key={`${rowId}-${flag.label}`}
                              title={flag.detail ? `${flag.label} — ${flag.detail}` : `Quality flag: ${flag.label}`}
                              style={{
                                fontSize: 10,
                                padding: '1px 5px',
                                borderRadius: 4,
                                border: `1px solid ${reliabilityColor(flag.tone)}55`,
                                color: reliabilityColor(flag.tone),
                                background: `${reliabilityColor(flag.tone)}15`,
                                whiteSpace: 'nowrap',
                              }}
                            >
                              {flag.label}
                            </span>
                          ))}
                        </div>
                        <div
                          style={{ marginTop: 5, fontSize: 10, fontWeight: 600, color: promotion.color }}
                          title={`Evidence checks ${promotion.evidenceCount}/${promotion.totalChecks}; missing: ${promotion.missing.length ? promotion.missing.join(', ') : 'none'}`}
                        >
                          Promotion confidence: {promotion.label} ({promotion.score}%)
                        </div>
                        <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                          Uncertainty {promotion.uncertaintyLabel}; runs {promotion.seenRuns}; std {promotion.std != null ? promotion.std.toFixed(3) : 'n/a'}
                        </div>
                        <div
                          style={{ marginTop: 2, fontSize: 10, color: reproPacket.color }}
                          title={reproPacket.missing.length ? `Missing packet fields: ${reproPacket.missing.join(', ')}` : 'Reproducibility packet has all required fields'}
                        >
                          Repro packet: {reproPacket.label} ({reproPacket.readyCount}/{reproPacket.totalChecks})
                        </div>
                      </>
                    )}
                  </td>
                  <td style={tdStyle} onClick={e => e.stopPropagation()}>
                    {eligibility.investigationEligible && (
                      <button
                        onClick={() => handleInvestigate([entry.result_id])}
                        style={{ ...actionBtnStyle, background: 'rgba(63, 185, 80, 0.12)', border: '1px solid rgba(63, 185, 80, 0.4)', color: 'var(--accent-green)' }}
                        title="Deep study with multiple training programs"
                      >
                        Investigate
                      </button>
                    )}
                    {!eligibility.investigationEligible && entry.tier === 'screening' && (
                      <span style={{
                        fontSize: 10, padding: '2px 6px', borderRadius: 4,
                        background: 'rgba(210,153,34,0.12)', color: 'var(--accent-yellow)',
                        whiteSpace: 'nowrap',
                      }} title="Candidate already has investigation evidence; wait for changed conditions before re-investigating">
                        Already investigated
                      </span>
                    )}
                    {eligibility.validationEligible && (
                      <button
                        onClick={() => handleValidate([entry.result_id])}
                        style={{ ...actionBtnStyle, background: 'rgba(188, 140, 255, 0.12)', border: '1px solid rgba(188, 140, 255, 0.4)', color: 'var(--accent-purple)' }}
                        title="Publication-grade multi-seed validation"
                      >
                        Validate
                      </button>
                    )}
                    {entry.tier === 'investigation' && !entry.investigation_passed && (
                      <span style={{
                        fontSize: 10, padding: '2px 6px', borderRadius: 4,
                        background: 'rgba(248,81,73,0.12)', color: 'var(--accent-red, #e74c3c)',
                        whiteSpace: 'nowrap',
                      }} title="Investigation did not pass — search for new candidates or review failure details">
                        Investigation failed
                      </span>
                    )}
                    <div style={{ display: 'flex', gap: 4, marginTop: 4 }}>
                      <button
                        onClick={() => setExpandedRowId(isExpanded ? null : rowId)}
                        style={{
                          ...actionBtnStyle,
                          borderColor: 'var(--accent-blue)',
                          color: 'var(--accent-blue)',
                          background: isExpanded ? 'rgba(88, 166, 255, 0.12)' : 'transparent',
                        }}
                      >
                        {isExpanded ? 'Hide details' : 'Details'}
                      </button>
                      {onOpenInDesigner && (
                        <button
                          onClick={() => onOpenInDesigner(entry.result_id)}
                          style={{ ...actionBtnStyle, background: 'rgba(188, 140, 255, 0.12)', border: '1px solid rgba(188, 140, 255, 0.4)', color: 'var(--accent-purple)' }}
                          title="Open architecture in visual designer"
                        >
                          Designer
                        </button>
                      )}
                    </div>
                    {isExpanded && (
                      <div style={{ marginTop: 6 }}>
                        <div style={{ marginBottom: 4 }}>
                          <span
                            style={{
                              fontSize: 10,
                              fontWeight: 600,
                              textTransform: 'uppercase',
                              padding: '2px 6px',
                              borderRadius: 4,
                              color: gate.color,
                              background: `${gate.color}22`,
                              border: `1px solid ${gate.color}55`,
                            }}
                            title={gate.decisionReady
                              ? 'All evidence checks passed.'
                              : `Missing checks: ${gate.missing.join(', ')}`}
                          >
                            {gate.label}
                          </span>
                        </div>
                        {entry.result_id && (onQueueAdd || onQueueRemove) && (
                          <button
                            onClick={() => {
                              if (isQueued) {
                                onQueueRemove && onQueueRemove(entry.result_id);
                                return;
                              }
                              if (!eligibility.queueEligible) {
                                return;
                              }
                              onQueueAdd && onQueueAdd({
                                resultId: entry.result_id,
                                fingerprint: entry.graph_fingerprint,
                                source: 'leaderboard',
                                architectureFamily: entry.architecture_family,
                                intent: queueIntent,
                                queueEligible: eligibility.queueEligible,
                                investigationEligible: eligibility.investigationEligible,
                                validationEligible: eligibility.validationEligible,
                                queueReason: eligibility.queueReason,
                              });
                            }}
                            disabled={!isQueued && !eligibility.queueEligible}
                            style={{
                              ...actionBtnStyle,
                              marginTop: 4,
                              borderColor: !isQueued && !eligibility.queueEligible
                                ? 'var(--border)'
                                : isQueued
                                  ? 'var(--accent-yellow)'
                                  : 'var(--accent-blue)',
                              color: !isQueued && !eligibility.queueEligible
                                ? 'var(--text-muted)'
                                : isQueued
                                  ? 'var(--accent-yellow)'
                                  : 'var(--accent-blue)',
                              opacity: !isQueued && !eligibility.queueEligible ? 0.6 : 1,
                            }}
                            title={isQueued
                              ? 'Remove from investigation queue'
                              : !eligibility.queueEligible
                                ? (entry.tier === 'validation' || entry.tier === 'breakthrough' 
                                    ? 'Architecture is fully validated.' 
                                    : 'Not eligible for investigation/validation queue actions')
                                : queueAddTitle}
                          >
                            {isQueued 
                              ? 'Queued' 
                              : !eligibility.queueEligible 
                                ? (entry.tier === 'validation' || entry.tier === 'breakthrough' ? 'Validated' : 'Ineligible') 
                                : queueAddLabel}
                          </button>
                        )}
                      </div>
                    )}
                  </td>
                </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

const thStyle = {
  padding: '6px 8px',
  textAlign: 'left',
  fontSize: 11,
  color: 'var(--text-muted)',
  fontWeight: 600,
  textTransform: 'uppercase',
  whiteSpace: 'nowrap',
};

const tdStyle = {
  padding: '6px 8px',
  whiteSpace: 'nowrap',
};

const actionBtnStyle = {
  padding: '4px 10px',
  fontSize: 11,
  border: '1px solid rgba(88, 166, 255, 0.4)',
  borderRadius: 4,
  background: 'rgba(88, 166, 255, 0.12)',
  color: 'var(--accent-blue)',
  cursor: 'pointer',
};

export default Leaderboard;
