import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { scoreColor } from '../utils/format';
import { lossColor, noveltyColor, reliabilityColor } from '../utils/colors';
import { candidateScore, candidateScoreBreakdown, promotionEvidence, TIER_ORDER } from '../utils/scoringEngine';

const API_BASE = process.env.REACT_APP_API_URL || '';
const DISCOVERIES_PREFS_KEY = 'aria_discoveries_prefs_v1';

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

const STATUS_LABELS = {
  screening: 'Screening',
  investigation: 'Investigating',
  validation: 'Validated',
  breakthrough: 'Breakthrough',
};

// ── Summary Bar ────────────────────────────────────────────────────

function SummaryBar({ tierCounts }) {
  const total = tierCounts?.total_survivors || 0;
  const validated = (tierCounts?.validation || 0) + (tierCounts?.breakthrough || 0);
  const breakthroughs = tierCounts?.breakthrough || 0;

  return (
    <div style={{
      display: 'flex', gap: 24, alignItems: 'center', flexWrap: 'wrap',
      padding: '10px 14px', marginBottom: 12,
      background: 'var(--bg-secondary)', borderRadius: 8,
      border: '1px solid var(--border)', fontSize: 13,
    }}>
      <Stat value={total} label="unique architectures" />
      <Stat value={tierCounts?.screening || 0} label="screening" color="var(--accent-blue)" />
      <Stat value={tierCounts?.investigation || 0} label="investigating" color="var(--accent-yellow)" />
      <Stat value={validated} label="validated" color="var(--accent-purple)" />
      <Stat value={breakthroughs} label="breakthroughs" color="var(--accent-green)" />
    </div>
  );
}

function Stat({ value, label, color }) {
  return (
    <span>
      <strong style={{ fontSize: 16, color: color || 'var(--text-primary)', marginRight: 4 }}>
        {value}
      </strong>
      <span style={{ color: 'var(--text-muted)' }}>{label}</span>
    </span>
  );
}

// ── Status Badge ───────────────────────────────────────────────────

function StatusBadge({ tier }) {
  const color = TIER_COLORS[tier] || 'var(--text-muted)';
  return (
    <span style={{
      padding: '2px 8px', borderRadius: 4, fontSize: 11, fontWeight: 600,
      color, background: `${color}22`, border: `1px solid ${color}`,
      textTransform: 'uppercase',
    }}>
      {STATUS_LABELS[tier] || tier || 'Unknown'}
    </span>
  );
}

// ── Score with hover breakdown ─────────────────────────────────────

function ScoreCell({ entry }) {
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
    .filter(([, w]) => w > 0)
    .map(([key, weight]) => ({ key, weight, ...(keyMap[key] || { label: key, color: 'var(--border)' }) }));

  const total = components.reduce((acc, c) => acc + (Number(c.weight) || 0), 0) || 1;

  return (
    <div
      style={{ minWidth: 70, position: 'relative', display: 'inline-block' }}
      onMouseEnter={() => setShow(true)}
      onMouseLeave={() => setShow(false)}
    >
      <div style={{ fontWeight: 600, color: scoreColor(score) }}>{score}</div>
      <div style={{ display: 'flex', height: 3, borderRadius: 2, overflow: 'hidden', background: 'var(--bg-tertiary)', marginTop: 2 }}>
        {components.map(c => (
          <div key={c.key} style={{ width: `${c.weight}%`, background: c.color, height: '100%' }} />
        ))}
      </div>
      {show && (
        <div style={{
          position: 'absolute', top: '100%', left: '50%', transform: 'translateX(-50%)',
          marginTop: 6, padding: '8px 10px', background: '#161b22',
          border: '1px solid var(--border)', borderRadius: 6,
          boxShadow: '0 6px 16px rgba(0,0,0,0.45)', zIndex: 1000,
          minWidth: 200, fontSize: 11, color: 'var(--text-primary)',
        }}>
          <div style={{ fontWeight: 600, marginBottom: 4 }}>Score Breakdown</div>
          {components.map(c => (
            <div key={c.key} style={{ marginBottom: 4 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 1 }}>
                <span>{c.label}</span>
                <span>{Number(c.weight).toFixed(1)}</span>
              </div>
              <div style={{ height: 3, background: 'var(--bg-tertiary)', borderRadius: 2, overflow: 'hidden' }}>
                <div style={{ width: `${(c.weight / total) * 100}%`, height: '100%', background: c.color }} />
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ── Expanded Row Detail ────────────────────────────────────────────

function ExpandedDetail({ entry, onInvestigate, onValidate, onQueueAdd, onQueueRemove, isQueued, eligibility }) {
  const promotion = promotionEvidence(entry);
  const fmt = (v, d = 4) => v != null ? Number(v).toFixed(d) : '--';

  return (
    <tr>
      <td colSpan={8} style={{ padding: '12px 16px', background: 'var(--bg-secondary)', borderBottom: '1px solid var(--border)' }}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 16, fontSize: 12 }}>
          {/* Metrics detail */}
          <div>
            <div style={{ fontWeight: 600, marginBottom: 6, textTransform: 'uppercase', fontSize: 10, color: 'var(--text-muted)' }}>Full Metrics</div>
            <MetricRow label="Screening Loss" value={fmt(entry.screening_loss_ratio)} color={lossColor(entry.screening_loss_ratio)} />
            <MetricRow label="Screening Novelty" value={fmt(entry.screening_novelty, 3)} color={noveltyColor(entry.screening_novelty)} />
            <MetricRow label="Investigation Loss" value={fmt(entry.investigation_loss_ratio)} />
            <MetricRow label="Robustness" value={fmt(entry.investigation_robustness, 2)}
              color={entry.investigation_robustness != null
                ? (entry.investigation_robustness >= 0.5 ? 'var(--accent-green)' : 'var(--accent-red)')
                : undefined} />
            <MetricRow label="Validation Loss" value={fmt(entry.validation_loss_ratio)} />
            <MetricRow label="Validation Baseline" value={fmt(entry.validation_baseline_ratio)}
              color={entry.validation_baseline_ratio != null
                ? (entry.validation_baseline_ratio < 1 ? 'var(--accent-green)' : 'var(--accent-red)')
                : undefined} />
            <MetricRow label="Multi-seed Std" value={fmt(entry.validation_multi_seed_std, 3)} />
            <MetricRow label="Composite" value={fmt(entry.composite_score, 3)} color="var(--accent-green)" />
          </div>

          {/* Evidence & promotion */}
          <div>
            <div style={{ fontWeight: 600, marginBottom: 6, textTransform: 'uppercase', fontSize: 10, color: 'var(--text-muted)' }}>Evidence</div>
            <div style={{ marginBottom: 6, color: promotion.color, fontWeight: 600 }}>
              Promotion: {promotion.label} ({promotion.score}%)
            </div>
            {entry.cka_source && (
              <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>
                CKA: {entry.cka_source === 'artifact' ? 'artifact-backed' : 'heuristic'}
              </div>
            )}
            {entry.novelty_confidence != null && (
              <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>
                Novelty confidence: {Number(entry.novelty_confidence).toFixed(2)}
              </div>
            )}
            {entry.param_count != null && (
              <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>
                Parameters: {(entry.param_count / 1e6).toFixed(1)}M
              </div>
            )}
            {entry.graph_fingerprint && (
              <div style={{ fontSize: 11, color: 'var(--text-muted)', fontFamily: 'monospace' }}>
                FP: {entry.graph_fingerprint}
              </div>
            )}
            {entry.result_id && (
              <div style={{ fontSize: 11, color: 'var(--text-muted)', fontFamily: 'monospace' }}>
                ID: {entry.result_id}
              </div>
            )}
          </div>

          {/* Actions */}
          <div>
            <div style={{ fontWeight: 600, marginBottom: 6, textTransform: 'uppercase', fontSize: 10, color: 'var(--text-muted)' }}>Actions</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              {eligibility?.investigationEligible && (
                <button onClick={() => onInvestigate([entry.result_id])} style={{ ...actionBtnStyle, background: 'rgba(63, 185, 80, 0.12)', border: '1px solid rgba(63, 185, 80, 0.4)', color: 'var(--accent-green)' }}>
                  Investigate
                </button>
              )}
              {eligibility?.validationEligible && (
                <button
                  onClick={() => onValidate([entry.result_id])}
                  style={{ ...actionBtnStyle, background: 'rgba(188, 140, 255, 0.12)', border: '1px solid rgba(188, 140, 255, 0.4)', color: 'var(--accent-purple)' }}
                >
                  Validate
                </button>
              )}
              {entry.result_id && (onQueueAdd || onQueueRemove) && (
                <button
                  onClick={() => {
                    if (isQueued) {
                      onQueueRemove?.(entry.result_id);
                    } else if (eligibility?.queueEligible) {
                      onQueueAdd?.({
                        resultId: entry.result_id,
                        fingerprint: entry.graph_fingerprint,
                        source: 'discoveries',
                        architectureFamily: entry.architecture_family,
                        intent: eligibility?.validationEligible ? 'validation' : 'investigation',
                        queueEligible: true,
                        investigationEligible: eligibility?.investigationEligible,
                        validationEligible: eligibility?.validationEligible,
                      });
                    }
                  }}
                  disabled={!isQueued && !eligibility?.queueEligible}
                  style={{
                    ...actionBtnStyle,
                    borderColor: isQueued ? 'var(--accent-yellow)' : 'var(--accent-blue)',
                    color: isQueued ? 'var(--accent-yellow)' : 'var(--accent-blue)',
                    opacity: !isQueued && !eligibility?.queueEligible ? 0.5 : 1,
                  }}
                >
                  {isQueued 
                    ? 'Queued' 
                    : (!eligibility?.queueEligible && (entry.tier === 'validation' || entry.tier === 'breakthrough'))
                      ? 'Fully Validated'
                      : !eligibility?.queueEligible 
                        ? 'Ineligible' 
                        : 'Add to Queue'}
                </button>
              )}
            </div>
          </div>
        </div>
      </td>
    </tr>
  );
}

function MetricRow({ label, value, color }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3 }}>
      <span style={{ color: 'var(--text-muted)' }}>{label}</span>
      <span style={{ color: color || 'var(--text-primary)', fontFamily: 'monospace' }}>{value}</span>
    </div>
  );
}

// ── Fingerprint Leaderboard Chart ─────────────────────────────────

function FingerprintLeaderboardChart({ entries }) {
  if (!entries || entries.length < 2) return null;
  
  // Take top 15 for the chart
  const top = entries.slice(0, 15);
  const W = 600;
  const H = 160;
  const PAD_X = 40;
  const PAD_Y = 20;
  const barW = (W - 2 * PAD_X) / top.length - 8;
  
  const maxScore = Math.max(...top.map(e => e._score), 80);
  
  return (
    <div style={{ marginBottom: 20, padding: '10px 0' }}>
      <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8, textTransform: 'uppercase', fontWeight: 600 }}>
        Fingerprint Performance Ranking (Top {top.length})
      </div>
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto', maxWidth: W }}>
        {/* Y-axis labels */}
        {[0, 0.5, 1].map(frac => (
          <text key={frac} x={PAD_X - 5} y={H - PAD_Y - frac * (H - 2 * PAD_Y)} 
                fontSize={9} fill="var(--text-muted)" textAnchor="end" alignmentBaseline="middle">
            {Math.round(frac * maxScore)}
          </text>
        ))}
        
        {/* Horizontal grid lines */}
        {[0, 0.5, 1].map(frac => (
          <line key={`grid-${frac}`} x1={PAD_X} y1={H - PAD_Y - frac * (H - 2 * PAD_Y)} 
                x2={W - PAD_X} y2={H - PAD_Y - frac * (H - 2 * PAD_Y)} 
                stroke="var(--border)" strokeWidth={0.5} strokeDasharray="2 2" />
        ))}

        {top.map((e, i) => {
          const score = e._score || 0;
          const barH = (score / maxScore) * (H - 2 * PAD_Y);
          const x = PAD_X + i * (barW + 8);
          const y = H - PAD_Y - barH;
          const color = scoreColor(score);
          
          return (
            <g key={e.result_id || i}>
              <rect x={x} y={y} width={barW} height={barH} fill={`${color}88`} stroke={color} strokeWidth={1} rx={2} />
              <text x={x + barW / 2} y={H - 5} fontSize={8} fill="var(--text-muted)" 
                    textAnchor="middle" transform={`rotate(45 ${x + barW / 2} ${H - 5})`}>
                {e.display_name?.slice(0, 8) || e.graph_fingerprint?.slice(0, 6)}
              </text>
              <title>{e.display_name || e.graph_fingerprint}: Score {score}</title>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

// ── Main Component ─────────────────────────────────────────────────

const COLUMNS = [
  { key: '_score', label: 'Score' },
  { key: 'display_name', label: 'Architecture' },
  { key: 'architecture_family', label: 'Family' },
  { key: '_best_loss', label: 'Loss' },
  { key: '_novelty', label: 'Novelty' },
  { key: 'tier', label: 'Status' },
  { key: '_actions', label: '' },
];

function Discoveries({
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
  const [expandedRowId, setExpandedRowId] = useState(null);
  const [highlightId, setHighlightId] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [showChart, setShowChart] = useState(true);
  const queuedSet = useMemo(() => new Set(queuedResultIds || []), [queuedResultIds]);
  const highlightRef = useRef(null);

  // Persist preferences
  useEffect(() => {
    try {
      if (typeof window === 'undefined') return;
      window.localStorage.setItem(DISCOVERIES_PREFS_KEY, JSON.stringify({
        activeTier, sortKey, sortDesc, searchQuery, showChart,
      }));
    } catch {}
  }, [activeTier, sortKey, sortDesc, searchQuery, showChart]);

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

  const fetchData = useCallback(async () => {
    try {
      const params = new URLSearchParams({ sort: 'composite_score', limit: '100', view: 'ranked' });
      if (activeTier !== 'all') params.set('tier', activeTier);
      const res = await fetch(`${API_BASE}/api/discoveries?${params}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = await res.json();
      setData(json);
      setLastUpdated(new Date());
      setError(null);
    } catch (e) {
      setError('Failed to load discoveries: ' + e.message);
    }
    setLoading(false);
  }, [activeTier]);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 15000);
    return () => clearInterval(interval);
  }, [fetchData]);

  const handleSort = (key) => {
    if (key === '_actions') return;
    if (sortKey === key) setSortDesc(!sortDesc);
    else { setSortKey(key); setSortDesc(true); }
  };

  // Sort & augment entries
  const sorted = useMemo(() => {
    const entries = data?.entries || [];
    const augmented = entries.map(e => ({
      ...e,
      _score: candidateScore(e, TIER_ORDER),
      _best_loss: e.screening_loss_ratio ?? e.investigation_loss_ratio ?? e.validation_loss_ratio ?? e.loss_ratio ?? null,
      _novelty: e.screening_novelty ?? e.novelty_score ?? null,
    }));
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

  // Search filter
  const filtered = useMemo(() => {
    if (!searchQuery.trim()) return sorted;
    const q = searchQuery.trim().toLowerCase();
    return sorted.filter(e =>
      (e.display_name && e.display_name.toLowerCase().includes(q)) ||
      (e.architecture_family && e.architecture_family.toLowerCase().includes(q)) ||
      (e.graph_fingerprint && e.graph_fingerprint.toLowerCase().includes(q)) ||
      (e.result_id && e.result_id.toLowerCase().includes(q)) ||
      (e.architecture_desc && e.architecture_desc.toLowerCase().includes(q))
    );
  }, [sorted, searchQuery]);

  const tierCounts = data?.tier_counts || {};
  const tiers = ['all', 'screening', 'investigation', 'validation', 'breakthrough'];

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
          <SummaryBar tierCounts={tierCounts} />
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
            ? (data?.total || 0)
            : (tierCounts[tier] || 0);
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
          onClick={fetchData}
          aria-label="Refresh discoveries"
          style={{ marginLeft: 'auto', fontSize: 11, padding: '5px 12px', cursor: 'pointer', background: 'transparent', border: '1px solid var(--border)', borderRadius: 4, color: 'var(--text-secondary)' }}
        >
          Refresh
        </button>
      </div>

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
            {filtered.length} of {sorted.length} entries
          </span>
        )}
      </div>

      {error && <p style={{ color: 'var(--accent-red)', fontSize: 13, marginBottom: 8 }}>{error}</p>}

      {loading ? (
        <p style={{ color: 'var(--text-muted)' }}>Loading discoveries...</p>
      ) : filtered.length === 0 && !error ? (
        <div style={{ color: 'var(--text-muted)', fontSize: 13, lineHeight: 1.6 }}>
          {searchQuery.trim() ? (
            <p>No entries match "{searchQuery}".</p>
          ) : activeTier === 'all' ? (
            <p>No discoveries yet. Run experiments to generate candidates.</p>
          ) : (
            <p>No entries in {TIER_LABELS[activeTier]} tier yet.</p>
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
                const rowId = entry.entry_id || entry.result_id || i;
                const isExpanded = expandedRowId === rowId;
                const isHighlighted = highlightId && entry.result_id === highlightId;
                const isQueued = !!entry.result_id && queuedSet.has(entry.result_id);
                const eligibility = eligibilityByResultId?.[entry.result_id] || null;
                const displayName = entry.display_name || entry.architecture_desc || entry.graph_fingerprint?.slice(0, 10) || '--';

                return (
                  <React.Fragment key={rowId}>
                    <tr
                      ref={isHighlighted ? highlightRef : undefined}
                      style={{
                        borderBottom: '1px solid var(--border)',
                        cursor: 'pointer',
                        background: isHighlighted
                          ? 'rgba(88, 166, 255, 0.2)'
                          : entry.tier === 'breakthrough' ? 'rgba(63, 185, 80, 0.08)' : undefined,
                        animation: isHighlighted ? 'leaderboard-pulse 1.5s ease-in-out 2' : undefined,
                      }}
                      onClick={() => onSelectProgram?.(entry.result_id)}
                    >
                      <td style={tdStyle}>{i + 1}</td>
                      <td style={tdStyle}><ScoreCell entry={entry} /></td>
                      <td style={{ ...tdStyle, maxWidth: 200 }}>
                        <div style={{ fontWeight: 500 }}>{displayName}</div>
                        {entry.graph_fingerprint && (
                          <div style={{ fontSize: 10, color: 'var(--text-muted)', fontFamily: 'monospace' }}>
                            {entry.graph_fingerprint.slice(0, 12)}
                          </div>
                        )}
                      </td>
                      <td style={tdStyle}>
                        <span style={{
                          fontSize: 11, padding: '1px 6px', borderRadius: 3,
                          background: 'var(--bg-tertiary)', color: 'var(--text-secondary)',
                        }}>
                          {entry.architecture_family || '--'}
                        </span>
                      </td>
                      <td style={{ ...tdStyle, color: lossColor(entry._best_loss), fontFamily: 'monospace' }}>
                        {entry._best_loss != null ? Number(entry._best_loss).toFixed(4) : '--'}
                      </td>
                      <td style={{ ...tdStyle, color: noveltyColor(entry._novelty), fontFamily: 'monospace' }}>
                        {entry._novelty != null ? Number(entry._novelty).toFixed(3) : '--'}
                      </td>
                      <td style={tdStyle}><StatusBadge tier={entry.tier} /></td>
                      <td style={tdStyle} onClick={e => e.stopPropagation()}>
                        <div style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
                          <button
                            onClick={() => setExpandedRowId(isExpanded ? null : rowId)}
                            style={{
                              ...actionBtnStyle,
                              borderColor: 'var(--accent-blue)',
                              color: 'var(--accent-blue)',
                              background: isExpanded ? 'rgba(88, 166, 255, 0.12)' : 'transparent',
                            }}
                          >
                            {isExpanded ? 'Collapse' : 'Details'}
                          </button>
                          {onOpenInDesigner && (
                            <button
                              onClick={() => {
                                if (entry.result_id) onOpenInDesigner(entry.result_id)
                              }}
                              disabled={!entry.result_id}
                              style={{
                                ...actionBtnStyle,
                                borderColor: 'var(--accent-purple)',
                                color: 'var(--accent-purple)',
                                opacity: entry.result_id ? 1 : 0.5,
                                cursor: entry.result_id ? 'pointer' : 'not-allowed',
                              }}
                              title={entry.result_id ? 'Open architecture in visual designer' : 'Designer unavailable: missing result ID'}
                            >
                              Designer
                            </button>
                          )}
                        </div>
                      </td>
                    </tr>
                    {isExpanded && (
                      <ExpandedDetail
                        entry={entry}
                        onInvestigate={onInvestigate}
                        onValidate={onValidate}
                        onQueueAdd={onQueueAdd}
                        onQueueRemove={onQueueRemove}
                        isQueued={isQueued}
                        eligibility={eligibility}
                      />
                    )}
                  </React.Fragment>
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
  padding: '6px 8px', textAlign: 'left', fontSize: 11,
  color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', whiteSpace: 'nowrap',
};

const tdStyle = {
  padding: '6px 8px', whiteSpace: 'nowrap',
};

const actionBtnStyle = {
  padding: '4px 10px', fontSize: 11,
  border: '1px solid rgba(88, 166, 255, 0.4)', borderRadius: 4,
  background: 'rgba(88, 166, 255, 0.12)', color: 'var(--accent-blue)', cursor: 'pointer',
};

export default Discoveries;
