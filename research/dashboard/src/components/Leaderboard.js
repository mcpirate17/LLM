import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { scoreColor } from '../utils/format';

const API_BASE = process.env.REACT_APP_API_URL || '';

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

function TierBadge({ tier }) {
  return (
    <span style={{
      padding: '2px 8px',
      borderRadius: 4,
      fontSize: 11,
      fontWeight: 600,
      color: TIER_COLORS[tier] || 'var(--text-muted)',
      background: `${TIER_COLORS[tier] || 'var(--text-muted)'}22`,
      border: `1px solid ${TIER_COLORS[tier] || 'var(--border)'}`,
      textTransform: 'uppercase',
    }}>
      {TIER_LABELS[tier] || tier}
    </span>
  );
}

/**
 * Compute a 0-100 overall score for a leaderboard entry.
 * Weights shift as the entry advances through tiers.
 *   Screening only:    loss (35%) + novelty (25%) + tier bonus (40%)
 *   + Investigation:   inv_loss (20%) + robustness (15%) replaces some tier weight
 *   + Validation:      val_baseline (25%) + consistency (15%) replaces more
 */
function entryScore(entry) {
  // Screening loss: lower is better
  const sLoss = entry.screening_loss_ratio != null
    ? Math.max(0, 1 - (entry.screening_loss_ratio - 0.2) / 0.8)
    : 0;

  // Novelty
  const novelty = entry.screening_novelty != null
    ? Math.min(entry.screening_novelty, 1.0)
    : 0;

  // Investigation loss
  const iLoss = entry.investigation_loss_ratio != null
    ? Math.max(0, 1 - (entry.investigation_loss_ratio - 0.2) / 0.8)
    : 0;

  // Robustness
  const robust = entry.investigation_robustness != null
    ? Math.min(entry.investigation_robustness, 1.0)
    : 0;

  // Validation baseline: < 1 = beats transformer
  const vBase = entry.validation_baseline_ratio != null
    ? Math.max(0, Math.min(1, 1.5 - entry.validation_baseline_ratio))
    : 0;

  // Consistency (inverse of multi-seed std)
  const consistency = entry.validation_multi_seed_std != null
    ? Math.max(0, 1 - entry.validation_multi_seed_std * 10)
    : 0;

  // Tier bonus: higher tiers get a base boost
  const tierBonus = (TIER_ORDER[entry.tier] || 0) / 4;

  const tier = entry.tier || 'screening';
  let score;
  if (tier === 'breakthrough' || tier === 'validation') {
    score = sLoss * 10 + novelty * 10 + iLoss * 10 + robust * 10 + vBase * 25 + consistency * 15 + tierBonus * 20;
  } else if (tier === 'investigation') {
    score = sLoss * 15 + novelty * 15 + iLoss * 20 + robust * 15 + tierBonus * 35;
  } else {
    score = sLoss * 35 + novelty * 25 + tierBonus * 40;
  }

  return Math.round(Math.max(0, Math.min(100, score)));
}

function entryScoreBreakdown(entry) {
  const sLoss = entry.screening_loss_ratio != null
    ? Math.max(0, 1 - (entry.screening_loss_ratio - 0.2) / 0.8)
    : 0;
  const novelty = entry.screening_novelty != null
    ? Math.min(entry.screening_novelty, 1.0)
    : 0;
  const iLoss = entry.investigation_loss_ratio != null
    ? Math.max(0, 1 - (entry.investigation_loss_ratio - 0.2) / 0.8)
    : 0;
  const robust = entry.investigation_robustness != null
    ? Math.min(entry.investigation_robustness, 1.0)
    : 0;
  const vBase = entry.validation_baseline_ratio != null
    ? Math.max(0, Math.min(1, 1.5 - entry.validation_baseline_ratio))
    : 0;
  const consistency = entry.validation_multi_seed_std != null
    ? Math.max(0, 1 - entry.validation_multi_seed_std * 10)
    : 0;
  const tierBonus = (TIER_ORDER[entry.tier] || 0) / 4;
  const tier = entry.tier || 'screening';

  if (tier === 'breakthrough' || tier === 'validation') {
    return {
      sLoss: sLoss * 10,
      novelty: novelty * 10,
      iLoss: iLoss * 10,
      robust: robust * 10,
      vBase: vBase * 25,
      consistency: consistency * 15,
      tierBonus: tierBonus * 20,
    };
  }
  if (tier === 'investigation') {
    return {
      sLoss: sLoss * 15,
      novelty: novelty * 15,
      iLoss: iLoss * 20,
      robust: robust * 15,
      tierBonus: tierBonus * 35,
    };
  }
  return {
    sLoss: sLoss * 35,
    novelty: novelty * 25,
    tierBonus: tierBonus * 40,
  };
}

const COLUMNS = [
  { key: '_score', label: 'Score' },
  { key: 'tier', label: 'Tier' },
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
  { key: '_actions', label: 'Actions' },
];

function Leaderboard({ onSelectProgram, onInvestigate, onValidate }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTier, setActiveTier] = useState('all');
  const [sortKey, setSortKey] = useState('_score');
  const [sortDesc, setSortDesc] = useState(true);
  const [actionError, setActionError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);

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

  // Count by tier for tab badges (from raw unfiltered data)
  const tierCounts = {};
  for (const entry of rawEntries) {
    const t = entry.tier || 'screening';
    tierCounts[t] = (tierCounts[t] || 0) + 1;
  }

  // Augment with computed score and sort client-side
  const sorted = useMemo(() => {
    const augmented = rawEntries.map(e => ({ ...e, _score: entryScore(e) }));
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

  return (
    <div className="card" style={{ padding: 16 }}>
      <div className="card-title" style={{ marginBottom: 12 }}>
        Leaderboard
        <span style={{ fontSize: 12, color: 'var(--text-muted)', marginLeft: 8 }}>
          {rawEntries.length} entries
        </span>
      </div>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
        Ranked candidates that survived screening, investigation, or validation. Higher-tier rows have stronger evidence
        that the architecture is robust, novel, and competitive with transformer baselines.
      </p>
      <p style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 10, lineHeight: 1.5 }}>
        Curated decision view: use this tab for promote/investigate/validate decisions. For broad raw survivor browsing,
        use Programs (Raw).
      </p>
      <p style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 10 }}>
        Last updated: {lastUpdated ? lastUpdated.toLocaleTimeString() : 'loading'} · Source: /api/leaderboard
      </p>
      <p style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 10 }}>
        Glossary: S.Loss = screening loss ratio, I.Loss = investigation loss ratio, V.Loss = validation loss ratio, V.Base {'<'} 1 means better than baseline.
      </p>

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
          style={{ marginLeft: 'auto', fontSize: 11, padding: '4px 10px', cursor: 'pointer' }}
        >
          Refresh
        </button>
      </div>

      {error && (
        <p style={{ color: 'var(--accent-red)', fontSize: 13, marginBottom: 8 }}>{error}</p>
      )}
      {actionError && (
        <p style={{ color: 'var(--accent-red)', fontSize: 13, marginBottom: 8 }}>{actionError}</p>
      )}

      {loading ? (
        <p style={{ color: 'var(--text-muted)' }}>Loading leaderboard...</p>
      ) : sorted.length === 0 && !error ? (
        <div style={{ color: 'var(--text-muted)', fontSize: 13, lineHeight: 1.6 }}>
          {activeTier === 'all' ? (
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
              {sorted.map((entry, i) => (
                <tr
                  key={entry.entry_id}
                  style={{
                    borderBottom: '1px solid var(--border)',
                    cursor: 'pointer',
                    background: entry.tier === 'breakthrough' ? 'rgba(63, 185, 80, 0.08)' : undefined,
                  }}
                  onClick={() => onSelectProgram && onSelectProgram(entry.result_id)}
                >
                  <td style={tdStyle}>{i + 1}</td>
                  <td style={{ ...tdStyle, fontWeight: 600, color: scoreColor(entry._score) }}>
                    <span title={Object.entries(entryScoreBreakdown(entry)).map(([k, v]) => `${k} ${Number(v || 0).toFixed(1)}`).join(' | ')}>
                      {entry._score}
                    </span>
                  </td>
                  <td style={tdStyle}><TierBadge tier={entry.tier} /></td>
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
                  <td style={tdStyle} onClick={e => e.stopPropagation()}>
                    {entry.tier === 'screening' && (
                      <button
                        onClick={() => handleInvestigate([entry.result_id])}
                        style={actionBtnStyle}
                        title="Deep study with multiple training programs"
                      >
                        Investigate
                      </button>
                    )}
                    {entry.tier === 'investigation' && entry.investigation_passed && (
                      <button
                        onClick={() => handleValidate([entry.result_id])}
                        style={{ ...actionBtnStyle, borderColor: 'var(--accent-purple)', color: 'var(--accent-purple)' }}
                        title="Publication-grade multi-seed validation"
                      >
                        Validate
                      </button>
                    )}
                  </td>
                </tr>
              ))}
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
  padding: '2px 8px',
  fontSize: 11,
  border: '1px solid var(--accent-blue)',
  borderRadius: 4,
  background: 'transparent',
  color: 'var(--accent-blue)',
  cursor: 'pointer',
};

export default Leaderboard;
