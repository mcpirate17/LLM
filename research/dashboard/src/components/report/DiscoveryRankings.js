import React, { useState, useEffect, useMemo } from 'react';
import useCopyToClipboard from '../../hooks/useCopyToClipboard';
import { discoveryScore, discoveryScoreBreakdown, promotionEvidence } from '../../utils/scoringEngine';
import { scoreColor } from '../../utils/format';
import { reliabilityColor } from '../../utils/colors';
import RatingBadge from './RatingBadge';
import { filterRowsByQuery } from '../../utils/tableFiltering';
import {
  compressionSummary, metricChips, qkvUsageDescriptor,
  resolveLossRatio,
  decisionGate, reproducibilityPacketStatus,
  DISC_COLUMNS, DISC_RATING_ORDER,
  REPORT_DISCOVERY_SORT_PREFS_KEY, REPORT_DISCOVERY_VIEW_PREFS_KEY,
  reportQueueReasonLabel,
} from './reportUtils';

export default function DiscoveryRankings({
  programs,
  expandedPrograms,
  onSelectProgram,
  onInvestigate,
  onValidate,
  onQueueAdd,
  onQueueRemove,
  queuedResultIds,
  eligibilityByResultId,
  onOpenInDesigner,
}) {
  const [viewMode, setViewMode] = useState(() => {
    try {
      const stored = localStorage.getItem(REPORT_DISCOVERY_VIEW_PREFS_KEY);
      return stored === 'expanded' ? 'expanded' : 'grouped';
    } catch {}
    return 'grouped';
  });
  const [sortKey, setSortKey] = useState(() => {
    try {
      const stored = JSON.parse(localStorage.getItem(REPORT_DISCOVERY_SORT_PREFS_KEY) || '{}');
      const validKeys = new Set([...DISC_COLUMNS.map((column) => column.key), '_ratingOrder']);
      if (typeof stored.sortKey === 'string' && validKeys.has(stored.sortKey)) {
        return stored.sortKey;
      }
    } catch {}
    return '_score';
  });
  const [sortDesc, setSortDesc] = useState(() => {
    try {
      const stored = JSON.parse(localStorage.getItem(REPORT_DISCOVERY_SORT_PREFS_KEY) || '{}');
      if (typeof stored.sortDesc === 'boolean') {
        return stored.sortDesc;
      }
    } catch {}
    return true;
  });
  const [filterQuery, setFilterQuery] = useState('');
  const [copiedValue, copyText] = useCopyToClipboard();
  const queuedSet = useMemo(() => new Set(queuedResultIds || []), [queuedResultIds]);

  useEffect(() => {
    try {
      localStorage.setItem(REPORT_DISCOVERY_SORT_PREFS_KEY, JSON.stringify({ sortKey, sortDesc }));
    } catch {}
  }, [sortKey, sortDesc]);

  useEffect(() => {
    try {
      localStorage.setItem(REPORT_DISCOVERY_VIEW_PREFS_KEY, viewMode);
    } catch {}
  }, [viewMode]);

  const groupedRows = Array.isArray(programs) ? programs : [];
  const expandedRows = Array.isArray(expandedPrograms) && expandedPrograms.length > 0
    ? expandedPrograms
    : groupedRows;
  const isExpanded = viewMode === 'expanded';
  const sourceRows = isExpanded ? expandedRows : groupedRows;

  const groupedUnique = groupedRows.length;
  const expandedTotal = expandedRows.length;
  const rerunRows = expandedRows.filter(p => Number(p.group_repeat_count || p.repeat_count || 1) > 1).length;
  const rerunRatio = expandedTotal > 0 ? Math.round((rerunRows / expandedTotal) * 100) : 0;

  const sortAriaValue = (columnKey) => {
    const normalized = columnKey === 'rating' ? '_ratingOrder' : columnKey;
    if (sortKey !== normalized) return 'none';
    return sortDesc ? 'descending' : 'ascending';
  };

  const handleSort = (key) => {
    if (key === 'rating') key = '_ratingOrder';
    if (sortKey === key) setSortDesc(!sortDesc);
    else { setSortKey(key); setSortDesc(true); }
  };

  const filtered = useMemo(() => (
    filterRowsByQuery(sourceRows, filterQuery, [
      'graph_fingerprint',
      'result_id',
      'display_name',
      'architecture_family',
      'most_similar_to',
    ])
  ), [sourceRows, filterQuery]);

  const sorted = useMemo(() => {
    const aug = filtered.map(p => {
      const repeatCount = Number(p.repeat_count || p.group_repeat_count || 1);
      const repeatIndex = Number(p.group_repeat_index || 1);
      const lr = resolveLossRatio(p) ?? p.loss_ratio;
      const nov = p.novelty_score || 0;
      const bl = p.baseline_loss_ratio;
      let rLabel;
      if (bl != null && bl < 1 && lr < 0.5 && nov > 0.7) rLabel = 'S1 - Exceptional';
      else if (lr < 0.5 && nov > 0.5) rLabel = 'S1 - Strong';
      else if (lr < 0.7) rLabel = 'S1 - Moderate';
      else rLabel = 'S1 - Marginal';
      const gate = decisionGate(p);
      const compression = compressionSummary(p);
      const chips = metricChips(p);
      const promotion = promotionEvidence(p);
      const reproPacket = reproducibilityPacketStatus(p);
      const qkv = qkvUsageDescriptor(p);
      return {
        ...p,
        repeat_count: repeatCount,
        group_repeat_count: repeatCount,
        group_repeat_index: repeatIndex,
        _score: discoveryScore(p),
        _scoreBreakdown: discoveryScoreBreakdown(p),
        _ratingOrder: DISC_RATING_ORDER[rLabel] || 0,
        _compressionRatio: compression.ratio ?? -1,
        _compressionSummary: compression,
        _metricQuality: chips,
        _metricQualityOrder: chips.filter(ch => ch.reliability === 'high').length,
        _decisionGateOrder: gate.decisionReady ? 1 : 0,
        _promotionEvidence: promotion,
        _reproPacket: reproPacket,
        _qkvDescriptor: qkv,
      };
    });
    aug.sort((a, b) => {
      let va, vb;
      if (sortKey === 'graph_fingerprint' || sortKey === 'most_similar_to') {
        va = a[sortKey] || ''; vb = b[sortKey] || '';
        return sortDesc ? vb.localeCompare(va) : va.localeCompare(vb);
      }
      va = a[sortKey]; vb = b[sortKey];
      if (va == null && vb == null) return 0;
      if (va == null) return 1;
      if (vb == null) return -1;
      return sortDesc ? vb - va : va - vb;
    });
    return aug;
  }, [filtered, sortKey, sortDesc]);

  return (
    <div className="card">
      <div className="card-title" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12 }}>
        <span>Discovery Rankings</span>
        <input
          value={filterQuery}
          onChange={(e) => setFilterQuery(e.target.value)}
          placeholder="Filter fingerprints / names"
          style={{
            fontSize: 11,
            padding: '4px 8px',
            borderRadius: 4,
            border: '1px solid var(--border)',
            background: 'var(--bg-tertiary)',
            color: 'var(--text-primary)',
            minWidth: 200,
          }}
        />
      </div>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
        The strongest architectures discovered, ranked by a composite of learning speed, novelty, and baseline comparison.
        Higher score is better and is meant for triage (not a publication-grade metric).
      </p>
      <p style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
        Rankings are fingerprint-deduplicated: each row is one architecture identity (`graph_fingerprint`) with repeat/run-spread metadata.
      </p>
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: 8,
        flexWrap: 'wrap',
        marginBottom: 12,
      }}>
        <div style={{ fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.5 }}>
          Same architecture repeated means reruns of one fingerprint. Grouped view shows one representative per fingerprint; expanded view shows every rerun row.
          {expandedTotal > 0 && (
            <span> Current mix: {groupedUnique} unique architectures across {expandedTotal} rows ({rerunRatio}% reruns).</span>
          )}
        </div>
        <div style={{ display: 'inline-flex', gap: 6 }}>
          <button
            className="refresh-btn"
            style={{ fontSize: 11, padding: '4px 8px', opacity: isExpanded ? 0.8 : 1 }}
            onClick={() => setViewMode('grouped')}
            aria-pressed={!isExpanded}
          >
            Grouped view
          </button>
          <button
            className="refresh-btn"
            style={{ fontSize: 11, padding: '4px 8px', opacity: isExpanded ? 1 : 0.8 }}
            onClick={() => setViewMode('expanded')}
            aria-pressed={isExpanded}
          >
            Expanded reruns
          </button>
        </div>
      </div>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
        <strong>Score bands:</strong> 70+ strong follow-up, 40-69 promising, below 40 low priority. Click a fingerprint to open full program detail.
      </p>
      <p style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
        Decision gate: mark as <strong>Decision-Ready</strong> only when screening metrics are present, baseline ratio is {'<'} 1.00, and CKA source is artifact-backed.
      </p>
      <p style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
        Compression column shows estimated parameter ratio vs dense baseline and memory footprint; Metric Quality chips show provenance (`artifact-backed`/`heuristic`) and reliability.
      </p>
      <p style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
        QKV alternative labels are candidate-level: <strong>Full QKV</strong>, <strong>Q=K=V</strong>, or <strong>QKV-free</strong>.
      </p>
      <div style={{ overflowX: 'auto' }}>
        <table className="data-table table-compact">
          <thead>
            <tr style={{ borderBottom: '1px solid var(--border)', textAlign: 'left' }}>
              <th scope="col" style={{ padding: '8px 6px', color: 'var(--text-muted)' }}>#</th>
              {DISC_COLUMNS.map(col => (
                <th
                  key={col.key}
                  onClick={() => handleSort(col.key)}
                  scope="col"
                  aria-sort={sortAriaValue(col.key)}
                  aria-label={`Sort by ${col.label}`}
                  style={{ padding: '8px 6px', color: 'var(--text-muted)', cursor: 'pointer', userSelect: 'none', whiteSpace: 'nowrap' }}
                >
                  {col.label}
                  {(sortKey === col.key || (col.key === 'rating' && sortKey === '_ratingOrder')) && (
                    <span style={{ marginLeft: 4, fontSize: 10 }}>
                      {sortDesc ? '\u25BC' : '\u25B2'}
                    </span>
                  )}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sorted.map((p, i) => {
              const gate = decisionGate(p);
              const lrRaw = resolveLossRatio(p) ?? p.loss_ratio;
              const lr = Number.isFinite(Number(lrRaw)) ? Number(lrRaw) : null;
              const eligibility = (p.result_id && eligibilityByResultId?.[p.result_id]) || {
                investigationEligible: false,
                validationEligible: false,
                queueEligible: false,
                queueReason: 'not_progression_eligible',
              };
              const queueIntent = eligibility.validationEligible
                ? 'validation'
                : eligibility.investigationEligible
                  ? 'investigation'
                  : null;
              return (
              <tr key={p.result_id || i} style={{ borderBottom: '1px solid var(--border)' }}>
                <td style={{ padding: '6px', color: 'var(--text-muted)' }}>{i + 1}</td>
                <td style={{ padding: '6px', fontWeight: 600, color: scoreColor(p._score) }}>
                  <span title={`Loss ${(p._scoreBreakdown.loss || 0)}% | Novelty ${(p._scoreBreakdown.novelty || 0)}% | Baseline ${(p._scoreBreakdown.baseline || 0)}% | ID ${(p._scoreBreakdown.id || 0)}% | ParamEff ${(p._scoreBreakdown.paramEfficiency || 0)}% | Speed ${(p._scoreBreakdown.learningSpeed || 0)}% | Efficiency ${(p._scoreBreakdown.efficiencyBonus || 0).toFixed(1)} | Routing ${(p._scoreBreakdown.routingBonus || 0).toFixed(1)} | Adaptive ${(p._scoreBreakdown.adaptiveBonus || 0).toFixed(1)}`}>
                    {p._score}
                  </span>
                </td>
                <td style={{ padding: '6px' }}>
                  {p.result_id && onSelectProgram ? (
                    <>
                      <button
                        className="refresh-btn"
                        style={{ fontSize: 11, padding: '3px 8px', fontFamily: 'monospace' }}
                        onClick={() => onSelectProgram(p.result_id)}
                        aria-label={`Open program details for fingerprint ${(p.graph_fingerprint || '').slice(0, 12)}`}
                      >
                        {(p.graph_fingerprint || '').slice(0, 12)}
                      </button>
                      {p.graph_fingerprint && (
                        <button
                          className="refresh-btn"
                          style={{ fontSize: 10, padding: '1px 5px', marginLeft: 6 }}
                          onClick={() => copyText(p.graph_fingerprint)}
                          aria-label={`Copy fingerprint ${p.graph_fingerprint}`}
                        >
                          {copiedValue === p.graph_fingerprint ? 'Copied FP' : 'Copy FP'}
                        </button>
                      )}
                      {p.result_id && (
                        <button
                          className="refresh-btn"
                          style={{ fontSize: 10, padding: '1px 5px', marginLeft: 4 }}
                          onClick={() => copyText(p.result_id)}
                          aria-label={`Copy result id ${p.result_id}`}
                        >
                          {copiedValue === p.result_id ? 'Copied ID' : 'Copy ID'}
                        </button>
                      )}
                    </>
                  ) : (
                    <span style={{ fontFamily: 'monospace', color: 'var(--accent-blue)' }}>
                      {(p.graph_fingerprint || '').slice(0, 12)}
                    </span>
                  )}
                  {(p.repeat_count || 1) > 1 && (
                    <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 4 }}>
                      {isExpanded
                        ? `rerun ${p.group_repeat_index || 1} of ${p.group_repeat_count || p.repeat_count || 1}`
                        : `repeated ${p.repeat_count}x across ${p.repeat_experiment_span || 1} run${(p.repeat_experiment_span || 1) === 1 ? '' : 's'}`}
                    </div>
                  )}
                  {(p.repeat_loss_min != null || p.repeat_loss_max != null) && (
                    <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                      loss spread {p.repeat_loss_min != null ? p.repeat_loss_min.toFixed(4) : '--'} to {p.repeat_loss_max != null ? p.repeat_loss_max.toFixed(4) : '--'}
                    </div>
                  )}
                </td>
                <td style={{ padding: '6px' }}>
                  <span style={{ color: (p.repeat_count || 1) > 1 ? 'var(--accent-yellow)' : 'var(--text-muted)', fontWeight: 600 }}>
                    {p.repeat_count || 1}x
                  </span>
                  <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                    {isExpanded
                      ? `row ${p.group_repeat_index || 1} in fingerprint group`
                      : `span ${p.repeat_experiment_span || 1} run${(p.repeat_experiment_span || 1) === 1 ? '' : 's'}`}
                  </div>
                </td>
                <td style={{
                  padding: '6px', fontWeight: 600,
                  color: (lr || 1) < 0.5 ? 'var(--accent-green)' : (lr || 1) < 0.7 ? 'var(--accent-yellow)' : 'var(--text-secondary)',
                }}>
                  {lr != null ? lr.toFixed(4) : '--'}
                </td>
                <td style={{ padding: '6px', color: (p.novelty_score || 0) > 0.7 ? 'var(--accent-green)' : 'var(--text-secondary)' }}>
                  {p.novelty_score != null ? p.novelty_score.toFixed(3) : '--'}
                </td>
                <td style={{
                  padding: '6px',
                  color: p.baseline_loss_ratio != null && p.baseline_loss_ratio < 1 ? 'var(--accent-green)' : 'var(--text-secondary)',
                  fontWeight: p.baseline_loss_ratio != null && p.baseline_loss_ratio < 1 ? 600 : 'normal',
                }}>
                  {p.baseline_loss_ratio != null ? p.baseline_loss_ratio.toFixed(3) : '--'}
                </td>
                <td style={{ padding: '6px' }}>
                  <div style={{ fontSize: 11, color: 'var(--text-secondary)' }}>
                    {p._compressionSummary?.ratio != null ? `${(p._compressionSummary.ratio * 100).toFixed(0)}%` : '--'}
                  </div>
                  <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                    {p._compressionSummary?.memoryMb != null ? `${p._compressionSummary.memoryMb.toFixed(2)} MB` : 'n/a'} · {p._compressionSummary?.label || 'dense'}
                  </div>
                  <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                    retention {p._compressionSummary?.qualityRetention != null ? `${(p._compressionSummary.qualityRetention * 100).toFixed(0)}%` : 'n/a'}
                  </div>
                </td>
                <td style={{ padding: '6px' }}>
                  <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', maxWidth: 220 }}>
                    {(p._metricQuality || []).map(chip => (
                      <span
                        key={`${p.result_id || i}-${chip.label}`}
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
                  <div
                    style={{ marginTop: 5, fontSize: 10, fontWeight: 600, color: p._promotionEvidence?.color || 'var(--text-muted)' }}
                    title={`Evidence checks ${p._promotionEvidence?.evidenceCount || 0}/${p._promotionEvidence?.totalChecks || 0}`}
                  >
                    Promotion confidence: {p._promotionEvidence?.label || 'Low'} ({p._promotionEvidence?.score ?? 0}%)
                  </div>
                  <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                    Uncertainty {p._promotionEvidence?.uncertaintyLabel || 'unknown'}; runs {p._promotionEvidence?.seenRuns || 0}; std {p._promotionEvidence?.std != null ? p._promotionEvidence.std.toFixed(3) : 'n/a'}
                  </div>
                  <div style={{ marginTop: 2, fontSize: 10, color: p._reproPacket?.color || 'var(--text-muted)' }}>
                    Repro packet: {p._reproPacket?.label || 'Sparse'} ({p._reproPacket?.readyCount || 0}/{p._reproPacket?.totalChecks || 0})
                  </div>
                </td>
                <td style={{ padding: '6px' }}>
                  {p.cka_source ? (
                    <span style={{
                      fontSize: 10,
                      fontWeight: 600,
                      padding: '2px 6px',
                      borderRadius: 4,
                      background: p.cka_source === 'artifact' ? 'rgba(63, 185, 80, 0.15)' : 'rgba(248, 81, 73, 0.15)',
                      color: p.cka_source === 'artifact' ? 'var(--accent-green)' : 'var(--accent-red)',
                    }}>
                      {p.cka_source === 'artifact' ? 'artifact' : 'fallback'}
                    </span>
                  ) : '--'}
                  {p.cka_artifact_version && (
                    <span style={{ marginLeft: 6, fontSize: 10, color: 'var(--text-muted)' }}>
                      {p.cka_artifact_version}
                    </span>
                  )}
                </td>
                <td style={{ padding: '6px', color: 'var(--text-muted)', fontSize: 11 }}>
                  {p.most_similar_to || '--'}
                  <div
                    style={{ marginTop: 4, fontSize: 10, color: p._qkvDescriptor?.color || 'var(--text-muted)', fontWeight: 600 }}
                    title={p._qkvDescriptor?.detail || ''}
                  >
                    {p._qkvDescriptor?.label || 'QKV unknown'}
                  </div>
                </td>
                <td style={{ padding: '6px' }}>
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
                      ? 'All report-level evidence checks passed.'
                      : `Missing checks: ${gate.missing.join(', ')}`}
                  >
                    {gate.label}
                  </span>
                </td>
                <td style={{ padding: '6px' }}>
                  {lr != null && <RatingBadge program={p} />}
                  {p.result_id && (
                    <div style={{ marginTop: 6, display: 'flex', gap: 4, flexWrap: 'wrap' }}>
                      {onInvestigate && (
                        <button
                          style={{ fontSize: 10, padding: '1px 6px', borderRadius: 4, cursor: 'pointer', background: 'rgba(63, 185, 80, 0.12)', border: '1px solid rgba(63, 185, 80, 0.4)', color: 'var(--accent-green)' }}
                          onClick={() => onInvestigate([p.result_id])}
                          aria-label={`Investigate program ${p.result_id}`}
                          title={eligibility.investigationEligible ? 'Start investigation' : 'Currently ineligible; click to force override'}
                        >
                          {eligibility.investigationEligible ? 'Investigate' : 'Force Investigate'}
                        </button>
                      )}
                      {onValidate && (
                        <button
                          style={{ fontSize: 10, padding: '1px 6px', borderRadius: 4, cursor: 'pointer', background: 'rgba(188, 140, 255, 0.12)', border: '1px solid rgba(188, 140, 255, 0.4)', color: 'var(--accent-purple)' }}
                          onClick={() => onValidate([p.result_id])}
                          aria-label={`Validate program ${p.result_id}`}
                          title={eligibility.validationEligible ? 'Start validation' : 'Currently ineligible; click to force override'}
                        >
                          {eligibility.validationEligible ? 'Validate' : 'Force Validate'}
                        </button>
                      )}
                      {onSelectProgram && (
                        <button
                          className="refresh-btn"
                          style={{
                            fontSize: 10, padding: '1px 6px',
                            borderColor: 'var(--accent-purple)',
                            color: 'var(--accent-purple)',
                          }}
                          onClick={() => onSelectProgram(p.result_id)}
                          aria-label={`Open decision packet for ${p.result_id}`}
                        >
                          Packet
                        </button>
                      )}
                      {onOpenInDesigner && (
                        <button
                          className="refresh-btn"
                          style={{
                            fontSize: 10, padding: '1px 6px',
                            borderColor: 'var(--accent-blue)',
                            color: 'var(--accent-blue)',
                          }}
                          onClick={() => onOpenInDesigner(p.result_id)}
                          aria-label={`Open program ${p.result_id} in designer`}
                          title="Open architecture in visual designer"
                        >
                          Designer
                        </button>
                      )}
                      {(onQueueAdd || onQueueRemove) && (() => {
                        const isQueued = queuedSet.has(p.result_id);
                        const queueDisabled = !isQueued && !eligibility.queueEligible;
                        return (
                          <button
                            className="refresh-btn"
                            style={{ fontSize: 10, padding: '1px 6px' }}
                            disabled={queueDisabled}
                            onClick={() => {
                              if (isQueued) {
                                onQueueRemove && onQueueRemove(p.result_id);
                              } else {
                                if (!eligibility.queueEligible) {
                                  return;
                                }
                                onQueueAdd && onQueueAdd({
                                  resultId: p.result_id,
                                  fingerprint: p.graph_fingerprint,
                                  source: 'report',
                                  architectureFamily: null,
                                  intent: queueIntent,
                                  queueEligible: eligibility.queueEligible,
                                  investigationEligible: eligibility.investigationEligible,
                                  validationEligible: eligibility.validationEligible,
                                  queueReason: eligibility.queueReason,
                                });
                              }
                            }}
                            title={isQueued
                              ? 'Remove from progression queue'
                              : queueDisabled
                                ? (p.tier === 'validation' || p.tier === 'breakthrough' 
                                    ? 'Architecture is fully validated.' 
                                    : reportQueueReasonLabel(eligibility.queueReason))
                                : queueIntent === 'validation'
                                  ? 'Add to validation queue'
                                  : 'Add to investigation queue'}
                            aria-label={`${isQueued ? 'Remove' : 'Add'} ${p.result_id} ${isQueued ? 'from' : 'to'} investigation queue`}
                          >
                            {isQueued
                              ? 'Queued'
                              : queueDisabled
                                ? (p.tier === 'validation' || p.tier === 'breakthrough' ? 'Validated' : 'Ineligible')
                                : queueIntent === 'validation'
                                  ? 'Queue Validate'
                                  : 'Queue Investigate'}
                          </button>
                        );
                      })()}
                      {!eligibility.investigationEligible && !eligibility.validationEligible && (
                        <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                          {reportQueueReasonLabel(eligibility.queueReason)}
                        </span>
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
    </div>
  );
}
