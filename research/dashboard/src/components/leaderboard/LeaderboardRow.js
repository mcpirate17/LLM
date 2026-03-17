import React from 'react';
import { reliabilityColor } from '../../utils/colors';
import { candidateScore, TIER_COLORS, TIER_ORDER } from '../../utils/scoringEngine';
import TierBadge, { decisionGate } from '../shared/TierBadge';
import StatusBadge from '../shared/StatusBadge';
import Sparkline from '../shared/Sparkline';
import { compressionSummary } from '../report/reportUtils';
import ScoreBreakdown from './ScoreBreakdown';
import { metricChips, qualityFlags, reproducibilityPacketStatus, candidateEligibility } from './leaderboardUtils';
import { apiCall } from '../../services/apiService';

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

const fmt = (v, d = 4) => {
  if (v == null) return '--';
  const num = Number(v);
  if (num !== 0 && Math.abs(num) < 0.0001) return num.toExponential(2);
  return num.toFixed(d);
};

export const LeaderboardRow = React.memo(({
  entry,
  index,
  visibleColumns,
  isHighlighted,
  highlightRef,
  isQueued,
  isExpanded,
  onSelect,
  onTogglePin,
  onToggleExpand,
  onInvestigate,
  onValidate,
  onOpenInDesigner,
  onQueueAdd,
  onQueueRemove,
  onDelete,
  eligibilityFromParent
}) => {
  const gate = decisionGate(entry);
  const compression = visibleColumns.includes('_compression_ratio') ? (entry._compression_summary || compressionSummary(entry)) : null;
  const chips = visibleColumns.includes('_metric_quality') ? metricChips(entry) : [];
  const flags = visibleColumns.includes('_metric_quality') ? qualityFlags(entry) : [];
  const reproPacket = visibleColumns.includes('_metric_quality') ? reproducibilityPacketStatus(entry) : { label: '--' };
  const eligibility = eligibilityFromParent || candidateEligibility(entry);
  
  const queueIntent = eligibility.validationEligible ? 'validation' : (eligibility.investigationEligible ? 'investigation' : null);
  const rowId = entry.entry_id || entry.result_id || index;

  const hasBeenInvestigated = entry.investigation_loss_ratio != null || ['investigation', 'validation', 'breakthrough'].includes(entry.tier);
  const hasBeenValidated = entry.validation_loss_ratio != null || ['validation', 'breakthrough'].includes(entry.tier);
  const canDelete = !entry.is_reference && (entry.tier === 'screening' || entry.tier === 'failed' || entry.tier === 'rejected' || entry.screening_passed === false || entry.investigation_passed === false || entry.validation_passed === false);

  const handleActionClick = (e, action) => {
    e.stopPropagation();
    action();
  };

  return (
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
      onClick={() => onSelect(entry.result_id)}
    >
      <td style={tdStyle}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <button
            onClick={(e) => handleActionClick(e, () => onTogglePin(entry.entry_id, entry.is_pinned))}
            aria-label={entry.is_pinned ? 'Unpin from top' : 'Pin to top'}
            aria-pressed={Boolean(entry.is_pinned)}
            title={entry.is_pinned ? 'Unpin from top' : 'Pin to top'}
            style={{
              background: 'none',
              border: 'none',
              padding: '2px 3px',
              cursor: 'pointer',
              fontSize: 14,
              lineHeight: 1,
              color: entry.is_pinned ? 'var(--accent-yellow)' : 'var(--text-muted)',
              opacity: entry.is_pinned ? 1 : 0.35,
              transition: 'opacity 0.1s, color 0.1s',
            }}
          >
            {entry.is_pinned ? '\u2605' : '\u2606'}
          </button>
          <span style={{ tabularNums: true }}>{index + 1}</span>
          {(entry.model_source === 'reference' || entry.is_reference) && (
            <span style={{ color: 'var(--accent-purple)', fontSize: 12, marginLeft: 2 }} title="Reference Architecture">★</span>
          )}
        </div>
      </td>
      {visibleColumns.map(colKey => {
        switch (colKey) {
          case '_score':
            return <td key={colKey} style={tdStyle}><ScoreBreakdown entry={entry} /></td>;
          case 'tier':
            return <td key={colKey} style={tdStyle}><TierBadge tier={entry.tier} entry={entry} /></td>;
          case '_verified': {
            const tags = (entry.tags || '').toLowerCase();
            const isRef = entry.is_reference;
            const hasTiktoken = tags.includes('tiktoken_native') || isRef;
            const hasWikitext = tags.includes('wikitext103') || isRef;
            let vLabel, vColor;
            if (hasTiktoken && hasWikitext) { vLabel = '\u2713'; vColor = 'var(--accent-green)'; }
            else if (hasTiktoken) { vLabel = '\u26A0'; vColor = 'var(--accent-yellow)'; }
            else { vLabel = '\u2717'; vColor = 'var(--accent-red)'; }
            return <td key={colKey} style={tdStyle}><span style={{ fontSize: 12, color: vColor, fontWeight: 700 }}>{vLabel}</span></td>;
          }
          case '_rate': {
            const rate = entry.loss_improvement_rate;
            if (rate == null) return <td key={colKey} style={tdStyle}><span style={{ color: 'var(--text-muted)', fontSize: 10 }}>?</span></td>;
            const pct = (rate * 100).toFixed(1);
            const rColor = rate > 0.10 ? 'var(--accent-green)' : rate > 0.05 ? 'var(--accent-yellow)' : 'var(--accent-red)';
            return <td key={colKey} style={tdStyle}><span style={{ fontSize: 11, color: rColor, fontWeight: 600 }}>{pct}%</span></td>;
          }
          case '_gap': {
            const gap = entry.gap_vs_gpt2;
            if (gap == null) return <td key={colKey} style={tdStyle}><span style={{ color: 'var(--text-muted)', fontSize: 10 }}>--</span></td>;
            const gColor = gap < 0 ? 'var(--accent-green)' : gap < 0.1 ? 'var(--accent-yellow)' : 'var(--accent-red)';
            return <td key={colKey} style={tdStyle}><span style={{ fontSize: 11, color: gColor, fontWeight: 600 }}>{gap > 0 ? '+' : ''}{gap.toFixed(2)}</span></td>;
          }
          case '_stability': {
            const s = entry.cross_run_stability || {};
            const trend = s.trend || 'unknown';
            const sColor = trend === 'up' ? 'var(--accent-green)' : trend === 'down' ? 'var(--accent-red)' : trend === 'stable' ? 'var(--accent-yellow)' : 'var(--text-muted)';
            return (
              <td key={colKey} style={tdStyle}>
                <span style={{ fontSize: 10, fontWeight: 600, textTransform: 'uppercase', padding: '2px 6px', borderRadius: 4, color: sColor, background: `${sColor}22`, border: `1px solid ${sColor}55` }}>{trend}</span>
              </td>
            );
          }
          case 'model_source':
            return (
              <td key={colKey} style={tdStyle}>
                {entry.is_reference && <span style={{ fontSize: 10, color: 'var(--accent-purple)', border: '1px solid var(--accent-purple)', borderRadius: 4, padding: '1px 6px', marginRight: 6 }}>📌 REF</span>}
                <span style={{ fontSize: 10, color: entry.model_source === 'morphological_box' ? 'var(--accent-purple)' : 'var(--accent-blue)' }}>
                  {entry.model_source === 'reference' ? 'REF' : entry.model_source === 'morphological_box' ? 'MORPH' : 'GRAPH'}
                </span>
              </td>
            );
          case 'architecture_family':
            return <td key={colKey} style={tdStyle}>{entry.architecture_family || '--'}</td>;
          case 'architecture_desc':
            return <td key={colKey} style={{ ...tdStyle, maxWidth: 150, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{entry.reference_name || entry.architecture_desc || entry.result_id?.slice(0, 12)}</td>;
          case '_composition': {
            const templates = entry.applied_templates || [];
            if (templates.length === 0) return <td key={colKey} style={tdStyle}><span style={{ color: 'var(--text-muted)', fontSize: 10 }}>--</span></td>;
            return (
              <td key={colKey} style={tdStyle}>
                <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
                  {templates.slice(0, 3).map((t, i) => (
                    <span key={i} title={t.name} style={{ fontSize: 9, padding: '1px 4px', borderRadius: 3, background: 'rgba(88, 166, 255, 0.15)', color: 'var(--accent-blue)', border: '1px solid rgba(88, 166, 255, 0.3)' }}>
                      {t.name?.replace(/_template$/, '').replace(/apply_/, '')}
                    </span>
                  ))}
                  {templates.length > 3 && <span style={{ fontSize: 9, color: 'var(--text-muted)' }}>+{templates.length - 3}</span>}
                </div>
              </td>
            );
          }
          case '_vs_reference':
            return <td key={colKey} style={tdStyle}>{entry._vs_reference != null ? `${entry._vs_reference.toFixed(0)}%` : '--'}</td>;
          case 'composite_score': {
            const tags = (entry.tags || '').toLowerCase();
            const hasTiktoken = tags.includes('tiktoken_native') || entry.is_reference;
            const tokIcon = hasTiktoken ? '\u2713' : '\u26A0';
            const tokColor = hasTiktoken ? 'var(--accent-green)' : 'var(--accent-yellow)';
            return (
              <td key={colKey} style={{ ...tdStyle, color: 'var(--accent-green)' }}>
                {fmt(entry.composite_score, 3)}
                <span style={{ fontSize: 10, marginLeft: 4, color: tokColor }} title={hasTiktoken ? 'tiktoken-native' : 'byte-era'}>{tokIcon}</span>
              </td>
            );
          }
          case 'discovery_loss_ratio': return <td key={colKey} style={tdStyle}>{fmt(entry.discovery_loss_ratio)}</td>;
          case 'validation_loss_ratio': return <td key={colKey} style={tdStyle}>{fmt(entry.validation_loss_ratio)}</td>;
          case 'moe_routing_efficiency': return <td key={colKey} style={tdStyle}>{fmt(entry.moe_routing_efficiency, 3)}</td>;
          case 'arch_quality_score': return <td key={colKey} style={tdStyle}><span style={{ color: entry.arch_quality_score > 0.7 ? 'var(--accent-green)' : (entry.arch_quality_score < 0.4 ? 'var(--accent-red)' : 'var(--text-primary)') }}>{fmt(entry.arch_quality_score, 3)}</span></td>;
          case 'screening_loss_ratio': return <td key={colKey} style={tdStyle}>{fmt(entry.screening_loss_ratio)}</td>;
          case 'screening_novelty': return <td key={colKey} style={tdStyle}>{fmt(entry.screening_novelty, 3)}</td>;
          case 'investigation_loss_ratio': return <td key={colKey} style={tdStyle}>{fmt(entry.investigation_loss_ratio)}</td>;
          case 'investigation_robustness':
            return <td key={colKey} style={tdStyle}><span style={{ color: entry.investigation_robustness >= 0.5 ? 'var(--accent-green)' : 'var(--accent-red)' }}>{fmt(entry.investigation_robustness, 2)}</span></td>;
          case 'wikitext_ppl':
            return <td key={colKey} style={tdStyle}><span style={{ color: 'var(--accent-blue)', fontWeight: 600 }}>{fmt(entry.wikitext_ppl ?? entry.wikitext_perplexity, 2)}</span></td>;
          case 'peak_ppl':
            return <td key={colKey} style={tdStyle}><span style={{ color: 'var(--accent-cyan)', fontWeight: 600 }}>{fmt(entry.peak_ppl, 2)}</span></td>;
          case 'divergence_step':
            return <td key={colKey} style={tdStyle}>{entry.divergence_step || '--'}</td>;
          case 'wikitext_ppl_trajectory': {
            const data = Array.isArray(entry.wikitext_ppl_trajectory) ? entry.wikitext_ppl_trajectory : 
                         (typeof entry.wikitext_ppl_trajectory === 'string' ? entry.wikitext_ppl_trajectory.split(',').map(v => parseFloat(v.trim())) : null);
            return (
              <td key={colKey} style={tdStyle}>
                <Sparkline data={data} />
              </td>
            );
          }
          case 'evaluation_stage':
            return (
              <td key={colKey} style={tdStyle}>
                <div style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
                  <span style={{ fontSize: 10, fontWeight: 700, color: 'var(--text-secondary)' }}>{entry.evaluation_stage || '--'}</span>
                  {entry.is_frontier_signal && <StatusBadge type="FRONTIER_SIGNAL" label="FRONTIER" title="Model beats reference PPL at equal budget" />}
                  {(entry.improvement_ratio > 2.0 || entry.is_slow_burn) && <StatusBadge type="SLOW_BURN" label="SLOW-BURN" title="Sharp trajectory: PPL improved >2x" />}
                  {entry.divergence_step && <StatusBadge type="DIVERGED" label={`DIVERGED @ ${entry.divergence_step}`} title="PPL diverged >2x peak" />}
                  {!entry.divergence_step && entry.wikitext_ppl_trajectory?.length >= 4 && <StatusBadge type="STABLE_GENERALIZER" label="STABLE" title="No divergence seen in 4000 steps" />}
                </div>
              </td>
            );
          case 'robustness_grade': {
            const grade = entry.robustness_grade || (entry.investigation_robustness >= 0.8 ? 'A' : entry.investigation_robustness >= 0.5 ? 'B' : entry.investigation_robustness != null ? 'C' : null);
            const statusType = grade === 'A' ? 'ROBUST' : grade === 'B' ? 'STABLE' : grade === 'C' ? 'FRAGILE' : null;
            return (
              <td key={colKey} style={tdStyle}>
                {statusType && <StatusBadge type={statusType} label={grade} title={`Robustness Grade ${grade}`} />}
              </td>
            );
          }
          case 'validation_baseline_ratio':
            return <td key={colKey} style={tdStyle}><span style={{ color: entry.validation_baseline_ratio < 1 ? 'var(--accent-green)' : 'var(--accent-red)' }}>{fmt(entry.validation_baseline_ratio)}</span></td>;
          case 'robustness_noise_score': return <td key={colKey} style={tdStyle}>{fmt(entry.robustness_noise_score, 3)}</td>;
          case 'quant_int8_retention': return <td key={colKey} style={tdStyle}>{entry._quant_retention_pct != null ? `${entry._quant_retention_pct.toFixed(1)}%` : '--'}</td>;
          case 'robustness_long_ctx_score': return <td key={colKey} style={tdStyle}>{fmt(entry.robustness_long_ctx_score, 3)}</td>;
          case 'robustness_long_ctx_scaling_score': return <td key={colKey} style={tdStyle}>{fmt(entry.robustness_long_ctx_scaling_score, 3)}</td>;
          case 'robustness_long_ctx_assoc_score': return <td key={colKey} style={tdStyle}>{fmt(entry.robustness_long_ctx_assoc_score, 3)}</td>;
          case 'robustness_long_ctx_multi_hop_score': return <td key={colKey} style={tdStyle}>{fmt(entry.robustness_long_ctx_multi_hop_score, 3)}</td>;
          case 'robustness_long_ctx_passkey_score': return <td key={colKey} style={tdStyle}>{fmt(entry.robustness_long_ctx_passkey_score, 3)}</td>;
          case 'robustness_long_ctx_retrieval_aggregate': return <td key={colKey} style={tdStyle}>{fmt(entry.robustness_long_ctx_retrieval_aggregate, 3)}</td>;
          case 'robustness_long_ctx_combined_score': return <td key={colKey} style={tdStyle}>{fmt(entry.robustness_long_ctx_combined_score, 3)}</td>;
          case 'max_viable_seq_len': return <td key={colKey} style={tdStyle}>{entry.max_viable_seq_len != null ? Number(entry.max_viable_seq_len).toFixed(0) : '--'}</td>;
          case 'init_sensitivity_std': return <td key={colKey} style={tdStyle}>{fmt(entry.init_sensitivity_std, 4)}</td>;
          case 'pre_inv_score': {
            const pis = Number(entry.pre_inv_score);
            const pisColor = pis >= 50 ? 'var(--accent-green)' : pis >= 20 ? 'var(--accent-yellow)' : 'var(--accent-red)';
            return <td key={colKey} style={{ ...tdStyle, color: pisColor, fontWeight: 600 }}>{fmt(pis, 1)}</td>;
          }
          case 'jacobian_spectral_norm': return <td key={colKey} style={tdStyle}>{fmt(entry.jacobian_spectral_norm ?? entry.fp_jacobian_spectral_norm, 4)}</td>;
          case '_compression_ratio':
            return (
              <td key={colKey} style={tdStyle}>
                <div style={{ fontSize: 11 }}>{compression.ratio != null ? `${(compression.ratio * 100).toFixed(0)}%` : '--'}</div>
                <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>{compression.memoryMb?.toFixed(2)} MB · {compression.label}</div>
              </td>
            );
          case '_metric_quality':
            return (
              <td key={colKey} style={tdStyle}>
                <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>Q: {reproPacket.label} · R: {reproPacket.label}</div>
                {isExpanded && (
                  <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', marginTop: 4 }}>
                    {chips.map(c => <span key={c.label} style={{ fontSize: 10, padding: '1px 5px', borderRadius: 4, background: `${reliabilityColor(c.reliability)}22`, color: reliabilityColor(c.reliability) }}>{c.label}</span>)}
                  </div>
                )}
              </td>
            );
          case '_actions':
            return (
              <td key={colKey} style={tdStyle} onClick={e => e.stopPropagation()}>
                <div style={{ display: 'flex', gap: 4, flexWrap: 'nowrap' }}>
                  {!hasBeenInvestigated && (
                    <button
                      onClick={(e) => handleActionClick(e, () => onInvestigate([entry.result_id]))}
                      style={{
                        ...actionBtnStyle,
                        opacity: eligibility.investigationEligible ? 1 : 0.6,
                        borderStyle: eligibility.investigationEligible ? 'solid' : 'dashed',
                      }}
                      aria-label={
                        eligibility.investigationEligible
                          ? `Investigate ${entry.result_id}`
                          : `Force investigate ${entry.result_id} (currently ineligible)`
                      }
                      title={
                        eligibility.investigationEligible
                          ? 'Run investigation stage'
                          : 'Not yet eligible — click to override and force-investigate'
                      }
                    >
                      Investigate
                    </button>
                  )}
                  {!hasBeenValidated && (
                    <button
                      onClick={(e) => handleActionClick(e, () => onValidate([entry.result_id]))}
                      style={{
                        ...actionBtnStyle,
                        opacity: eligibility.validationEligible ? 1 : 0.6,
                        borderStyle: eligibility.validationEligible ? 'solid' : 'dashed',
                      }}
                      aria-label={
                        eligibility.validationEligible
                          ? `Validate ${entry.result_id}`
                          : `Force validate ${entry.result_id} (currently ineligible)`
                      }
                      title={
                        eligibility.validationEligible
                          ? 'Run validation stage'
                          : 'Not yet eligible — click to override and force-validate'
                      }
                    >
                      Validate
                    </button>
                  )}
                  <button
                    onClick={(e) => handleActionClick(e, () => onToggleExpand(rowId))}
                    style={actionBtnStyle}
                    aria-expanded={isExpanded}
                    aria-label={isExpanded ? 'Hide expanded details' : 'Show details'}
                  >
                    {isExpanded ? 'Hide' : 'Details'}
                  </button>
                  {canDelete && (
                    <button
                      onClick={(e) => handleActionClick(e, () => {
                        if (window.confirm(`Delete entry ${entry.entry_id?.slice(0, 11)} and all associated data? This cannot be undone.`)) {
                          onDelete(entry.entry_id);
                        }
                      })}
                      style={{
                        ...actionBtnStyle,
                        borderColor: 'rgba(248, 81, 73, 0.4)',
                        background: 'rgba(248, 81, 73, 0.12)',
                        color: 'var(--accent-red, #f85149)',
                      }}
                      title="Delete entry and all associated data"
                    >
                      Delete
                    </button>
                  )}
                </div>
              </td>
            );
          default: return null;
        }
      })}
    </tr>
  );
});

export default LeaderboardRow;
