import React from 'react';
import { reliabilityColor } from '../../utils/colors';
import { candidateScore, TIER_COLORS, TIER_ORDER } from '../../utils/scoringEngine';
import TierBadge, { decisionGate } from '../shared/TierBadge';
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
  eligibilityFromParent
}) => {
  const gate = decisionGate(entry);
  const compression = entry._compression_summary || compressionSummary(entry);
  const chips = metricChips(entry);
  const flags = qualityFlags(entry);
  const reproPacket = reproducibilityPacketStatus(entry);
  const eligibility = eligibilityFromParent || candidateEligibility(entry);
  
  const queueIntent = eligibility.validationEligible ? 'validation' : (eligibility.investigationEligible ? 'investigation' : null);
  const rowId = entry.entry_id || entry.result_id || index;

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
          <span 
            onClick={(e) => handleActionClick(e, () => onTogglePin(entry.entry_id, entry.is_pinned))}
            title={entry.is_pinned ? "Unpin from top" : "Pin to top"}
            style={{ 
              cursor: 'pointer', 
              fontSize: 14, 
              color: entry.is_pinned ? 'var(--accent-yellow)' : 'var(--text-muted)',
              opacity: entry.is_pinned ? 1 : 0.3
            }}
          >
            {entry.is_pinned ? '★' : '☆'}
          </span>
          {index + 1}
        </div>
      </td>
      {visibleColumns.map(colKey => {
        switch (colKey) {
          case '_score':
            return <td key={colKey} style={tdStyle}><ScoreBreakdown entry={entry} /></td>;
          case 'tier':
            return <td key={colKey} style={tdStyle}><TierBadge tier={entry.tier} entry={entry} /></td>;
          case '_stability':
            const s = entry.cross_run_stability || {};
            const trend = s.trend || 'unknown';
            const sColor = trend === 'up' ? 'var(--accent-green)' : trend === 'down' ? 'var(--accent-red)' : trend === 'stable' ? 'var(--accent-yellow)' : 'var(--text-muted)';
            return (
              <td key={colKey} style={tdStyle}>
                <span style={{ fontSize: 10, fontWeight: 600, textTransform: 'uppercase', padding: '2px 6px', borderRadius: 4, color: sColor, background: `\${sColor}22`, border: `1px solid \${sColor}55` }}>{trend}</span>
              </td>
            );
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
          case '_vs_reference':
            return <td key={colKey} style={tdStyle}>{entry._vs_reference != null ? `\${entry._vs_reference.toFixed(0)}%` : '--'}</td>;
          case 'composite_score':
            return <td key={colKey} style={{ ...tdStyle, color: 'var(--accent-green)' }}>{fmt(entry.composite_score, 3)}</td>;
          case 'discovery_loss_ratio': return <td key={colKey} style={tdStyle}>{fmt(entry.discovery_loss_ratio)}</td>;
          case 'validation_loss_ratio': return <td key={colKey} style={tdStyle}>{fmt(entry.validation_loss_ratio)}</td>;
          case 'screening_loss_ratio': return <td key={colKey} style={tdStyle}>{fmt(entry.screening_loss_ratio)}</td>;
          case 'screening_novelty': return <td key={colKey} style={tdStyle}>{fmt(entry.screening_novelty, 3)}</td>;
          case 'investigation_loss_ratio': return <td key={colKey} style={tdStyle}>{fmt(entry.investigation_loss_ratio)}</td>;
          case 'investigation_robustness':
            return <td key={colKey} style={tdStyle}><span style={{ color: entry.investigation_robustness >= 0.5 ? 'var(--accent-green)' : 'var(--accent-red)' }}>{fmt(entry.investigation_robustness, 2)}</span></td>;
          case 'validation_baseline_ratio':
            return <td key={colKey} style={tdStyle}><span style={{ color: entry.validation_baseline_ratio < 1 ? 'var(--accent-green)' : 'var(--accent-red)' }}>{fmt(entry.validation_baseline_ratio)}</span></td>;
          case 'robustness_noise_score': return <td key={colKey} style={tdStyle}>{fmt(entry.robustness_noise_score, 3)}</td>;
          case 'quant_int8_retention': return <td key={colKey} style={tdStyle}>{entry._quant_retention_pct != null ? `\${entry._quant_retention_pct.toFixed(1)}%` : '--'}</td>;
          case 'robustness_long_ctx_score': return <td key={colKey} style={tdStyle}>{fmt(entry.robustness_long_ctx_score, 3)}</td>;
          case 'robustness_long_ctx_scaling_score': return <td key={colKey} style={tdStyle}>{fmt(entry.robustness_long_ctx_scaling_score, 3)}</td>;
          case 'robustness_long_ctx_assoc_score': return <td key={colKey} style={tdStyle}>{fmt(entry.robustness_long_ctx_assoc_score, 3)}</td>;
          case 'robustness_long_ctx_multi_hop_score': return <td key={colKey} style={tdStyle}>{fmt(entry.robustness_long_ctx_multi_hop_score, 3)}</td>;
          case 'robustness_long_ctx_passkey_score': return <td key={colKey} style={tdStyle}>{fmt(entry.robustness_long_ctx_passkey_score, 3)}</td>;
          case 'robustness_long_ctx_retrieval_aggregate': return <td key={colKey} style={tdStyle}>{fmt(entry.robustness_long_ctx_retrieval_aggregate, 3)}</td>;
          case 'robustness_long_ctx_combined_score': return <td key={colKey} style={tdStyle}>{fmt(entry.robustness_long_ctx_combined_score, 3)}</td>;
          case 'max_viable_seq_len': return <td key={colKey} style={tdStyle}>{entry.max_viable_seq_len != null ? Number(entry.max_viable_seq_len).toFixed(0) : '--'}</td>;
          case 'init_sensitivity_std': return <td key={colKey} style={tdStyle}>{fmt(entry.init_sensitivity_std, 4)}</td>;
          case 'pre_inv_score':
            const pis = Number(entry.pre_inv_score);
            const pisColor = pis >= 50 ? 'var(--accent-green)' : pis >= 20 ? 'var(--accent-yellow)' : 'var(--accent-red)';
            return <td key={colKey} style={{ ...tdStyle, color: pisColor, fontWeight: 600 }}>{fmt(pis, 1)}</td>;
          case 'jacobian_spectral_norm': return <td key={colKey} style={tdStyle}>{fmt(entry.jacobian_spectral_norm ?? entry.fp_jacobian_spectral_norm, 4)}</td>;
          case '_compression_ratio':
            return (
              <td key={colKey} style={tdStyle}>
                <div style={{ fontSize: 11 }}>{compression.ratio != null ? `\${(compression.ratio * 100).toFixed(0)}%` : '--'}</div>
                <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>{compression.memoryMb?.toFixed(2)} MB · {compression.label}</div>
              </td>
            );
          case '_metric_quality':
            return (
              <td key={colKey} style={tdStyle}>
                <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>Q: {reproPacket.label} · R: {reproPacket.label}</div>
                {isExpanded && (
                  <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', marginTop: 4 }}>
                    {chips.map(c => <span key={c.label} style={{ fontSize: 10, padding: '1px 5px', borderRadius: 4, background: `\${reliabilityColor(c.reliability)}22`, color: reliabilityColor(c.reliability) }}>{c.label}</span>)}
                  </div>
                )}
              </td>
            );
          case '_actions':
            return (
              <td key={colKey} style={tdStyle} onClick={e => e.stopPropagation()}>
                <div style={{ display: 'flex', gap: 4 }}>
                  <button
                    onClick={(e) => handleActionClick(e, () => onInvestigate([entry.result_id]))}
                    style={{
                      ...actionBtnStyle,
                      opacity: eligibility.investigationEligible ? 1 : 0.85,
                    }}
                    title={eligibility.investigationEligible ? 'Start investigation' : 'Currently ineligible; click to force override'}
                  >
                    {eligibility.investigationEligible ? 'Investigate' : 'Force Investigate'}
                  </button>
                  <button
                    onClick={(e) => handleActionClick(e, () => onValidate([entry.result_id]))}
                    style={{
                      ...actionBtnStyle,
                      opacity: eligibility.validationEligible ? 1 : 0.85,
                    }}
                    title={eligibility.validationEligible ? 'Start validation' : 'Currently ineligible; click to force override'}
                  >
                    {eligibility.validationEligible ? 'Validate' : 'Force Validate'}
                  </button>
                  <button onClick={(e) => handleActionClick(e, () => onToggleExpand(rowId))} style={actionBtnStyle}>{isExpanded ? 'Hide' : 'Details'}</button>
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
