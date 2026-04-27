import React from 'react';
import { compressionSummary } from '../report/reportUtils';
import { metricChips, reproducibilityPacketStatus, candidateEligibility } from './leaderboardUtils';
import { RENDERERS, TD_STYLE_OVERRIDES } from './columnRenderers';

function rowBackground(entry, isHighlighted) {
  const score = Number(entry?.composite_score);
  if (isHighlighted) return 'color-mix(in srgb, var(--table-row-bg) 78%, var(--accent-blue))';
  if (entry.is_reference || entry.model_source === 'reference') {
    return 'color-mix(in srgb, var(--table-row-bg) 88%, var(--score-reference))';
  }
  if (Number.isFinite(score) && score >= 230) return 'color-mix(in srgb, var(--table-row-bg) 91%, var(--score-champion))';
  if (Number.isFinite(score) && score >= 200) return 'color-mix(in srgb, var(--table-row-bg) 93%, var(--score-elite))';
  if (Number.isFinite(score) && score >= 150) return 'color-mix(in srgb, var(--table-row-bg) 94%, var(--score-reference))';
  if (entry.tier === 'breakthrough') return 'color-mix(in srgb, var(--table-row-bg) 92%, var(--score-elite))';
  return undefined;
}

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

const deleteBtnStyle = {
  ...actionBtnStyle,
  borderColor: 'rgba(248, 81, 73, 0.4)',
  background: 'rgba(248, 81, 73, 0.12)',
  color: 'var(--accent-red, #f85149)',
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
  onConfirm,
  onDelete,
  eligibilityFromParent
}) => {
  const compression = visibleColumns.includes('_compression_ratio') ? (entry._compression_summary || compressionSummary(entry)) : null;
  const chips = visibleColumns.includes('_metric_quality') ? metricChips(entry) : [];
  const reproPacket = visibleColumns.includes('_metric_quality') ? reproducibilityPacketStatus(entry) : { label: '--' };
  const eligibility = eligibilityFromParent || candidateEligibility(entry);

  const rowId = entry.entry_id || entry.result_id || index;
  const hasBeenInvestigated = entry.investigation_loss_ratio != null || ['investigation', 'validation', 'breakthrough'].includes(entry.tier);
  const hasBeenValidated = entry.validation_loss_ratio != null || ['validation', 'breakthrough'].includes(entry.tier);
  const canDelete = !entry.is_reference && (entry.tier === 'screening' || entry.tier === 'failed' || entry.tier === 'rejected' || entry.screening_passed === false || entry.investigation_passed === false || entry.validation_passed === false);

  const handleActionClick = (e, action) => {
    e.stopPropagation();
    action();
  };

  // Shared context for renderers that need external state
  const ctx = { compression, chips, reproPacket, eligibility, isExpanded };

  return (
    <tr
      ref={isHighlighted ? highlightRef : undefined}
      style={{
        borderBottom: '1px solid var(--border)',
        cursor: 'pointer',
        background: rowBackground(entry, isHighlighted),
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
        // Actions column — complex, stays inline
        if (colKey === '_actions') {
          return (
            <td key={colKey} style={tdStyle} onClick={e => e.stopPropagation()}>
              <div style={{ display: 'flex', gap: 4, flexWrap: 'nowrap' }}>
                {!hasBeenInvestigated && (
                  <button
                    onClick={(e) => handleActionClick(e, () => onInvestigate([entry.result_id]))}
                    style={{ ...actionBtnStyle, opacity: eligibility.investigationEligible ? 1 : 0.6, borderStyle: eligibility.investigationEligible ? 'solid' : 'dashed' }}
                    aria-label={eligibility.investigationEligible ? `Investigate ${entry.result_id}` : `Force investigate ${entry.result_id} (currently ineligible)`}
                    title={eligibility.investigationEligible ? 'Run investigation stage' : 'Not yet eligible \u2014 click to override and force-investigate'}
                  >
                    Investigate
                  </button>
                )}
                {!hasBeenValidated && (
                  <button
                    onClick={(e) => handleActionClick(e, () => onValidate([entry.result_id]))}
                    style={{ ...actionBtnStyle, opacity: eligibility.validationEligible ? 1 : 0.6, borderStyle: eligibility.validationEligible ? 'solid' : 'dashed' }}
                    aria-label={eligibility.validationEligible ? `Validate ${entry.result_id}` : `Force validate ${entry.result_id} (currently ineligible)`}
                    title={eligibility.validationEligible ? 'Run validation stage' : 'Not yet eligible \u2014 click to override and force-validate'}
                  >
                    Validate
                  </button>
                )}
                {eligibility.confirmationEligible && (
                  <button
                    onClick={(e) => handleActionClick(e, () => onConfirm?.([entry.result_id]))}
                    style={{ ...actionBtnStyle, borderColor: 'rgba(255, 184, 108, 0.5)', color: 'var(--score-elite)' }}
                    aria-label={`Confirm ${entry.result_id}`}
                    title="Run post-validation champion confirmation"
                  >
                    Confirm
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
                    style={deleteBtnStyle}
                    title="Delete entry and all associated data"
                  >
                    Delete
                  </button>
                )}
              </div>
            </td>
          );
        }

        // Data columns — dispatch to renderer map
        const renderer = RENDERERS[colKey];
        if (!renderer) return null;
        const override = TD_STYLE_OVERRIDES[colKey];
        const style = override ? { ...tdStyle, ...override } : tdStyle;
        return <td key={colKey} style={style}>{renderer(entry, ctx)}</td>;
      })}
    </tr>
  );
});

export default LeaderboardRow;
