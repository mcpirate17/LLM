import React, { useState } from 'react';
import { scoreColor } from '../../utils/format';
import { lossColor, noveltyColor } from '../../utils/colors';
import {
  candidateScore,
  candidateScoreBreakdown,
  promotionEvidence,
  TIER_COLORS,
  TIER_ORDER,
} from '../../utils/scoringEngine';

const STATUS_LABELS = {
  screening: 'Screened',
  screened_out: 'Failed Investigation',
  investigation: 'Investigation',
  validation: 'Validated',
  breakthrough: 'Breakthrough',
};

const STATUS_OPTIONS = [
  { value: 'screening', label: 'Screened' },
  { value: 'screened_out', label: 'Failed Investigation' },
  { value: 'investigation', label: 'Investigating' },
  { value: 'validation', label: 'Validated' },
  { value: 'breakthrough', label: 'Breakthrough' },
];

export function SummaryBar({ tierCounts }) {
  const total = tierCounts?.all || 0;
  const validated = (tierCounts?.validation || 0) + (tierCounts?.breakthrough || 0);
  const breakthroughs = tierCounts?.breakthrough || 0;
  const references = tierCounts?.references || 0;

  return (
    <div
      style={{
        display: 'flex',
        gap: 24,
        alignItems: 'center',
        flexWrap: 'wrap',
        padding: '10px 14px',
        marginBottom: 12,
        background: 'var(--bg-secondary)',
        borderRadius: 8,
        border: '1px solid var(--border)',
        fontSize: 13,
      }}
    >
      <Stat value={total} label="discoveries" />
      <Stat value={tierCounts?.screening || 0} label="screened" color="var(--accent-blue)" />
      <Stat value={tierCounts?.screened_out || 0} label="failed investigation" color="var(--text-muted)" />
      <Stat value={tierCounts?.investigation || 0} label="investigation" color="var(--accent-yellow)" />
      <Stat value={validated} label="validated" color="var(--accent-purple)" />
      <Stat value={breakthroughs} label="breakthroughs" color="var(--accent-green)" />
      <Stat value={references} label="references" color="var(--accent-purple)" />
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

export function StatusBadge({ entry }) {
  const tier = entry.tier;
  const color = TIER_COLORS[tier] || 'var(--text-muted)';

  let label = STATUS_LABELS[tier] || tier || 'Unknown';
  if (tier === 'investigation' && entry.investigation_robustness != null && !entry.investigation_passed) {
    label = 'Brittle';
  } else if (tier === 'validation' && entry.validation_baseline_ratio != null && !entry.validation_passed) {
    label = 'Mediocre';
  }

  return (
    <span
      style={{
        padding: '2px 8px',
        borderRadius: 4,
        fontSize: 11,
        fontWeight: 600,
        color,
        background: `${color}22`,
        border: `1px solid ${color}`,
        textTransform: 'uppercase',
      }}
    >
      {label}
    </span>
  );
}

export function StatusEditor({ entry, currentValue, onChange }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
      <select
        value={currentValue || entry.tier || 'screening'}
        onChange={(e) => onChange(e.target.value)}
        onClick={(e) => e.stopPropagation()}
        style={{
          fontSize: 11,
          padding: '2px 6px',
          borderRadius: 4,
          border: '1px solid var(--border)',
          background: 'var(--bg-tertiary)',
          color: 'var(--text-primary)',
        }}
      >
        {STATUS_OPTIONS.map((option) => (
          <option key={option.value} value={option.value}>{option.label}</option>
        ))}
      </select>
    </div>
  );
}

export function ScoreCell({ entry }) {
  const [show, setShow] = useState(false);
  const breakdown = candidateScoreBreakdown(entry, TIER_ORDER);
  const score = entry?._score != null
    ? Number(entry._score)
    : entry?.composite_score != null
      ? Number(entry.composite_score)
      : candidateScore(entry, TIER_ORDER);

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
    robustnessBonus: { label: 'Robustness', color: 'var(--accent-yellow)' },
    referenceDeltaBonus: { label: 'Baseline Delta', color: 'var(--accent-orange)' },
  };

  const components = Object.entries(breakdown)
    .filter(([, weight]) => weight > 0)
    .map(([key, weight]) => ({ key, weight, ...(keyMap[key] || { label: key, color: 'var(--border)' }) }));
  const total = components.reduce((acc, component) => acc + (Number(component.weight) || 0), 0) || 1;

  return (
    <div
      style={{ minWidth: 92, position: 'relative', display: 'inline-block' }}
      onMouseEnter={() => setShow(true)}
      onMouseLeave={() => setShow(false)}
    >
      <div style={{ fontWeight: 600, color: scoreColor(score) }}>{Number(score).toFixed(1)}</div>
      <div style={{ display: 'flex', height: 3, borderRadius: 2, overflow: 'hidden', background: 'var(--bg-tertiary)', marginTop: 2 }}>
        {components.map((component) => (
          <div key={component.key} style={{ width: `${(component.weight / total) * 100}%`, background: component.color, height: '100%' }} />
        ))}
      </div>
      {show && (
        <div
          style={{
            position: 'absolute',
            top: '100%',
            left: '50%',
            transform: 'translateX(-50%)',
            marginTop: 6,
            padding: '8px 10px',
            background: '#161b22',
            border: '1px solid var(--border)',
            borderRadius: 6,
            boxShadow: '0 6px 16px rgba(0,0,0,0.45)',
            zIndex: 1000,
            minWidth: 200,
            fontSize: 11,
            color: 'var(--text-primary)',
          }}
        >
          <div style={{ fontWeight: 600, marginBottom: 4 }}>Score Breakdown</div>
          {components.map((component) => (
            <div key={component.key} style={{ marginBottom: 4 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 1 }}>
                <span>{component.label}</span>
                <span>{Number(component.weight).toFixed(1)}</span>
              </div>
              <div style={{ height: 3, background: 'var(--bg-tertiary)', borderRadius: 2, overflow: 'hidden' }}>
                <div style={{ width: `${(component.weight / total) * 100}%`, height: '100%', background: component.color }} />
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export function ExpandedDetail({
  entry,
  onInvestigate,
  onValidate,
  onQueueAdd,
  onQueueRemove,
  onDelete,
  isQueued,
  eligibility,
  statusDraft,
  onStatusDraftChange,
  onSaveStatus,
  savingStatus,
  actionBtnStyle,
}) {
  const promotion = promotionEvidence(entry);
  const fmt = (value, digits = 4) => {
    if (value == null) return '--';
    const num = Number(value);
    if (num !== 0 && Math.abs(num) < 0.0001) return num.toExponential(2);
    return num.toFixed(digits);
  };
  const hasBeenInvestigated = entry.investigation_loss_ratio != null || ['investigation', 'validation', 'breakthrough'].includes(entry.tier);
  const hasBeenValidated = entry.validation_loss_ratio != null || ['validation', 'breakthrough'].includes(entry.tier);
  const canDelete = !entry.is_reference && (entry.tier === 'screening' || entry.tier === 'failed' || entry.tier === 'rejected' || entry.screening_passed === false || entry.investigation_passed === false || entry.validation_passed === false);

  return (
    <tr>
      <td colSpan={8} style={{ padding: '12px 16px', background: 'var(--bg-secondary)', borderBottom: '1px solid var(--border)' }}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 16, fontSize: 12 }}>
          <div>
            <div style={{ fontWeight: 600, marginBottom: 6, textTransform: 'uppercase', fontSize: 10, color: 'var(--text-muted)' }}>Full Metrics</div>
            <MetricRow label="Screening Loss" value={fmt(entry.screening_loss_ratio)} color={lossColor(entry.screening_loss_ratio)} />
            <MetricRow label="Screening Novelty" value={fmt(entry.screening_novelty, 3)} color={noveltyColor(entry.screening_novelty)} />
            <MetricRow label="Investigation Loss" value={fmt(entry.investigation_loss_ratio)} />
            <MetricRow
              label="Robustness"
              value={fmt(entry.investigation_robustness, 2)}
              color={entry.investigation_robustness != null ? (entry.investigation_robustness >= 0.5 ? 'var(--accent-green)' : 'var(--accent-red)') : undefined}
            />
            <MetricRow label="Validation Loss" value={fmt(entry.validation_loss_ratio)} />
            <MetricRow
              label="Validation Baseline"
              value={fmt(entry.validation_baseline_ratio)}
              color={entry.validation_baseline_ratio != null ? (entry.validation_baseline_ratio < 1 ? 'var(--accent-green)' : 'var(--accent-red)') : undefined}
            />
            <MetricRow label="Multi-seed Std" value={fmt(entry.validation_multi_seed_std, 3)} />
            <MetricRow label="Composite" value={fmt(entry.composite_score, 3)} color="var(--accent-green)" />
          </div>

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
            <div style={{ marginTop: 8 }}>
              <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 4 }}>
                Edit Status
              </div>
              <StatusEditor entry={entry} currentValue={statusDraft || entry.tier} onChange={(tier) => onStatusDraftChange?.(tier)} />
            </div>
            <div style={{ marginTop: 8 }}>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onSaveStatus?.();
                }}
                disabled={savingStatus || (statusDraft || entry.tier) === entry.tier}
                style={{
                  ...actionBtnStyle,
                  padding: '3px 9px',
                  fontSize: 10,
                  borderColor: 'var(--accent-green)',
                  color: 'var(--accent-green)',
                  opacity: (savingStatus || (statusDraft || entry.tier) === entry.tier) ? 0.6 : 1,
                  cursor: (savingStatus || (statusDraft || entry.tier) === entry.tier) ? 'not-allowed' : 'pointer',
                }}
                title="Save status change"
              >
                {savingStatus ? 'Saving…' : 'Save Status'}
              </button>
            </div>
            {entry.result_id && (
              <div style={{ fontSize: 11, color: 'var(--text-muted)', fontFamily: 'monospace' }}>
                ID: {entry.result_id}
              </div>
            )}
          </div>

          <div>
            <div style={{ fontWeight: 600, marginBottom: 6, textTransform: 'uppercase', fontSize: 10, color: 'var(--text-muted)' }}>Actions</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              {!hasBeenInvestigated && (
                <button
                  onClick={() => onInvestigate([entry.result_id])}
                  style={{
                    ...actionBtnStyle,
                    background: 'rgba(63, 185, 80, 0.12)',
                    border: '1px solid rgba(63, 185, 80, 0.4)',
                    color: 'var(--accent-green)',
                    opacity: eligibility?.investigationEligible ? 1 : 0.85,
                  }}
                  title={eligibility?.investigationEligible ? 'Start investigation' : 'Currently ineligible; click to force override'}
                >
                  {eligibility?.investigationEligible ? 'Investigate' : 'Force Investigate'}
                </button>
              )}
              {!hasBeenValidated && (
                <button
                  onClick={() => onValidate([entry.result_id])}
                  style={{
                    ...actionBtnStyle,
                    background: 'rgba(188, 140, 255, 0.12)',
                    border: '1px solid rgba(188, 140, 255, 0.4)',
                    color: 'var(--accent-purple)',
                    opacity: eligibility?.validationEligible ? 1 : 0.85,
                  }}
                  title={eligibility?.validationEligible ? 'Start validation' : 'Currently ineligible; click to force override'}
                >
                  {eligibility?.validationEligible ? 'Validate' : 'Force Validate'}
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
              {canDelete && (
                <button
                  onClick={() => {
                    if (window.confirm(`Delete ${entry.entry_id?.slice(0, 11) || entry.result_id?.slice(0, 11)} and all associated data?`)) {
                      onDelete?.(entry.entry_id || entry.result_id);
                    }
                  }}
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

export function FingerprintLeaderboardChart({ entries }) {
  if (!entries || entries.length < 2) return null;

  const top = entries.slice(0, 15);
  const width = 600;
  const height = 160;
  const padX = 40;
  const padY = 20;
  const barWidth = (width - 2 * padX) / top.length - 8;
  const maxScore = Math.max(...top.map((entry) => entry._score), 80);

  return (
    <div style={{ marginBottom: 20, padding: '10px 0' }}>
      <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8, textTransform: 'uppercase', fontWeight: 600 }}>
        Fingerprint Performance Ranking (Top {top.length})
      </div>
      <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} style={{ width: '100%', height: 'auto', maxWidth: width }}>
        {[0, 0.5, 1].map((fraction) => (
          <text
            key={fraction}
            x={padX - 5}
            y={height - padY - fraction * (height - 2 * padY)}
            fontSize={9}
            fill="var(--text-muted)"
            textAnchor="end"
            alignmentBaseline="middle"
          >
            {Math.round(fraction * maxScore)}
          </text>
        ))}

        {[0, 0.5, 1].map((fraction) => (
          <line
            key={`grid-${fraction}`}
            x1={padX}
            y1={height - padY - fraction * (height - 2 * padY)}
            x2={width - padX}
            y2={height - padY - fraction * (height - 2 * padY)}
            stroke="var(--border)"
            strokeWidth={0.5}
            strokeDasharray="2 2"
          />
        ))}

        {top.map((entry, index) => {
          const score = entry._score || 0;
          const barHeight = (score / maxScore) * (height - 2 * padY);
          const x = padX + index * (barWidth + 8);
          const y = height - padY - barHeight;
          const isPinnedReference = Boolean(entry?.is_reference)
            || String(entry?.model_source || '').toLowerCase() === 'reference'
            || Boolean(entry?.reference_name);
          const color = isPinnedReference ? 'var(--accent-purple)' : scoreColor(score);

          return (
            <g key={entry.result_id || index}>
              <rect x={x} y={y} width={barWidth} height={barHeight} fill={`${color}88`} stroke={color} strokeWidth={1} rx={2} />
              <text
                x={x + barWidth / 2}
                y={height - 5}
                fontSize={8}
                fill="var(--text-muted)"
                textAnchor="middle"
                transform={`rotate(45 ${x + barWidth / 2} ${height - 5})`}
              >
                {entry.display_name?.slice(0, 8) || entry.graph_fingerprint?.slice(0, 6)}
              </text>
              <title>{entry.display_name || entry.graph_fingerprint}: Discovery Score {score}</title>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
