import React, { useState } from 'react';
import { SCORE_MAX, scoreColor, scoreGradient, scoreGradientStops, scoreScaleDomain, scoreScaleRatio, scoreToneLabel } from '../../utils/format';
import { lossColor, noveltyColor } from '../../utils/colors';
import { TIER_COLORS } from '../../utils/scoringEngine';
import {
  canonicalCompositeScore,
  canonicalScoreComponents,
  promotionEvidenceView,
} from '../../utils/backendScore';
import { getDiscoveryDisplayStatus } from '../../utils/discoveryStatus';

const STATUS_LABELS = {
  screening: 'Screened',
  screened_out: 'Failed Screening',
  investigation_failed: 'Failed Investigation',
  validation_failed: 'Failed Validation',
  validation_pending: 'Validation Pending',
  investigation: 'Investigation',
  validation: 'Validated',
  breakthrough: 'Breakthrough',
};

const STATUS_OPTIONS = [
  { value: 'screening', label: 'Screened' },
  { value: 'screened_out', label: 'Failed Screening' },
  { value: 'investigation', label: 'Investigating' },
  { value: 'validation', label: 'Validated' },
  { value: 'breakthrough', label: 'Breakthrough' },
];

function provenanceStatus(entry) {
  const cohort = String(entry?.result_cohort || '').trim().toLowerCase();
  const experimentType = String(entry?.experiment_type || '').trim().toLowerCase();
  const trustLabel = String(entry?.trust_label || '').trim().toLowerCase();

  if (experimentType === 'exact_graph_replay' || cohort === 'exact_graph_replay') {
    return { label: 'Replay', color: 'var(--accent-blue)' };
  }
  if (
    cohort === 'backfill'
    || trustLabel === 'backfill_observation'
    || String(entry?.comparability_label || '').trim().toLowerCase() === 'reconstructed_init_variant'
  ) {
    return { label: 'Backfill', color: 'var(--accent-orange, #d29922)' };
  }
  return null;
}

function capabilityBadgeTheme(status) {
  switch (status) {
    case 'qualified':
      return { color: 'var(--accent-green)', bg: 'rgba(63, 185, 80, 0.12)' };
    case 'breakthrough':
      return { color: 'var(--accent-green)', bg: 'rgba(63, 185, 80, 0.18)' };
    case 'training_only':
      return { color: 'var(--accent-yellow)', bg: 'rgba(210, 153, 34, 0.12)' };
    case 'pending':
      return { color: 'var(--accent-purple)', bg: 'rgba(188, 140, 255, 0.12)' };
    default:
      return { color: 'var(--text-muted)', bg: 'rgba(139, 148, 158, 0.10)' };
  }
}

export function SummaryBar({ tierCounts }) {
  const total = tierCounts?.all || 0;
  const validated = (tierCounts?.validation || 0) + (tierCounts?.breakthrough || 0);
  const breakthroughs = tierCounts?.breakthrough || 0;
  const references = tierCounts?.references || 0;
  const backfill = tierCounts?.backfill || 0;
  const replay = tierCounts?.replay || 0;

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
      <Stat value={tierCounts?.screened_out || 0} label="failed screening" color="var(--text-muted)" />
      {(tierCounts?.investigation_failed || 0) > 0 && (
        <Stat value={tierCounts?.investigation_failed || 0} label="failed investigation" color="var(--accent-red)" />
      )}
      <Stat value={tierCounts?.investigation || 0} label="investigation" color="var(--accent-yellow)" />
      {(tierCounts?.validation_pending || 0) > 0 && (
        <Stat value={tierCounts?.validation_pending || 0} label="validation pending" color="var(--accent-purple)" />
      )}
      <Stat value={validated} label="validated" color="var(--accent-purple)" />
      <Stat value={breakthroughs} label="breakthroughs" color="var(--accent-green)" />
      {backfill > 0 && <Stat value={backfill} label="backfill" color="var(--accent-orange, #d29922)" />}
      {replay > 0 && <Stat value={replay} label="replay" color="var(--accent-blue)" />}
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
  const status = getDiscoveryDisplayStatus(entry);
  const tier = status.tierKey;
  const provenance = provenanceStatus(entry);
  const color = provenance?.color || TIER_COLORS[tier] || 'var(--text-muted)';
  const capability = entry?.capability_quality || null;
  const semanticWarning = entry?.semantic_warning || null;
  const semanticWarningTitle = semanticWarning
    ? [semanticWarning.message, ...(semanticWarning.evidence || [])].join('\n')
    : '';

  let label = provenance?.label || status.label || STATUS_LABELS[tier] || tier || 'Unknown';
  if (tier === 'investigation' && entry.investigation_robustness != null && !entry.investigation_passed) {
    label = 'Brittle';
  } else if (tier === 'validation' && entry.validation_baseline_ratio != null && !entry.validation_passed) {
    label = 'Mediocre';
  }

  return (
    <div style={{ display: 'inline-flex', gap: 6, alignItems: 'center', flexWrap: 'wrap', minWidth: 0, maxWidth: '100%' }}>
      <span
        style={{
          display: 'inline-block',
          padding: '2px 8px',
          borderRadius: 4,
          fontSize: 11,
          fontWeight: 600,
          color,
          background: `${color}22`,
          border: `1px solid ${color}`,
          textTransform: 'uppercase',
          whiteSpace: 'nowrap',
          maxWidth: '100%',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
        }}
      >
        {label}
      </span>
      {capability?.status && !['exploratory', 'investigated'].includes(capability.status) && (
        <span
          title={(capability?.missing || []).length ? `Missing: ${(capability.missing || []).join(', ')}` : capability.label}
          style={{
            display: 'inline-block',
            padding: '2px 8px',
            borderRadius: 4,
            fontSize: 10,
            fontWeight: 600,
            color: capabilityBadgeTheme(capability.status).color,
            background: capabilityBadgeTheme(capability.status).bg,
            border: `1px solid ${capabilityBadgeTheme(capability.status).color}`,
            whiteSpace: 'nowrap',
            maxWidth: '100%',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
          }}
        >
          {capability.label}
        </span>
      )}
      {semanticWarning && (
        <span
          title={semanticWarningTitle}
          style={{
            display: 'inline-block',
            padding: '2px 8px',
            borderRadius: 4,
            fontSize: 10,
            fontWeight: 700,
            color: 'var(--accent-yellow)',
            background: 'rgba(210, 153, 34, 0.12)',
            border: '1px solid rgba(210, 153, 34, 0.45)',
            whiteSpace: 'nowrap',
            maxWidth: '100%',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
          }}
        >
          {semanticWarning.label || 'Warning'}
        </span>
      )}
    </div>
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
  const score = canonicalCompositeScore(entry);
  const components = canonicalScoreComponents(entry);
  const total = components.reduce((acc, component) => acc + (Number(component.weight) || 0), 0) || 1;
  const scorePercent = score == null ? 0 : Math.max(4, Math.min(100, (score / SCORE_MAX) * 100));

  return (
    <div
      style={{ minWidth: 92, position: 'relative', display: 'inline-block' }}
      onMouseEnter={() => setShow(true)}
      onMouseLeave={() => setShow(false)}
    >
      <div
        title={score != null ? `${scoreToneLabel(score)} score` : undefined}
        style={{
          fontWeight: 700,
          color: score != null ? scoreColor(score) : 'var(--text-muted)',
          fontVariantNumeric: 'tabular-nums',
        }}
      >
        {score != null ? Number(score).toFixed(1) : '—'}
      </div>
      {score != null && (
        <div className="champion-strip" style={{ marginTop: 2 }}>
          <div
            className="champion-strip-fill"
            style={{ width: `${scorePercent}%`, background: scoreGradient(score) }}
          />
        </div>
      )}
      {components.length > 0 && (
        <div style={{ display: 'flex', height: 2, borderRadius: 2, overflow: 'hidden', background: 'var(--bg-tertiary)', marginTop: 3 }}>
          {components.map((component) => (
            <div key={component.key} style={{ width: `${(component.weight / total) * 100}%`, background: component.color, height: '100%' }} />
          ))}
        </div>
      )}
      {show && components.length > 0 && (
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
          <div style={{ fontWeight: 600, marginBottom: 4 }}>Canonical Score Breakdown</div>
          {score != null && (
            <div style={{ color: scoreColor(score), fontSize: 10, marginBottom: 8 }}>
              {scoreToneLabel(score)} band after tiktoken/BPE rescore
            </div>
          )}
          {components.map((component) => (
            <div key={component.key} style={{ marginBottom: 4 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 1 }}>
                <span style={{ color: component.color }}>{component.label}</span>
                <span style={{ color: component.color, fontFamily: 'monospace' }}>{Number(component.weight).toFixed(1)}</span>
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

function ExpandedDetailBody({
  entry,
  onRescreen,
  onPromoteScreening,
  onInvestigate,
  onValidate,
  onConfirm,
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
  const promotion = promotionEvidenceView(entry);
  const score = canonicalCompositeScore(entry);
  const scoreComponents = canonicalScoreComponents(entry);
  const scoreTotal = scoreComponents.reduce((acc, component) => acc + component.weight, 0) || 1;
  const fmt = (value, digits = 4) => {
    if (value == null) return '--';
    const num = Number(value);
    if (num !== 0 && Math.abs(num) < 0.0001) return num.toExponential(2);
    return num.toFixed(digits);
  };
  const hasBeenInvestigated = entry.investigation_loss_ratio != null || ['investigation', 'validation', 'breakthrough'].includes(entry.tier);
  const hasBeenValidated = entry.validation_loss_ratio != null || ['validation', 'breakthrough'].includes(entry.tier);
  const isTrusted = ['candidate_screening', 'candidate_grade', 'reference'].includes(String(entry.trust_label || '').trim().toLowerCase());
  const screeningActionLabel = 'Replay';
  const showPromoteScreening = entry.result_id && !entry.is_reference && !isTrusted;
  const canDelete = !entry.is_reference && (entry.tier === 'screening' || entry.tier === 'failed' || entry.tier === 'rejected' || entry.screening_passed === false || entry.investigation_passed === false || entry.validation_passed === false);

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(190px, 1fr))', gap: 12, fontSize: 12 }}>
          <div style={{ gridColumn: '1 / -1', padding: '8px 10px', borderRadius: 8, border: '1px solid var(--border)', background: 'rgba(255,255,255,0.015)' }}>
            <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--text-primary)' }}>
              {entry.display_name || entry.architecture_desc || entry.graph_fingerprint || entry.result_id || 'Selected discovery'}
            </div>
            <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', marginTop: 4, fontSize: 10, color: 'var(--text-muted)', fontFamily: 'monospace' }}>
              {entry.result_id && <span>ID: {entry.result_id}</span>}
              {entry.graph_fingerprint && <span>FP: {entry.graph_fingerprint}</span>}
            </div>
          </div>
          <div style={{ padding: 10, borderRadius: 8, border: '1px solid var(--border)', background: 'rgba(255,255,255,0.02)' }}>
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

          <div style={{ padding: 10, borderRadius: 8, border: '1px solid var(--border)', background: 'rgba(255,255,255,0.02)' }}>
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
          </div>

          <div style={{ padding: 10, borderRadius: 8, border: '1px solid var(--border)', background: 'rgba(255,255,255,0.02)' }}>
            <div style={{ fontWeight: 600, marginBottom: 6, textTransform: 'uppercase', fontSize: 10, color: 'var(--text-muted)' }}>Score Totals</div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 10 }}>
              <span style={{ color: 'var(--text-muted)' }}>Composite total</span>
              <span style={{ color: score != null ? scoreColor(score) : 'var(--text-primary)', fontWeight: 700, fontFamily: 'monospace' }}>
                {score != null ? Number(score).toFixed(1) : '--'}
              </span>
            </div>
            {scoreComponents.length > 0 ? (
              <>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                  {scoreComponents.map((component) => (
                    <div key={component.key}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8, marginBottom: 3 }}>
                        <span style={{ color: component.color }}>{component.label}</span>
                        <span style={{ color: component.color, fontFamily: 'monospace' }}>{component.weight.toFixed(1)}</span>
                      </div>
                      <div style={{ height: 5, background: 'var(--bg-tertiary)', borderRadius: 999, overflow: 'hidden' }}>
                        <div style={{ width: `${(component.weight / scoreTotal) * 100}%`, height: '100%', background: component.color }} />
                      </div>
                    </div>
                  ))}
                </div>
                <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 8 }}>
                  These are additive v10 subtotals. Detailed metrics live inside the subtotals and are not added again.
                </div>
              </>
            ) : (
              <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>No score breakdown recorded for this row.</div>
            )}
          </div>

          <div style={{ padding: 10, borderRadius: 8, border: '1px solid var(--border)', background: 'rgba(255,255,255,0.02)' }}>
            <div style={{ fontWeight: 600, marginBottom: 6, textTransform: 'uppercase', fontSize: 10, color: 'var(--text-muted)' }}>Actions</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              {entry.result_id && !entry.is_reference && (
                <button
                  onClick={() => onRescreen?.(entry.result_id)}
                  style={{
                    ...actionBtnStyle,
                    background: 'rgba(88, 166, 255, 0.12)',
                    border: '1px solid rgba(88, 166, 255, 0.4)',
                    color: 'var(--accent-blue)',
                  }}
                  title={isTrusted ? 'Replay this fingerprint through the current loss-oriented screening replay path' : 'Replay this fingerprint through the current loss-oriented screening replay path'}
                >
                  {screeningActionLabel}
                </button>
              )}
              {showPromoteScreening && (
                <button
                  onClick={() => onPromoteScreening?.(entry.result_id)}
                  style={{
                    ...actionBtnStyle,
                    background: 'rgba(63, 185, 80, 0.12)',
                    border: '1px solid rgba(63, 185, 80, 0.4)',
                    color: 'var(--accent-green)',
                  }}
                  title="Promote this row into the trusted screening candidate pool"
                >
                  Promote
                </button>
              )}
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
              {eligibility?.confirmationEligible && (
                <button
                  onClick={() => onConfirm?.([entry.result_id])}
                  style={{
                    ...actionBtnStyle,
                    background: 'rgba(255, 184, 108, 0.12)',
                    border: '1px solid rgba(255, 184, 108, 0.48)',
                    color: 'var(--score-elite)',
                  }}
                  title="Run post-validation champion confirmation at 4x validation steps"
                >
                  Confirm
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
                        intent: eligibility?.confirmationEligible
                          ? 'confirmation'
                          : eligibility?.validationEligible
                            ? 'validation'
                            : 'investigation',
                        queueEligible: true,
                        investigationEligible: eligibility?.investigationEligible,
                        validationEligible: eligibility?.validationEligible,
                        confirmationEligible: eligibility?.confirmationEligible,
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
                      ? 'No Action'
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
  );
}

export function ExpandedDetail({
  entry,
  colSpan,
  ...props
}) {
  return (
    <tr>
      <td colSpan={colSpan || 8} style={{ padding: '10px 14px', background: 'var(--bg-secondary)', borderBottom: '1px solid var(--border)' }}>
        <ExpandedDetailBody entry={entry} {...props} />
      </td>
    </tr>
  );
}

export function ExpandedDetailPanel({
  entry,
  onClose,
  ...props
}) {
  if (!entry) return null;

  return (
    <div
      style={{
        marginTop: 14,
        position: 'sticky',
        bottom: 12,
        zIndex: 6,
        border: '1px solid var(--border)',
        borderRadius: 10,
        background: 'var(--bg-secondary)',
        overflow: 'hidden',
        boxShadow: '0 14px 36px rgba(0, 0, 0, 0.34)',
        backdropFilter: 'blur(10px)',
      }}
    >
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          gap: 12,
          padding: '10px 14px',
          borderBottom: '1px solid var(--border)',
          background: 'rgba(255,255,255,0.02)',
        }}
      >
        <div style={{ minWidth: 0 }}>
          <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--text-primary)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
            Discovery Detail
          </div>
          <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginTop: 4, minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis' }}>
            {entry.display_name || entry.architecture_desc || entry.graph_fingerprint || entry.result_id || 'Selected discovery'}
          </div>
        </div>
        <button
          type="button"
          onClick={onClose}
          style={{
            border: '1px solid var(--border)',
            background: 'transparent',
            color: 'var(--text-secondary)',
            borderRadius: 6,
            padding: '5px 10px',
            cursor: 'pointer',
            fontSize: 11,
            flexShrink: 0,
          }}
        >
          Close
        </button>
      </div>
      <div style={{ padding: '10px 14px', maxHeight: 'min(46vh, 420px)', overflowY: 'auto' }}>
        <ExpandedDetailBody entry={entry} {...props} />
      </div>
    </div>
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

export function FingerprintLeaderboardChart({ entries, scoreScale }) {
  if (!entries || entries.length < 2) return null;

  const top = entries.slice(0, 15);
  const width = 1040;
  const height = 190;
  const padX = 48;
  const padTop = 22;
  const padBottom = 34;
  const barWidth = (width - 2 * padX) / top.length - 8;
  const apiMin = Number(scoreScale?.p25);
  const apiMax = Number(scoreScale?.max_possible);
  const scoreDomain = Number.isFinite(apiMin) && Number.isFinite(apiMax) && apiMax > apiMin
    ? { min: apiMin, max: apiMax }
    : scoreScaleDomain(entries.map((entry) => entry._score), { minMode: 'p25' });
  const minScore = scoreDomain.min;
  const maxScore = scoreDomain.max;
  const chartHeight = height - padTop - padBottom;

  return (
    <div style={{ marginBottom: 20, padding: '10px 0 4px' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 14, flexWrap: 'wrap', marginBottom: 8 }}>
        <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600 }}>
          Canonical Composite Ranking (Top {top.length})
        </div>
        <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>
          Chart range: DB p25 ({Math.round(minScore)}) to scorer ceiling ({Math.round(maxScore)}). Colors use the same rubric.
        </div>
        <div
          aria-hidden="true"
          style={{
            width: 150,
            height: 5,
            borderRadius: 999,
            background: 'linear-gradient(90deg, #58a6ff, #2dd4bf, #e3b341, #ffd166, #ff7b72)',
            boxShadow: '0 0 16px rgba(255, 209, 102, 0.12)',
          }}
        />
      </div>
      <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} style={{ width: '100%', height: 'auto', maxWidth: width }}>
        <defs>
          {top.map((entry, index) => {
            const score = Number(entry._score) || 0;
            const isPinnedReference = Boolean(entry?.is_reference)
              || String(entry?.model_source || '').toLowerCase() === 'reference'
              || Boolean(entry?.reference_name);
            const [start, end] = isPinnedReference ? ['#00d4ff', '#2dd4bf'] : scoreGradientStops(score);
            return (
              <linearGradient key={`grad-${entry.result_id || index}`} id={`score-grad-${index}`} x1="0" x2="0" y1="1" y2="0">
                <stop offset="0%" stopColor={start} stopOpacity="0.72" />
                <stop offset="100%" stopColor={end} stopOpacity="0.98" />
              </linearGradient>
            );
          })}
        </defs>
        {[0, 0.5, 1].map((fraction) => (
          <text
            key={fraction}
            x={padX - 5}
            y={height - padBottom - fraction * chartHeight}
            fontSize={9}
            fill="var(--text-muted)"
            textAnchor="end"
            alignmentBaseline="middle"
          >
            {Math.round(minScore + fraction * (maxScore - minScore))}
          </text>
        ))}

        {[0, 0.5, 1].map((fraction) => (
          <line
            key={`grid-${fraction}`}
            x1={padX}
            y1={height - padBottom - fraction * chartHeight}
            x2={width - padX}
            y2={height - padBottom - fraction * chartHeight}
            stroke="var(--border)"
            strokeWidth={0.5}
            strokeDasharray="2 2"
          />
        ))}

        {top.map((entry, index) => {
          const score = entry._score || 0;
          const barHeight = Math.max(3, scoreScaleRatio(score, scoreDomain) * chartHeight);
          const x = padX + index * (barWidth + 8);
          const y = height - padBottom - barHeight;
          const isPinnedReference = Boolean(entry?.is_reference)
            || String(entry?.model_source || '').toLowerCase() === 'reference'
            || Boolean(entry?.reference_name);
          const color = isPinnedReference ? 'var(--accent-purple)' : scoreColor(score);

          return (
            <g key={entry.result_id || index}>
              <rect
                x={x}
                y={y}
                width={barWidth}
                height={barHeight}
                fill={`url(#score-grad-${index})`}
                stroke={color}
                strokeWidth={isPinnedReference ? 1.5 : 1}
                strokeDasharray={isPinnedReference ? '3 2' : undefined}
                rx={3}
              />
              <text
                x={x + barWidth / 2}
                y={Math.max(10, y - 5)}
                fontSize={9}
                fill={color}
                textAnchor="middle"
                fontWeight="700"
              >
                {Math.round(score)}
              </text>
              <text
                x={x + barWidth / 2}
                y={height - 12}
                fontSize={8}
                fill="var(--text-muted)"
                textAnchor="middle"
              >
                {index + 1}
              </text>
              <title>{entry.display_name || entry.graph_fingerprint}: Composite Score {score}</title>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
