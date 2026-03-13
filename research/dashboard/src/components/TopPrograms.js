import React, { useState, useMemo } from 'react';
import { scoreColor } from '../utils/format';
import { lossColor, noveltyColor, reliabilityColor } from '../utils/colors';
import { qkvUsageDescriptor, detectQkvFree } from '../utils/architecture';
import { candidateScore, candidateScoreBreakdown } from '../utils/scoringEngine';
import useCopyToClipboard from '../hooks/useCopyToClipboard';
import useRenderPerf from '../hooks/useRenderPerf';
import { filterRowsByQuery } from '../utils/tableFiltering';
import { SortableHeader, TableFilterInput } from './shared/DataTableControls';
import useInteractiveTable from './shared/useInteractiveTable';

const TOP_PROGRAMS_SORT_KEY = 'aria_top_programs_sort_v1';

/** Rate a program: green (excellent), amber (promising), red (weak) */
function programRating(p) {
  const lr = p.loss_ratio;
  const nov = p.novelty_score || 0;
  const bl = p.baseline_loss_ratio;

  // Beat the transformer baseline = green
  if (bl != null && bl < 1.0) return { color: 'var(--accent-green)', label: 'Excellent', tip: 'Outperforms a standard transformer of the same size', order: 4 };
  // Low loss ratio + high novelty
  if (lr != null && lr < 0.5 && nov > 0.7) return { color: 'var(--accent-green)', label: 'Strong', tip: 'Learns fast and is structurally novel', order: 3 };
  if (lr != null && lr < 0.6) return { color: 'var(--accent-yellow)', label: 'Promising', tip: 'Learns but hasn\'t beaten the transformer baseline yet', order: 2 };
  if (nov > 0.8) return { color: 'var(--accent-yellow)', label: 'Novel', tip: 'Very different structure but learning is modest', order: 1 };
  return { color: 'var(--accent-orange, #f0883e)', label: 'Marginal', tip: 'Passed all stages but performance is weak', order: 0 };
}

function metricText(value, fallbackReason, formatter) {
  if (value == null) return fallbackReason;
  return formatter(value);
}

function programMetricChips(program) {
  const noveltyConfidence = program.novelty_confidence;
  return [
    {
      label: 'Loss',
      source: 'measured',
      reliability: program.loss_ratio != null ? 'high' : 'low',
    },
    {
      label: 'Novelty',
      source: program.cka_source === 'artifact' ? 'artifact-backed' : 'heuristic',
      reliability: noveltyConfidence != null
        ? (noveltyConfidence >= 0.7 ? 'high' : noveltyConfidence >= 0.4 ? 'medium' : 'low')
        : 'low',
    },
    {
      label: 'Baseline',
      source: program.baseline_loss_ratio != null ? 'baseline-run' : 'not-available',
      reliability: program.baseline_loss_ratio != null ? 'medium' : 'low',
    },
  ];
}

function programQualityFlags(program) {
  const flags = [];
  if (program.cka_source === 'artifact') {
    flags.push({ label: 'CKA artifact-backed', tone: 'high' });
  } else {
    flags.push({ label: 'CKA fallback heuristic', tone: 'low' });
  }
  if (program.baseline_loss_ratio != null) {
    flags.push({ label: 'Baseline measured', tone: 'medium' });
  } else {
    flags.push({ label: 'Baseline unavailable', tone: 'low' });
  }
  const qkv = qkvUsageDescriptor(program);
  flags.push({ label: qkv.label, tone: qkv.tone, detail: qkv.detail });
  return flags;
}

const COLUMNS_FULL = [
  { key: 'score', label: 'Utility Score', title: 'Internal discovery score (0-100) based on performance, novelty, and stability.' },
  { key: 'rating', label: 'Rating', title: 'Aria\'s qualitative assessment of this candidate\'s potential.' },
  { key: 'graph_fingerprint', label: 'Program Fingerprint ID', title: 'The unique architectural identity for this candidate.' },
  { key: 'novelty_score', label: 'Novelty', title: 'How different this architecture is from known frontier models (0-1).' },
  { key: 'structural_novelty', label: 'Structural', title: 'Measures topological differences in the compute graph.' },
  { key: 'behavioral_novelty', label: 'Behavioral', title: 'Measures differences in how the model processes information (via CKA).' },
  { key: 'loss_ratio', label: 'Loss Ratio', title: 'How much the loss decreased during micro-training (lower is better).' },
  { key: 'jacobian_spectral_norm', label: 'Spectral', title: 'Jacobian Spectral Norm: gradient stability indicator (lower is better).' },
  { key: 'init_sensitivity_std', label: 'InitStd', title: 'Sensitivity to weight initialization (lower is better).' },
  { key: 'param_count', label: 'Params', title: 'Total trainable parameter count.' },
  { key: 'most_similar_to', label: 'Similar To', title: 'The known architecture most closely resembling this candidate.' },
  { key: 'throughput_tok_s', label: 'Throughput', title: 'Processing speed in tokens per second.' },
];

const COLUMNS_COMPACT = [
  { key: 'score', label: 'Utility Score', title: 'Internal discovery score (0-100).' },
  { key: 'rating', label: 'Rating', title: 'Aria\'s qualitative assessment.' },
  { key: 'graph_fingerprint', label: 'Program Fingerprint ID', title: 'Architectural identity.' },
  { key: 'novelty_score', label: 'Novelty', title: 'Architectural uniqueness.' },
  { key: 'loss_ratio', label: 'Loss Ratio', title: 'Training performance.' },
];

const PROGRAM_FINGERPRINT_HEADER_TOOLTIP = 'Architecture identity for each program row; the same fingerprint can appear multiple times when rerun.';

function TopPrograms({
  programs,
  compact,
  onSelectProgram,
  totalCount,
  onQueueAdd,
  onQueueRemove,
  queuedResultIds,
  eligibilityByResultId,
  onOpenInDesigner,
}) {
  useRenderPerf(compact ? 'TopPrograms(compact)' : 'TopPrograms');

  const [copiedValue, copyText] = useCopyToClipboard();
  const [fingerprintFilter, setFingerprintFilter] = useState('');
  const queuedSet = useMemo(() => new Set(queuedResultIds || []), [queuedResultIds]);

  const augmented = useMemo(() => {
    if (!programs) return [];
    return programs.map(p => ({
      ...p,
      _score: candidateScore(p),
      _rating: programRating(p),
    }));
  }, [programs]);

  const {
    sortKey,
    sortDesc,
    filterQuery,
    setFilterQuery,
    sortedRows: sorted,
    handleSort,
  } = useInteractiveTable({
    rows: augmented,
    filterFields: [
      'graph_fingerprint',
      'result_id',
      'architecture_name',
      'program_id',
      'experiment_id',
      'tags',
    ],
    initialSortKey: 'score',
    initialSortDesc: true,
    storageKey: TOP_PROGRAMS_SORT_KEY,
    getSortValue: (row, key) => {
      if (key === 'score') return row._score;
      if (key === 'rating') return row._rating.order;
      return row?.[key];
    },
  });

  const leadingFingerprints = useMemo(() => {
    const groups = new Map();
    for (const p of augmented) {
      const fp = String(p.graph_fingerprint || '').trim();
      if (!fp) continue;
      const current = groups.get(fp) || {
        fingerprint: fp,
        count: 0,
        bestScore: -Infinity,
        bestLossRatio: null,
        bestNovelty: null,
      };
      current.count += 1;
      if (p._score > current.bestScore) {
        current.bestScore = p._score;
        current.bestLossRatio = p.loss_ratio;
        current.bestNovelty = p.novelty_score;
      }
      groups.set(fp, current);
    }
    return [...groups.values()]
      .sort((a, b) => {
        if (b.count !== a.count) return b.count - a.count;
        return b.bestScore - a.bestScore;
      })
      .slice(0, 10);
  }, [augmented]);

  const filteredFingerprints = useMemo(() => (
    filterRowsByQuery(leadingFingerprints, fingerprintFilter, ['fingerprint'])
  ), [leadingFingerprints, fingerprintFilter]);

  const columns = compact ? COLUMNS_COMPACT : COLUMNS_FULL;
  const showRating = columns.some(col => col.key === 'rating');
  const showParamCount = columns.some(col => col.key === 'param_count');

  if (!programs || programs.length === 0) {
    return (
      <div className="card">
        <div className="card-title">Top Programs {compact ? '(Preview)' : ''}</div>
        <p style={{ color: 'var(--text-muted)', fontSize: 13 }}>
          No surviving programs yet.
        </p>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-title" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12 }}>
        <span>
          Candidate Programs (Raw Survivors) {compact
            ? `(${programs.length}${totalCount > programs.length ? ` of ${totalCount}` : ''})`
            : `— ${programs.length} Survivors`}
        </span>
        <TableFilterInput
          value={filterQuery}
          onChange={setFilterQuery}
          placeholder="Filter programs"
          ariaLabel="Filter programs"
        />
      </div>
      {!compact && (
        <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
          Raw survivor view: each row is one Stage-1-passing candidate program run.
          Program Fingerprint ID is the architecture identity for that row (not a separate object), so the same fingerprint can appear on multiple rows when rerun.
          For sortable decision-ready tiers with promotion evidence, use the <span style={{ color: 'var(--accent-blue)', textDecoration: 'underline', cursor: 'pointer' }} onClick={() => onSelectProgram && onSelectProgram('_QUALIFIED_TAB_')}>Qualified Models</span> tab.
          Lower loss ratio = learned faster. Higher novelty = more structurally different from known architectures.
          Click any row to inspect full graph and metrics.
        </p>
      )}
      {!compact && leadingFingerprints.length > 0 && (
        <div style={{ marginBottom: 12, border: '1px solid var(--border)', borderRadius: 6, padding: 8 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12, marginBottom: 6 }}>
            <div style={{ fontSize: 12, fontWeight: 600 }}>Fingerprint Leaderboard (Deduplicated Architecture IDs)</div>
            <TableFilterInput
              value={fingerprintFilter}
              onChange={setFingerprintFilter}
              placeholder="Filter fingerprints"
              ariaLabel="Filter fingerprints"
            />
          </div>
          <table className="data-table" style={{ margin: 0 }}>
            <thead>
              <tr>
                <th>Fingerprint (ID)</th>
                <th>Appearances</th>
                <th>Best Utility Score</th>
                <th>Best Loss</th>
                <th>Best Novelty</th>
              </tr>
            </thead>
            <tbody>
              {filteredFingerprints.slice(0, 6).map((row) => (
                <tr key={row.fingerprint}>
                  <td style={{ fontFamily: 'monospace', fontSize: 12 }}>{row.fingerprint.slice(0, 10)}</td>
                  <td>{row.count}</td>
                  <td>{Number(row.bestScore || 0).toFixed(0)}</td>
                  <td>{row.bestLossRatio != null ? Number(row.bestLossRatio).toFixed(4) : '--'}</td>
                  <td>{row.bestNovelty != null ? Number(row.bestNovelty).toFixed(3) : '--'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <table className="data-table">
        <thead>
          <tr>
            {columns.map(col => (
              <SortableHeader
                key={col.key}
                sortKey={col.key}
                activeSortKey={sortKey}
                sortDesc={sortDesc}
                onSort={handleSort}
                label={col.label}
                title={col.title || (col.key === 'graph_fingerprint' ? PROGRAM_FINGERPRINT_HEADER_TOOLTIP : undefined)}
              />
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map((p, i) => {
            const rating = p._rating;
            const score = p._score;
            const chips = programMetricChips(p);
            const qualityFlags = programQualityFlags(p);
            const isQueued = !!p.result_id && queuedSet.has(p.result_id);
            const eligibility = p.result_id ? (eligibilityByResultId?.[p.result_id] || null) : null;
            const queueEligible = eligibility ? Boolean(eligibility.queueEligible) : true;
            const queueIntent = eligibility?.validationEligible
              ? 'validation'
              : eligibility?.investigationEligible
                ? 'investigation'
                : null;
            const queueAddLabel = queueIntent === 'validation' ? 'Queue Validate' : 'Queue Investigate';
            const queueAddTitle = queueIntent === 'validation'
              ? 'Add to validation queue'
              : 'Add to investigation queue';
            return (
              <tr key={p.result_id || i}
                style={{ cursor: onSelectProgram ? 'pointer' : 'default' }}
                onClick={() => onSelectProgram && onSelectProgram(p.result_id)}>
                <td style={{ fontWeight: 600, color: scoreColor(score) }}>
                  <span title={Object.entries(candidateScoreBreakdown(p)).map(([k, v]) => `${k} ${Number(v || 0).toFixed(1)}`).join(' | ')}>
                    {score}
                  </span>
                </td>
                {showRating && (
                  <td title={rating.tip}>
                    <span style={{
                      display: 'inline-block', width: 10, height: 10, borderRadius: '50%',
                      background: rating.color, marginRight: 6,
                    }} />
                    <span style={{ fontSize: 11, color: rating.color }}>{rating.label}</span>
                  </td>
                )}
                <td
                  style={{
                    fontFamily: 'monospace',
                    fontSize: 12,
                    color: onSelectProgram ? 'var(--accent-blue)' : 'inherit',
                    whiteSpace: 'normal',
                    overflowWrap: 'anywhere',
                    lineHeight: 1.35,
                    minWidth: 160,
                  }}
                  title={p.graph_fingerprint || 'not available'}
                >
                  {p.graph_fingerprint || '--'}
                  {p.graph_fingerprint && (
                    <button
                      className="refresh-btn"
                      style={{ fontSize: 10, padding: '1px 5px', marginLeft: 6 }}
                      onClick={(e) => {
                        e.stopPropagation();
                        copyText(p.graph_fingerprint);
                      }}
                      aria-label={`Copy fingerprint ${p.graph_fingerprint}`}
                    >
                      {copiedValue === p.graph_fingerprint ? 'Copied' : 'Copy'}
                    </button>
                  )}
                  {p.result_id && (
                    <button
                      className="refresh-btn"
                      style={{ fontSize: 10, padding: '1px 5px', marginLeft: 4 }}
                      onClick={(e) => {
                        e.stopPropagation();
                        copyText(p.result_id);
                      }}
                      aria-label={`Copy result id ${p.result_id}`}
                    >
                      {copiedValue === p.result_id ? 'Copied ID' : 'Copy ID'}
                    </button>
                  )}
                  {p.result_id && (onQueueAdd || onQueueRemove) && (
                    <button
                      className="refresh-btn"
                      style={{
                        fontSize: 10, padding: '1px 5px', marginLeft: 4,
                        opacity: !isQueued && !queueEligible ? 0.5 : 1,
                        cursor: !isQueued && !queueEligible ? 'not-allowed' : 'pointer',
                      }}
                      onClick={(e) => {
                        e.stopPropagation();
                        if (isQueued) {
                          onQueueRemove && onQueueRemove(p.result_id);
                          return;
                        }
                        if (!queueEligible) {
                          return;
                        }
                        onQueueAdd && onQueueAdd({
                          resultId: p.result_id,
                          fingerprint: p.graph_fingerprint,
                          source: 'programs',
                          architectureFamily: p.architecture_family,
                          intent: queueIntent,
                          queueEligible,
                          investigationEligible: eligibility?.investigationEligible,
                          validationEligible: eligibility?.validationEligible,
                          queueReason: eligibility?.queueReason,
                        });
                      }}
                      disabled={!isQueued && !queueEligible}
                      aria-label={`${isQueued ? 'Remove' : 'Add'} ${p.result_id} ${isQueued ? 'from' : 'to'} investigation queue`}
                      title={isQueued
                        ? 'Remove from investigation queue'
                        : !queueEligible
                          ? (p.tier === 'validation' || p.tier === 'breakthrough' 
                              ? 'Architecture is fully validated.' 
                              : 'Not eligible for investigation/validation queue actions')
                          : queueAddTitle}
                    >
                      {isQueued 
                        ? 'Queued' 
                        : !queueEligible 
                          ? (p.tier === 'validation' || p.tier === 'breakthrough' ? 'Validated' : 'Ineligible') 
                          : queueAddLabel}
                    </button>
                  )}
                  {p.result_id && onOpenInDesigner && (
                    <button
                      className="refresh-btn"
                      style={{
                        fontSize: 10,
                        padding: '1px 5px',
                        marginLeft: 4,
                        borderColor: 'var(--accent-purple)',
                        color: 'var(--accent-purple)',
                      }}
                      onClick={(e) => {
                        e.stopPropagation();
                        onOpenInDesigner(p.result_id);
                      }}
                      aria-label={`Open ${p.result_id} in designer`}
                      title="Open architecture in visual designer"
                    >
                      Designer
                    </button>
                  )}
                </td>
                <td>
                  <span style={{ color: noveltyColor(p.novelty_score) }}>
                    {metricText(p.novelty_score, 'not computed', (v) => v.toFixed(3))}
                  </span>
                </td>
                {!compact && <td>{p.structural_novelty?.toFixed(3) || '--'}</td>}
                {!compact && <td>{p.behavioral_novelty?.toFixed(3) || '--'}</td>}
                <td style={{ color: lossColor(p.loss_ratio) }}>
                  {metricText(p.loss_ratio, 'not computed', (v) => v.toFixed(4))}
                  {!compact && (
                    <div style={{ marginTop: 4, display: 'flex', gap: 4, flexWrap: 'wrap', maxWidth: 220 }}>
                      {chips.map(chip => (
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
                      {qualityFlags.map(flag => (
                        <span
                          key={`${p.result_id || i}-${flag.label}`}
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
                  )}
                </td>
                {columns.some(c => c.key === 'jacobian_spectral_norm') && (
                  <td>{metricText(p.jacobian_spectral_norm ?? p.fp_jacobian_spectral_norm, '--', (v) => v.toFixed(4))}</td>
                )}
                {columns.some(c => c.key === 'init_sensitivity_std') && (
                  <td>{metricText(p.init_sensitivity_std, '--', (v) => v.toFixed(4))}</td>
                )}
                {showParamCount && (
                  <td>{p.param_count ? `${(p.param_count / 1e6).toFixed(1)}M` : 'not available'}</td>
                )}
                {!compact && <td title={p.most_similar_to || 'not available'}>{p.most_similar_to || '--'}</td>}
                {!compact && <td>{p.throughput_tok_s ? `${Number(p.throughput_tok_s).toFixed(0)} tok/s` : 'not measured'}</td>}
              </tr>
            );
          })}
        </tbody>
      </table>
      {!compact && (
        <div style={{ marginTop: 8, fontSize: 11, color: 'var(--text-muted)' }}>
          Tip: use Copy on fingerprint cells to reuse IDs in validation/investigation workflows.
        </div>
      )}
      {!compact && (
        <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 8, display: 'flex', gap: 16 }}>
          <span><span style={{ color: 'var(--accent-green)' }}>Green</span> = outperforms transformer or high novelty + fast learning</span>
          <span><span style={{ color: 'var(--accent-yellow)' }}>Amber</span> = promising but hasn't beaten baseline</span>
          <span>Loss ratio: lower = better (how much loss decreased during training)</span>
        </div>
      )}
      {onSelectProgram && (
        <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: compact ? 8 : 0, textAlign: 'right' }}>
          Click a row to view program details
        </div>
      )}
    </div>
  );
}

export default TopPrograms;
