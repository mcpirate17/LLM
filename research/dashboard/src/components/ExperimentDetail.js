import { apiCall } from "../services/apiService";
import React, { useState, useEffect, useMemo } from 'react';
import FailureAnalysis from './FailureAnalysis';
import ProgramDetail from './ProgramDetail';
import { SCORE_MAX, formatTime, formatDuration, scoreColor, scoreGradient, scoreToneLabel } from '../utils/format';
import { lossColor, noveltyColor } from '../utils/colors';
import useInteractiveTable from './shared/useInteractiveTable';
import SortIndicator from './shared/SortIndicator';
import useResizableColumns from './shared/useResizableColumns';


/**
 * ExperimentDetail — Full experiment breakdown with hypothesis, funnel,
 * all programs table, failure analysis, and Aria's LLM analysis.
 */

/** Extract a compact architecture summary from API-provided ops_summary or category histogram. */
function extractArchSummary(p) {
  // Backend extracts unique op names and sends as ops_summary
  if (p.ops_summary) return p.ops_summary;
  // Fall back to category histogram if ops_summary is missing
  const hist = p.graph_category_histogram;
  if (hist) {
    try {
      const cats = typeof hist === 'string' ? JSON.parse(hist) : hist;
      const parts = Object.entries(cats)
        .filter(([, v]) => v > 0)
        .sort((a, b) => b[1] - a[1])
        .map(([k, v]) => `${v}\u00d7${k.replace(/_/g, ' ')}`);
      if (parts.length > 0) return parts.join(' ');
    } catch { /* fall through */ }
  }
  return null;
}

/** Rate a program row: green (learned well), amber (compiles), red (failed early) */
function programRowRating(p) {
  if (p.stage1_passed) {
    if (p.baseline_loss_ratio != null && p.baseline_loss_ratio < 1.0)
      return { color: 'var(--accent-green)', label: 'Excellent', tip: 'Outperforms a standard transformer of the same size' };
    if ((p.loss_ratio || 1) < 0.5)
      return { color: 'var(--accent-green)', label: 'Strong', tip: 'Learned quickly — loss dropped significantly' };
    return { color: 'var(--accent-yellow)', label: 'Learned', tip: 'Passed Stage 1 — demonstrated learning ability' };
  }
  if (p.stage05_passed)
    return { color: 'var(--accent-orange, #f0883e)', label: 'Stable', tip: 'Numerically stable but didn\'t learn — gradient signal too weak' };
  if (p.stage0_passed)
    return { color: 'var(--accent-orange, #f0883e)', label: 'Compiled', tip: 'Compiled and ran but produced NaN or unstable gradients' };
  return { color: 'var(--accent-red)', label: 'Failed', tip: 'Failed to compile or crashed — invalid operation combination', order: 0 };
}

function progScoreColor(score) {
  if (score == null) return 'var(--text-muted)';
  return scoreColor(score);
}

function metricTone(value, kind) {
  const n = Number(value);
  if (!Number.isFinite(n)) return { tone: 'missing', color: 'var(--text-muted)', label: '--' };
  if (kind === 'loss') {
    if (n < 0.7) return { tone: 'positive', color: 'var(--accent-green)', label: 'positive' };
    if (n < 1.0) return { tone: 'neutral', color: 'var(--accent-yellow)', label: 'neutral' };
    return { tone: 'negative', color: 'var(--accent-red)', label: 'negative' };
  }
  if (kind === 'baseline') {
    if (n < 1.0) return { tone: 'positive', color: 'var(--accent-green)', label: 'beats baseline' };
    if (n < 1.1) return { tone: 'neutral', color: 'var(--accent-yellow)', label: 'near baseline' };
    return { tone: 'negative', color: 'var(--accent-red)', label: 'below baseline' };
  }
  if (kind === 'novelty') {
    if (n >= 0.8) return { tone: 'positive', color: 'var(--accent-green)', label: 'novel' };
    if (n >= 0.5) return { tone: 'neutral', color: 'var(--accent-yellow)', label: 'mixed' };
    return { tone: 'negative', color: 'var(--text-muted)', label: 'familiar' };
  }
  return { tone: 'neutral', color: 'var(--text-muted)', label: 'recorded' };
}

function canonicalScore(row) {
  if (row?.composite_score == null) return null;
  const score = Number(row.composite_score);
  return Number.isFinite(score) ? score : null;
}

const PROG_COLUMNS = [
  { key: '_score', label: 'Score', initWidth: 52 },
  { key: 'rating', label: 'Rating', initWidth: 80 },
  { key: 'graph_fingerprint', label: 'Fingerprint', initWidth: 90 },
  { key: '_arch', label: 'Architecture', initWidth: 220 },
  { key: 'stage0_passed', label: 'S0', initWidth: 32 },
  { key: 'stage05_passed', label: 'S0.5', initWidth: 36 },
  { key: 'stage1_passed', label: 'S1', initWidth: 32 },
  { key: 'novelty_score', label: 'Novelty', initWidth: 60 },
  { key: 'loss_ratio', label: 'Loss Ratio', initWidth: 72 },
  { key: 'param_count', label: 'Params', initWidth: 56 },
  { key: 'peak_memory_mb', label: 'Memory', initWidth: 64 },
  { key: 'flops_forward', label: 'FLOPs', initWidth: 56 },
  { key: 'baseline_loss_ratio', label: 'Baseline', initWidth: 64 },
];

const EXPERIMENT_DETAIL_PROGRAM_SORT_PREFS_KEY = 'dashboard.experiment-detail.programs.sort.v1';

const ROW_RATING_ORDER = { Excellent: 5, Strong: 4, Learned: 3, Stable: 2, Compiled: 1, Failed: 0 };

function FunnelViz({ experiment }) {
  const stages = [
    { label: 'Generated', value: experiment.n_programs_generated || 0, color: 'var(--accent-blue)', icon: 'Σ' },
    { label: 'Compiled', value: experiment.n_stage0_passed || 0, color: 'var(--accent-green)', icon: '✓' },
    { label: 'Stable', value: experiment.n_stage05_passed || 0, color: 'var(--accent-yellow)', icon: '±' },
    { label: 'Learned', value: experiment.n_stage1_passed || 0, color: 'var(--accent-purple)', icon: '★' },
  ];

  const max = stages[0].value || 1;

  return (
    <div style={{ display: 'flex', gap: 12, alignItems: 'center', background: 'var(--bg-tertiary)', padding: '16px 20px', borderRadius: 8, border: '1px solid var(--border)' }}>
      {stages.map((stage, i) => {
        const percent = Math.round((stage.value / max) * 100);
        return (
          <React.Fragment key={i}>
            {i > 0 && <div style={{ color: 'var(--text-muted)', fontSize: 20 }}>&rarr;</div>}
            <div style={{ flex: 1, textAlign: 'center' }}>
              <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 4 }}>{stage.label}</div>
              <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'center', gap: 4 }}>
                <div style={{ fontSize: 22, fontWeight: 800, color: 'var(--text-primary)' }}>{stage.value}</div>
                {i > 0 && <div style={{ fontSize: 12, color: stage.color }}>{percent}%</div>}
              </div>
              <div style={{ height: 4, background: 'var(--border)', borderRadius: 2, marginTop: 8, overflow: 'hidden' }}>
                <div style={{ width: `${percent}%`, height: '100%', background: stage.color }} />
              </div>
            </div>
          </React.Fragment>
        );
      })}
    </div>
  );
}

function ExperimentSummaryHeader({ experiment, programs }) {
  const bestProgram = useMemo(() => {
    if (!programs || programs.length === 0) return null;
    return [...programs].sort((a, b) => {
      const aScore = canonicalScore(a);
      const bScore = canonicalScore(b);
      if (aScore == null && bScore == null) return 0;
      if (aScore == null) return 1;
      if (bScore == null) return -1;
      return bScore - aScore;
    })[0];
  }, [programs]);

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 16, marginBottom: 16 }}>
      <div className="card" style={{ borderLeft: '4px solid var(--accent-blue)' }}>
        <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 4 }}>RESEARCH HYPOTHESIS</div>
        <div style={{ fontSize: 16, fontWeight: 600, color: 'var(--text-primary)', lineHeight: 1.4 }}>
          {experiment.hypothesis || "No hypothesis recorded for this session."}
        </div>
        {experiment.aria_summary && (
          <div style={{ marginTop: 12, fontSize: 13, color: 'var(--text-secondary)', borderTop: '1px solid var(--border)', paddingTop: 12 }}>
            <strong>Outcome:</strong> {experiment.aria_summary}
          </div>
        )}
      </div>

      {bestProgram && (
        <div className="card" style={{ borderLeft: `4px solid ${scoreColor(canonicalScore(bestProgram))}`, background: 'linear-gradient(135deg, var(--bg-secondary) 0%, rgba(63, 185, 80, 0.05) 100%)' }}>
          <div style={{ fontSize: 10, color: scoreColor(canonicalScore(bestProgram)), fontWeight: 700, marginBottom: 4 }}>TOP DISCOVERY</div>
          <div style={{ fontSize: 14, fontWeight: 700, fontFamily: 'monospace' }}>{bestProgram.graph_fingerprint?.slice(0, 12)}</div>
          <div style={{ marginTop: 8, display: 'flex', flexWrap: 'wrap', gap: 8 }}>
            <div style={{ fontSize: 11 }}>
              <span style={{ color: 'var(--text-muted)' }}>Loss Ratio:</span> 
              <span style={{ color: 'var(--accent-green)', marginLeft: 4, fontWeight: 600 }}>{bestProgram.loss_ratio?.toFixed(4)}</span>
            </div>
            <div style={{ fontSize: 11 }}>
              <span style={{ color: 'var(--text-muted)' }}>Novelty:</span> 
              <span style={{ color: 'var(--accent-purple)', marginLeft: 4, fontWeight: 600 }}>{bestProgram.novelty_score?.toFixed(3)}</span>
            </div>
          </div>
          <div style={{ marginTop: 10, fontSize: 10, color: 'var(--text-muted)' }}>
            Score: <span style={{ color: scoreColor(canonicalScore(bestProgram)), fontWeight: 700 }}>{canonicalScore(bestProgram)?.toFixed(1) ?? '—'}</span>
            {canonicalScore(bestProgram) != null && (
              <span style={{ marginLeft: 6, color: 'var(--text-muted)' }}>{scoreToneLabel(canonicalScore(bestProgram))}</span>
            )}
          </div>
          {canonicalScore(bestProgram) != null && (
            <div className="champion-strip" style={{ marginTop: 8 }}>
              <div
                className="champion-strip-fill"
                style={{
                  width: `${Math.max(4, Math.min(100, (canonicalScore(bestProgram) / SCORE_MAX) * 100))}%`,
                  background: scoreGradient(canonicalScore(bestProgram)),
                }}
              />
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function ExperimentSignalStrip({ programs }) {
  const total = programs.length || 0;
  const learned = programs.filter(p => p.stage1_passed).length;
  const baselineWins = programs.filter(p => p.baseline_loss_ratio != null && p.baseline_loss_ratio < 1).length;
  const scored = programs.map(canonicalScore).filter(v => v != null);
  const bestScore = scored.length ? Math.max(...scored) : null;
  const strongLoss = programs.filter(p => p.loss_ratio != null && Number(p.loss_ratio) < 0.7).length;
  const cells = [
    { label: 'Learned', value: total ? `${learned}/${total}` : '--', color: learned > 0 ? 'var(--accent-green)' : 'var(--text-muted)', hint: 'Stage 1 pass count' },
    { label: 'Baseline Wins', value: baselineWins, color: baselineWins > 0 ? 'var(--accent-green)' : 'var(--text-muted)', hint: 'Baseline ratio < 1.0' },
    { label: 'Strong Loss', value: strongLoss, color: strongLoss > 0 ? 'var(--accent-green)' : 'var(--text-muted)', hint: 'Loss ratio < 0.7' },
    { label: 'Best Score', value: bestScore != null ? bestScore.toFixed(1) : '--', color: scoreColor(bestScore), hint: 'Highest composite score in this experiment' },
  ];
  return (
    <div className="card" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: 12 }}>
      {cells.map(cell => (
        <div key={cell.label} title={cell.hint} style={{ minWidth: 0 }}>
          <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 4 }}>{cell.label}</div>
          <div style={{ fontSize: 20, fontWeight: 750, color: cell.color, fontVariantNumeric: 'tabular-nums' }}>{cell.value}</div>
        </div>
      ))}
    </div>
  );
}
const PROG_FILTER_FIELDS = [
  'graph_fingerprint',
  'result_id',
  'architecture_name',
  'program_id',
  'notes',
  '_arch',
];

function getProgSortValue(row, key) {
  if (key === '_score') return canonicalScore(row);
  if (key === 'rating') return ROW_RATING_ORDER[programRowRating(row).label] || 0;
  return row[key];
}

function ProgramsTable({ programs, onSelectProgram }) {
  const defaultColWidths = useMemo(
    () => Object.fromEntries(PROG_COLUMNS.map((c) => [c.key, c.initWidth])),
    []
  );
  const {
    columnWidths: storedColWidths,
    onResizeStart,
    activeResizeKey,
  } = useResizableColumns(EXPERIMENT_DETAIL_PROGRAM_SORT_PREFS_KEY + '.widths');
  const colWidths = useMemo(
    () => ({ ...defaultColWidths, ...storedColWidths }),
    [defaultColWidths, storedColWidths]
  );

  // Pre-compute architecture summaries so they're available for filtering
  const augmented = useMemo(() =>
    programs.map(p => ({ ...p, _arch: extractArchSummary(p) })),
    [programs]
  );

  const { sortKey, sortDesc, filterQuery, setFilterQuery, sortedRows, handleSort } = useInteractiveTable({
    rows: augmented,
    filterFields: PROG_FILTER_FIELDS,
    initialSortKey: '_score',
    initialSortDesc: true,
    storageKey: EXPERIMENT_DETAIL_PROGRAM_SORT_PREFS_KEY,
    getSortValue: getProgSortValue,
  });

  const sorted = useMemo(() =>
    sortedRows.map(p => ({ ...p, _score: canonicalScore(p), _rating: programRowRating(p) })),
    [sortedRows]
  );

  const cellStyle = (key) => ({
    width: colWidths[key],
    minWidth: 28,
    maxWidth: colWidths[key],
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    wordBreak: key === 'graph_fingerprint' || key === '_arch' ? 'break-all' : undefined,
    whiteSpace: key === 'graph_fingerprint' || key === '_arch' ? 'normal' : 'nowrap',
  });

  return (
    <div className="card">
      <div className="card-title" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12 }}>
        <span>All Programs ({programs.length})</span>
        <input
          value={filterQuery}
          onChange={(e) => setFilterQuery(e.target.value)}
          placeholder="Filter programs"
          className="filter-input"
        />
      </div>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 8, lineHeight: 1.5 }}>
        Every architecture tested in this experiment. P = passed, F = failed at that stage.
        Baseline {'<'} 1.0 means it outperformed a standard transformer of the same size.
        Click any row for the full computation graph and detailed metrics.
        Drag column borders to resize.
      </p>
      <div style={{ maxHeight: 400, overflow: 'auto' }}>
        <table className="data-table" style={{ tableLayout: 'fixed', width: 'max-content', minWidth: '100%' }}>
          <thead>
            <tr>
              {PROG_COLUMNS.map(col => (
                <th
                  key={col.key}
                  style={{
                    cursor: 'pointer',
                    userSelect: 'none',
                    position: 'relative',
                    width: colWidths[col.key],
                    minWidth: 28,
                    maxWidth: colWidths[col.key],
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                  }}
                  onClick={() => handleSort(col.key)}
                >
                  {col.label}
                  <SortIndicator active={sortKey === col.key} desc={sortDesc} />
                  {/* Resize handle */}
                  <span
                    onMouseDown={(e) => onResizeStart(e, col.key)}
                    style={{
                      position: 'absolute',
                      right: 0,
                      top: 0,
                      bottom: 0,
                      width: 5,
                      cursor: 'col-resize',
                      background: activeResizeKey === col.key ? 'var(--accent-blue)' : 'transparent',
                    }}
                    onClick={(e) => e.stopPropagation()}
                  />
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sorted.map((p, i) => {
              const rating = p._rating;
              return (
                <tr key={p.result_id || i}
                  style={{ cursor: 'pointer' }}
                  onClick={() => onSelectProgram && onSelectProgram(p.result_id)}>
                  <td style={{ ...cellStyle('_score'), fontWeight: 600, color: progScoreColor(p._score) }}>
                    {p._score != null ? Number(p._score).toFixed(1) : '--'}
                  </td>
                  <td style={cellStyle('rating')} title={rating.tip}>
                    <span style={{
                      display: 'inline-block', width: 10, height: 10, borderRadius: '50%',
                      background: rating.color, marginRight: 6, verticalAlign: 'middle',
                    }} />
                    <span style={{ fontSize: 11, color: rating.color }}>{rating.label}</span>
                  </td>
                  <td style={{ ...cellStyle('graph_fingerprint'), fontFamily: 'monospace', fontSize: 11, color: 'var(--accent-blue)' }}>
                    {p.graph_fingerprint?.slice(0, 10) || '--'}
                  </td>
                  <td style={{ ...cellStyle('_arch'), fontSize: 11, color: 'var(--text-secondary)' }}
                      title={p._arch || ''}>
                    {p._arch || '--'}
                  </td>
                  <td style={cellStyle('stage0_passed')}><span className={`badge ${p.stage0_passed ? 'pass' : 'fail'}`}>{p.stage0_passed ? 'P' : 'F'}</span></td>
                  <td style={cellStyle('stage05_passed')}><span className={`badge ${p.stage05_passed ? 'pass' : 'fail'}`}>{p.stage05_passed ? 'P' : 'F'}</span></td>
                  <td style={cellStyle('stage1_passed')}><span className={`badge ${p.stage1_passed ? 'pass' : 'fail'}`}>{p.stage1_passed ? 'P' : 'F'}</span></td>
                  <td
                    style={{ ...cellStyle('novelty_score'), color: metricTone(p.novelty_score, 'novelty').color }}
                    title={p.novelty_score != null ? `${metricTone(p.novelty_score, 'novelty').label}: higher is more novel` : undefined}
                  >
                    {p.novelty_score?.toFixed(3) || '--'}
                  </td>
                  <td
                    style={{ ...cellStyle('loss_ratio'), color: lossColor(p.loss_ratio), fontWeight: p.loss_ratio != null && Number(p.loss_ratio) < 0.7 ? 650 : 400 }}
                    title={p.loss_ratio != null ? `${metricTone(p.loss_ratio, 'loss').label}: lower is better` : undefined}
                  >
                    {p.loss_ratio?.toFixed(4) || '--'}
                  </td>
                  <td style={cellStyle('param_count')}>{p.param_count ? `${(p.param_count / 1e6).toFixed(1)}M` : '--'}</td>
                  <td style={{ ...cellStyle('peak_memory_mb'), fontSize: 11 }}>{p.peak_memory_mb ? `${Number(p.peak_memory_mb).toFixed(0)}MB` : '--'}</td>
                  <td style={{ ...cellStyle('flops_forward'), fontSize: 11 }}>{p.flops_forward ? `${(p.flops_forward / 1e6).toFixed(1)}M` : '--'}</td>
                  <td style={{
                    ...cellStyle('baseline_loss_ratio'),
                    fontSize: 11,
                    fontWeight: p.baseline_loss_ratio != null && p.baseline_loss_ratio < 1 ? 600 : 'normal',
                    color: metricTone(p.baseline_loss_ratio, 'baseline').color,
                  }}
                    title={p.baseline_loss_ratio != null ? `${metricTone(p.baseline_loss_ratio, 'baseline').label}: < 1.0 beats baseline` : undefined}
                  >
                    {p.baseline_loss_ratio?.toFixed(3) || '--'}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 8, display: 'flex', gap: 16, flexWrap: 'wrap' }}>
        <span><span style={{ color: 'var(--accent-green)' }}>Green</span> = learned from data (S1 pass)</span>
        <span><span style={{ color: 'var(--accent-yellow)' }}>Amber</span> = passed learning stage</span>
        <span><span style={{ color: 'var(--accent-orange, #f0883e)' }}>Orange</span> = compiled but didn't learn</span>
        <span><span style={{ color: 'var(--accent-red)' }}>Red</span> = failed to compile</span>
      </div>
    </div>
  );
}

function ExperimentDetail({ experimentId, onBack, onSelectProgram }) {
  const [data, setData] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedProgramId, setSelectedProgramId] = useState(null);
  const [isRerunning, setIsRerunning] = useState(false);
  const [rerunConfirm, setRerunConfirm] = useState(false);

  useEffect(() => {
    if (!experimentId) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    setData(null);
    setAnalysis(null);

    apiCall(`/api/experiments/${experimentId}`).then(r => {
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return r.json();
    }).then((expData) => {
      if (cancelled) return;
      setData(expData);
      setLoading(false);
    }).catch(e => {
      if (cancelled) return;
      const msg = e.message.includes('404')
        ? 'This experiment could not be found. It may have been deleted.'
        : e.message.includes('500')
        ? 'Server error while loading experiment. Try again later.'
        : 'Failed to load experiment: ' + e.message;
      setError(msg);
      setLoading(false);
    });

    apiCall(`/api/experiments/${experimentId}/analysis`)
      .then(r => r.ok ? r.json() : null)
      .then((analysisData) => {
        if (!cancelled) setAnalysis(analysisData);
      })
      .catch(() => {
        if (!cancelled) setAnalysis(null);
      });

    return () => {
      cancelled = true;
    };
  }, [experimentId]);

  if (loading) return <div className="card"><p style={{ color: 'var(--text-muted)' }}>Loading experiment...</p></div>;
  if (error) return <div className="card"><p style={{ color: 'var(--accent-red)' }}>{error}</p></div>;
  if (!data || !data.experiment) return <div className="card"><p style={{ color: 'var(--accent-red)' }}>Experiment not found</p></div>;

  const exp = data.experiment;
  const programs = data.programs || [];
  const entries = data.entries || [];
  const prereg = data.preregistration || null;
  const preregDeviations = data.preregistration_deviations || [];

  const handleRerun = async () => {
    if (!rerunConfirm) {
      setRerunConfirm(true);
      setTimeout(() => setRerunConfirm(false), 3000);
      return;
    }
    setIsRerunning(true);
    try {
      const res = await apiCall(`/api/experiments/${experimentId}/rerun`, { method: 'POST' });
      const payload = await res.json();
      if (!res.ok) throw new Error(payload.error || 'Failed to rerun');
      // On success, go back to command view which shows the active run
      if (onBack) onBack();
    } catch (err) {
      alert(err.message);
    } finally {
      setIsRerunning(false);
      setRerunConfirm(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {/* Header Info */}
      <div className="card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap', minWidth: 0 }}>
            <button className="refresh-btn" onClick={() => onBack && onBack()} style={{ marginRight: 12 }}>&larr; Back</button>
            <span style={{ fontFamily: 'monospace', color: 'var(--accent-blue)', marginRight: 8, overflowWrap: 'anywhere' }}>{experimentId}</span>
            <span className={`badge ${exp.status === 'completed' ? 'pass' : exp.status === 'running' ? 'running' : 'fail'}`}>
              {exp.status}
            </span>
            <button
              className="refresh-btn"
              style={{
                marginLeft: 12,
                fontSize: 11,
                borderColor: rerunConfirm ? 'var(--accent-yellow)' : 'var(--border)',
                color: rerunConfirm ? 'var(--accent-yellow)' : 'inherit',
              }}
              disabled={isRerunning || exp.status === 'running'}
              onClick={handleRerun}
            >
              {isRerunning ? 'Starting...' : rerunConfirm ? 'Click to confirm Rerun' : 'Rerun Experiment'}
            </button>
          </div>
          <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
            {formatTime(exp.timestamp)} | {formatDuration(exp.duration_seconds)}
          </div>
        </div>
      </div>

      <ExperimentSummaryHeader experiment={exp} programs={programs} />
      <ExperimentSignalStrip programs={programs} />

      {/* Funnel */}
      <div className="card">
        <div className="card-title">Evaluation Funnel</div>
        <FunnelViz experiment={exp} />
      </div>

      {/* LLM Analysis */}
      {analysis?.analysis && (
        <div className="card">
          <div className="card-title">
            Deep Analysis
            <span className="badge novel" style={{ marginLeft: 8, fontSize: 10 }}>
              {analysis.source === 'stored' ? 'cached' : 'live'}
            </span>
          </div>
          <div style={{ fontSize: 13, color: 'var(--text-secondary)', whiteSpace: 'pre-wrap', lineHeight: 1.6 }}>
            {analysis.analysis}
          </div>
        </div>
      )}

      {/* Insights */}
      <div className="card">
        <div className="card-title">Insights</div>
        <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
          Patterns and takeaways Aria extracted from this experiment's results.
        </p>
        {exp.insights && exp.insights.length > 0 ? (
          exp.insights.map((insight, i) => (
            <div key={i} className="insight-card">
              <div className="insight-content">{insight}</div>
            </div>
          ))
        ) : (
          <p style={{ fontSize: 12, color: 'var(--text-muted)', fontStyle: 'italic' }}>
            No insights generated for this experiment yet.
          </p>
        )}
      </div>

      {/* Two-column: Failure Analysis + Programs Table */}
      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(260px, 1fr) minmax(0, 2fr)', gap: 16 }}>
        <FailureAnalysis experimentId={experimentId} />

        {/* Programs Table */}
        <ProgramsTable
          programs={programs}
          onSelectProgram={setSelectedProgramId}
        />
      </div>

      {/* Notebook Entries */}
      <div className="card">
        <div className="card-title">Notebook Entries</div>
        <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
          Chronological log of observations, decisions, and errors recorded during this experiment.
        </p>
        {entries.length > 0 ? (
          entries.slice(0, 10).map((entry, i) => (
            <div key={i} className={`notebook-entry ${entry.entry_type}`}>
              <div className="entry-header">
                <span className="entry-title">{entry.title}</span>
                <span className="entry-type">{entry.entry_type}</span>
              </div>
              <div className="entry-content">{entry.content}</div>
            </div>
          ))
        ) : (
          <p style={{ fontSize: 12, color: 'var(--text-muted)', fontStyle: 'italic' }}>
            No notebook entries recorded for this experiment.
          </p>
        )}
      </div>

      {/* Program Detail Modal */}
      {selectedProgramId && (
        <ProgramDetail
          resultId={selectedProgramId}
          onClose={() => setSelectedProgramId(null)}
        />
      )}
    </div>
  );
}

export default ExperimentDetail;
